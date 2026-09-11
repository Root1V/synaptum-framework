"""Conformidad · costura de durabilidad.

Corre los **casos dorados compartidos** contra nuestras dos implementaciones del
``Checkpointer``. Los casos los publica Aeon en `contratos/costura-durabilidad`;
aquí no se copian ni se ajustan — se leen del fichero y se ejecutan tal cual.

El runner lo trae cada proyecto, que es la parte del acuerdo que importa: si dos
implementaciones sin una línea de código en común reproducen los mismos
resultados observables, la equivalencia deja de ser una afirmación. Por eso los
casos describen resultados y nunca internos.

Si los contratos no están montados, la suite se salta con un aviso — no se
inventa un veredicto verde.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from synaptum import (
    Decision,
    Disposition,
    FinishReason,
    MemoryCheckpointer,
    ModelStep,
    Phase,
    RunState,
    SqliteCheckpointer,
    ToolStep,
    UncertainEffect,
    make_step_id,
)

_CONTRACTS = Path(
    "/Users/emericespiritusantiago/Documents/Victor/coordinacion_project/contratos"
)
_FIXTURES = _CONTRACTS / "costura-durabilidad" / "fixtures"

pytestmark = pytest.mark.skipif(
    not _FIXTURES.exists(), reason="carpeta de contratos compartidos no disponible"
)


def _load_cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    for path in sorted(_FIXTURES.glob("*.json")):
        document = json.loads(path.read_text())
        for case in document.get("cases", []):
            cases.append((f"{path.stem}::{case['name']}", case))
    return cases


CASES = _load_cases()


def _event(step_id: str, phase: str, payload: dict | None) -> ModelStep:
    """Traduce una operación del caso a nuestro evento tipado.

    El caso habla de ``step_id``, ``phase`` y un ``payload`` opaco.  Nuestro
    evento es más rico, así que el payload viaja en ``meta`` — que es
    precisamente el campo que se propaga sin interpretarse.
    """
    return ModelStep(
        run_id="conformance",
        step_id=step_id,
        step_seq=0,
        phase=Phase(phase),
        meta=payload if payload is not None else {},
    )


async def _run_case(store, case: dict, run_id: str) -> None:
    for index, operation in enumerate(case["operations"]):
        where = f"{case['name']} · operación {index}"
        kind = operation["op"]

        if kind == "append":
            if operation.get("expect_error"):
                with pytest.raises(ValueError):
                    _event(operation["step_id"], operation["phase"], operation.get("payload"))
                continue

            result = await store.append(
                run_id,
                _event(operation["step_id"], operation["phase"], operation.get("payload")),
            )
            expected = operation["expect"]
            assert result.seq == expected["seq"], f"{where}: seq"
            assert result.duplicate is expected["duplicate"], f"{where}: duplicate"
            assert result.payload_diverged is expected["payload_diverged"], (
                f"{where}: payload_diverged"
            )

        elif kind == "load":
            state = await store.load(run_id)
            expected = operation["expect"]
            assert state.next_seq == expected["next_seq"], f"{where}: next_seq"
            if "records" in expected:
                actual = [
                    {
                        "step_id": event.step_id,
                        "phase": event.phase.value,
                        "seq": position,
                        "payload": dict(event.meta),
                    }
                    for position, event in enumerate(state.events)
                ]
                wanted = [
                    {**record, "payload": record.get("payload", {})}
                    for record in expected["records"]
                ]
                assert actual == wanted, f"{where}: records"

        elif kind == "query":
            state = await store.load(run_id)
            expected = operation["expect"]
            step_id = operation["step_id"]
            assert state.completed(step_id) is expected["completed"], f"{where}: completed"
            assert state.attempted(step_id) is expected["attempted"], f"{where}: attempted"

        else:  # pragma: no cover
            pytest.fail(f"{where}: operación desconocida {kind!r}")


@pytest.mark.parametrize("name,case", CASES, ids=[name for name, _ in CASES])
def test_memory_checkpointer_matches_the_golden_case(name: str, case: dict):
    asyncio.run(_run_case(MemoryCheckpointer(), case, "conformance"))


@pytest.mark.parametrize("name,case", CASES, ids=[name for name, _ in CASES])
def test_sqlite_checkpointer_matches_the_golden_case(name: str, case: dict):
    with SqliteCheckpointer() as store:
        asyncio.run(_run_case(store, case, "conformance"))


def test_the_shared_cases_are_actually_being_read():
    """Si la carpeta se mueve, la suite debe fallar en vez de pasar vacía."""
    assert CASES, "no se leyó ningún caso dorado"


# ── Contrato: identidad determinista de paso ──────────────────────────────────

_IDENTITY = _CONTRACTS / "identidad-de-paso" / "fixtures"


def _load_identity_cases() -> list[tuple[str, dict]]:
    if not _IDENTITY.exists():
        return []
    cases: list[tuple[str, dict]] = []
    for path in sorted(_IDENTITY.glob("*.json")):
        document = json.loads(path.read_text())
        for case in document.get("cases", []):
            cases.append((f"{path.stem}::{case['name']}", case))
    return cases


IDENTITY_CASES = _load_identity_cases()


def _state_of(store_state, step_id: str, *, idempotent: bool) -> str:
    """Traduce el estado del diario al vocabulario del contrato."""
    from synaptum.run.journal import Replay

    replay = Replay(store_state)
    try:
        done = replay.resolve(step_id, idempotent=idempotent)
    except UncertainEffect:
        return "uncertain"
    return "done" if done is not None else "new"


def _journal_event(entry: dict) -> ModelStep:
    ordinal = int(entry["step_id"].split("-", 1)[0])
    kind = entry["step_id"].split("-", 1)[1]
    decision = entry.get("decision")
    cls = ToolStep if kind == "tool" else ModelStep
    return cls(
        run_id="conformance",
        step_id=entry["step_id"],
        step_seq=ordinal,
        phase=Phase(entry["phase"]),
        decision=Decision(Disposition(decision)) if decision else None,
    )


@pytest.mark.parametrize(
    "name,case", IDENTITY_CASES, ids=[name for name, _ in IDENTITY_CASES]
)
def test_step_identity_matches_the_golden_case(name: str, case: dict):
    journal: list = []

    for index, operation in enumerate(case["operations"]):
        where = f"{case['name']} · operación {index}"
        kind = operation["op"]

        if kind == "mint":
            if operation.get("expect_error"):
                with pytest.raises(ValueError):
                    make_step_id(operation["ordinal"], operation["kind"])
                continue
            minted = make_step_id(operation["ordinal"], operation["kind"])
            assert minted == operation["expect"], f"{where}: acuñación"

        elif kind == "distinct":
            assert (operation["a"] != operation["b"]) is operation["expect"], where

        elif kind == "sorted":
            minted = [make_step_id(n, operation["kind"]) for n in operation["ordinals"]]
            assert minted == operation["expect"], f"{where}: acuñación"
            assert minted == sorted(minted), f"{where}: el orden lexicográfico no coincide"

        elif kind == "journal":
            journal = [_journal_event(entry) for entry in operation["entries"]]

        elif kind == "state":
            state = RunState("conformance", tuple(journal))
            actual = _state_of(
                state, operation["step_id"], idempotent=operation.get("idempotent", False)
            )
            assert actual == operation["expect"], f"{where}: estado"

        else:  # pragma: no cover
            pytest.fail(f"{where}: operación desconocida {kind!r}")


def test_the_identity_cases_are_actually_being_read():
    assert IDENTITY_CASES, "no se leyó ningún caso de identidad de paso"


# ── Contrato: normalización entre proveedores ─────────────────────────────────
#
# Todavía no hay adaptador que ejecute estos casos por nuestro lado — llega con
# `SYN-18`. Lo que sí se comprueba ahora es que el corpus **no sea ficción**: que
# cada caso referencie un cuerpo que existe, que ese cuerpo parsee, y que la
# proyección esperada esté bien formada contra nuestro vocabulario.
#
# Un corpus que nadie puede ejecutar y que además no se valida es peor que no
# tenerlo: da la impresión de cobertura sin ninguna.

_NORMALIZATION = _CONTRACTS / "normalizacion" / "fixtures"

_UNIFIED_COUNTERS = {"input", "output", "reasoning", "cache_read", "cache_write"}
_CONTENT_KINDS = {
    "text", "image", "audio", "document",
    "tool_call", "tool_result", "thinking", "redacted_thinking",
}
_STREAM_KINDS = {
    "stream_start", "text_start", "text_delta", "text_end",
    "reasoning_start", "reasoning_delta", "reasoning_end",
    "tool_call_start", "tool_call_delta", "tool_call_end", "finish",
}


def _load_normalization_cases() -> list[tuple[str, dict, Path]]:
    if not _NORMALIZATION.exists():
        return []
    cases: list[tuple[str, dict, Path]] = []
    for path in sorted(_NORMALIZATION.glob("*.json")):
        document = json.loads(path.read_text())
        root = (path.parent / document["body_root"]).resolve()
        for case in document.get("cases", []):
            cases.append((f"{path.stem}::{case['name']}", case, root))
    return cases


NORMALIZATION_CASES = _load_normalization_cases()


def _chunks_of(body: Path) -> list[dict]:
    """Separa los fragmentos de un SSE.  El transporte no es normalización."""
    return [
        json.loads(payload)
        for line in body.read_text().splitlines()
        if line.startswith("data: ")
        for payload in [line.removeprefix("data: ").strip()]
        if payload != "[DONE]"
    ]


@pytest.mark.parametrize(
    "name,case,root",
    NORMALIZATION_CASES,
    ids=[name for name, _, _ in NORMALIZATION_CASES],
)
def test_the_normalization_corpus_runs_against_the_python_adapter(
    name: str, case: dict, root: Path
):
    """SYN-18 · La otra mitad de la equivalencia de `H1 = D`.

    El gateway implementa la misma especificación en Go. Lo que impide que
    diverjan no es la confianza: es que ambas ejecutan esto contra los mismos
    cuerpos.
    """
    from synaptum import ProviderError, providers

    adapter = providers.get(case.get("dialect", "openai-compatible"))
    body = root / case["body_file"]
    expected = case["expect"]

    if case.get("stream"):
        events: list = []
        try:
            for event in adapter.stream_from_wire(_chunks_of(body)):
                events.append(event)
        except ProviderError:
            assert case.get("expect_error"), f"{case['name']}: error no esperado"
            partial = "".join(e.text for e in events if e.kind == "text_delta")
            assert partial == expected["partial_text"], f"{case['name']}: parcial"
            return

        assert not case.get("expect_error"), f"{case['name']}: se esperaba un error"
        response = events[-1].response
        if "event_kinds" in expected:
            assert [e.kind for e in events] == expected["event_kinds"], f"{case['name']}: eventos"
    else:
        response = adapter.from_wire(json.loads(body.read_text()))

    if "text" in expected:
        assert response.text == expected["text"], f"{case['name']}: texto"
    if "model" in expected:
        assert response.model == expected["model"], f"{case['name']}: modelo"
    if "finish_reason" in expected:
        assert response.finish_reason.value == expected["finish_reason"], f"{case['name']}: motivo"
    if "content_kinds" in expected:
        assert [p.kind for p in response.message.content] == expected["content_kinds"], (
            f"{case['name']}: partes de contenido"
        )
    if "tool_calls" in expected:
        assert [
            {"id": c.id, "name": c.name, "arguments": dict(c.arguments)}
            for c in response.tool_calls
        ] == expected["tool_calls"], f"{case['name']}: tool calls"
    for counter, value in expected.get("usage", {}).items():
        assert getattr(response.usage, counter) == value, f"{case['name']}: usage.{counter}"


@pytest.mark.parametrize(
    "name,case,root",
    NORMALIZATION_CASES,
    ids=[name for name, _, _ in NORMALIZATION_CASES],
)
def test_the_normalization_corpus_is_well_formed(name: str, case: dict, root: Path):
    body = root / case["body_file"]
    assert body.exists(), f"{case['name']}: falta el cuerpo {case['body_file']}"

    if case.get("stream"):
        payloads = [
            line.removeprefix("data: ").strip()
            for line in body.read_text().splitlines()
            if line.startswith("data: ")
        ]
        assert payloads, f"{case['name']}: el .sse no trae ningún evento"
        for payload in payloads:
            if payload != "[DONE]":
                json.loads(payload)
    else:
        json.loads(body.read_text())

    expected = case["expect"]

    for counter, value in expected.get("usage", {}).items():
        if counter == "estimated":
            assert isinstance(value, bool), f"{case['name']}: 'estimated' no es booleano"
            continue
        assert counter in _UNIFIED_COUNTERS, (
            f"{case['name']}: '{counter}' no es un contador del vocabulario"
        )
        assert value is None or isinstance(value, int), (
            f"{case['name']}: un contador es un entero o null, nunca otra cosa"
        )

    for kind in expected.get("content_kinds", []):
        assert kind in _CONTENT_KINDS, f"{case['name']}: parte de contenido desconocida {kind!r}"

    for kind in expected.get("event_kinds", []):
        assert kind in _STREAM_KINDS, f"{case['name']}: evento de stream desconocido {kind!r}"

    if "finish_reason" in expected:
        FinishReason(expected["finish_reason"])

    if case.get("stream") and not case.get("expect_error"):
        assert expected.get("event_kinds", ["finish"])[-1] == "finish", (
            f"{case['name']}: un stream que termina bien acaba en 'finish'"
        )


def test_the_normalization_cases_are_actually_being_read():
    assert NORMALIZATION_CASES, "no se leyó ningún caso de normalización"


def test_the_inclusive_input_convention_holds_in_the_corpus():
    """``input`` incluye lo cacheado, así que nunca puede ser menor que ``cache_read``.

    Es la ambigüedad que destapó ``chat_stream_ok.sse``: con ``prompt_n`` y
    ``cache_n`` por separado, las dos convenciones dan cifras distintas y
    ninguna falla.  Fijada la inclusiva, el corpus tiene que respetarla.
    """
    for name, case, _ in NORMALIZATION_CASES:
        usage = case["expect"].get("usage", {})
        entrada, cacheado = usage.get("input"), usage.get("cache_read")
        if entrada is not None and cacheado is not None:
            assert cacheado <= entrada, f"{name}: cache_read excede input"
