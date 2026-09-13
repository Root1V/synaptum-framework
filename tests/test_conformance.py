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
    """Resuelve cada caso a su cuerpo.

    Hay dos raíces. La normal son las grabaciones de Axonium; la otra son los
    cuerpos **escritos a mano**, que existen solo cuando la propiedad a fijar ya
    no se puede grabar contra el despliegue. Cada uno lo declara en el propio
    fichero, así que un verde nunca deja dudas sobre de qué es evidencia.
    """
    if not _NORMALIZATION.exists():
        return []
    cases: list[tuple[str, dict, Path]] = []
    for path in sorted(_NORMALIZATION.glob("*.json")):
        document = json.loads(path.read_text())
        recorded = (path.parent / document["body_root"]).resolve()
        authored = (path.parent / document.get("authored_root", ".")).resolve()
        for case in document.get("cases", []):
            root = authored if case.get("authored") else recorded
            cases.append((f"{path.stem}::{case['name']}", case, root))
    return cases


def _collapse(kinds: list[str]) -> list[str]:
    """Quita las repeticiones seguidas: el *ciclo*, sin contar los deltas.

    Cuántos deltas trae un stream es una propiedad de la grabación — cambia al
    volver a grabar y no lo dice la especificación. El orden en que se abren y
    se cierran los ciclos sí lo dice, y es donde estaba el fallo del
    `reasoning_end` a destiempo.
    """
    return [kind for index, kind in enumerate(kinds) if index == 0 or kinds[index - 1] != kind]


def _check_usage(usage, expected: dict, where: str) -> None:
    """Las tres formas de hablar de consumo, en orden de lo que cada una afirma.

    `usage` fija un valor exacto, y solo se usa donde el valor **es** la
    afirmación: `null` y `0`, que son las que distinguen los tres estados. Una
    cifra positiva pertenece a la grabación, no al contrato — fijarla convierte
    el caso dorado en un guardián del fixture.
    """
    for counter, value in expected.get("usage", {}).items():
        assert getattr(usage, counter) == value, f"{where}: usage.{counter}"

    for counter, state in expected.get("usage_state", {}).items():
        actual = getattr(usage, counter)
        if state == "measured":
            assert actual is not None, f"{where}: usage.{counter} debería estar medido"
        elif state == "unmeasured":
            assert actual is None, (
                f"{where}: usage.{counter} es {actual!r} y la fuente no lo mide "
                "— un cero aquí diría «no hubo»"
            )
        else:  # pragma: no cover - lo cubre el test de buena formación
            raise AssertionError(f"{where}: estado de contador desconocido {state!r}")

    for left, operator, right in expected.get("usage_relations", []):
        a = left if isinstance(left, int) else getattr(usage, left)
        b = right if isinstance(right, int) else getattr(usage, right)
        assert a is not None and b is not None, f"{where}: relación sobre un contador sin medir"
        ok = {">=": a >= b, "<=": a <= b, "==": a == b}[operator]
        assert ok, f"{where}: {left} {operator} {right} → {a} {operator} {b}"


NORMALIZATION_CASES = _load_normalization_cases()


def _assert_stops_at_first_sentinel(adapter, body: Path, where: str) -> None:
    """El resultado no puede depender de lo que venga tras el primer centinela.

    Se comprueba contra el cuerpo real y contra el mismo cuerpo con un segundo
    centinela y basura detrás — que es literalmente lo que mandaba el gateway
    antes del v8. Si las dos normalizaciones no coinciden, el cliente está
    leyendo de más.
    """
    from synaptum.testing import split_sse

    original = body.read_text()
    contaminado = original.rstrip("\n") + (
        '\ndata: {"choices":[{"delta":{"content":" BASURA"}}]}\ndata: [DONE]\n'
    )
    limpio = adapter.stream_from_wire(split_sse(original))
    sucio = adapter.stream_from_wire(split_sse(contaminado))
    assert [e.kind for e in limpio] == [e.kind for e in sucio], (
        f"{where}: lo que hay tras el primer centinela cambió el resultado"
    )


def _chunks_of(body: Path) -> list[dict]:
    """Separa los fragmentos de un SSE.  El transporte no es normalización."""
    from synaptum.testing import split_sse

    return split_sse(body.read_text())


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
        kinds = [e.kind for e in events]
        if "event_kinds" in expected:
            assert kinds == expected["event_kinds"], f"{case['name']}: eventos"
        if "event_kinds_collapsed" in expected:
            assert _collapse(kinds) == expected["event_kinds_collapsed"], (
                f"{case['name']}: ciclo de eventos"
            )
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
        # El `id` no se compara: lo genera el proveedor y cambia en cada
        # grabación.  Que exista y no venga vacío sí es del contrato, y hay un
        # caso que lo pide aparte.
        assert [
            {"name": c.name, "arguments": dict(c.arguments)} for c in response.tool_calls
        ] == [
            {"name": c["name"], "arguments": c["arguments"]} for c in expected["tool_calls"]
        ], f"{case['name']}: tool calls"
    if expected.get("tool_call_ids_are_nonempty"):
        assert response.tool_calls, f"{case['name']}: no se reconstruyó ninguna tool call"
        for call in response.tool_calls:
            assert call.id, (
                f"{case['name']}: la identidad llega solo en el primer fragmento "
                "y hay que recordarla"
            )
    if expected.get("stops_at_first_sentinel"):
        _assert_stops_at_first_sentinel(adapter, body, case["name"])

    _check_usage(response.usage, expected, case["name"])


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

    for key in ("event_kinds", "event_kinds_collapsed"):
        for kind in expected.get(key, []):
            assert kind in _STREAM_KINDS, f"{case['name']}: evento de stream desconocido {kind!r}"

    for counter, state in expected.get("usage_state", {}).items():
        assert counter in _UNIFIED_COUNTERS, (
            f"{case['name']}: '{counter}' no es un contador del vocabulario"
        )
        assert state in {"measured", "unmeasured"}, (
            f"{case['name']}: estado {state!r} — solo hay medido y sin medir"
        )
        assert counter not in expected.get("usage", {}), (
            f"{case['name']}: '{counter}' se fija por valor y por estado a la vez"
        )

    for relation in expected.get("usage_relations", []):
        left, operator, right = relation
        assert operator in {">=", "<=", "=="}, f"{case['name']}: operador {operator!r}"
        for operand in (left, right):
            assert isinstance(operand, int) or operand in _UNIFIED_COUNTERS, (
                f"{case['name']}: operando {operand!r} no es contador ni entero"
            )

    for counter, positivo in expected.get("usage", {}).items():
        if counter == "estimated":   # es un bool, y en Python un bool es un int
            continue
        assert not (isinstance(positivo, int) and positivo > 0), (
            f"{case['name']}: un contador positivo pertenece a la grabación, no al "
            "contrato — usa usage_state o usage_relations"
        )

    if "finish_reason" in expected:
        FinishReason(expected["finish_reason"])

    if case.get("stream") and not case.get("expect_error"):
        assert expected.get("event_kinds", ["finish"])[-1] == "finish", (
            f"{case['name']}: un stream que termina bien acaba en 'finish'"
        )


def test_the_normalization_cases_are_actually_being_read():
    assert NORMALIZATION_CASES, "no se leyó ningún caso de normalización"


def test_the_inclusive_input_convention_holds_in_the_recorded_bodies():
    """``input`` incluye lo cacheado, y quien lo demuestra son los cuerpos.

    Es la ambigüedad que destapó ``chat_stream_ok.sse``: con ``prompt_n`` y
    ``cache_n`` por separado, las dos convenciones dan cifras distintas y
    ninguna falla.  La versión anterior de este test recorría lo que el propio
    corpus *afirmaba*, que es circular — y al quitar del corpus las cifras
    incidentales se quedó además sin nada que recorrer, pasando en verde.

    Esto mira el cable: en cada grabación no-streaming, ``prompt_tokens`` tiene
    que ser exactamente ``prompt_n + cache_n``.  Si una regrabación futura
    cambia de convención, salta aquí en vez de cuadrar mal en la factura.
    """
    if not _NORMALIZATION.exists():
        pytest.skip("contratos compartidos no disponibles")

    comprobados = 0
    for cuerpo in sorted((_CONTRACTS / "gateway-prometheus" / "fixtures").glob("chat_completion*.json")):
        body = json.loads(cuerpo.read_text())
        reported, timings = body.get("usage") or {}, body.get("timings") or {}
        if not reported or not timings:
            continue
        prompt, cached = timings.get("prompt_n"), timings.get("cache_n")
        if prompt is None or cached is None:
            continue
        assert reported["prompt_tokens"] == prompt + cached, (
            f"{cuerpo.name}: prompt_tokens={reported['prompt_tokens']} no es "
            f"prompt_n({prompt}) + cache_n({cached}) — la convención inclusiva dejó de valer"
        )
        assert (reported.get("prompt_tokens_details") or {}).get("cached_tokens") == cached
        comprobados += 1

    assert comprobados >= 3, f"solo {comprobados} grabaciones confirman la convención"
