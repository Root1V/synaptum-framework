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
import hashlib
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
    Role,
    RunState,
    SqliteCheckpointer,
    ToolStep,
    InvalidToolCallError,
    UncertainEffect,
    dumps,
    tool_call_hash,
    make_step_id,
)

from contratos import SIN_CONTRATOS, corpus

_FIXTURES = corpus("costura-durabilidad", "fixtures")

pytestmark = pytest.mark.skipif(_FIXTURES is None, reason=SIN_CONTRATOS)


def _load_cases() -> list[tuple[str, dict]]:
    if _FIXTURES is None:
        return []
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

_IDENTITY = corpus("identidad-de-paso", "fixtures")


def _load_identity_cases() -> list[tuple[str, dict]]:
    """Los casos de **acuñación**, y solo esos.

    Antes se leía todo `*.json` de la carpeta dando por hecho que cualquier
    fichero de ahí tenía esta forma. El día que Aeon publicó el corpus de
    hashes dorados en la misma carpeta, sus once casos entraron aquí y
    fallaron: no tienen `operations` porque no describen una acuñación.

    Suponer la forma por la ubicación es cómodo hasta que alguien añade un
    vecino. Ahora se selecciona por lo que el documento **declara ser**.
    """
    if _IDENTITY is None:
        return []
    cases: list[tuple[str, dict]] = []
    for path in sorted(_IDENTITY.glob("*.json")):
        document = json.loads(path.read_text())
        if document.get("contract") != "identidad-de-paso":
            continue
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


# ── Contrato: los hashes dorados de una llamada a herramienta ─────────────────
#
# Lo que ata una aprobación a lo que se aprobó. El arnés escribe el hash de la
# llamada en el payload de la decisión; al reanudar se compara contra el de la
# llamada que se va a ejecutar, y si no coinciden es que alguien cambió los
# argumentos después de que una persona dijera que sí.
#
# Por eso la equivalencia entre implementaciones **es** la garantía: si el Go
# del arnés y este Python no producen el mismo hash del mismo paso, una
# aprobación legítima parece manipulada — una falsa alarma en el peor sitio.

_HASHES = _IDENTITY / "hashes-dorados.json" if _IDENTITY else None

#: Casos que **no deben producir hash**: se rechaza construirlo.
#:
#: La divergencia que medimos el 27 —nosotros conservábamos `2^53+1` y la
#: canonicalización compartida lo pliega— se cerró decidiendo que un número de
#: esa magnitud no se ata: se rechaza. Ver `tool_call_hash`.
#:
#: Esta lista es local **hasta que el corpus traiga su propio `expect`**. En
#: cuanto lo traiga, manda el fichero: el acuerdo tiene que vivir donde los dos
#: lo leen, no en una constante de cada repositorio.
RECHAZADOS_EN_LOCAL = {"big-integer-beyond-double-precision"}


def _se_espera_rechazo(case: dict) -> bool:
    declarado = case.get("expect")
    if declarado is not None:
        return declarado == "reject"
    return case["name"] in RECHAZADOS_EN_LOCAL


def _hash_de(paso: dict) -> tuple[str, str]:
    """La forma hasheada del contrato, serializada como la serializamos."""
    forma = {
        "step_id": paso["step_id"],
        "tool_args": paso["tool_args"],
        "tool_name": paso["tool_name"],
    }
    canonico = dumps(forma)
    return canonico, hashlib.sha256(canonico.encode()).hexdigest()


def _casos_de_hash() -> list[tuple[str, dict]]:
    if _HASHES is None or not _HASHES.exists():
        return []
    return [(c["name"], c) for c in json.loads(_HASHES.read_text())["cases"]]


HASH_CASES = _casos_de_hash()


@pytest.mark.skipif(not HASH_CASES, reason=SIN_CONTRATOS)
@pytest.mark.parametrize("name,case", HASH_CASES, ids=[n for n, _ in HASH_CASES])
def test_the_golden_tool_call_hashes_match(name: str, case: dict):
    paso = case["step"]

    if _se_espera_rechazo(case):
        # No es que divergamos: es que este paso **no se puede atar**, y el
        # hash se niega a construirlo en vez de elegir entre plegar y divergir.
        with pytest.raises(InvalidToolCallError, match="2\\^53"):
            tool_call_hash(paso["step_id"], paso["tool_name"], paso["tool_args"])
        return

    canonico, digest = _hash_de(paso)
    assert canonico == case["canonical_json"], (
        f"{name}: nuestra forma canónica difiere de la del corpus.\n"
        f"  corpus : {case['canonical_json']}\n"
        f"  nuestra: {canonico}\n  ({case['why']})"
    )
    assert digest == case["sha256"], f"{name}: mismo canónico y distinto hash"

    # Y el hash público produce lo mismo que la forma de arriba: si algún día
    # dejaran de coincidir, el corpus estaría validando algo que nadie usa.
    assert tool_call_hash(paso["step_id"], paso["tool_name"], paso["tool_args"]) == digest


@pytest.mark.skipif(not HASH_CASES, reason=SIN_CONTRATOS)
def test_the_golden_hash_corpus_is_not_empty():
    """Un corpus que no se lee pasa igual que uno que sí."""
    assert len(HASH_CASES) >= 11, f"solo se leyeron {len(HASH_CASES)} casos de hash"


# ── Contrato: normalización entre proveedores ─────────────────────────────────
#
# Todavía no hay adaptador que ejecute estos casos por nuestro lado — llega con
# `SYN-18`. Lo que sí se comprueba ahora es que el corpus **no sea ficción**: que
# cada caso referencie un cuerpo que existe, que ese cuerpo parsee, y que la
# proyección esperada esté bien formada contra nuestro vocabulario.
#
# Un corpus que nadie puede ejecutar y que además no se valida es peor que no
# tenerlo: da la impresión de cobertura sin ninguna.

_NORMALIZATION = corpus("normalizacion", "fixtures")

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
    if _NORMALIZATION is None:
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

    if case.get("direction") == "request":
        _check_request_direction(adapter, body, expected, case["name"])
        _assert_every_key_was_read(expected, case["name"])
        return

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
    _assert_every_key_was_read(expected, case["name"])


def _check_request_direction(adapter, body: Path, expected: dict, where: str) -> None:
    """La dirección que el contrato decía que no había que normalizar.

    El cuerpo es un ``Request`` del vocabulario compartido y lo que se fija es
    el **cuerpo que sale al cable**. Existe desde `VRT-SYN-004`, que falló justo
    aquí: las dos direcciones no son independientes —la *response* produce la
    parte que la *request* no sabe mandar— así que ningún caso del corpus podía
    ver el fallo en ninguno de los dos lenguajes.
    """
    from synaptum.core.codec import decode
    from synaptum.core.types import Request

    wire = adapter.to_wire(decode(Request, json.loads(body.read_text())))
    mensajes = wire["messages"]

    if "wire_roles" in expected:
        assert [m["role"] for m in mensajes] == expected["wire_roles"], (
            f"{where}: los roles del cuerpo\n"
            f"  esperado: {expected['wire_roles']}\n"
            f"  salió   : {[m['role'] for m in mensajes]}"
        )

    if "wire_assistant_contents" in expected:
        assert [
            m.get("content") for m in mensajes if m["role"] == "assistant"
        ] == expected["wire_assistant_contents"], f"{where}: el `content` del asistente"

    if expected.get("assistant_messages_sendable"):
        # La regla del dialecto, medida contra el servidor real por los dos
        # lados: `content` tiene que ser una cadena —vacía vale— o haber
        # `tool_calls`. `null` sin tool calls es un 400 no reintentable, y es
        # exactamente la forma que se colaba.
        for mensaje in mensajes:
            if mensaje["role"] != "assistant":
                continue
            contenido = mensaje.get("content")
            assert isinstance(contenido, (str, list)) or mensaje.get("tool_calls"), (
                f"{where}: un mensaje del asistente sale sin `content` enviable "
                f"y sin `tool_calls`: {mensaje}"
            )


#: Las claves de expectativa que este runner sabe comprobar.
#:
#: Existe para que una clave **nueva** falle en vez de ignorarse. Un
#: comprobador que se salta en silencio lo que no entiende desarrolla puntos
#: ciegos exactamente donde el contrato crece — y el contrato crece ahí, porque
#: las claves nuevas son las que describen lo recién acordado.
#:
#: Aeon lo encontró en su runner: leía 5 de 13, y tres casos no tenían ni una
#: sola clave que comprobara. Pasaban sin verificar **ninguna** de sus
#: afirmaciones, y ese número se publicó en el canal como evidencia. Lo avisaron
#: por si aplicaba a los nuestros: la pregunta no es «¿pasan mis casos?», es
#: «¿hay alguna clave del fichero que mi código no lea?».
CLAVES_QUE_SE_COMPRUEBAN = {
    "text", "model", "finish_reason", "content_kinds",
    "tool_calls", "tool_call_ids_are_nonempty",
    "event_kinds", "event_kinds_collapsed", "partial_text",
    "stops_at_first_sentinel",
    "usage", "usage_state", "usage_relations",
    # Dirección request — `VRT-SYN-004`.
    "wire_roles", "wire_assistant_contents", "assistant_messages_sendable",
}


def _assert_every_key_was_read(expected: dict, where: str) -> None:
    desconocidas = sorted(set(expected) - CLAVES_QUE_SE_COMPRUEBAN)
    assert not desconocidas, (
        f"{where}: el corpus afirma {desconocidas} y este runner no sabe "
        "comprobarlo. Impleméntalo o quítalo del corpus — ignorarlo en silencio "
        "haría pasar el caso sin verificar lo que dice."
    )


@pytest.mark.parametrize(
    "name,case,root",
    NORMALIZATION_CASES,
    ids=[name for name, _, _ in NORMALIZATION_CASES],
)
def test_the_normalization_corpus_is_well_formed(name: str, case: dict, root: Path):
    body = root / case["body_file"]
    assert body.exists(), f"{case['name']}: falta el cuerpo {case['body_file']}"

    # Un caso sin afirmaciones no es un caso: es un cuerpo que se parsea y un
    # verde que no significa nada. Lo avisó Aeon de su propio runner —ahí un
    # caso sin `expect` vuelve temprano y pasa sin comprobar nada— y en el
    # nuestro reventaba por `KeyError`, que es fallar por suerte y no por
    # regla. Dicho como regla, el mensaje explica qué hacer.
    assert case.get("expect"), (
        f"{case['name']}: el caso no trae `expect`. Un caso sin afirmaciones "
        "pasa siempre y no es evidencia de nada: dale afirmaciones o quítalo."
    )

    direccion = case.get("direction", "response")
    assert direccion in {"request", "response"}, (
        f"{case['name']}: dirección {direccion!r} — solo hay request y response"
    )
    if direccion == "request":
        # El cuerpo de esta dirección **no es del cable**: es el vocabulario
        # compartido, y tiene que poder volver a ser un `Request`. Un cuerpo de
        # cable puesto aquí por error decodificaría a un `Request` vacío y el
        # caso pasaría comprobando nada.
        from synaptum.core.codec import decode
        from synaptum.core.types import Request

        peticion = decode(Request, json.loads(body.read_text()))
        assert peticion.messages, (
            f"{case['name']}: el cuerpo no trae mensajes — ¿es un cuerpo del cable?"
        )
        assert not case.get("stream"), (
            f"{case['name']}: la dirección request no se trocea"
        )
        for rol in case["expect"].get("wire_roles", []):
            Role(rol)
        return

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
    if _NORMALIZATION is None:
        pytest.skip(SIN_CONTRATOS)

    comprobados = 0
    grabaciones = corpus("gateway-prometheus", "fixtures")
    if grabaciones is None:
        pytest.skip(SIN_CONTRATOS)

    for cuerpo in sorted(grabaciones.glob("chat_completion*.json")):
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


def test_a_float_at_the_limit_cannot_be_told_from_one_that_was_folded():
    """La imagen espejo del límite de Go, en Python.

    Aeon midió que su `encoding/json` decodifica a `float64` salvo que pidas
    `UseNumber`, así que un `2^53+1` leído de un fichero **llega ya plegado** y
    su guard no tiene nada que mirar. Fuimos a comprobar el nuestro y la mitad
    buena se confirmó: Python conserva los enteros de precisión arbitraria, así
    que el dígito llega intacto.

    La mitad mala no la esperaba. Un **literal flotante** sí llega plegado:

        json.loads('{"reference": 9007199254740993.0}')  ->  9007199254740992.0

    …que vale exactamente `2^53` y pasaba el corte. Por eso el límite es `>=`
    para un flotante y `>` para un entero: con el entero no hay ambigüedad, con
    el flotante no se puede saber de dónde viene.
    """
    plegado = json.loads('{"reference": 9007199254740993.0}')
    assert plegado["reference"] == float(2**53), "el pliegue ocurre al decodificar"

    with pytest.raises(InvalidToolCallError, match=r"2\^53"):
        tool_call_hash("s8", "payments.capture", plegado)

    # Y el entero del mismo valor sí se ata: ahí no hay nada que dudar.
    assert tool_call_hash("s8", "payments.capture", {"reference": 2**53})


def test_the_corpus_integers_survive_being_read_from_the_file():
    """Lo que Aeon pidió comprobar: que nuestro lector no normalice números."""
    if _HASHES is None or not _HASHES.exists():
        pytest.skip(SIN_CONTRATOS)

    casos = {c["name"]: c for c in json.loads(_HASHES.read_text())["cases"]}
    leido = casos["big-integer-beyond-double-precision"]["step"]["tool_args"]["reference"]

    assert isinstance(leido, int), "un lector que normalice números borra el caso"
    assert leido == 2**53 + 1, f"llegó {leido}, así que el dígito se perdió al leer"
