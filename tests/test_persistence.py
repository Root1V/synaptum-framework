"""SYN-24 · SYN-26 · El journal sobrevive al proceso, y la costura local avisa.

La prueba que importa es ``test_a_run_resumes_after_the_process_is_gone``: hasta
ahora la reanudación solo funcionaba mientras el proceso viviera, que es
justamente el caso que la ejecución durable no cubre.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Annotated

import pytest

from synaptum import (
    Agent,
    ApprovalStep,
    Check,
    Decision,
    Disposition,
    FinalStep,
    FinishReason,
    LocalGateway,
    Message,
    ModelStep,
    Phase,
    Request,
    Response,
    Risk,
    Role,
    Session,
    SqliteCheckpointer,
    Text,
    Thinking,
    ToolCall,
    ToolStep,
    Usage,
    dumps,
    make_step_id,
    tool,
)
from synaptum.core.codec import decode_event
from synaptum.testing import FakeGateway, calls, says


@tool(idempotent=True)
def leer(path: Annotated[str, "Ruta"]) -> str:
    """Lee un fichero."""
    return f"contenido de {path}"


@tool(risk=Risk.DESTRUCTIVE)
def borrar(path: str) -> str:
    """Borra un fichero."""
    return f"borrado {path}"


def drain(agent: Agent, task: str, session: Session) -> list:
    async def go():
        return [e async for e in agent.run(task, session=session)]

    return asyncio.run(go())


# ── Codec · sin esto no hay journal fuera de memoria ──────────────────────────

def test_an_event_survives_a_round_trip_through_json():
    original = ModelStep(
        run_id="run-1", step_id=make_step_id(0, "model"), seq=0, phase=Phase.RESULT,
        response=Response(
            message=Message(Role.ASSISTANT, (Thinking(text="pensando", signature="s1"), Text("hola"))),
            finish_reason=FinishReason.STOP,
            usage=Usage(input=10, output=5, cache_read=90),
            model="p:m",
        ),
        usage=Usage(input=10, output=5, cache_read=90),
        meta={"traceparent": "00-abc-def-01"},
    )
    restored = decode_event(json.loads(dumps(original)))

    assert dumps(restored) == dumps(original)
    assert isinstance(restored, ModelStep)
    assert restored.response.message.text == "hola"
    assert isinstance(restored.response.message.content[0], Thinking)
    assert restored.response.message.content[0].signature == "s1"
    assert restored.meta["traceparent"] == "00-abc-def-01"


def test_an_unmeasured_counter_survives_as_none_not_zero():
    """El bit que Axonium pidió tiene que cruzar el disco intacto."""
    original = ModelStep(
        run_id="r", step_id="000000-model", seq=0, phase=Phase.RESULT,
        usage=Usage(input=100, output=20, cache_read=80, estimated=True),
    )
    restored = decode_event(json.loads(dumps(original)))
    assert restored.usage.cache_write is None
    assert restored.usage.estimated is True


def test_a_tool_step_keeps_its_risk_and_idempotence():
    original = ToolStep(
        run_id="r", step_id="000001-tool", seq=1, phase=Phase.RESULT,
        call=ToolCall(id="c1", name="borrar", arguments={"path": "/x"}),
        risk=Risk.DESTRUCTIVE, idempotent=False,
    )
    restored = decode_event(json.loads(dumps(original)))
    assert restored.risk is Risk.DESTRUCTIVE
    assert restored.call.arguments == {"path": "/x"}


def test_an_unknown_event_kind_fails_loudly():
    """Mejor que devolver un evento a medias que el replay lea como no hecho."""
    with pytest.raises(ValueError, match="Evento desconocido"):
        decode_event({"kind": "inventado", "run_id": "r", "step_id": "x", "seq": 0, "phase": "result"})


def test_a_journal_written_by_an_older_version_stays_readable():
    """Un campo que no existía se queda en su default, no revienta la lectura."""
    payload = {"kind": "model", "run_id": "r", "step_id": "000000-model", "seq": 0, "phase": "result"}
    restored = decode_event(payload)
    assert restored.response is None
    assert restored.usage.input is None


# ── SYN-24 · SQLite ───────────────────────────────────────────────────────────

def test_the_primary_key_is_the_idempotency_guarantee():
    """No es una comprobación antes de insertar: es que la tabla no lo admite."""
    store = SqliteCheckpointer()
    event = ModelStep(run_id="r", step_id="000000-model", seq=0, phase=Phase.RESULT)

    async def go():
        await store.append("r", event)
        await store.append("r", event)
        return await store.load("r")

    state = asyncio.run(go())
    assert state is not None and len(state.events) == 1


def test_events_come_back_in_write_order():
    store = SqliteCheckpointer()

    async def go():
        for n in (0, 1, 2):
            await store.append(
                "r", ModelStep(run_id="r", step_id=make_step_id(n, "model"), seq=n, phase=Phase.RESULT)
            )
        return await store.load("r")

    state = asyncio.run(go())
    assert [e.seq for e in state.events] == [0, 1, 2]


def test_loading_an_unknown_run_returns_none():
    assert asyncio.run(SqliteCheckpointer().load("no-existe")) is None


def test_runs_are_kept_apart():
    store = SqliteCheckpointer()

    async def go():
        for run in ("a", "b"):
            await store.append(
                run, ModelStep(run_id=run, step_id="000000-model", seq=0, phase=Phase.RESULT)
            )
        return store.runs()

    assert asyncio.run(go()) == ["a", "b"]


def test_a_run_resumes_after_the_process_is_gone(tmp_path):
    """Lo que el almacén en memoria no podía dar: sobrevivir al proceso.

    Se simula el reinicio cerrando la conexión y abriendo otra sobre el mismo
    fichero — el objeto en memoria desaparece, el journal no.
    """
    db = tmp_path / "runs.db"
    agent = Agent("a", model="fake:m", tools=[leer])

    first = FakeGateway(calls("leer", path="/x"), says("resultado"), tools=[leer])
    with SqliteCheckpointer(db) as store:
        drain(agent, "lee /x", Session("run-1", first, store))
    assert first.model_calls == 2 and first.tool_calls == 1

    # Proceso nuevo: otra conexión, otro gateway, y el guion vacío a propósito.
    second = FakeGateway(tools=[leer])
    with SqliteCheckpointer(db) as store:
        events = drain(agent, "lee /x", Session("run-1", second, store))

    assert second.model_calls == 0, "no debe volver a inferir tras reiniciar"
    assert second.tool_calls == 0
    assert events[-1].output == "resultado"
    assert [type(e).__name__ for e in events] == ["FinalStep"]


def test_a_crash_mid_run_resumes_from_disk(tmp_path):
    db = tmp_path / "runs.db"
    agent = Agent("a", model="fake:m", tools=[leer])

    from synaptum import ProviderError

    dying = FakeGateway(calls("leer", path="/x"), ProviderError("se cayó", status=400), tools=[leer])
    with SqliteCheckpointer(db) as store:
        with pytest.raises(ProviderError):
            drain(agent, "lee /x", Session("run-1", dying, store))

    resumed = FakeGateway(says("por fin"), tools=[leer])
    with SqliteCheckpointer(db) as store:
        events = drain(agent, "lee /x", Session("run-1", resumed, store))

    assert resumed.model_calls == 1, "solo la inferencia que quedó pendiente"
    assert resumed.tool_calls == 0, "la tool ya estaba hecha y en disco"
    assert events[-1].output == "por fin"


# ── SYN-26 · La costura local no aplica nada, y lo dice ───────────────────────

async def _model(request: Request) -> Response:
    return Response(
        message=Message.assistant("respuesta local"),
        finish_reason=FinishReason.STOP,
        usage=Usage(input=10, output=5),
        model=request.model,
    )


def test_it_warns_that_it_enforces_nothing():
    """Una comprobación dentro del proceso gobernado es advisoria."""
    with pytest.warns(UserWarning, match="no aplica política"):
        LocalGateway(model=_model)


def test_every_recorded_check_says_it_was_not_enforced():
    """Está en el dato, no solo en la documentación."""
    gateway = LocalGateway(model=_model, warn=False)
    drain(Agent("a", model="local:m"), "hola", Session("run-1", gateway))

    assert gateway.checks
    assert all(check.enforced is False for check in gateway.checks)
    assert "NO aplicadas" in gateway.report()


def test_it_records_the_risk_a_real_gateway_would_have_seen():
    async def model(request: Request) -> Response:
        if any(m.role is Role.TOOL for m in request.messages):
            return await _model(request)
        return Response(
            message=Message(Role.ASSISTANT, (ToolCall(id="c1", name="borrar", arguments={"path": "/x"}),)),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input=10, output=5),
        )

    gateway = LocalGateway(model=model, tools=[borrar], warn=False)
    drain(Agent("a", model="local:m", tools=[borrar]), "borra /x", Session("run-1", gateway))

    tool_checks = [c for c in gateway.checks if c.kind == "tool"]
    assert [c.risk for c in tool_checks] == [Risk.DESTRUCTIVE]
    assert tool_checks[0].detail["arguments"] == {"path": "/x"}


def test_it_runs_the_real_tool():
    async def model(request: Request) -> Response:
        if any(m.role is Role.TOOL for m in request.messages):
            return await _model(request)
        return Response(
            message=Message(Role.ASSISTANT, (ToolCall(id="c1", name="leer", arguments={"path": "/x"}),)),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input=10, output=5),
        )

    gateway = LocalGateway(model=model, tools=[leer], warn=False)
    events = drain(Agent("a", model="local:m", tools=[leer]), "lee /x", Session("run-1", gateway))

    result = next(e for e in events if isinstance(e, ToolStep) and e.result is not None)
    assert result.result.content[0].text == "contenido de /x"


def test_a_simulated_policy_exercises_the_three_denial_paths():
    """Sirve para ver la forma de las decisiones, no para confiar en ellas."""
    def deny_destructive(check: Check) -> Decision:
        if check.risk is Risk.DESTRUCTIVE:
            return Decision(Disposition.REQUIRE_APPROVAL, reason_code="risk.destructive")
        return Decision(Disposition.ALLOW)

    async def model(request: Request) -> Response:
        return Response(
            message=Message(Role.ASSISTANT, (ToolCall(id="c1", name="borrar", arguments={"path": "/x"}),)),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input=10, output=5),
        )

    gateway = LocalGateway(model=model, tools=[borrar], policy=deny_destructive, warn=False)
    events = drain(Agent("a", model="local:m", tools=[borrar]), "borra", Session("run-1", gateway))

    assert isinstance(events[-1], ApprovalStep)
    assert not any(isinstance(e, FinalStep) for e in events)


def test_a_provider_without_streaming_is_served_anyway_and_says_so():
    """Ocultar la diferencia haría creer que se ahorra cuando ya se pagó todo."""
    gateway = LocalGateway(model=_model, warn=False)

    async def go():
        from synaptum import CallContext

        ctx = CallContext(run_id="run-1", step_id="000000-model")
        return [e async for e in gateway.stream_model(Request(model="local:m"), ctx)]

    events = asyncio.run(go())
    assert events[-1].kind == "finish"
    assert events[-1].response.provider_metadata["streamed"] is False


# ── Reanudar un run suspendido por una aprobación ─────────────────────────────

def test_a_run_paused_for_approval_can_be_resumed(tmp_path):
    """Un paso denegado antes de ejecutar es un desenlace conocido, no un silencio.

    Sin esto, el replay veía intención sin resultado —el patrón del caso
    incierto— y levantaba ``UncertainEffect`` sobre un efecto que sabemos con
    certeza que no ocurrió. Solo se ve cuando el journal sobrevive al proceso.
    """
    db = tmp_path / "runs.db"
    agent = Agent("a", model="fake:m", tools=[borrar])

    pausing = FakeGateway(
        calls("borrar", path="/x"),
        tools=[borrar],
        deny_tools={"borrar": Decision(Disposition.REQUIRE_APPROVAL, reason_code="risk.destructive")},
    )
    with SqliteCheckpointer(db) as store:
        events = drain(agent, "borra /x", Session("run-1", pausing, store))

    assert isinstance(events[-1], ApprovalStep)
    denied = next(e for e in events if isinstance(e, ToolStep) and e.phase is Phase.RESULT)
    assert denied.decision is not None
    assert denied.decision.disposition is Disposition.REQUIRE_APPROVAL
    assert denied.result is None, "no hubo efecto que registrar"

    # La persona aprueba: el gateway ya no deniega y el paso se reintenta.
    approved = FakeGateway(says("borrado y confirmado"), tools=[borrar])
    with SqliteCheckpointer(db) as store:
        events = drain(agent, "borra /x", Session("run-1", approved, store))

    assert approved.tool_calls == 1, "la tool se ejecuta ahora que hay permiso"
    assert approved.model_calls == 1, "solo la inferencia que faltaba"
    assert events[-1].output == "borrado y confirmado"


def test_a_denied_step_is_not_mistaken_for_an_uncertain_one(tmp_path):
    """La diferencia entre «no ocurrió» y «no se sabe» tiene que sobrevivir al disco."""
    db = tmp_path / "runs.db"
    agent = Agent("a", model="fake:m", tools=[borrar])

    pausing = FakeGateway(
        calls("borrar", path="/x"),
        tools=[borrar],
        deny_tools={"borrar": Decision(Disposition.REQUIRE_APPROVAL)},
    )
    with SqliteCheckpointer(db) as store:
        drain(agent, "borra", Session("run-1", pausing, store))

    from synaptum import UncertainEffect

    again = FakeGateway(
        tools=[borrar],
        deny_tools={"borrar": Decision(Disposition.REQUIRE_APPROVAL)},
    )
    with SqliteCheckpointer(db) as store:
        try:
            drain(agent, "borra", Session("run-1", again, store))
        except UncertainEffect:  # pragma: no cover
            pytest.fail("un paso denegado no es un paso incierto")


def test_a_closed_run_returns_its_recorded_outcome(tmp_path):
    """Un run cerrado no se reabre: devuelve lo que pasó."""
    db = tmp_path / "runs.db"
    agent = Agent("a", model="fake:m", tools=[borrar])

    terminating = FakeGateway(
        calls("borrar", path="/x"),
        tools=[borrar],
        deny_tools={"borrar": Decision(Disposition.TERMINATE_RUN, reason_code="budget.exhausted")},
    )
    with SqliteCheckpointer(db) as store:
        drain(agent, "borra", Session("run-1", terminating, store))

    empty = FakeGateway(tools=[borrar])
    with SqliteCheckpointer(db) as store:
        events = drain(agent, "borra", Session("run-1", empty, store))

    assert [type(e).__name__ for e in events] == ["FinalStep"]
    assert events[-1].meta["reason_code"] == "budget.exhausted"
    assert empty.model_calls == 0 and empty.tool_calls == 0
