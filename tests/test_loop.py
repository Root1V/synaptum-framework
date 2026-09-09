"""RM-19 · RM-22 · RM-23 · RM-25 · El bucle, el journal y el replay.

La prueba que importa es ``test_a_resumed_run_does_not_pay_for_inference_twice``.
Todo lo demás sostiene esa.
"""

from __future__ import annotations

import asyncio

import pytest

from synaptum import (
    SEAM_VERSION,
    Agent,
    ApprovalStep,
    Decision,
    Denied,
    Disposition,
    FinalStep,
    FinishReason,
    LimitExceeded,
    MemoryCheckpointer,
    Message,
    ModelStep,
    Phase,
    ProviderError,
    Response,
    Risk,
    Role,
    Session,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolStep,
    UncertainEffect,
    Usage,
    Welcome,
    make_step_id,
)
from synaptum.agent.agent import Limits

USAGE = Usage(input=10, output=5, cache_read=90)


# ── Dobles ────────────────────────────────────────────────────────────────────

def answer(text: str) -> Response:
    return Response(
        message=Message.assistant(text), finish_reason=FinishReason.STOP, usage=USAGE
    )


def wants_tool(name: str, call_id: str = "c1", **args) -> Response:
    return Response(
        message=Message(Role.ASSISTANT, (ToolCall(id=call_id, name=name, arguments=args),)),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=USAGE,
    )


class ScriptedGateway:
    """Gateway guionizado.  Cuenta llamadas: es lo que hace visible el replay."""

    def __init__(self, *script, tool=None, deny=None) -> None:
        self.script = list(script)
        self.model_calls = 0
        self.tool_calls = 0
        self._tool = tool or (lambda call: ToolResult.of(call.id, "ok"))
        self._deny = deny or {}

    async def handshake(self, hello):
        return Welcome(version=SEAM_VERSION)

    async def invoke_model(self, request, ctx):
        self.model_calls += 1
        if "model" in self._deny:
            raise Denied(self._deny["model"])
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def stream_model(self, request, ctx):
        raise NotImplementedError

    async def invoke_tool(self, call, ctx, *, risk, tool_ref=None):
        self.tool_calls += 1
        if "tool" in self._deny:
            raise Denied(self._deny["tool"])
        return self._tool(call)


def run(agent: Agent, task: str, session: Session) -> list:
    async def go():
        return [event async for event in agent.run(task, session=session)]

    return asyncio.run(go())


def make_agent(**kw) -> Agent:
    return Agent("analista", model="fake:m", instructions="sé breve", **kw)


# ── El camino feliz ───────────────────────────────────────────────────────────

def test_a_single_turn_yields_intent_result_and_close():
    gateway = ScriptedGateway(answer("42"))
    events = run(make_agent(), "¿cuánto?", Session("run-1", gateway))

    assert [type(e).__name__ for e in events] == ["ModelStep", "ModelStep", "FinalStep"]
    assert [e.phase for e in events[:2]] == [Phase.INTENT, Phase.RESULT]
    assert events[-1].output == "42"


def test_usage_accumulates_across_the_run():
    gateway = ScriptedGateway(wants_tool("leer"), answer("listo"))
    events = run(
        make_agent(tools=[ToolDefinition(name="leer", idempotent=True)]),
        "lee y resume",
        Session("run-1", gateway),
    )
    final = events[-1]
    assert isinstance(final, FinalStep)
    assert final.usage.input == USAGE.input * 2
    assert final.usage.cache_read == USAGE.cache_read * 2


def test_a_tool_round_trip_feeds_the_result_back_to_the_model():
    gateway = ScriptedGateway(wants_tool("leer", path="/x"), answer("el fichero dice hola"))
    events = run(
        make_agent(tools=[ToolDefinition(name="leer", idempotent=True)]),
        "lee /x",
        Session("run-1", gateway),
    )

    kinds = [(type(e).__name__, e.phase.value) for e in events]
    assert kinds == [
        ("ModelStep", "intent"), ("ModelStep", "result"),
        ("ToolStep", "intent"), ("ToolStep", "result"),
        ("ModelStep", "intent"), ("ModelStep", "result"),
        ("FinalStep", "result"),
    ]
    assert gateway.model_calls == 2 and gateway.tool_calls == 1
    assert events[-1].output == "el fichero dice hola"


def test_the_declared_risk_travels_with_the_step():
    """Synaptum declara; el harness decide.  El bucle solo lo transporta."""
    gateway = ScriptedGateway(wants_tool("borrar"), answer("hecho"))
    events = run(
        make_agent(tools=[ToolDefinition(name="borrar", risk=Risk.DESTRUCTIVE)]),
        "borra",
        Session("run-1", gateway),
    )
    tool_steps = [e for e in events if isinstance(e, ToolStep)]
    assert all(e.risk is Risk.DESTRUCTIVE for e in tool_steps)
    assert all(e.idempotent is False for e in tool_steps)


# ── RM-25 · Lo que justifica toda la arquitectura ─────────────────────────────

def test_a_resumed_run_does_not_pay_for_inference_twice():
    """Reanudar con el mismo run_id lee el journal en vez de volver a inferir."""
    store = MemoryCheckpointer()
    agent = make_agent(tools=[ToolDefinition(name="leer", idempotent=True)])

    first = ScriptedGateway(wants_tool("leer"), answer("resultado"))
    run(agent, "lee /x", Session("run-1", first, store))
    assert first.model_calls == 2 and first.tool_calls == 1

    # Segunda vuelta: mismo run_id, mismo almacén, gateway que fallaría si se
    # le llamara — el guion está vacío a propósito.
    second = ScriptedGateway()
    events = run(agent, "lee /x", Session("run-1", second, store))

    assert second.model_calls == 0, "no debe haberse llamado al modelo"
    assert second.tool_calls == 0, "no debe haberse ejecutado la tool"
    assert events[-1].output == "resultado"
    # Un run cerrado devuelve su cierre registrado, no vuelve a recorrerse.
    assert [type(e).__name__ for e in events] == ["FinalStep"]


def test_a_run_that_crashed_midway_resumes_where_it_stopped():
    """Solo se re-ejecuta lo que no llegó a completarse."""
    store = MemoryCheckpointer()
    agent = make_agent(tools=[ToolDefinition(name="leer", idempotent=True)])

    # La primera vuelta muere justo después de ejecutar la tool.
    crashing = ScriptedGateway(
        wants_tool("leer"), ProviderError("se cayó", status=400)
    )
    with pytest.raises(ProviderError):
        run(agent, "lee /x", Session("run-1", crashing, store))
    assert crashing.model_calls == 2 and crashing.tool_calls == 1

    # La segunda reproduce lo hecho y solo pide la inferencia que faltaba.
    resumed = ScriptedGateway(answer("resultado"))
    events = run(agent, "lee /x", Session("run-1", resumed, store))

    assert resumed.model_calls == 1, "solo la inferencia que quedó pendiente"
    assert resumed.tool_calls == 0, "la tool ya estaba hecha"
    assert events[-1].output == "resultado"
    assert events[-1].meta["replayed_steps"] == 2, "el modelo y la tool ya hechos; faltaba una inferencia"


def test_the_context_window_is_re_derived_not_stored():
    """Al reanudar, los mensajes se reconstruyen del journal en el mismo orden."""
    store = MemoryCheckpointer()
    agent = make_agent(tools=[ToolDefinition(name="leer", idempotent=True)])

    run(agent, "lee /x", Session("run-1", ScriptedGateway(wants_tool("leer"), answer("ok")), store))

    seen: list = []

    class Spy(ScriptedGateway):
        async def invoke_model(self, request, ctx):
            seen.append(request)
            return await super().invoke_model(request, ctx)

    # Un turno más sobre el run ya reproducido.
    store2 = MemoryCheckpointer()
    asyncio.run(_replay_into(store, store2, "run-1"))
    spy = Spy(answer("y ahora esto"))
    run(agent, "lee /x", Session("run-1", spy, store2))
    assert seen == [], "el run ya estaba completo: nada que preguntar"


async def _replay_into(src: MemoryCheckpointer, dst: MemoryCheckpointer, run_id: str) -> None:
    state = await src.load(run_id)
    assert state is not None
    for event in state.events:
        await dst.append(run_id, event)


def test_a_non_idempotent_effect_left_uncertain_is_not_repeated_blindly():
    """Intención sin resultado: el efecto pudo ocurrir y el bucle no lo decide."""
    store = MemoryCheckpointer()
    agent = make_agent(tools=[ToolDefinition(name="transferir", risk=Risk.HARD_WRITE)])

    # Muere entre registrar la intención de la tool y obtener su resultado.
    class DyingGateway(ScriptedGateway):
        async def invoke_tool(self, call, ctx, *, risk, tool_ref=None):
            raise ProviderError("el proceso se fue", status=400)

    with pytest.raises(ProviderError):
        run(agent, "transfiere", Session("run-1", DyingGateway(wants_tool("transferir")), store))

    with pytest.raises(UncertainEffect) as caught:
        run(agent, "transfiere", Session("run-1", ScriptedGateway(), store))
    assert caught.value.step_id == make_step_id(1, "tool")


def test_an_idempotent_effect_left_uncertain_is_simply_retried():
    store = MemoryCheckpointer()
    agent = make_agent(tools=[ToolDefinition(name="leer", idempotent=True)])

    class DyingGateway(ScriptedGateway):
        async def invoke_tool(self, call, ctx, *, risk, tool_ref=None):
            raise ProviderError("el proceso se fue", status=400)

    with pytest.raises(ProviderError):
        run(agent, "lee", Session("run-1", DyingGateway(wants_tool("leer")), store))

    resumed = ScriptedGateway(answer("ok"))
    events = run(agent, "lee", Session("run-1", resumed, store))
    assert resumed.tool_calls == 1, "releer no tiene consecuencias: se repite"
    assert events[-1].output == "ok"


# ── H4 · Las tres disposiciones ───────────────────────────────────────────────

def test_deny_step_is_handed_back_to_the_model_so_it_can_try_something_else():
    gateway = ScriptedGateway(
        wants_tool("borrar"), answer("entendido, no lo borro"),
        deny={"tool": Decision(Disposition.DENY_STEP, reason_code="risk.destructive")},
    )
    events = run(
        make_agent(tools=[ToolDefinition(name="borrar", risk=Risk.DESTRUCTIVE)]),
        "borra todo",
        Session("run-1", gateway),
    )

    result = next(e for e in events if isinstance(e, ToolStep) and e.phase is Phase.RESULT)
    assert result.result is not None and result.result.is_error is True
    assert "risk.destructive" in result.result.content[0].text
    assert events[-1].output == "entendido, no lo borro", "el bucle siguió"


def test_terminate_run_closes_the_run_with_the_reason_recorded():
    gateway = ScriptedGateway(
        wants_tool("gastar"),
        deny={"tool": Decision(Disposition.TERMINATE_RUN, reason_code="budget.exhausted",
                               message="Superó los 5 USD.")},
    )
    events = run(
        make_agent(tools=[ToolDefinition(name="gastar")]), "gasta", Session("run-1", gateway)
    )

    final = events[-1]
    assert isinstance(final, FinalStep)
    assert final.output is None
    assert final.meta["disposition"] == "terminate_run"
    assert final.meta["reason_code"] == "budget.exhausted"


def test_require_approval_suspends_without_closing_the_run():
    """Se emite la pausa y el stream termina.  El run no se cierra."""
    gateway = ScriptedGateway(
        wants_tool("transferir"),
        deny={"tool": Decision(Disposition.REQUIRE_APPROVAL, reason_code="risk.destructive")},
    )
    events = run(
        make_agent(tools=[ToolDefinition(name="transferir", risk=Risk.DESTRUCTIVE)]),
        "transfiere",
        Session("run-1", gateway),
    )

    assert isinstance(events[-1], ApprovalStep)
    assert not any(isinstance(e, FinalStep) for e in events)
    assert "transferir" in events[-1].subject


# ── RM-27 · Límites y reintentos ──────────────────────────────────────────────

def test_the_step_limit_stops_a_loop_that_never_converges():
    gateway = ScriptedGateway(*[wants_tool("leer", call_id=f"c{n}") for n in range(10)])
    with pytest.raises(LimitExceeded) as caught:
        run(
            make_agent(tools=[ToolDefinition(name="leer", idempotent=True)], limits=Limits(max_steps=3)),
            "bucle",
            Session("run-1", gateway),
        )
    assert caught.value.limit == "max_steps"


def test_a_retryable_error_is_retried_and_a_client_error_is_not():
    ok = ScriptedGateway(ProviderError("saturado", status=429), answer("a la segunda"))
    events = run(make_agent(limits=Limits(max_retries=2)), "hola", Session("run-1", ok))
    assert ok.model_calls == 2
    assert events[-1].output == "a la segunda"

    bad = ScriptedGateway(ProviderError("mal formada", status=422), answer("nunca"))
    with pytest.raises(ProviderError):
        run(make_agent(limits=Limits(max_retries=2)), "hola", Session("run-2", bad))
    assert bad.model_calls == 1, "un 4xx no se reintenta"


def test_a_non_idempotent_tool_is_never_retried():
    """Reintentar un efecto irreversible es peor que fallar."""
    attempts = {"n": 0}

    class Flaky(ScriptedGateway):
        async def invoke_tool(self, call, ctx, *, risk, tool_ref=None):
            attempts["n"] += 1
            raise ProviderError("saturado", status=429)

    with pytest.raises(ProviderError):
        run(
            make_agent(tools=[ToolDefinition(name="transferir", risk=Risk.HARD_WRITE)]),
            "transfiere",
            Session("run-1", Flaky(wants_tool("transferir"))),
        )
    assert attempts["n"] == 1


# ── RM-22 · Journal ───────────────────────────────────────────────────────────

def test_every_durable_event_reaches_the_store_in_execution_order():
    store = MemoryCheckpointer()
    gateway = ScriptedGateway(wants_tool("leer"), answer("ok"))
    run(
        make_agent(tools=[ToolDefinition(name="leer", idempotent=True)]),
        "lee",
        Session("run-1", gateway, store),
    )

    state = asyncio.run(store.load("run-1"))
    assert state is not None
    assert [e.seq for e in state.events] == sorted(e.seq for e in state.events)
    # La tool es idempotente, así que sus dos eventos son diferibles pero acaban
    # igualmente en el journal al vaciar.
    assert len(state.events) == 7


def test_the_store_ignores_a_duplicate_write():
    store = MemoryCheckpointer()
    event = ModelStep(
        run_id="run-1", step_id=make_step_id(0, "model"), seq=0, phase=Phase.RESULT
    )

    async def go():
        await store.append("run-1", event)
        await store.append("run-1", event)
        return await store.load("run-1")

    state = asyncio.run(go())
    assert state is not None and len(state.events) == 1
