"""RM-04 · RM-07 · RM-08 · Las garantías de la frontera con el harness.

Estas pruebas cubren exactamente lo que otro equipo va a implementar contra el
contrato: clasificación de errores, negociación de versión, y el estado de run
sobre el que se apoya el replay.
"""

from __future__ import annotations

import pytest

from synaptum import (
    SEAM_VERSION,
    CallContext,
    Checkpointer,
    Decision,
    Denied,
    Disposition,
    Gateway,
    Hello,
    ModelStep,
    Phase,
    ProviderError,
    Request,
    Response,
    RunState,
    SeamVersionError,
    ToolCall,
    ToolResult,
    Welcome,
    make_step_id,
    negotiate,
    supported_versions,
)
from synaptum.core.events import Risk


# ── RM-04 · Clasificación de errores ──────────────────────────────────────────

@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_client_errors_are_not_retried(status: int):
    """La petición es el problema; repetirla da lo mismo y cuesta lo mismo."""
    assert ProviderError("nope", status=status).retryable is False


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_rate_limits_and_server_errors_are_retried(status: int):
    assert ProviderError("later", status=status).retryable is True


def test_an_error_without_a_status_is_retried():
    """Nunca llegó a haber respuesta: se parece a un problema de camino."""
    assert ProviderError("se cayó la conexión").retryable is True


def test_an_unknown_client_error_is_not_retried():
    """La familia 4xx entera significa «tu petición es el problema»."""
    assert ProviderError("raro", status=418).retryable is False


def test_an_explicit_flag_overrides_the_status_rule():
    """El adaptador sabe más que la tabla cuando el proveedor se sale de norma."""
    assert ProviderError("cuota diaria", status=403, retryable=True).retryable is True


# ── RM-04 · La denegación es una decisión, no un fallo ────────────────────────

def test_a_denial_is_never_retried():
    """Repetir la llamada no cambia la política."""
    denied = Denied(Decision(Disposition.DENY_STEP, reason_code="tool.blocked"))
    assert denied.retryable is False


def test_a_denial_carries_the_whole_decision():
    denied = Denied(
        Decision(
            Disposition.TERMINATE_RUN,
            reason_code="budget.exhausted",
            message="El run superó el límite de 5 USD.",
        )
    )
    assert denied.disposition is Disposition.TERMINATE_RUN
    assert denied.decision.reason_code == "budget.exhausted"
    assert "5 USD" in str(denied)


def test_only_terminate_run_ends_the_run():
    """deny_step admite intentar otra cosa; require_approval suspende, no falla."""
    assert Denied(Decision(Disposition.TERMINATE_RUN)).terminal is True
    assert Denied(Decision(Disposition.DENY_STEP)).terminal is False
    assert Denied(Decision(Disposition.REQUIRE_APPROVAL)).terminal is False


# ── RM-12 · Negociación de versión ────────────────────────────────────────────

def test_the_support_window_counts_the_current_version():
    """N = 2 significa dos versiones vivas, no la actual más dos."""
    assert supported_versions("1.7", window=2) == ("1.7", "1.6")
    assert supported_versions("1.7", window=3) == ("1.7", "1.6", "1.5")


def test_the_window_does_not_run_past_the_first_minor():
    assert supported_versions("0.1", window=2) == ("0.1", "0.0")


def test_negotiation_picks_the_highest_version_both_speak():
    """Dos extremos al día no deben quedarse en una antigua que ambos soportan."""
    assert negotiate(["1.5", "1.6", "1.7"], current="1.7") == "1.7"
    assert negotiate(["1.5", "1.6"], current="1.7") == "1.6"


def test_negotiation_fails_loudly_when_there_is_no_overlap():
    """Mejor fallar aquí que descubrir la incompatibilidad campo a campo."""
    with pytest.raises(SeamVersionError, match="Sin versión de costura común"):
        negotiate(["2.0"], current="1.7")


def test_hello_offers_our_supported_versions_by_default():
    assert Hello().versions[0] == SEAM_VERSION


# ── RM-11 · Contexto de traza ─────────────────────────────────────────────────

def test_the_call_context_carries_w3c_trace_headers():
    """Sin esto los spans del bucle y los de la E/S quedan en árboles distintos."""
    ctx = CallContext(
        run_id="run-1",
        step_id=make_step_id(0, "model"),
        traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )
    assert ctx.traceparent is not None
    assert ctx.tracestate is None


# ── RM-07 · Estado de run y fast-forward ──────────────────────────────────────

def _step(seq: int, phase: Phase) -> ModelStep:
    return ModelStep(
        run_id="run-1", step_id=make_step_id(seq, "model"), step_seq=seq, phase=phase
    )


def test_next_seq_is_the_size_of_the_journal_not_the_step_ordinal():
    """Posiciones contiguas desde cero: «por dónde continúa el diario»."""
    state = RunState("run-1", (_step(0, Phase.ATTEMPTED), _step(0, Phase.COMPLETED), _step(1, Phase.ATTEMPTED)))
    assert state.next_seq == 3


def test_next_seq_starts_at_zero_for_an_empty_run():
    assert RunState("run-1").next_seq == 0


def test_a_step_with_a_recorded_result_is_complete():
    """Que exista el resultado significa que el efecto ocurrió: no se repite."""
    state = RunState("run-1", (_step(0, Phase.ATTEMPTED), _step(0, Phase.COMPLETED)))
    step_id = make_step_id(0, "model")
    assert state.completed(step_id) is True
    assert state.result_of(step_id) is not None


def test_attempted_is_exclusive_of_completed():
    """Un paso terminado no está intentado, está hecho.

    Quien pregunta «¿qué hago ahora?» necesita una respuesta, no dos.
    """
    state = RunState("run-1", (_step(0, Phase.ATTEMPTED), _step(0, Phase.COMPLETED)))
    step_id = make_step_id(0, "model")
    assert state.completed(step_id) is True
    assert state.attempted(step_id) is False


def test_an_intent_without_a_result_is_the_uncertain_case():
    """El proceso cayó entre el registro y el efecto: pudo haber ocurrido."""
    state = RunState("run-1", (_step(0, Phase.ATTEMPTED),))
    step_id = make_step_id(0, "model")
    assert state.attempted(step_id) is True
    assert state.completed(step_id) is False


def test_an_unseen_step_was_never_attempted():
    state = RunState("run-1", (_step(0, Phase.COMPLETED),))
    assert state.attempted(make_step_id(9, "model")) is False


# ── RM-07 · RM-08 · Conformidad estructural ───────────────────────────────────
#
# Los protocolos no se heredan: quien los implementa no importa nada de
# Synaptum en tiempo de ejecución.  Estas dos clases lo demuestran.

class _Journal:
    """Checkpointer mínimo, con la deduplicación que exige el contrato."""

    def __init__(self) -> None:
        self._by_run: dict[str, list] = {}
        self._seen: set[tuple[str, str, str]] = set()

    async def append(self, run_id, event) -> None:
        if event.key in self._seen:
            return
        self._seen.add(event.key)
        self._by_run.setdefault(run_id, []).append(event)

    async def load(self, run_id):
        events = self._by_run.get(run_id)
        return RunState(run_id, tuple(events)) if events else None


class _Gate:
    """Gateway mínimo que deniega todo lo destructivo."""

    async def handshake(self, hello: Hello) -> Welcome:
        return Welcome(version=negotiate(hello.versions), tool_refs={t.name: f"{t.name}@1" for t in hello.tools})

    async def invoke_model(self, request: Request, ctx: CallContext) -> Response:
        raise NotImplementedError

    def stream_model(self, request: Request, ctx: CallContext):
        raise NotImplementedError

    async def invoke_tool(self, call: ToolCall, ctx: CallContext, *, risk: Risk, tool_ref=None) -> ToolResult:
        if risk is Risk.DESTRUCTIVE:
            raise Denied(Decision(Disposition.REQUIRE_APPROVAL, reason_code="risk.destructive"))
        return ToolResult.of(call.id, "ok")


def test_a_plain_class_satisfies_the_protocols_without_inheriting():
    assert isinstance(_Journal(), Checkpointer)
    assert isinstance(_Gate(), Gateway)


def test_append_is_idempotent_under_a_repeated_write():
    """Un motor de workflows puede reintentar una unidad que ya escribió."""
    import asyncio

    async def scenario():
        journal = _Journal()
        event = _step(0, Phase.COMPLETED)
        await journal.append("run-1", event)
        await journal.append("run-1", event)
        state = await journal.load("run-1")
        assert state is not None
        return state

    state = asyncio.run(scenario())
    assert len(state.events) == 1


def test_the_gateway_denies_instead_of_returning_a_result():
    """La denegación interrumpe el efecto; no es un valor que se pueda ignorar."""
    import asyncio

    ctx = CallContext(run_id="run-1", step_id=make_step_id(1, "tool"))
    gate = _Gate()

    allowed = asyncio.run(
        gate.invoke_tool(ToolCall(id="c1", name="leer"), ctx, risk=Risk.READ)
    )
    assert allowed.is_error is False

    with pytest.raises(Denied) as caught:
        asyncio.run(
            gate.invoke_tool(ToolCall(id="c2", name="borrar"), ctx, risk=Risk.DESTRUCTIVE)
        )
    assert caught.value.disposition is Disposition.REQUIRE_APPROVAL
    assert caught.value.terminal is False


def test_the_handshake_returns_versioned_tool_references():
    """A partir de aquí el esquema no viaja en cada llamada — H5."""
    import asyncio

    from synaptum import ToolDefinition

    welcome = asyncio.run(
        _Gate().handshake(Hello(tools=(ToolDefinition(name="buscar", description="busca"),)))
    )
    assert welcome.version == SEAM_VERSION
    assert welcome.tool_refs["buscar"] == "buscar@1"
