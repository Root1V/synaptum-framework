"""RM-05 · Las garantías de la taxonomía de eventos.

Estas pruebas cubren el contrato que Aeon implementa al escribir su
``Checkpointer``: identidad determinista, clave de idempotencia y política de
durabilidad.
"""

from __future__ import annotations

import pytest

from synaptum import (
    ApprovalStep,
    Decision,
    DelegateStep,
    Disposition,
    Durability,
    FinalStep,
    ModelStep,
    Phase,
    Request,
    Risk,
    ToolCall,
    ToolStep,
    Usage,
    dumps,
    idempotency_key,
    make_step_id,
)


# ── Identidad determinista — RM-06 ────────────────────────────────────────────

def test_step_id_depends_only_on_position_and_kind():
    """Un run reproducido debe acuñar exactamente los mismos identificadores."""
    assert make_step_id(7, "model") == make_step_id(7, "model")
    assert make_step_id(7, "model") != make_step_id(8, "model")
    assert make_step_id(7, "model") != make_step_id(7, "tool")


def test_step_ids_sort_lexicographically_in_execution_order():
    """Un almacén que ordene por clave devuelve el journal en orden, sin índice."""
    ids = [make_step_id(n, "model") for n in (0, 2, 9, 10, 100, 1000)]
    assert ids == sorted(ids)


def test_step_id_rejects_a_negative_sequence():
    with pytest.raises(ValueError):
        make_step_id(-1, "model")


# ── Clave de idempotencia — A3 ────────────────────────────────────────────────

def _model_step(phase: Phase, *, at: float | None = None) -> ModelStep:
    return ModelStep(
        run_id="run-1",
        step_id=make_step_id(3, "model"),
        step_seq=3,
        phase=phase,
        at=at,
        request=Request(model="p:m") if phase is Phase.ATTEMPTED else None,
    )


def test_the_key_is_run_step_and_phase():
    step = _model_step(Phase.ATTEMPTED)
    assert idempotency_key(step) == ("run-1", "000003-model", "attempted")
    assert step.key == idempotency_key(step)


def test_the_timestamp_is_not_part_of_the_identity():
    """Un reintento con distinto reloj sigue siendo el mismo paso."""
    first = _model_step(Phase.ATTEMPTED, at=1000.0)
    again = _model_step(Phase.ATTEMPTED, at=2000.0)
    assert first.key == again.key


def test_intent_and_result_share_the_step_and_differ_in_phase():
    intent = _model_step(Phase.ATTEMPTED)
    result = _model_step(Phase.COMPLETED)
    assert intent.step_id == result.step_id
    assert intent.key != result.key


# ── Política de durabilidad — RM-14 ───────────────────────────────────────────

def test_a_model_result_is_always_durable():
    """Cuesta dinero y no es reproducible: una vez escrito, no se repite."""
    assert _model_step(Phase.COMPLETED).durability is Durability.DURABLE


def _tool_step(*, idempotent: bool, risk: Risk = Risk.READ) -> ToolStep:
    return ToolStep(
        run_id="run-1",
        step_id=make_step_id(4, "tool"),
        step_seq=4,
        phase=Phase.ATTEMPTED,
        call=ToolCall(id="c1", name="leer"),
        risk=risk,
        idempotent=idempotent,
    )


def test_tool_durability_follows_the_idempotency_of_the_effect():
    """Releer un fichero se repite sin consecuencias; ordenar una transferencia no."""
    assert _tool_step(idempotent=True).durability is Durability.DEFERRABLE
    assert _tool_step(idempotent=False).durability is Durability.DURABLE


def test_a_tool_that_declares_nothing_pays_durability():
    """Conservador por defecto: el caso silencioso es el seguro."""
    step = ToolStep(run_id="r", step_id="000001-tool", step_seq=1, phase=Phase.ATTEMPTED)
    assert step.idempotent is False
    assert step.durability is Durability.DURABLE


def test_an_approval_is_durable_so_nobody_is_asked_twice():
    step = ApprovalStep(
        run_id="run-1", step_id="000005-approval", step_seq=5,
        phase=Phase.ATTEMPTED, subject="transferencia de 420.000 EUR",
    )
    assert step.durability is Durability.DURABLE


def test_delegation_and_closure_are_durable():
    delegate = DelegateStep(
        run_id="r", step_id="000006-delegate", step_seq=6,
        phase=Phase.ATTEMPTED, agent="analista", brief="evalúa el riesgo",
    )
    final = FinalStep(run_id="r", step_id="000007-final", step_seq=7, phase=Phase.COMPLETED)
    assert delegate.durability is Durability.DURABLE
    assert final.durability is Durability.DURABLE


# ── Riesgo — RM-21 ────────────────────────────────────────────────────────────

def test_risk_is_declared_not_decided():
    """Synaptum declara el nivel; la decisión pertenece al harness."""
    step = _tool_step(idempotent=False, risk=Risk.DESTRUCTIVE)
    assert step.risk is Risk.DESTRUCTIVE
    assert not hasattr(step, "approved")


# ── Disposición de la denegación — H4 ─────────────────────────────────────────

def test_only_allow_counts_as_permission():
    assert Decision(Disposition.ALLOW).allowed is True
    for refusal in (Disposition.DENY_STEP, Disposition.TERMINATE_RUN, Disposition.REQUIRE_APPROVAL):
        assert Decision(refusal).allowed is False


def test_a_refusal_carries_a_stable_code_and_a_readable_message():
    """El código va a métricas; el mensaje puede acabar delante de una persona."""
    decision = Decision(
        Disposition.TERMINATE_RUN,
        reason_code="budget.exhausted",
        message="El run superó el límite de 5 USD.",
    )
    assert decision.reason_code == "budget.exhausted"
    assert "5 USD" in decision.message


# ── Serialización del journal ─────────────────────────────────────────────────

def test_events_serialise_deterministically():
    """El journal es append-only y comparable entre ejecución y replay."""
    a = _model_step(Phase.COMPLETED)
    b = _model_step(Phase.COMPLETED)
    assert dumps(a) == dumps(b)


def test_an_event_carries_opaque_metadata_for_trace_context():
    step = ModelStep(
        run_id="r", step_id="000001-model", step_seq=1, phase=Phase.ATTEMPTED,
        meta={"traceparent": "00-abc-def-01"},
        usage=Usage(input=10),
    )
    assert step.meta["traceparent"] == "00-abc-def-01"


# ── Durabilidad por fase, no solo por tipo ────────────────────────────────────

def test_a_model_intent_is_deferrable_and_its_result_is_not():
    """Lo que se ahorra no es una escritura: es una espera antes del efecto."""
    assert _model_step(Phase.ATTEMPTED).durability is Durability.DEFERRABLE
    assert _model_step(Phase.COMPLETED).durability is Durability.DURABLE


def test_a_non_idempotent_tool_blocks_on_both_phases():
    """Escritura anticipada: la intención en disco antes de que ocurra el efecto."""
    for phase in (Phase.ATTEMPTED, Phase.COMPLETED):
        step = ToolStep(
            run_id="r", step_id="000001-tool", step_seq=1, phase=phase, idempotent=False
        )
        assert step.durability is Durability.DURABLE
