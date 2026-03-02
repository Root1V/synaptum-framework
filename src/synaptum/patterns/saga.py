"""
SagaAgent — Saga / Compensating Transactions pattern.

Design
------
The Saga pattern manages a sequence of steps that each have *side effects*
(external systems, ledger entries, SWIFT messages, etc.).  If any step fails
the saga automatically runs the *compensating action* for every previously
completed step, in reverse order, restoring the system to its pre-saga state.

  1. **Forward execution** — steps run sequentially.  Each step executor
     receives: the original transaction payload + a summary of all prior
     step results so that later steps can reference earlier outputs.

  2. **Failure detection** — an executor signals failure by returning a
     ``StepResult`` with ``success=False`` and a ``failure_reason``.
     Any Python exception also triggers the same rollback path.

  3. **Compensation (rollback)** — steps are compensated in LIFO order.
     Each compensator receives: the original payload + the step's own
     forward output (so it knows exactly what to undo).

  4. **Outcome** — a ``SagaOutcome`` reports whether the saga committed
     fully or was rolled back, which step failed, which compensations were
     applied, and the per-step audit log.

Difference from other patterns
-------------------------------
  Plan-and-Execute → planner decides the steps dynamically; no rollback
  Swarm            → agents hand off control; no compensation concept
  HITL             → single pause/gate; not a multi-step transaction
  Saga             → FIXED, ordered steps each with a paired compensator;
                     failure at step K triggers rollback of steps K-1…0
                     in reverse order — the classic distributed-txn pattern

Classic use-cases
-----------------
  • Cross-border wire transfer (debit → FX → SWIFT → credit)
  • E-commerce order (inventory reserve → payment → shipping → loyalty points)
  • Loan disbursement (approval record → GL debit → core-banking credit → notify)

Usage
-----
::

    from synaptum.patterns.saga import SagaAgent, SagaStep, StepResult

    saga = SagaAgent(
        "cross-border-wire",
        steps = [
            SagaStep("debit-source",       "Debit USD from payer account",
                     executor=debit_agent,   compensator=credit_back_agent),
            SagaStep("fx-conversion",       "Convert USD to EUR at spot rate",
                     executor=fx_agent,      compensator=fx_reverse_agent),
            SagaStep("swift-transmission",  "Send SWIFT MT103 to correspondent",
                     executor=swift_agent,   compensator=swift_cancel_agent),
            SagaStep("credit-destination",  "Credit EUR to beneficiary account",
                     executor=credit_agent,  compensator=debit_back_agent),
        ],
        submit_type = "wire.saga.started",
        result_type = "wire.saga.completed",
        verbose     = True,
    )
    runtime.register(saga)
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef


# ── Step result schema ─────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Structured result that every executor AND compensator agent must return.

    Fields
    ------
    success : bool
        True → step completed; the saga may proceed to the next step.
        False → step failed; the saga should begin compensation.
    data : dict
        Output data produced by this step (e.g. debit reference, FX rate used,
        SWIFT UETR, credited amount).  Forwarded to subsequent steps and stored
        in the audit log for the compensator's use.
    failure_reason : str
        Human-readable explanation of the failure.  Empty string when success=True.
    reference_id : str
        External reference / transaction ID assigned by this step
        (e.g. journal entry ID, SWIFT UETR).  Empty string if not applicable.
    """
    success: bool = Field(
        description="True if the step completed successfully; False if it failed.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output data produced by this step (forwarded to subsequent steps).",
    )
    failure_reason: str = Field(
        default="",
        description="Explanation of the failure. Empty string when success=True.",
    )
    reference_id: str = Field(
        default="",
        description="External reference ID assigned by this step (e.g. UETR, journal ID).",
    )


# ── Saga outcome schema ────────────────────────────────────────────────────────

class StepAuditEntry(BaseModel):
    """Audit record for one step in the saga execution log."""
    step_name: str
    status: str = Field(
        description="COMPLETED | FAILED | COMPENSATED | COMPENSATION_FAILED",
    )
    reference_id: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    failure_reason: str = ""


class SagaOutcome(BaseModel):
    """
    Final outcome of a saga, delivered to the caller.

    Fields
    ------
    status : str
        COMMITTED — all steps completed successfully.
        ROLLED_BACK — a step failed; all prior steps have been compensated.
        PARTIAL_ROLLBACK — compensation itself failed for one or more steps
                           (system requires manual intervention).
    failed_step : str
        Name of the step that triggered rollback.  Empty if status=COMMITTED.
    failure_reason : str
        Reason reported by the failed step.  Empty if status=COMMITTED.
    steps_completed : int
        Number of forward steps that ran successfully before failure.
    compensations_applied : list[str]
        Names of steps whose compensating actions were executed.
    audit_log : list[StepAuditEntry]
        Full per-step execution history in chronological order.
    summary : str
        One-paragraph narrative of what happened (human-readable).
    """
    status: str = Field(
        description="COMMITTED | ROLLED_BACK | PARTIAL_ROLLBACK",
    )
    failed_step: str = Field(
        default="",
        description="Name of the step that triggered rollback.",
    )
    failure_reason: str = Field(
        default="",
        description="Failure reason from the failed step.",
    )
    steps_completed: int = Field(
        description="Number of forward steps that completed before any failure.",
    )
    compensations_applied: List[str] = Field(
        default_factory=list,
        description="Names of compensated steps.",
    )
    audit_log: List[StepAuditEntry] = Field(
        default_factory=list,
        description="Full execution history in chronological order.",
    )
    summary: str = Field(
        description="One-paragraph narrative summary of the saga execution.",
    )


# ── SagaStep dataclass ─────────────────────────────────────────────────────────

@dataclass
class SagaStep:
    """
    Defines one step in a saga: a named forward action + its compensating action.

    Parameters
    ----------
    name : str
        Unique identifier for this step (e.g. "debit-source").
    description : str
        Human-readable description of what this step does.
    executor : Agent
        LLM agent that performs the forward action.
        Must return a ``StepResult``-compatible JSON response.
    compensator : Agent
        LLM agent that undoes the forward action.
        Must return a ``StepResult``-compatible JSON response.
    """
    name: str
    description: str
    executor: Agent
    compensator: Agent


# ── SagaAgent ─────────────────────────────────────────────────────────────────

class SagaAgent(Agent):
    """
    Orchestrates a fixed sequence of steps with automatic rollback.

    If any step's executor returns ``success=False`` (or raises an exception),
    the saga immediately runs the compensator for each previously completed step
    in reverse order, then delivers a ``SagaOutcome(status="ROLLED_BACK")``.

    If all steps complete successfully it delivers
    ``SagaOutcome(status="COMMITTED")``.

    Parameters
    ----------
    name : str
        Bus address for this agent.
    steps : list[SagaStep]
        Ordered list of saga steps.  Executed left-to-right; compensated right-to-left.
    submit_type : str
        Message type that triggers a saga run.
    result_type : str
        Message type for the final ``SagaOutcome``.
    verbose : bool
        Print execution progress to stdout.  Default: False.
    """

    def __init__(
        self,
        name: str,
        *,
        steps: List[SagaStep],
        submit_type: str = "saga.started",
        result_type: str = "saga.completed",
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.steps       = steps
        self.submit_type = submit_type
        self.result_type = result_type
        self.verbose     = verbose
        self._ref: Optional[AgentRef] = None

    # ── Runtime binding ───────────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        for step in self.steps:
            runtime.register(step.executor)
            runtime.register(step.compensator)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute(message)

    # ── Core execution ────────────────────────────────────────────────────────

    async def _execute(self, message: Message) -> None:
        if self._ref is None:
            raise RuntimeError(
                f"SagaAgent '{self.name}' has not been bound to a runtime."
            )

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender

        _say = self._say
        _say(
            f"\n{'━' * 56}\n"
            f"  SAGA START — {self.name}  ({len(self.steps)} steps)\n"
            f"{'━' * 56}"
        )

        # Completed steps stack: (SagaStep, StepResult)
        completed: List[tuple[SagaStep, StepResult]] = []
        audit_log: List[StepAuditEntry] = []
        prior_results: Dict[str, Any] = {}  # step_name → data dict

        outcome: SagaOutcome

        # ── 1. Forward execution ──────────────────────────────────────────────
        for step in self.steps:
            _say(f"  ▶  [{step.name}]  {step.description}")
            prompt = self._forward_prompt(payload, step, prior_results)

            try:
                raw = await step.executor.think(prompt)
                result = self._coerce_step_result(raw, step.name, is_compensator=False)
            except Exception as exc:
                result = StepResult(
                    success=False,
                    failure_reason=f"Executor raised exception: {exc}\n{traceback.format_exc(limit=3)}",
                )

            if result.success:
                _say(
                    f"     ✓  {step.name} — OK"
                    + (f"  [ref: {result.reference_id}]" if result.reference_id else "")
                )
                audit_log.append(StepAuditEntry(
                    step_name=step.name,
                    status="COMPLETED",
                    reference_id=result.reference_id,
                    data=result.data,
                ))
                completed.append((step, result))
                prior_results[step.name] = result.data
            else:
                _say(f"     ✗  {step.name} — FAILED: {result.failure_reason[:100]}")
                audit_log.append(StepAuditEntry(
                    step_name=step.name,
                    status="FAILED",
                    failure_reason=result.failure_reason,
                ))

                # ── 2. Compensation (rollback) ─────────────────────────────
                outcome = await self._compensate(
                    payload, completed, audit_log,
                    failed_step=step.name,
                    failure_reason=result.failure_reason,
                )
                await self._deliver(caller, message, outcome)
                return

        # ── All steps committed ───────────────────────────────────────────────
        _say(f"\n  ✅  All {len(self.steps)} steps COMMITTED successfully.")
        outcome = SagaOutcome(
            status="COMMITTED",
            steps_completed=len(self.steps),
            audit_log=audit_log,
            summary=(
                f"Saga '{self.name}' committed successfully. "
                f"All {len(self.steps)} steps completed: "
                + ", ".join(s.name for s in self.steps) + "."
            ),
        )
        await self._deliver(caller, message, outcome)

    # ── Compensation orchestrator ─────────────────────────────────────────────

    async def _compensate(
        self,
        payload: Dict[str, Any],
        completed: List[tuple[SagaStep, StepResult]],
        audit_log: List[StepAuditEntry],
        failed_step: str,
        failure_reason: str,
    ) -> SagaOutcome:
        _say = self._say
        _say(
            f"\n  ↩  ROLLING BACK — compensating {len(completed)} completed step(s) "
            f"in reverse order…"
        )

        compensation_failures: List[str] = []
        compensations_applied: List[str] = []

        # LIFO reverse
        for step, forward_result in reversed(completed):
            _say(f"  ↩  compensating [{step.name}]…")
            prompt = self._compensation_prompt(payload, step, forward_result)

            try:
                raw = await step.compensator.think(prompt)
                comp_result = self._coerce_step_result(raw, step.name, is_compensator=True)
            except Exception as exc:
                comp_result = StepResult(
                    success=False,
                    failure_reason=f"Compensator raised exception: {exc}",
                )

            if comp_result.success:
                _say(f"     ✓  {step.name} compensated")
                audit_log.append(StepAuditEntry(
                    step_name=step.name,
                    status="COMPENSATED",
                    reference_id=comp_result.reference_id,
                    data=comp_result.data,
                ))
                compensations_applied.append(step.name)
            else:
                reason = comp_result.failure_reason[:120]
                _say(f"     ⚠  {step.name} COMPENSATION FAILED: {reason}")
                audit_log.append(StepAuditEntry(
                    step_name=step.name,
                    status="COMPENSATION_FAILED",
                    failure_reason=comp_result.failure_reason,
                ))
                compensation_failures.append(step.name)

        final_status = "PARTIAL_ROLLBACK" if compensation_failures else "ROLLED_BACK"
        _say(f"\n  {'⚠' if compensation_failures else '↩'}  {final_status}")

        return SagaOutcome(
            status=final_status,
            failed_step=failed_step,
            failure_reason=failure_reason,
            steps_completed=len(completed),
            compensations_applied=compensations_applied,
            audit_log=audit_log,
            summary=(
                f"Saga '{self.name}' rolled back after failure at step '{failed_step}'. "
                f"Reason: {failure_reason[:120]}. "
                f"{len(compensations_applied)} compensation(s) applied: "
                + (", ".join(compensations_applied) or "none")
                + ("." if not compensation_failures else
                   f".  WARNING — {len(compensation_failures)} compensation(s) failed "
                   f"and require manual intervention: {', '.join(compensation_failures)}.")
            ),
        )

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _forward_prompt(
        payload: Dict[str, Any],
        step: SagaStep,
        prior_results: Dict[str, Any],
    ) -> str:
        parts = [
            f"SAGA STEP: {step.name}",
            f"YOUR TASK: {step.description}",
            "",
            "TRANSACTION PAYLOAD:",
            json.dumps(payload, indent=2, default=str),
        ]
        if prior_results:
            parts += [
                "",
                "PRIOR STEP OUTPUTS (use these to inform your action):",
                json.dumps(prior_results, indent=2, default=str),
            ]
        parts += [
            "",
            "Execute this step now.  Respond ONLY with a valid JSON object with these keys:",
            '  "success"        : true if the step completed, false if it failed',
            '  "data"           : dict of outputs produced (references, amounts, rates, etc.)',
            '  "failure_reason" : explanation if success=false, else empty string ""',
            '  "reference_id"   : external reference / transaction ID, or empty string ""',
            "",
            "Be specific and realistic.  Generate plausible reference IDs (e.g. JE-2026-XXXXX, "
            "UETR: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX).",
        ]
        return "\n".join(parts)

    @staticmethod
    def _compensation_prompt(
        payload: Dict[str, Any],
        step: SagaStep,
        forward_result: StepResult,
    ) -> str:
        parts = [
            f"SAGA COMPENSATION: {step.name}",
            f"YOUR TASK: You are undoing / reversing the step '{step.name}'.",
            f"ORIGINAL STEP DESCRIPTION: {step.description}",
            "",
            "ORIGINAL TRANSACTION PAYLOAD:",
            json.dumps(payload, indent=2, default=str),
            "",
            "OUTPUTS FROM THE ORIGINAL STEP (what you need to reverse):",
            json.dumps({
                "reference_id": forward_result.reference_id,
                "data":         forward_result.data,
            }, indent=2, default=str),
            "",
            "Reverse/undo exactly what the original step did.  "
            "Respond ONLY with a valid JSON object with these keys:",
            '  "success"        : true if the compensation completed, false if it also failed',
            '  "data"           : dict of compensation outputs (reversal refs, amounts, etc.)',
            '  "failure_reason" : explanation if success=false, else empty string ""',
            '  "reference_id"   : reversal reference ID, or empty string ""',
        ]
        return "\n".join(parts)

    # ── Delivery ──────────────────────────────────────────────────────────────

    async def _deliver(
        self,
        caller: str,
        message: Message,
        outcome: SagaOutcome,
    ) -> None:
        self._say(f"  ■  Sending '{self.result_type}' → '{caller}'")
        await self._ref.send(
            to       = caller,
            type     = self.result_type,
            payload  = outcome.model_dump(),
            metadata = {"in_reply_to": message.id},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {msg}")

    def _coerce_step_result(
        self,
        raw: Any,
        step_name: str,
        is_compensator: bool,
    ) -> StepResult:
        label = "compensator" if is_compensator else "executor"
        if isinstance(raw, StepResult):
            return raw
        if isinstance(raw, dict):
            return StepResult.model_validate(raw)
        try:
            import json as _json
            parsed = _json.loads(raw) if isinstance(raw, str) else raw
            return StepResult.model_validate(parsed)
        except Exception as exc:
            raise ValueError(
                f"[{self.name}] {label} for step '{step_name}' returned output "
                f"that cannot be coerced to StepResult: {raw!r}"
            ) from exc
