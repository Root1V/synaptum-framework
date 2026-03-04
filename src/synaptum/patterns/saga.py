"""
SagaAgent — Saga / Compensating Transactions  (choreography model)
===================================================================

Architecture
------------
Classic implementations run sagas as a central loop that calls each step
in turn.  This implementation is fundamentally different: **there is no
orchestrator**.  The saga is a chain of autonomous agents connected by
message passing.  Once started, the ``SagaAgent`` only launches the first
step and steps back.  The chain evolves entirely through the bus.

Execution graph
---------------

  SagaAgent  ─[͟͟saga.fwd͟͟]─▶  _fwd.step-0
                                    │  success  ─[͟͟saga.fwd͟͟]─▶  _fwd.step-1
                                    │                           │  success  ─▶  …  ─▶  _outcome (COMMITTED)
                                    │                           │  failure  ─[͟͟saga.cmp͟͟]─▶  _cmp.step-0  ─▶  _outcome
                                    │  failure  ─[͟͟saga.cmp͟͟]─▶  _outcome (step-0 failed, nothing to compensate)

Each ``_SagaForwardAgent``:
  - Receives ``__saga.fwd__`` on the bus
  - Calls its LLM via ``self.think()`` (inherited from ``LLMAgent``)
  - On success → sends ``__saga.fwd__`` to the next step (or ``_outcome``)
  - On failure → sends ``__saga.cmp__`` to start the compensation chain

Each ``_SagaCompensatorAgent``:
  - Receives ``__saga.cmp__`` on the bus
  - Calls its LLM to undo the forward action
  - Sends ``__saga.cmp__`` to the previous compensator (LIFO) or to ``_outcome``

``_SagaOutcomeAgent``:
  - Terminal node: receives final ``__saga.fwd__`` (COMMITTED) or
    ``__saga.cmp__`` (ROLLED_BACK / PARTIAL_ROLLBACK) and delivers
    ``SagaOutcome`` to the original caller.

Saga state
----------
All state is carried **in the message payload** under a reserved
``__saga__`` key, evolving immutably as messages flow through the chain.
No shared mutable state outside the bus.

Public API
----------
::

    from synaptum.patterns.saga import SagaAgent, SagaStep, StepResult, SagaOutcome

    saga = SagaAgent(
        "wire-saga",
        steps = [
            SagaStep(
                name              = "debit-source",
                description       = "Debit USD from payer account",
                forward_prompt    = "bank.saga.debit_source.system",
                compensate_prompt = "bank.saga.credit_source.system",
            ),
            ...
        ],
        trigger_type = "wire.saga.started",
        result_type  = "wire.saga.completed",
        verbose      = True,
    )
    runtime.register(saga)   # registers saga + all internal step/compensator agents
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.composite_agent import CompositeAgent
from ..agents.llm_agent import LLMAgent
from ..agents.message_agent import MessageAgent


# ── Internal message types ─────────────────────────────────────────────────────────────────────────────
_SAGA_FWD = "__saga.fwd__"
_SAGA_CMP = "__saga.cmp__"


# ── Public result schemas ─────────────────────────────────────────────────────────────────────────────────
class StepResult(BaseModel):
    """
    Structured result that every forward executor and compensator must return.

    Fields
    ------
    success        : True → step completed; False → step failed / compensation applied.
    data           : Output data produced (e.g. debit reference, FX rate, UETR).
    failure_reason : Human-readable failure description.  Empty when success=True.
    reference_id   : External reference assigned by this step (e.g. "JE-2026-001").
    """
    success: bool = Field(
        description="True if the step completed; False if it failed.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output data produced by this step.",
    )
    failure_reason: str = Field(
        default="",
        description="Failure description.  Empty when success=True.",
    )
    reference_id: str = Field(
        default="",
        description="External reference ID assigned by this step.",
    )


class StepAuditEntry(BaseModel):
    """Immutable audit record written for every step execution in the saga log."""
    step_name: str = Field(
        description="Name of the saga step this entry belongs to (e.g. 'debit-source').",
    )
    status: str = Field(
        description=(
            "Execution result of this step. "
            "One of: COMPLETED, FAILED, COMPENSATED, COMPENSATION_FAILED."
        ),
    )
    reference_id: str = Field(
        default="",
        description="External reference ID produced by this step execution (e.g. 'JE-2026-001').  Empty if none.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output data snapshot captured at the time of execution.",
    )
    failure_reason: str = Field(
        default="",
        description="Failure description when status is FAILED or COMPENSATION_FAILED.  Empty otherwise.",
    )


class SagaOutcome(BaseModel):
    """
    Final outcome delivered to the saga caller once the chain terminates.

    status
    ------
    COMMITTED        — all steps completed successfully; no rollback needed.
    ROLLED_BACK      — a step failed; all previously completed steps were
                       successfully compensated.
    PARTIAL_ROLLBACK — a step failed AND at least one compensator also failed;
                       manual intervention is required to restore consistency.
    """
    status: str = Field(
        description="Terminal status of the saga: COMMITTED | ROLLED_BACK | PARTIAL_ROLLBACK.",
    )
    failed_step: str = Field(
        default="",
        description="Name of the step that triggered rollback.  Empty when status=COMMITTED.",
    )
    failure_reason: str = Field(
        default="",
        description="Failure reason reported by the failed step.  Empty when status=COMMITTED.",
    )
    steps_completed: int = Field(
        description="Number of forward steps that completed successfully before any failure.",
    )
    compensations_applied: List[str] = Field(
        default_factory=list,
        description="Names of the steps whose compensating actions were executed successfully.",
    )
    audit_log: List[StepAuditEntry] = Field(
        default_factory=list,
        description="Full per-step execution history in chronological order (forward + compensation).",
    )
    summary: str = Field(
        description="One-paragraph narrative summary of the saga execution, suitable for logging or display.",
    )


# ── SagaStep definition ─────────────────────────────────────────────────────────────────────────────────────────

class SagaStep(BaseModel):
    """
    Declares one step in a saga: a named forward action paired with its
    compensating action.  Both are expressed as prompt keys resolved at
    runtime from the configured ``PromptProvider``.
    """
    name: str = Field(
        description="Unique identifier for this step, used as the bus agent name suffix (e.g. 'debit-source').",
    )
    description: str = Field(
        description="Human-readable description of what this step does, injected into the LLM prompt.",
    )
    forward_prompt: str = Field(
        description="Prompt registry key for the executor LLM agent (the forward action).",
    )
    compensate_prompt: str = Field(
        description="Prompt registry key for the compensator LLM agent (the undo action).",
    )


# ── Internal agent: forward step ─────────────────────────────────────────────────────────────────────────────────────

class _SagaForwardAgent(LLMAgent):
    """
    Autonomous forward-execution node in the saga chain.

    Inherits LLM infrastructure (``think()``, prompt resolution, ``_ref``)
    from ``LLMAgent`` and overrides ``on_message`` to react only to
    ``__saga.fwd__`` messages.

    - On success → emits ``__saga.fwd__`` to ``success_target``
    - On failure → emits ``__saga.cmp__`` to ``failure_target``

    All saga state is immutably propagated through the ``__saga__`` key in
    the message payload — this agent keeps no local state between runs.
    """

    def __init__(
        self,
        name: str,
        *,
        step_name:        str,
        step_description: str,
        prompt_name:      str,
        success_target:   str,
        failure_target:   str,
        verbose:          bool = False,
    ) -> None:
        super().__init__(name, prompt_name=prompt_name, output_model=StepResult)
        self.step_name        = step_name
        self.step_description = step_description
        self.success_target   = success_target
        self.failure_target   = failure_target
        self.verbose          = verbose
        # _bind_runtime, _ref and think() are all inherited from LLMAgent

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _SAGA_FWD:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload = message.payload
        saga    = dict(payload.get("__saga__", {}))
        prior   = saga.get("steps_state", {})

        if self.verbose:
            print(f"[saga] ▶  {self.step_name}")

        user_prompt = _forward_prompt(
            payload, self.step_name, self.step_description, prior
        )
        try:
            result: StepResult = await self.think(user_prompt)
        except Exception as exc:
            result = StepResult(
                success=False,
                failure_reason=f"{exc}\n{traceback.format_exc(limit=2)}",
            )

        new_payload = _apply_forward_result(payload, saga, self.step_name, result)

        if result.success:
            if self.verbose:
                ref = f"  [ref: {result.reference_id}]" if result.reference_id else ""
                print(f"[saga]    ✓  {self.step_name}{ref}")
            await self._ref.send(
                to       = self.success_target,
                type     = _SAGA_FWD,
                payload  = new_payload,
                reply_to = message.reply_to,
            )
        else:
            if self.verbose:
                print(f"[saga]    ✗  {self.step_name}: {result.failure_reason[:90]}")
            await self._ref.send(
                to       = self.failure_target,
                type     = _SAGA_CMP,
                payload  = new_payload,
                reply_to = message.reply_to,
            )


# ── Internal agent: compensator step ──────────────────────────────────────────────────────────────────────────────────

class _SagaCompensatorAgent(LLMAgent):
    """
    Autonomous compensation node in the saga rollback chain.

    Inherits LLM infrastructure (``think()``, prompt resolution, ``_ref``)
    from ``LLMAgent`` and overrides ``on_message`` to react only to
    ``__saga.cmp__`` messages.

    Receives ``__saga.cmp__``, calls the LLM to undo the forward step, then
    emits ``__saga.cmp__`` to the previous compensator in LIFO order, or to
    ``_SagaOutcomeAgent`` when the rollback chain is exhausted.

    The forward step's output is read from
    ``__saga__["steps_state"][step_name]`` so the compensator knows exactly
    what to undo.
    """

    def __init__(
        self,
        name: str,
        *,
        step_name:        str,
        step_description: str,
        prompt_name:      str,
        next_target:      str,
        verbose:          bool = False,
    ) -> None:
        super().__init__(name, prompt_name=prompt_name, output_model=StepResult)
        self.step_name        = step_name
        self.step_description = step_description
        self.next_target      = next_target
        self.verbose          = verbose
        # _bind_runtime, _ref and think() are all inherited from LLMAgent

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _SAGA_CMP:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload      = message.payload
        saga         = dict(payload.get("__saga__", {}))
        fwd_raw      = saga.get("steps_state", {}).get(self.step_name, {})

        if self.verbose:
            print(f"[saga] ↩  compensating {self.step_name}…")

        user_prompt = _compensation_prompt(
            payload, self.step_name, self.step_description, fwd_raw
        )
        try:
            result: StepResult = await self.think(user_prompt)
        except Exception as exc:
            result = StepResult(
                success=False,
                failure_reason=f"{exc}\n{traceback.format_exc(limit=2)}",
            )

        new_payload = _apply_compensation_result(payload, saga, self.step_name, result)

        if self.verbose:
            if result.success:
                ref = f"  [ref: {result.reference_id}]" if result.reference_id else ""
                print(f"[saga]    ✓  {self.step_name} compensated{ref}")
            else:
                print(f"[saga]    ⚠  {self.step_name} compensation FAILED: {result.failure_reason[:80]}")

        # Continue LIFO chain → previous compensator or outcome
        await self._ref.send(
            to       = self.next_target,
            type     = _SAGA_CMP,
            payload  = new_payload,
            reply_to = message.reply_to,
        )


# ── Internal agent: outcome finalizer ───────────────────────────────────────────────────────────────────────────

class _SagaOutcomeAgent(MessageAgent):
    """
    Terminal node of the saga chain.

    Receives:
    - ``__saga.fwd__`` from the last forward step (COMMITTED)
    - ``__saga.cmp__`` from the first compensator / or directly from a failing
      step-0 (ROLLED_BACK / PARTIAL_ROLLBACK)

    Assembles the ``SagaOutcome`` and delivers it to the original caller.
    """

    def __init__(
        self,
        name: str,
        *,
        saga_name:   str,
        result_type: str,
        step_names:  List[str],
        verbose:     bool = False,
    ) -> None:
        super().__init__(name)
        self.saga_name   = saga_name
        self.result_type = result_type
        self.step_names  = step_names
        self.verbose     = verbose
        # self._ref inherited from MessageAgent

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type in (_SAGA_FWD, _SAGA_CMP):
            await self._finalize(message)

    async def _finalize(self, message: Message) -> None:
        payload  = message.payload
        saga     = payload.get("__saga__", {})
        caller   = message.reply_to or saga.get("caller", "")

        audit: List[StepAuditEntry] = [
            StepAuditEntry.model_validate(e) for e in saga.get("audit", [])
        ]
        failed_step:   str  = saga.get("failed_step", "")
        failure_reason: str = saga.get("failure_reason", "")
        completed:     List = saga.get("completed", [])
        compensations: List = saga.get("compensations", [])
        comp_failures: List = saga.get("compensation_failures", [])

        if not failed_step:
            # ── COMMITTED ──────────────────────────────────────────────────────────────────
            if self.verbose:
                print(f"[saga] ✅  All {len(self.step_names)} steps COMMITTED.")
            outcome = SagaOutcome(
                status          = "COMMITTED",
                steps_completed = len(completed),
                audit_log       = audit,
                summary         = (
                    f"Saga \'{self.saga_name}\' committed successfully. "
                    f"All {len(self.step_names)} steps completed: "
                    + ", ".join(self.step_names) + "."
                ),
            )
        else:
            # ── ROLLED_BACK / PARTIAL_ROLLBACK ────────────────────────────────────────────────
            final_status = "PARTIAL_ROLLBACK" if comp_failures else "ROLLED_BACK"
            if self.verbose:
                icon = "⚠" if comp_failures else "↩"
                print(f"[saga] {icon}  {final_status}")
            outcome = SagaOutcome(
                status                 = final_status,
                failed_step            = failed_step,
                failure_reason         = failure_reason,
                steps_completed        = len(completed),
                compensations_applied  = compensations,
                audit_log              = audit,
                summary                = (
                    f"Saga '{self.saga_name}' rolled back after failure at "
                    f"step '{failed_step}'. Reason: {failure_reason[:120]}. "
                    f"{len(compensations)} compensation(s) applied: "
                    + (", ".join(compensations) or "none")
                    + ("." if not comp_failures else
                       f". WARNING — {len(comp_failures)} compensation(s) failed "
                       f"(manual intervention required): {', '.join(comp_failures)}.")
                ),
            )

        if self.verbose:
            print(f"[saga] ■  Delivering '{self.result_type}' → '{caller}'")

        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = outcome.model_dump(),
        )


# ── Public entry point ─────────────────────────────────────────────────────────────────────────────────────────
# El patrón tiene nombre: Wiring Factory. _bind_runtime es un constructor de topología, no un contenedor. 
class SagaAgent(CompositeAgent):
    """
    Entry point and topology hub for a choreographed saga.

    Inherits from ``CompositeAgent``: the internal topology
    (``_SagaForwardAgent``, ``_SagaCompensatorAgent``, ``_SagaOutcomeAgent``)
    is built in ``__init__`` and declared via ``sub_agents()``. The runtime
    registers all sub-agents automatically when ``runtime.register(saga)``
    is called — no manual ``runtime.register()`` inside ``_bind_runtime``.

    At runtime, ``SagaAgent`` acts as the public entry point: it receives
    the trigger message, initialises the ``__saga__`` state envelope, and
    fires the first ``__saga.fwd__`` message. After that **it does nothing**
    — the chain is self-driving.

    Parameters
    ----------
    name         : Bus address for this agent (e.g. ``"wire-saga"``).
    steps        : Ordered ``SagaStep`` list.  Forward order left-to-right;
                   compensation order right-to-left on failure.
    trigger_type : Message type that starts a saga run.
    result_type  : Message type for the delivered ``SagaOutcome``.
    verbose      : Print execution progress to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        steps:        List[SagaStep],
        trigger_type: str = "saga.started",
        result_type:  str = "saga.completed",
        verbose:      bool = False,
    ) -> None:
        super().__init__(name)
        self.steps        = steps
        self.trigger_type = trigger_type
        self.result_type  = result_type
        self.verbose      = verbose
        # _ref inherited from CompositeAgent -> MessageAgent

        # Pre-compute internal agent names
        self._outcome_name = f"{name}._outcome"
        self._fwd_names    = [f"{name}._fwd.{s.name}" for s in steps]
        self._cmp_names    = [f"{name}._cmp.{s.name}" for s in steps]

        # Build topology at construction time — runtime registers them via sub_agents()
        self._topology: List[Agent] = self._build_topology()

    # ── Topology declaration ───────────────────────────────────────────────────────────────────────────

    def sub_agents(self) -> List[Agent]:
        """Declares the internal agents that form this saga's topology."""
        return self._topology

    def _build_topology(self) -> List[Agent]:
        """Constructs all internal agents. Called once during ``__init__``."""
        steps = self.steps
        n     = len(steps)
        agents: List[Agent] = []

        # Outcome agent (terminal node)
        agents.append(_SagaOutcomeAgent(
            self._outcome_name,
            saga_name   = self.name,
            result_type = self.result_type,
            step_names  = [s.name for s in steps],
            verbose     = self.verbose,
        ))

        # Compensator agents — cmp[i] sends to cmp[i-1] or outcome (LIFO)
        for i, step in enumerate(steps):
            next_cmp = self._cmp_names[i - 1] if i > 0 else self._outcome_name
            agents.append(_SagaCompensatorAgent(
                self._cmp_names[i],
                step_name        = step.name,
                step_description = step.description,
                prompt_name      = step.compensate_prompt,
                next_target      = next_cmp,
                verbose          = self.verbose,
            ))

        # Forward agents
        # success_target[i] = fwd[i+1]  (outcome for last step)
        # failure_target[i] = cmp[i-1]  (outcome for step 0 — nothing to compensate)
        for i, step in enumerate(steps):
            success_target = self._fwd_names[i + 1] if i < n - 1 else self._outcome_name
            failure_target = self._cmp_names[i - 1] if i > 0 else self._outcome_name
            agents.append(_SagaForwardAgent(
                self._fwd_names[i],
                step_name        = step.name,
                step_description = step.description,
                prompt_name      = step.forward_prompt,
                success_target   = success_target,
                failure_target   = failure_target,
                verbose          = self.verbose,
            ))

        return agents

    # ── Message handling ─────────────────────────────────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type != self.trigger_type:
            return

        caller = message.reply_to or message.sender

        if self.verbose:
            sep = "━" * 56
            print(f"\n{sep}\n  SAGA START — {self.name}  ({len(self.steps)} steps)\n{sep}")

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )

        # Initialise immutable saga-state envelope
        saga_state: Dict[str, Any] = {
            "id":                    self.name,
            "caller":                caller,
            "result_type":           self.result_type,
            "steps_state":           {},
            "completed":             [],
            "compensations":         [],
            "compensation_failures": [],
            "audit":                 [],
            "failed_step":           "",
            "failure_reason":        "",
        }

        # Fire the chain — we step back; the bus does the rest
        await self._ref.send(
            to       = self._fwd_names[0],
            type     = _SAGA_FWD,
            payload  = {**payload, "__saga__": saga_state},
            reply_to = caller,
        )


# ── Prompt builders ─────────────────────────────────────────────────────────────────────────────────────────

def _forward_prompt(
    payload:          Dict[str, Any],
    step_name:        str,
    step_description: str,
    prior_results:    Dict[str, Any],
) -> str:
    clean = {k: v for k, v in payload.items() if k != "__saga__"}
    parts = [
        f"SAGA STEP: {step_name}",
        f"YOUR TASK: {step_description}",
        "",
        "TRANSACTION PAYLOAD:",
        json.dumps(clean, indent=2, default=str),
    ]
    if prior_results:
        parts += [
            "",
            "PRIOR STEP OUTPUTS (use these references in your response):",
            json.dumps(prior_results, indent=2, default=str),
        ]
    parts += [
        "",
        "Execute this step.  Respond ONLY with a valid JSON object:",
        '  "success"        : true if the step completed, false if it failed',
        '  "data"           : dict of outputs (references, amounts, rates, etc.)',
        '  "failure_reason" : explanation if success=false, else ""',
        '  "reference_id"   : external reference ID or ""',
        "",
        "Generate realistic reference IDs (e.g. JE-2026-XXXXX, FXD-2026-XXXXX).",
    ]
    return "\n".join(parts)


def _compensation_prompt(
    payload:          Dict[str, Any],
    step_name:        str,
    step_description: str,
    forward_result:   Dict[str, Any],
) -> str:
    clean = {k: v for k, v in payload.items() if k != "__saga__"}
    parts = [
        f"SAGA COMPENSATION: {step_name}",
        f"YOUR TASK: Reverse / undo step '{step_name}'.",
        f"ORIGINAL STEP: {step_description}",
        "",
        "ORIGINAL TRANSACTION PAYLOAD:",
        json.dumps(clean, indent=2, default=str),
        "",
        "WHAT THE ORIGINAL STEP PRODUCED (undo exactly this):",
        json.dumps(forward_result, indent=2, default=str),
        "",
        "Reverse/undo exactly what the original step did.",
        "Respond ONLY with a valid JSON object:",
        '  "success"        : true if compensation completed, false if it also failed',
        '  "data"           : compensation outputs (reversal refs, amounts, etc.)',
        '  "failure_reason" : explanation if success=false, else ""',
        '  "reference_id"   : reversal reference ID or ""',
    ]
    return "\n".join(parts)


# ── Payload state helpers (pure / immutable) ────────────────────────────────────────────────────────────────

def _apply_forward_result(
    payload:   Dict[str, Any],
    saga:      Dict[str, Any],
    step_name: str,
    result:    StepResult,
) -> Dict[str, Any]:
    new_steps = {**saga.get("steps_state", {}), step_name: result.model_dump()}
    new_audit = list(saga.get("audit", [])) + [
        StepAuditEntry(
            step_name      = step_name,
            status         = "COMPLETED" if result.success else "FAILED",
            reference_id   = result.reference_id,
            data           = result.data,
            failure_reason = result.failure_reason,
        ).model_dump()
    ]
    new_completed = list(saga.get("completed", [])) + ([step_name] if result.success else [])
    patch: Dict[str, Any] = {
        **saga,
        "steps_state": new_steps,
        "audit":       new_audit,
        "completed":   new_completed,
    }
    if not result.success:
        patch["failed_step"]    = step_name
        patch["failure_reason"] = result.failure_reason
    return {**payload, "__saga__": patch}


def _apply_compensation_result(
    payload:   Dict[str, Any],
    saga:      Dict[str, Any],
    step_name: str,
    result:    StepResult,
) -> Dict[str, Any]:
    status    = "COMPENSATED" if result.success else "COMPENSATION_FAILED"
    new_audit = list(saga.get("audit", [])) + [
        StepAuditEntry(
            step_name      = step_name,
            status         = status,
            reference_id   = result.reference_id,
            data           = result.data,
            failure_reason = result.failure_reason,
        ).model_dump()
    ]
    new_comps    = list(saga.get("compensations", []))
    new_failures = list(saga.get("compensation_failures", []))
    if result.success:
        new_comps.append(step_name)
    else:
        new_failures.append(step_name)
    patch: Dict[str, Any] = {
        **saga,
        "audit":                 new_audit,
        "compensations":         new_comps,
        "compensation_failures": new_failures,
    }
    return {**payload, "__saga__": patch}
