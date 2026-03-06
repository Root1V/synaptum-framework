"""
PlanAndExecuteAgent — explicit-plan, adaptive-execution coordination pattern  (choreography model).

Design
------
The Plan-and-Execute pattern separates *planning* from *execution*:

  1. **Plan**    — a ``planner`` agent receives the goal and produces an
                   explicit, ordered list of steps.  Each step declares which
                   executor (specialist agent) handles it and why.

  2. **Execute** — steps are run sequentially via the message bus.  After each
                   step the accumulated context grows; the executor receives
                   the full history so far.

  3. **Replan**  — before the next step, the replanner evaluates the latest
                   result and can revise, add, or remove the remaining steps.
                   This is the key difference from Supervisor/Worker: the plan
                   is a living artefact, not a one-shot assignment.

  4. **Finalise** — once no more steps remain a ``finalizer`` agent synthesises
                    all step results into the final deliverable.

Architecture — pure message choreography
-----------------------------------------
``PlanAndExecuteAgent`` never calls ``agent.think()`` directly.  Every exchange
travels through the message bus:

  Client       ──[submit_type]──►  PlanAndExecuteAgent
  P&E          ──[__pe.plan__]──►  planner (LLMAgent)
  planner      ──[agent.output]──► P&E    → parse Plan, start steps
  P&E          ──[__pe.execute__]──► executor (LLMAgent)
  executor     ──[agent.output]──► P&E    → if remaining: send replan or next step
  P&E          ──[__pe.replan__]──► replanner (LLMAgent)
  replanner    ──[agent.output]──► P&E    → send next execute
  P&E          ──[__pe.finalize__]──► finalizer (LLMAgent)
  finalizer    ──[agent.output]──► P&E    → deliver result_type to caller

In-flight state per run:
  ``_runs``         run_id → _RunState
  ``_pending_msgs`` sent_msg_id → (run_id, phase)

Phases: ``plan`` | ``execute`` | ``replan`` | ``finalize``

All agents (planner, replanner, finalizer, executors) are **external agents
registered independently** by the caller — same pattern as ``ReflectionAgent``.

Difference from other patterns
-------------------------------
  Supervisor/Worker   → dynamic dispatch, no upfront plan, workers are
                        unaware of each other
  Plan-and-Execute    → explicit plan first, sequential execution, mid-run
                        replanning based on emerging information

Usage
-----
::

    from synaptum.patterns.plan_execute import PlanAndExecuteAgent, Plan, ReplanDecision

    planner   = LLMAgent("deal-planner",  prompt_name="...", output_model=Plan)
    replanner = LLMAgent("deal-replanner",prompt_name="...", output_model=ReplanDecision)
    finalizer = LLMAgent("deal-writer",   prompt_name="...")
    analyst   = LLMAgent("financial-analyst", prompt_name="...", output_model=FinancialAnalysis)
    legal     = LLMAgent("legal-advisor",     prompt_name="...", output_model=LegalAnalysis)

    processor = PlanAndExecuteAgent(
        "debt-restructuring",
        planner             = planner,
        replanner           = replanner,
        finalizer           = finalizer,
        executors           = {"financial-analyst": analyst, "legal-advisor": legal},
        submit_type         = "restructuring.submitted",
        result_type         = "restructuring.report",
        plan_prompt_name    = "bank.plan_execute.plan_user_prompt",
        exec_prompt_name    = "bank.plan_execute.exec_user_prompt",
        replan_prompt_name  = "bank.plan_execute.replan_user_prompt",
        final_prompt_name   = "bank.plan_execute.final_user_prompt",
        max_replans         = 2,
        verbose             = True,
    )

    # Register all agents independently, then the coordinator.
    runtime.register(planner)
    runtime.register(replanner)
    runtime.register(finalizer)
    runtime.register(analyst)
    runtime.register(legal)
    runtime.register(processor)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..agents.message_agent import MessageAgent
from ..core.context import AgentContext
from ..core.message import Message
from ..prompts.template import PromptTemplate
from ..prompts.provider import PromptProvider
from ..utils.formatting import fmt_dict, fmt_list, fmt_records


# ── Internal message types ────────────────────────────────────────────────────

_PE_PLAN     = "__pe.plan__"
_PE_EXECUTE  = "__pe.execute__"
_PE_REPLAN   = "__pe.replan__"
_PE_FINALIZE = "__pe.finalize__"


# ── Plan data structures ──────────────────────────────────────────────────────

class Step(BaseModel):
    """A single step in the execution plan."""
    id: str = Field(description="Short unique identifier, e.g. 'step-1'.")
    executor: str = Field(
        description="Name of the executor agent that will handle this step."
    )
    description: str = Field(
        description="Clear instruction telling the executor what to do and what to produce."
    )
    rationale: str = Field(
        description="Why this step is needed at this point in the plan."
    )


class Plan(BaseModel):
    """Initial plan produced by the planner."""
    goal: str = Field(description="Restatement of the overall objective.")
    rationale: str = Field(
        description="Why these steps were chosen in this order."
    )
    steps: List[Step] = Field(
        description="Ordered list of steps to execute. At least 2."
    )


class ReplanDecision(BaseModel):
    """Decision produced after each step — should the remaining plan change?"""
    should_replan: bool = Field(
        description=(
            "True if the latest result reveals new information that requires "
            "changing the remaining steps."
        )
    )
    reason: str = Field(
        description="Brief explanation of why a replan is or is not needed."
    )
    revised_steps: List[Step] = Field(
        description=(
            "Updated list of remaining steps (may be identical to original if "
            "should_replan is false). Must not include already-completed steps."
        )
    )


# ── Run state ─────────────────────────────────────────────────────────────────

@dataclass
class _RunState:
    """Per-run mutable context held by PlanAndExecuteAgent between messages."""
    caller:          str
    original_msg_id: str
    payload:         Dict[str, Any]
    goal:            str                    = ""
    remaining_steps: List[Step]             = field(default_factory=list)
    completed:       List[Dict[str, Any]]   = field(default_factory=list)
    replan_count:    int                    = 0
    t_start:         float                  = 0.0
    t_phase:         float                  = 0.0
    _current_step:   Optional[Step]         = None   # step sent to executor, awaiting reply


# ── Completed-steps serialiser ────────────────────────────────────────────────

def _fmt_completed(completed: List[Dict[str, Any]]) -> str:
    """Serialise completed step results for prompt injection.  Pure data — no headers."""
    lines: List[str] = []
    for entry in completed:
        s = entry["step"]
        r = entry["result"]
        lines.append(f"[{s['id']}] {s['executor']}")
        if isinstance(r, dict):
            for k, v in r.items():
                if isinstance(v, list):
                    lines.append(f"  {k}: {', '.join(str(x) for x in v)}")
                else:
                    lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {r}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ── PlanAndExecuteAgent ───────────────────────────────────────────────────────

class PlanAndExecuteAgent(MessageAgent):
    """
    Choreographs an explicit plan → sequential execution → adaptive replan
    pipeline via pure message bus communication.

    Parameters
    ----------
    name : str
        Bus address for this coordinator.
    planner : Agent
        Produces the initial ``Plan``.  Must use ``output_model=Plan``.
    finalizer : Agent
        Synthesises all step results into the final deliverable.
    executors : dict[str, Agent]
        Pool of specialist agents keyed by bus name.  Planner assigns steps
        to these agents; all must be registered independently.
    replanner : Agent, optional
        Produces ``ReplanDecision`` after each step.  Defaults to ``planner``
        if not provided (same agent, different prompt).
    submit_type : str
        Message type that triggers a run.
    result_type : str
        Message type for the final result.
    reply_type : str
        Message type expected back from all child agents (default: ``agent.output``).
    plan_type / exec_type / replan_type / final_type : str
        Internal message types sent to planner / executor / replanner / finalizer.
    plan_prompt_name : str, optional
        YAML key for the planner user-turn prompt.
        Variables: ``{payload}``, ``{executor_names}``.
    exec_prompt_name : str, optional
        YAML key for the executor user-turn prompt.
        Variables: ``{payload}``, ``{goal}``, ``{step_id}``, ``{step_description}``,
        ``{step_rationale}``, ``{completed_steps}``.
    replan_prompt_name : str, optional
        YAML key for the replanner user-turn prompt.
        Variables: ``{payload}``, ``{goal}``, ``{last_step_id}``, ``{last_executor}``,
        ``{last_result}``, ``{remaining_steps}``, ``{executor_names}``.
    final_prompt_name : str, optional
        YAML key for the finalizer user-turn prompt.
        Variables: ``{payload}``, ``{goal}``, ``{completed_steps}``.
    max_replans : int
        Maximum replan cycles allowed (default: 2).
    verbose : bool
        Print plan and execution progress to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        planner:   Any,
        finalizer: Any,
        executors: Dict[str, Any],
        replanner: Optional[Any] = None,
        submit_type:          str = "plan_execute.submitted",
        result_type:          str = "plan_execute.result",
        reply_type:           str = "agent.output",
        plan_type:            str = _PE_PLAN,
        exec_type:            str = _PE_EXECUTE,
        replan_type:          str = _PE_REPLAN,
        final_type:           str = _PE_FINALIZE,
        plan_prompt_name:     Optional[str] = None,
        exec_prompt_name:     Optional[str] = None,
        replan_prompt_name:   Optional[str] = None,
        final_prompt_name:    Optional[str] = None,
        max_replans:          int  = 2,
        verbose:              bool = False,
    ) -> None:
        super().__init__(name)
        self._planner_name   = planner.name
        self._replanner_name = (replanner or planner).name
        self._finalizer_name = finalizer.name
        self._executor_names = frozenset(executors.keys())

        self.submit_type  = submit_type
        self.result_type  = result_type
        self.reply_type   = reply_type
        self.plan_type    = plan_type
        self.exec_type    = exec_type
        self.replan_type  = replan_type
        self.final_type   = final_type
        self.max_replans  = max_replans
        self.verbose      = verbose

        self._plan_prompt_tpl:   Optional[PromptTemplate] = None
        self._exec_prompt_tpl:   Optional[PromptTemplate] = None
        self._replan_prompt_tpl: Optional[PromptTemplate] = None
        self._final_prompt_tpl:  Optional[PromptTemplate] = None

        self._pending_plan_name:   Optional[str] = plan_prompt_name
        self._pending_exec_name:   Optional[str] = exec_prompt_name
        self._pending_replan_name: Optional[str] = replan_prompt_name
        self._pending_final_name:  Optional[str] = final_prompt_name

        self._runs:         Dict[str, _RunState]        = {}
        self._pending_msgs: Dict[str, Tuple[str, str]]  = {}  # msg_id → (run_id, phase)

    # ── Prompt resolution ─────────────────────────────────────────────────────

    def _inject_prompt_registry(self, provider: PromptProvider) -> None:
        """Called by AgentRuntime.register() to resolve deferred prompt names."""
        if self._pending_plan_name:
            self._plan_prompt_tpl = provider.get(self._pending_plan_name)
            self._pending_plan_name = None
        if self._pending_exec_name:
            self._exec_prompt_tpl = provider.get(self._pending_exec_name)
            self._pending_exec_name = None
        if self._pending_replan_name:
            self._replan_prompt_tpl = provider.get(self._pending_replan_name)
            self._pending_replan_name = None
        if self._pending_final_name:
            self._final_prompt_tpl = provider.get(self._pending_final_name)
            self._pending_final_name = None

    # ── Message routing ───────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._handle_submit(message)
        elif message.type == self.reply_type:
            await self._handle_reply(message)

    # ── Initial request ───────────────────────────────────────────────────────

    async def _handle_submit(self, message: Message) -> None:
        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        run_id = message.id
        self._runs[run_id] = _RunState(
            caller          = message.reply_to or message.sender,
            original_msg_id = message.id,
            payload         = payload,
            t_start         = time.perf_counter(),
        )

        if self.verbose:
            print(f"\n── [{self.name}]  PLAN & EXECUTE  START ──")

        await self._send_plan(run_id)

    # ── Reply router ──────────────────────────────────────────────────────────

    async def _handle_reply(self, message: Message) -> None:
        in_reply_to = message.metadata.get("in_reply_to")
        entry = self._pending_msgs.pop(in_reply_to, None)
        if entry is None:
            return

        run_id, phase = entry
        state = self._runs.get(run_id)
        if state is None:
            return

        if phase == "plan":
            await self._on_plan(run_id, state, message)
        elif phase == "execute":
            await self._on_executed(run_id, state, message)
        elif phase == "replan":
            await self._on_replanned(run_id, state, message)
        elif phase == "finalize":
            await self._on_finalized(run_id, state, message)

    # ── Phase handlers ────────────────────────────────────────────────────────

    async def _on_plan(
        self, run_id: str, state: _RunState, message: Message
    ) -> None:
        elapsed_s = round(time.perf_counter() - state.t_phase, 2)

        raw  = message.payload.get("answer", message.payload)
        plan = self._coerce_plan(raw)

        state.goal            = plan.goal
        state.remaining_steps = list(plan.steps)

        if self.verbose:
            print(f"   Goal: {plan.goal}")
            print(f"   Initial plan ({len(state.remaining_steps)} steps)  ⏱ {elapsed_s:.2f}s")
            for s in state.remaining_steps:
                print(f"     [{s.id}] {s.executor} — {s.description[:70]}")

        if state.remaining_steps:
            await self._send_execute(run_id)
        else:
            await self._send_finalize(run_id)

    async def _on_executed(
        self, run_id: str, state: _RunState, message: Message
    ) -> None:
        elapsed_s = round(time.perf_counter() - state.t_phase, 2)

        raw         = message.payload.get("answer", message.payload)
        result_dict = (
            raw.model_dump() if hasattr(raw, "model_dump")
            else (raw if isinstance(raw, dict) else {"text": str(raw)})
        )
        step = state._current_step
        state.completed.append({
            "step":      step.model_dump(),
            "result":    result_dict,
            "elapsed_s": elapsed_s,
        })

        if self.verbose:
            preview = str(result_dict)[:120].replace("\n", " ")
            print(f"     ✓ {preview}…")
            print(f"     ⏱  {elapsed_s:.2f}s")

        if state.remaining_steps and state.replan_count < self.max_replans:
            await self._send_replan(run_id)
        elif state.remaining_steps:
            # Replan budget exhausted — proceed directly
            await self._send_execute(run_id)
        else:
            await self._send_finalize(run_id)

    async def _on_replanned(
        self, run_id: str, state: _RunState, message: Message
    ) -> None:
        elapsed_s = round(time.perf_counter() - state.t_phase, 2)

        raw    = message.payload.get("answer", message.payload)
        replan = self._coerce_replan(raw)

        if replan.should_replan:
            state.replan_count += 1
            state.remaining_steps = list(replan.revised_steps)
            if self.verbose:
                print(
                    f"\n   ↺  REPLAN #{state.replan_count}: {replan.reason}  ⏱ {elapsed_s:.2f}s"
                )
                for s in state.remaining_steps:
                    print(f"       [{s.id}] {s.executor} — {s.description[:70]}")
        else:
            if self.verbose:
                print(f"     → no replan needed: {replan.reason}  ⏱ {elapsed_s:.2f}s")

        if state.remaining_steps:
            await self._send_execute(run_id)
        else:
            await self._send_finalize(run_id)

    async def _on_finalized(
        self, run_id: str, state: _RunState, message: Message
    ) -> None:
        elapsed_s = round(time.perf_counter() - state.t_phase, 2)

        raw    = message.payload.get("answer", message.payload)
        result = (
            raw.model_dump() if hasattr(raw, "model_dump")
            else (raw if isinstance(raw, dict) else {"text": str(raw)})
        )

        elapsed_total_s = round(time.perf_counter() - state.t_start, 2)

        if self.verbose:
            print(f"\n   ■  Finalised  ⏱ {elapsed_s:.2f}s")
            print(f"   ■  Sending '{self.result_type}' to '{state.caller}'  ({elapsed_total_s:.2f}s total)")

        await self._ref.send(
            to      = state.caller,
            type    = self.result_type,
            payload = {
                "result":          result,
                "plan":            {"goal": state.goal, "steps": [e["step"] for e in state.completed]},
                "steps":           state.completed,
                "replans":         state.replan_count,
                "elapsed_total_s": elapsed_total_s,
                "input":           state.payload,
            },
            metadata = {"in_reply_to": state.original_msg_id},
        )
        self._runs.pop(run_id, None)

    # ── Send helpers ──────────────────────────────────────────────────────────

    async def _send_plan(self, run_id: str) -> None:
        state = self._runs[run_id]
        if self._plan_prompt_tpl is None:
            raise RuntimeError(
                f"PlanAndExecuteAgent '{self.name}': no plan_prompt_name configured."
            )
        prompt = self._plan_prompt_tpl.render(
            payload        = fmt_dict(state.payload),
            executor_names = fmt_list(sorted(self._executor_names)),
        )
        msg_id = await self._ref.send(
            to       = self._planner_name,
            type     = self.plan_type,
            payload  = {"text": prompt},
            reply_to = self.name,
            metadata = {"run_id": run_id},
        )
        state.t_phase = time.perf_counter()
        self._pending_msgs[msg_id] = (run_id, "plan")

    async def _send_execute(self, run_id: str) -> None:
        state = self._runs[run_id]
        step  = state.remaining_steps.pop(0)
        state._current_step = step

        if self.verbose:
            print(f"\n   ▶ [{step.id}] {step.executor}: {step.description[:60]}…")

        if self._exec_prompt_tpl is None:
            raise RuntimeError(
                f"PlanAndExecuteAgent '{self.name}': no exec_prompt_name configured."
            )
        completed_str = _fmt_completed(state.completed) if state.completed else "(none)"
        prompt = self._exec_prompt_tpl.render(
            payload          = fmt_dict(state.payload),
            goal             = state.goal,
            step_id          = step.id,
            step_description = step.description,
            step_rationale   = step.rationale,
            completed_steps  = completed_str,
        )
        msg_id = await self._ref.send(
            to       = step.executor,
            type     = self.exec_type,
            payload  = {"text": prompt},
            reply_to = self.name,
            metadata = {"run_id": run_id, "step_id": step.id},
        )
        state.t_phase = time.perf_counter()
        self._pending_msgs[msg_id] = (run_id, "execute")

    async def _send_replan(self, run_id: str) -> None:
        state    = self._runs[run_id]
        last     = state.completed[-1]
        last_s   = last["step"]
        last_r   = last["result"]

        if self._replan_prompt_tpl is None:
            raise RuntimeError(
                f"PlanAndExecuteAgent '{self.name}': no replan_prompt_name configured."
            )
        prompt = self._replan_prompt_tpl.render(
            payload         = fmt_dict(state.payload),
            goal            = state.goal,
            last_step_id    = last_s["id"],
            last_executor   = last_s["executor"],
            last_result     = fmt_dict(last_r) if isinstance(last_r, dict) else str(last_r),
            remaining_steps = fmt_records(
                [{"step_id": s.id, "executor": s.executor, "description": s.description}
                 for s in state.remaining_steps],
                "[{step_id}] {executor}: {description}",
            ),
            executor_names  = fmt_list(sorted(self._executor_names)),
        )
        msg_id = await self._ref.send(
            to       = self._replanner_name,
            type     = self.replan_type,
            payload  = {"text": prompt},
            reply_to = self.name,
            metadata = {"run_id": run_id},
        )
        state.t_phase = time.perf_counter()
        self._pending_msgs[msg_id] = (run_id, "replan")

    async def _send_finalize(self, run_id: str) -> None:
        state = self._runs[run_id]

        if self.verbose:
            print(f"\n   ■  All steps complete. Finalising…")

        if self._final_prompt_tpl is None:
            raise RuntimeError(
                f"PlanAndExecuteAgent '{self.name}': no final_prompt_name configured."
            )
        prompt = self._final_prompt_tpl.render(
            payload         = fmt_dict(state.payload),
            goal            = state.goal,
            completed_steps = _fmt_completed(state.completed),
        )
        msg_id = await self._ref.send(
            to       = self._finalizer_name,
            type     = self.final_type,
            payload  = {"text": prompt},
            reply_to = self.name,
            metadata = {"run_id": run_id},
        )
        state.t_phase = time.perf_counter()
        self._pending_msgs[msg_id] = (run_id, "finalize")

    # ── Coercion helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _coerce_plan(raw: Any) -> Plan:
        if isinstance(raw, Plan):
            return raw
        if isinstance(raw, dict):
            return Plan.model_validate(raw)
        if isinstance(raw, str):
            return Plan.model_validate_json(raw)
        raise TypeError(f"Cannot coerce {type(raw)} to Plan")

    @staticmethod
    def _coerce_replan(raw: Any) -> ReplanDecision:
        if isinstance(raw, ReplanDecision):
            return raw
        if isinstance(raw, dict):
            return ReplanDecision.model_validate(raw)
        if isinstance(raw, str):
            return ReplanDecision.model_validate_json(raw)
        raise TypeError(f"Cannot coerce {type(raw)} to ReplanDecision")
