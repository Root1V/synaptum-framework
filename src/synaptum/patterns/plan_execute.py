"""
PlanAndExecuteAgent — explicit-plan, adaptive-execution coordination pattern.

Design
------
The Plan-and-Execute pattern separates *planning* from *execution*:

  1. **Plan**    — a ``planner`` agent receives the goal and produces an
                   explicit, ordered list of steps.  Each step declares which
                   executor (specialist agent) handles it and why.

  2. **Execute** — steps are run sequentially.  After each step the accumulated
                   context grows; the executor receives the full history so far.

  3. **Replan**  — before executing the *next* step, the planner evaluates the
                   latest result and can revise, add, or remove the remaining
                   steps.  This is the key difference from Supervisor/Worker:
                   the plan is a living artefact, not a one-shot assignment.

  4. **Finalise** — once no more steps remain a ``finalizer`` agent synthesises
                    all step results into the final deliverable.

Difference from other patterns
-------------------------------
  Supervisor/Worker   → dynamic dispatch, no upfront plan, workers are
                        unaware of each other
  Plan-and-Execute    → explicit plan first, sequential execution, mid-run
                        replanning based on emerging information

Usage
-----
::

    from synaptum.patterns.plan_execute import PlanAndExecuteAgent

    analyst    = SimpleAgent("financial-analyst", ...)
    legal      = SimpleAgent("legal-advisor", ...)
    market     = SimpleAgent("market-analyst", ...)
    planner    = SimpleAgent("deal-planner",  ...)
    finalizer  = SimpleAgent("deal-writer",   ...)

    processor = PlanAndExecuteAgent(
        "debt-restructuring",
        planner   = planner,
        finalizer = finalizer,
        executors = {
            "financial-analyst": analyst,
            "legal-advisor":     legal,
            "market-analyst":    market,
        },
        submit_type = "restructuring.submitted",
        result_type = "restructuring.report",
        max_replans = 2,
        verbose     = True,
    )

    runtime.register(processor)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef


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


# ── Agent ─────────────────────────────────────────────────────────────────────

class PlanAndExecuteAgent(Agent):
    """
    Generates an explicit step-by-step plan, executes each step with
    specialist agents, and adaptively replans based on intermediate results.

    Parameters
    ----------
    name : str
        Bus address for this agent.
    planner : Agent
        Produces the initial ``Plan`` and ``ReplanDecision`` objects.
        Must use ``output_model=Plan`` for the initial call; the agent is
        re-used with a different prompt for replan decisions.
    finalizer : Agent
        Synthesises all step results into the final deliverable.
    executors : dict[str, Agent]
        Pool of specialist agents keyed by name.  Planner assigns steps to
        these agents by name.
    submit_type : str
        Message type that triggers a run.
    result_type : str
        Message type for the final result.
    max_replans : int
        Maximum number of replan cycles allowed (default: 2).
    verbose : bool
        Print plan and execution progress to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        planner: Agent,
        finalizer: Agent,
        executors: Dict[str, Agent],
        replanner: Optional[Agent] = None,
        submit_type: str = "plan_execute.submitted",
        result_type: str = "plan_execute.result",
        max_replans: int = 2,
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.planner     = planner
        self.replanner   = replanner or planner  # separate agent for ReplanDecision
        self.finalizer   = finalizer
        self.executors   = executors
        self.submit_type = submit_type
        self.result_type = result_type
        self.max_replans = max_replans
        self.verbose     = verbose
        self._ref: Optional[AgentRef] = None

    # ── Runtime binding ───────────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        runtime.register(self.planner)
        if self.replanner is not self.planner:
            runtime.register(self.replanner)
        runtime.register(self.finalizer)
        for agent in self.executors.values():
            runtime.register(agent)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute(message)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute(self, message: Message) -> None:
        if self._ref is None:
            raise RuntimeError(
                f"PlanAndExecuteAgent '{self.name}' has not been bound to a runtime."
            )

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender

        # ── 1. Plan ───────────────────────────────────────────────────────────
        plan_prompt = self._plan_prompt(payload)
        raw_plan    = await self.planner.think(plan_prompt)
        plan        = self._coerce_plan(raw_plan)

        remaining_steps: List[Step] = list(plan.steps)
        completed: List[Dict[str, Any]] = []
        replan_count = 0

        if self.verbose:
            print(f"\n── [{self.name}]  GOAL: {plan.goal} ──")
            print(f"   Initial plan ({len(remaining_steps)} steps):")
            for s in remaining_steps:
                print(f"     [{s.id}] {s.executor} — {s.description[:70]}")

        # ── 2. Execute + Replan loop ──────────────────────────────────────────
        while remaining_steps:
            step = remaining_steps.pop(0)
            executor = self.executors.get(step.executor)
            if executor is None:
                raise KeyError(
                    f"No executor registered for '{step.executor}'. "
                    f"Available: {list(self.executors.keys())}"
                )

            if self.verbose:
                print(f"\n   ▶ [{step.id}] {step.executor}: {step.description[:60]}…")

            # Execute the step
            exec_prompt = self._exec_prompt(payload, plan.goal, step, completed)
            raw_result  = await executor.think(exec_prompt)
            result_dict = (
                raw_result.model_dump()
                if hasattr(raw_result, "model_dump")
                else (raw_result if isinstance(raw_result, dict) else {"text": raw_result})
            )
            completed.append({"step": step.model_dump(), "result": result_dict})

            if self.verbose:
                preview = str(result_dict)[:120].replace("\n", " ")
                print(f"     ✓ result: {preview}…")

            # Replan if there are steps remaining and budget allows
            if remaining_steps and replan_count < self.max_replans:
                replan_prompt   = self._replan_prompt(
                    payload, plan.goal, step, result_dict, remaining_steps, completed,
                    available_executors=list(self.executors.keys()),
                )
                raw_replan      = await self.replanner.think(replan_prompt)
                replan          = self._coerce_replan(raw_replan)

                if replan.should_replan:
                    replan_count += 1
                    remaining_steps = list(replan.revised_steps)
                    if self.verbose:
                        print(
                            f"\n   ↺  REPLAN #{replan_count}: {replan.reason}\n"
                            f"     Revised plan ({len(remaining_steps)} remaining steps):"
                        )
                        for s in remaining_steps:
                            print(f"       [{s.id}] {s.executor} — {s.description[:70]}")
                else:
                    if self.verbose:
                        print(f"     → no replan needed: {replan.reason}")

        # ── 3. Finalise ───────────────────────────────────────────────────────
        if self.verbose:
            print(f"\n   ■  All steps complete. Finalising…")

        final_prompt = self._final_prompt(payload, plan.goal, completed)
        raw_final    = await self.finalizer.think(final_prompt)
        final_result = (
            raw_final.model_dump()
            if hasattr(raw_final, "model_dump")
            else (raw_final if isinstance(raw_final, dict) else {"text": raw_final})
        )

        if self.verbose:
            print(f"   ■  Sending '{self.result_type}' to '{caller}'")

        # ── 4. Deliver ────────────────────────────────────────────────────────
        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = {
                "result":    final_result,
                "plan":      plan.model_dump(),
                "steps":     completed,
                "replans":   replan_count,
                "input":     payload,
            },
            metadata = {"in_reply_to": message.id},
        )

    # ── Coercion helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _coerce_plan(raw: Any) -> Plan:
        """Accept a Plan instance, a dict, or a JSON string and return Plan."""
        if isinstance(raw, Plan):
            return raw
        if isinstance(raw, dict):
            return Plan.model_validate(raw)
        if isinstance(raw, str):
            return Plan.model_validate_json(raw)
        raise TypeError(f"Cannot coerce {type(raw)} to Plan")

    @staticmethod
    def _coerce_replan(raw: Any) -> ReplanDecision:
        """Accept a ReplanDecision, a dict, or a JSON string."""
        if isinstance(raw, ReplanDecision):
            return raw
        if isinstance(raw, dict):
            return ReplanDecision.model_validate(raw)
        if isinstance(raw, str):
            return ReplanDecision.model_validate_json(raw)
        raise TypeError(f"Cannot coerce {type(raw)} to ReplanDecision")

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _plan_prompt(self, payload: Dict[str, Any]) -> str:
        context = "\n".join(
            f"  {k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}"
            for k, v in payload.items()
        )
        executor_note = (
            "\nAvailable executors (use exact names in the 'executor' field):\n"
            + "\n".join(f"  · {e}" for e in self.executors.keys())
        )
        return (
            "You are the planning agent. Analyse the goal below and produce a\n"
            "structured plan: an ordered list of steps, each assigned to a\n"
            f"specialist executor by name.{executor_note}\n\n"
            f"Goal context:\n{context}\n\n"
            "Respond ONLY with a valid JSON object matching the Plan schema."
        )

    @staticmethod
    def _exec_prompt(
        payload: Dict[str, Any],
        goal: str,
        step: Step,
        completed: List[Dict[str, Any]],
    ) -> str:
        lines = [
            f"Overall goal: {goal}",
            f"\nYour task ({step.id}): {step.description}",
            f"Rationale: {step.rationale}",
        ]
        if completed:
            lines.append("\nContext from previous steps:")
            for entry in completed:
                s = entry["step"]
                r = entry["result"]
                lines.append(f"\n  [{s['id']}] {s['executor']}:")
                if isinstance(r, dict):
                    for k, v in r.items():
                        if isinstance(v, list):
                            lines.append(f"    {k}: {', '.join(str(x) for x in v)}")
                        else:
                            lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"    {r}")

        # Include raw payload for reference
        lines.append("\nOriginal request:")
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                lines.append(f"  {k}: {json.dumps(v, ensure_ascii=False)[:300]}")
            else:
                lines.append(f"  {k}: {v}")

        lines.append("\nRespond ONLY with a valid JSON object matching the required schema.")
        return "\n".join(lines)

    @staticmethod
    def _replan_prompt(
        payload: Dict[str, Any],
        goal: str,
        completed_step: Step,
        step_result: Dict[str, Any],
        remaining: List[Step],
        all_completed: List[Dict[str, Any]],
        available_executors: Optional[List[str]] = None,
    ) -> str:
        remaining_summary = "\n".join(
            f"  [{s.id}] {s.executor}: {s.description[:80]}" for s in remaining
        )
        result_summary = "\n".join(
            f"  {k}: {v}" for k, v in step_result.items()
            if not isinstance(v, (dict, list))
        )
        executor_note = (
            f"\nIMPORTANT: You may ONLY assign steps to these executors (exact names):\n"
            + "\n".join(f"  · {e}" for e in (available_executors or []))
            + "\nDo NOT invent new executor names."
            if available_executors
            else ""
        )
        lines = [
            f"Overall goal: {goal}",
            f"\nJust completed: [{completed_step.id}] {completed_step.executor}",
            f"Result summary:\n{result_summary}",
            f"\nRemaining steps in current plan:\n{remaining_summary}",
            executor_note,
            "\nDecide: does the result above reveal new information that requires",
            "changing the remaining steps? Set should_replan=true only if",
            "the plan genuinely needs adjustment — avoid unnecessary replans.",
            "If replanning, provide the complete revised list of remaining steps.",
            "\nRespond ONLY with a valid JSON object matching the ReplanDecision schema.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _final_prompt(
        payload: Dict[str, Any],
        goal: str,
        completed: List[Dict[str, Any]],
    ) -> str:
        lines = [
            f"Overall goal: {goal}",
            f"\nAll {len(completed)} steps have been executed. Their results are below.",
            "Synthesise everything into the final deliverable.\n",
        ]
        for entry in completed:
            s = entry["step"]
            r = entry["result"]
            lines.append(f"── [{s['id']}] {s['executor']}: {s['description'][:80]}")
            if isinstance(r, dict):
                for k, v in r.items():
                    if isinstance(v, list):
                        lines.append(f"  {k}: {', '.join(str(x) for x in v)}")
                    else:
                        lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {r}")
            lines.append("")

        lines.append("Respond ONLY with a valid JSON object matching the required schema.")
        return "\n".join(lines)
