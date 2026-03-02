"""
ReflectionAgent — iterative self-evaluation / reflection loop pattern.

Design
------
The Reflection pattern separates *generation* from *evaluation*:

  1. **Generate** — a ``generator`` agent produces an initial output given
                    the task and any prior critique.

  2. **Critique**  — a ``critic`` agent evaluates the output against explicit
                     criteria and returns a numeric score plus structured
                     feedback.

  3. **Reflect**   — if the score is below ``pass_threshold``, the generator
                     receives the critique and produces a revised output.
                     This continues up to ``max_iterations`` times.

  4. **Deliver**   — the first output that meets the threshold (or the best
                     output if the limit is hit) is sent to the caller.

Difference from other patterns
-------------------------------
  Supervisor/Worker → workers execute tasks; no quality gate
  Plan-and-Execute  → planner revises the plan, not the output quality
  Swarm             → agents hand off control; no iterative refinement
  Reflection        → SAME output is revised repeatedly based on a score
                      and structured critique — a quality improvement loop

Usage
-----
::

    from synaptum.patterns.reflection import ReflectionAgent

    writer  = SimpleAgent("report-writer",   output_model=CreditReport, ...)
    critic  = SimpleAgent("report-reviewer", output_model=Critique,     ...)

    loop = ReflectionAgent(
        "credit-report-loop",
        generator       = writer,
        critic          = critic,
        pass_threshold  = 8.0,        # score out of 10
        max_iterations  = 3,
        submit_type     = "report.requested",
        result_type     = "report.final",
        verbose         = True,
    )
    runtime.register(loop)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef


# ── Critique data structure ───────────────────────────────────────────────────

class Critique(BaseModel):
    """
    Structured evaluation returned by the critic agent.

    Fields
    ------
    score : float
        Overall quality score from 0.0 to 10.0.
    passed : bool
        True if the output meets the required standard.
    dimension_scores : dict[str, float]
        Per-criterion scores (criterion name → 0.0–10.0).
    strengths : list[str]
        What the output does well.  At least one item.
    weaknesses : list[str]
        Specific, actionable issues.  Empty list if score >= 9.
    revision_instructions : str
        Concrete, prioritised instructions for the generator to follow
        when producing the next revision.  Empty string if passed=true.
    """
    score: float = Field(
        ge=0.0, le=10.0,
        description="Overall quality score 0.0–10.0.",
    )
    passed: bool = Field(
        description="True if the output meets the required quality standard.",
    )
    dimension_scores: Dict[str, float] = Field(
        description="Per-criterion scores, e.g. {'completeness': 8.5, 'accuracy': 7.0}.",
    )
    strengths: List[str] = Field(
        description="Positive aspects of the output (at least one).",
    )
    weaknesses: List[str] = Field(
        description="Specific, actionable issues to address. Empty if score >= 9.",
    )
    revision_instructions: str = Field(
        description=(
            "Prioritised, concrete instructions for the next revision. "
            "Empty string when passed=true."
        ),
    )


# ── ReflectionAgent ───────────────────────────────────────────────────────────

class ReflectionAgent(Agent):
    """
    Runs a generate → critique → revise loop until the output passes quality
    criteria or the iteration budget is exhausted.

    Parameters
    ----------
    name : str
        Bus address for this agent.
    generator : Agent
        Produces the initial output and all revisions.
        Should be configured with the appropriate ``output_model``.
    critic : Agent
        Evaluates the generator's output.
        Should be configured with ``output_model=Critique``.
    pass_threshold : float
        Minimum score (0–10) to accept an output without further revision.
        Default: 7.5.
    max_iterations : int
        Maximum number of generate + critique cycles.  Default: 3.
    submit_type : str
        Message type that triggers a run.
    result_type : str
        Message type for the final result.
    verbose : bool
        Print iteration trace to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        generator: Agent,
        critic: Agent,
        pass_threshold: float = 7.5,
        max_iterations: int = 3,
        submit_type: str = "reflection.submitted",
        result_type: str = "reflection.result",
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.generator      = generator
        self.critic         = critic
        self.pass_threshold = pass_threshold
        self.max_iterations = max_iterations
        self.submit_type    = submit_type
        self.result_type    = result_type
        self.verbose        = verbose
        self._ref: Optional[AgentRef] = None

    # ── Runtime binding ───────────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        runtime.register(self.generator)
        runtime.register(self.critic)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute(message)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute(self, message: Message) -> None:
        if self._ref is None:
            raise RuntimeError(
                f"ReflectionAgent '{self.name}' has not been bound to a runtime."
            )

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender

        if self.verbose:
            print(f"\n── [{self.name}]  REFLECTION LOOP  threshold={self.pass_threshold}/10 ──")

        history: List[Dict[str, Any]] = []   # [{iteration, output, critique}]
        best_output: Any   = None
        best_score: float  = -1.0
        best_critique: Optional[Critique] = None
        passed = False

        for iteration in range(1, self.max_iterations + 1):
            # ── Generate ──────────────────────────────────────────────────────
            gen_prompt = self._gen_prompt(payload, history)

            if self.verbose:
                label = "Initial draft" if iteration == 1 else f"Revision {iteration - 1}"
                print(f"\n   ▶ Iteration {iteration}/{self.max_iterations} — {label}")

            raw_output = await self.generator.think(gen_prompt)
            output = (
                raw_output.model_dump()
                if hasattr(raw_output, "model_dump")
                else (raw_output if isinstance(raw_output, dict) else {"text": raw_output})
            )

            if self.verbose:
                preview = str(output)[:120].replace("\n", " ")
                print(f"     ✎ Generated: {preview}…")

            # ── Critique ──────────────────────────────────────────────────────
            crit_prompt = self._crit_prompt(payload, output, iteration, history)
            raw_critique = await self.critic.think(crit_prompt)
            critique = self._coerce_critique(raw_critique)

            if self.verbose:
                dims = ", ".join(f"{k}={v:.1f}" for k, v in critique.dimension_scores.items())
                print(f"     ✦ Score: {critique.score:.1f}/10  [{dims}]")
                if critique.weaknesses:
                    print(f"     ✗ Weaknesses: {'; '.join(critique.weaknesses[:2])}")

            history.append({
                "iteration": iteration,
                "output":    output,
                "critique":  critique.model_dump(),
            })

            # Track best
            if critique.score > best_score:
                best_score   = critique.score
                best_output  = output
                best_critique = critique

            if critique.passed or critique.score >= self.pass_threshold:
                passed = True
                if self.verbose:
                    print(
                        f"     ✓ PASSED  ({critique.score:.1f} >= {self.pass_threshold}) "
                        f"after {iteration} iteration(s)"
                    )
                break

            if iteration < self.max_iterations and self.verbose:
                print(f"     → Below threshold — regenerating with critique feedback…")

        if not passed and self.verbose:
            print(
                f"\n   ⚠  Max iterations reached. "
                f"Best score: {best_score:.1f}/10 (threshold: {self.pass_threshold}). "
                f"Returning best output."
            )

        if self.verbose:
            print(f"\n   ■  Sending '{self.result_type}' to '{caller}'")

        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = {
                "result":          best_output,
                "score":           best_score,
                "passed":          passed,
                "iterations_used": len(history),
                "history":         history,
                "input":           payload,
            },
            metadata = {"in_reply_to": message.id},
        )

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _gen_prompt(
        payload: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []

        # Task context
        lines.append("── TASK ────────────────────────────────────────────────")
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:500]}")
            else:
                lines.append(f"{k}: {v}")

        # Prior attempts with critique
        if history:
            last = history[-1]
            critique = last["critique"]
            lines.append("")
            lines.append("── PREVIOUS ATTEMPT AND CRITIQUE ───────────────────────")
            lines.append(f"Iteration {last['iteration']} score: {critique['score']:.1f}/10")

            if critique.get("weaknesses"):
                lines.append("Weaknesses identified:")
                for w in critique["weaknesses"]:
                    lines.append(f"  · {w}")

            if critique.get("revision_instructions"):
                lines.append("")
                lines.append("REVISION INSTRUCTIONS (follow these precisely):")
                lines.append(critique["revision_instructions"])

            lines.append("")
            lines.append(
                "Your previous output did NOT meet the quality standard. "
                "Address ALL weaknesses above in your new response."
            )
        else:
            lines.append("")
            lines.append("Produce your best possible output for the task above.")

        lines.append("")
        lines.append("Respond ONLY with a valid JSON object matching the required schema.")
        return "\n".join(lines)

    @staticmethod
    def _crit_prompt(
        payload: Dict[str, Any],
        output: Dict[str, Any],
        iteration: int,
        history: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []

        lines.append("── ORIGINAL TASK ───────────────────────────────────────")
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:400]}")
            else:
                lines.append(f"{k}: {v}")

        lines.append("")
        lines.append(f"── OUTPUT TO EVALUATE (Iteration {iteration}) ──────────────")
        for k, v in output.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:400]}")
            elif isinstance(v, str) and len(v) > 300:
                lines.append(f"{k}:")
                lines.append(f"  {v}")
            else:
                lines.append(f"{k}: {v}")

        # Show improvement trend if prior iterations exist
        if history:
            prior_scores = [h["critique"]["score"] for h in history]
            lines.append("")
            lines.append(
                f"── PRIOR SCORES (for context) ─────────────────────────────"
            )
            for i, s in enumerate(prior_scores, 1):
                lines.append(f"  Iteration {i}: {s:.1f}/10")

        lines.append("")
        lines.append(
            "Evaluate the output above rigorously against the task requirements. "
            "Be strict — a score >= 8.0 means genuinely publication-ready quality."
        )
        lines.append("Respond ONLY with a valid JSON object matching the Critique schema.")
        return "\n".join(lines)

    # ── Coercion ──────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_critique(raw: Any) -> Critique:
        if isinstance(raw, Critique):
            return raw
        if isinstance(raw, dict):
            return Critique.model_validate(raw)
        if isinstance(raw, str):
            return Critique.model_validate_json(raw)
        raise TypeError(f"Cannot coerce {type(raw)} to Critique")
