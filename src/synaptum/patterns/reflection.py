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

Architecture — pure message choreography
-----------------------------------------
All communication happens via the message bus.  ``ReflectionAgent`` never
calls ``generator.think()`` or ``critic.think()`` directly.  Instead it
sends request messages and reacts to replies:

  Client           ──[submit_type]──►  ReflectionAgent
  ReflectionAgent  ──[generate_type]──►  generator (LLMAgent)
  generator        ──[agent.output]──►  ReflectionAgent
  ReflectionAgent  ──[critique_type]──►  critic (LLMAgent)
  critic           ──[agent.output]──►  ReflectionAgent
                        │
                   score >= threshold?
                   YES → send result_type to Client
                   NO  → send next generate_type (up to max_iterations)

In-flight state is tracked in two dicts:
  ``_runs``         run_id → _RunState      (full loop context per run)
  ``_pending_msgs`` sent_msg_id → (run_id, phase)   (correlate replies)

The generator and critic are independent agents registered on the same
runtime.  They must be registered explicitly:

    runtime.register(writer)
    runtime.register(reviewer)
    runtime.register(reflection_loop)

Difference from other patterns
-------------------------------
  Supervisor/Worker → workers execute tasks; no quality gate
  Plan-and-Execute  → planner revises steps mid-run based on results
  Swarm             → agents hand off control; no iterative refinement
  Reflection        → SAME output is revised repeatedly based on a score
                      and structured critique — a quality improvement loop

Usage
-----
::

    from synaptum.patterns.reflection import ReflectionAgent

    writer   = LLMAgent("report-writer",   output_model=CreditReport, ...)
    reviewer = LLMAgent("report-reviewer", output_model=Critique,     ...)

    loop = ReflectionAgent(
        "credit-report-loop",
        generator      = writer,
        critic         = reviewer,
        pass_threshold = 8.0,
        max_iterations = 3,
        submit_type    = "report.requested",
        result_type    = "report.final",
        verbose        = True,
    )

    runtime.register(writer)    # registered independently
    runtime.register(reviewer)  # registered independently
    runtime.register(loop)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..core.context import AgentContext
from ..core.message import Message
from ..agents.message_agent import MessageAgent
from ..agents.llm_agent import LLMAgent
from ..prompts.template import PromptTemplate
from ..prompts.provider import PromptProvider


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


# ── Run state ─────────────────────────────────────────────────────────────────

@dataclass
class _RunState:
    """Per-run mutable context held by ReflectionAgent between messages."""
    caller:          str
    original_msg_id: str
    payload:         Dict[str, Any]
    history:         List[Dict[str, Any]] = field(default_factory=list)
    current_output:  Optional[Dict[str, Any]] = None   # output awaiting critique
    best_output:     Any                       = None
    best_score:      float                     = -1.0
    iteration:       int                       = 0
    passed:          bool                      = False


# ── ReflectionAgent ───────────────────────────────────────────────────────────

class ReflectionAgent(MessageAgent):
    """
    Runs a generate → critique → revise loop via pure message choreography.

    Never calls ``generator.think()`` or ``critic.think()`` directly.
    All communication happens through the message bus:

      ┌─────────────────────────────────────────────────────────┐
      │  ReflectionAgent ──[generate_type]──► generator         │
      │  generator       ──[agent.output] ──► ReflectionAgent   │
      │  ReflectionAgent ──[critique_type]──► critic            │
      │  critic          ──[agent.output] ──► ReflectionAgent   │
      │       └─ score >= threshold?                            │
      │            YES ──► deliver result_type to caller        │
      │            NO  ──► next generate_type (up to max_iters) │
      └─────────────────────────────────────────────────────────┘

    In-flight state is tracked in:
      ``_runs``         run_id → _RunState
      ``_pending_msgs`` sent_msg_id → (run_id, phase)

    Parameters
    ----------
    name : str
        Bus address for this agent.
    generator : LLMAgent
        Produces the initial output and all revisions.  Must be registered
        independently: ``runtime.register(generator)``.
    critic : LLMAgent
        Evaluates the generator's output.  Must be registered independently.
        Should be configured with ``output_model=Critique``.
    pass_threshold : float
        Minimum score (0–10) to accept output without further revision. Default: 7.5.
    max_iterations : int
        Maximum number of generate + critique cycles.  Default: 3.
    submit_type : str
        Message type that triggers a run.
    result_type : str
        Message type for the final result.
    generate_type : str
        Message type sent to the generator.
    critique_type : str
        Message type sent to the critic.
    reply_type : str
        Message type expected back from generator and critic. Default: ``agent.output``.
    gen_prompt : PromptTemplate, optional
        Template for the user turn sent to the generator on the FIRST iteration.
        Variables available: ``{payload}``.
    gen_prompt_name : str, optional
        Name to resolve via the runtime's PromptProvider (deferred resolution).
    revision_prompt : PromptTemplate, optional
        Template for the user turn sent to the generator on RETRY iterations.
        Variables available: ``{payload}``, ``{last_score}``, ``{weaknesses}``,
        ``{revision_instructions}``.
    revision_prompt_name : str, optional
        Name to resolve via the runtime's PromptProvider (deferred resolution).
    crit_prompt : PromptTemplate, optional
        Template for the user turn sent to the critic.
        Variables available: ``{payload}``, ``{output}``, ``{iteration}``,
        ``{prior_scores}``.
    crit_prompt_name : str, optional
        Name to resolve via the runtime's PromptProvider (deferred resolution).
    verbose : bool
        Print iteration trace to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        generator: LLMAgent,
        critic: LLMAgent,
        pass_threshold: float = 7.5,
        max_iterations: int = 3,
        submit_type:    str = "reflection.submitted",
        result_type:    str = "reflection.result",
        generate_type:  str = "reflect.generate.requested",
        critique_type:  str = "reflect.critique.requested",
        reply_type:     str = "agent.output",
        gen_prompt:            Optional[PromptTemplate] = None,
        gen_prompt_name:       Optional[str] = None,
        revision_prompt:       Optional[PromptTemplate] = None,
        revision_prompt_name:  Optional[str] = None,
        crit_prompt:           Optional[PromptTemplate] = None,
        crit_prompt_name:      Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self._generator_name = generator.name
        self._critic_name    = critic.name
        self.pass_threshold  = pass_threshold
        self.max_iterations  = max_iterations
        self.submit_type     = submit_type
        self.result_type     = result_type
        self.generate_type   = generate_type
        self.critique_type   = critique_type
        self.reply_type      = reply_type
        self.verbose         = verbose

        self._gen_prompt_tpl:       Optional[PromptTemplate] = gen_prompt
        self._revision_prompt_tpl:  Optional[PromptTemplate] = revision_prompt
        self._crit_prompt_tpl:      Optional[PromptTemplate] = crit_prompt
        self._pending_gen_prompt_name:       Optional[str] = gen_prompt_name
        self._pending_revision_prompt_name:  Optional[str] = revision_prompt_name
        self._pending_crit_prompt_name:      Optional[str] = crit_prompt_name

        self._runs:         Dict[str, _RunState]       = {}
        self._pending_msgs: Dict[str, Tuple[str, str]] = {}  # msg_id → (run_id, phase)

    # ── Prompt resolution ─────────────────────────────────────────────────────

    def _inject_prompt_registry(self, provider: PromptProvider) -> None:
        """Called by AgentRuntime.register() to resolve deferred prompt names."""
        if self._pending_gen_prompt_name is not None:
            self._gen_prompt_tpl = provider.get(self._pending_gen_prompt_name)
            self._pending_gen_prompt_name = None
        if self._pending_revision_prompt_name is not None:
            self._revision_prompt_tpl = provider.get(self._pending_revision_prompt_name)
            self._pending_revision_prompt_name = None
        if self._pending_crit_prompt_name is not None:
            self._crit_prompt_tpl = provider.get(self._pending_crit_prompt_name)
            self._pending_crit_prompt_name = None

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
        )

        if self.verbose:
            print(f"\n── [{self.name}]  REFLECTION LOOP  threshold={self.pass_threshold}/10 ──")

        await self._send_generate(run_id)

    # ── Reply router ──────────────────────────────────────────────────────────

    async def _handle_reply(self, message: Message) -> None:
        in_reply_to = message.metadata.get("in_reply_to")
        entry = self._pending_msgs.pop(in_reply_to, None)
        if entry is None:
            return  # reply not for this agent

        run_id, phase = entry
        state = self._runs.get(run_id)
        if state is None:
            return

        if phase == "generate":
            await self._on_generated(run_id, state, message)
        elif phase == "critique":
            await self._on_critiqued(run_id, state, message)

    # ── Handle generation reply ───────────────────────────────────────────────

    async def _on_generated(
        self, run_id: str, state: _RunState, message: Message
    ) -> None:
        raw = message.payload.get("answer", message.payload)
        output: Dict[str, Any] = (
            raw.model_dump()
            if hasattr(raw, "model_dump")
            else (raw if isinstance(raw, dict) else {"text": str(raw)})
        )
        state.current_output = output

        if self.verbose:
            preview = str(output)[:120].replace("\n", " ")
            print(f"     ✎ Generated: {preview}…")

        await self._send_critique(run_id, state, output)

    # ── Handle critique reply ─────────────────────────────────────────────────

    async def _on_critiqued(
        self, run_id: str, state: _RunState, message: Message
    ) -> None:
        raw_critique = message.payload.get("answer", message.payload)
        critique     = self._coerce_critique(raw_critique)
        output       = state.current_output or {}

        if self.verbose:
            dims = ", ".join(f"{k}={v:.1f}" for k, v in critique.dimension_scores.items())
            print(f"     ✦ Score: {critique.score:.1f}/10  [{dims}]")
            if critique.weaknesses:
                print(f"     ✗ Weaknesses: {'; '.join(critique.weaknesses[:2])}")

        state.history.append({
            "iteration": state.iteration,
            "output":    output,
            "critique":  critique.model_dump(),
        })

        if critique.score > state.best_score:
            state.best_score  = critique.score
            state.best_output = output

        if critique.passed or critique.score >= self.pass_threshold:
            state.passed = True
            if self.verbose:
                print(
                    f"     ✓ PASSED  ({critique.score:.1f} >= {self.pass_threshold}) "
                    f"after {state.iteration} iteration(s)"
                )
            await self._deliver(run_id, state)
            return

        if state.iteration < self.max_iterations:
            if self.verbose:
                print(f"     → Below threshold — regenerating with critique feedback…")
            await self._send_generate(run_id)
        else:
            if self.verbose:
                print(
                    f"\n   ⚠  Max iterations reached. "
                    f"Best score: {state.best_score:.1f}/10 (threshold: {self.pass_threshold}). "
                    f"Returning best output."
                )
            await self._deliver(run_id, state)

    # ── Send helpers ──────────────────────────────────────────────────────────

    async def _send_generate(self, run_id: str) -> None:
        state = self._runs[run_id]
        state.iteration += 1

        if self.verbose:
            label = "Initial draft" if state.iteration == 1 else f"Revision {state.iteration - 1}"
            print(f"\n   ▶ Iteration {state.iteration}/{self.max_iterations} — {label}")

        if state.iteration == 1:
            tpl = self._gen_prompt_tpl
            if tpl is None:
                raise RuntimeError(
                    f"ReflectionAgent '{self.name}': no gen_prompt configured. "
                    "Pass 'gen_prompt' or 'gen_prompt_name'."
                )
            prompt = tpl.render(payload=self._fmt_dict(state.payload))
        else:
            tpl = self._revision_prompt_tpl
            if tpl is None:
                raise RuntimeError(
                    f"ReflectionAgent '{self.name}': no revision_prompt configured. "
                    "Pass 'revision_prompt' or 'revision_prompt_name'."
                )
            last     = state.history[-1]
            critique = last["critique"]
            prompt = tpl.render(
                payload               = self._fmt_dict(state.payload),
                last_score            = f"{critique['score']:.1f}/10",
                weaknesses            = self._fmt_list(critique.get("weaknesses", [])),
                revision_instructions = critique.get("revision_instructions", ""),
            )
        msg_id = await self._ref.send(
            to       = self._generator_name,
            type     = self.generate_type,
            payload  = {"text": prompt},
            reply_to = self.name,
            metadata = {"run_id": run_id},
        )
        self._pending_msgs[msg_id] = (run_id, "generate")

    async def _send_critique(
        self, run_id: str, state: _RunState, output: Dict[str, Any]
    ) -> None:
        if self._crit_prompt_tpl is None:
            raise RuntimeError(
                f"ReflectionAgent '{self.name}': no crit_prompt configured. "
                "Pass 'crit_prompt' or 'crit_prompt_name'."
            )
        prompt = self._crit_prompt_tpl.render(
            payload       = self._fmt_dict(state.payload),
            output        = self._fmt_dict(output),
            iteration     = str(state.iteration),
            prior_scores  = self._fmt_scores(state.history),
        )
        msg_id = await self._ref.send(
            to       = self._critic_name,
            type     = self.critique_type,
            payload  = {"text": prompt},
            reply_to = self.name,
            metadata = {"run_id": run_id},
        )
        self._pending_msgs[msg_id] = (run_id, "critique")

    async def _deliver(self, run_id: str, state: _RunState) -> None:
        if self.verbose:
            print(f"\n   ■  Sending '{self.result_type}' to '{state.caller}'")

        await self._ref.send(
            to      = state.caller,
            type    = self.result_type,
            payload = {
                "result":          state.best_output,
                "score":           state.best_score,
                "passed":          state.passed,
                "iterations_used": len(state.history),
                "history":         state.history,
                "input":           state.payload,
            },
            metadata = {"in_reply_to": state.original_msg_id},
        )
        self._runs.pop(run_id, None)

    # ── Context serializers ───────────────────────────────────────────────────
    # Python serialises raw data only — no headers, bullets, or framing text.
    # All structural prompt text lives in the YAML templates.

    @staticmethod
    def _fmt_dict(d: Dict[str, Any], max_value_len: int = 500) -> str:
        """Serialise a dict as 'key: value' lines."""
        lines: List[str] = []
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:max_value_len]}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_list(items: List[str], prefix: str = "· ") -> str:
        """Serialise a list as prefixed lines."""
        return "\n".join(f"{prefix}{item}" for item in items)

    @staticmethod
    def _fmt_scores(history: List[Dict[str, Any]]) -> str:
        """Serialise iteration scores as plain lines."""
        return "\n".join(
            f"Iteration {h['iteration']}: {h['critique']['score']:.1f}/10"
            for h in history
        )

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
