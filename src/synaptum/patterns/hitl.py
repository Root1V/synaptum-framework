"""
HITLAgent — Human-in-the-Loop pause pattern.

Design
------
The HITL pattern introduces a mandatory human checkpoint inside an otherwise
automated pipeline:

  1. **Analyze**  — an ``analyzer`` agent runs automated screening / risk
                    assessment on the incoming request and returns a structured
                    ``ScreeningResult``.

  2. **PAUSE**    — the engine builds a ``HumanReviewRequest`` from the
                    screening findings and passes it to ``review_handler``.
                    This is an *async callable* supplied by the application —
                    it may block on a terminal prompt, open a web form, send
                    a Slack message and poll for a response, etc.
                    Execution halts here until the human responds.

  3. **Route**    — based on ``HumanReviewResponse.decision``:
                      • APPROVE / APPROVE_WITH_CONDITIONS → ``executor`` runs
                      • REJECT / REJECT_WITH_FINDINGS    → short-circuit;
                        executor prompt signals rejection
                      • REQUEST_ADDITIONAL_INFO          → treated as soft
                        rejection (caller should resubmit with extras)

  4. **Execute**  — the ``executor`` agent produces the final output,
                    incorporating the original request, screening findings,
                    and the human's decision and any conditions.

  5. **Deliver**  — result is returned to the original caller.

Why HITL matters
-----------------
  Automated pipelines may miss edge cases; high-stakes decisions (large wires,
  credit approvals above threshold, sanctions overrides) often require a human
  signature.  HITL formalises that checkpoint without sacrificing the
  surrounding automation.

Difference from other patterns
-------------------------------
  Reflection       → automated critic scores output and revises it; no human
  Consensus        → multiple LLM panelists vote; no human veto
  Plan-and-Execute → planner + executors; no mandatory pause
  HITL             → execution pause with a HUMAN decision gate; the reviewer
                     can inject free-form conditions / modifications that LLMs
                     then incorporate into the final output

Usage
-----
::

    from synaptum.patterns.hitl import HITLAgent, HumanReviewRequest, HumanReviewResponse

    async def my_review_handler(request: HumanReviewRequest) -> HumanReviewResponse:
        print(request.model_dump_json(indent=2))
        decision = input("Decision: ").strip().upper()
        comments = input("Comments: ").strip()
        return HumanReviewResponse(decision=decision, comments=comments)

    screener = SimpleAgent("aml-screener",    output_model=ScreeningResult, ...)
    ops      = SimpleAgent("wire-operations", output_model=WireInstruction, ...)

    hitl = HITLAgent(
        "wire-edd-review",
        analyzer       = screener,
        executor       = ops,
        review_handler = my_review_handler,
        submit_type    = "wire.transfer.submitted",
        result_type    = "wire.transfer.processed",
        verbose        = True,
    )
    runtime.register(hitl)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef


# ── Public data models ────────────────────────────────────────────────────────

class ScreeningResult(BaseModel):
    """
    Structured output that the *analyzer* agent must return.

    The ``HITLAgent`` maps this directly to a ``HumanReviewRequest``.
    """
    summary: str = Field(
        description="One-sentence summary of the request and its key risk factors.",
    )
    automated_findings: str = Field(
        description=(
            "Detailed findings from the automated screening: all flags raised, "
            "checks passed, data points considered."
        ),
    )
    risk_level: str = Field(
        description="Aggregate risk level: LOW | MEDIUM | HIGH | CRITICAL.",
    )
    question: str = Field(
        description="The specific yes/no (or multi-option) question the human must answer.",
    )
    options: List[str] = Field(
        description=(
            "Decision options available to the reviewer, e.g. "
            "[\"APPROVE\", \"APPROVE_WITH_CONDITIONS\", \"REJECT\"]."
        ),
    )


class HumanReviewRequest(BaseModel):
    """
    Package delivered to the *review_handler*.  Contains everything a human
    reviewer needs to make an informed decision.
    """
    task_id: str = Field(
        description="Unique identifier for this review task (e.g. HITL-A3F8C201).",
    )
    summary: str = Field(
        description="One-sentence case summary.",
    )
    automated_findings: str = Field(
        description="Full text of automated screening findings.",
    )
    risk_level: str = Field(
        description="Risk level as assessed by the automated screener.",
    )
    question: str = Field(
        description="The question the reviewer must answer.",
    )
    options: List[str] = Field(
        description="Available decision options.",
    )
    context_payload: dict = Field(
        default_factory=dict,
        description="Original request payload, serialised.",
    )


class HumanReviewResponse(BaseModel):
    """
    The human reviewer's response.
    """
    decision: str = Field(
        description="One of the options offered in HumanReviewRequest.options.",
    )
    comments: str = Field(
        description="Free-text rationale, notes, or instructions.",
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Explicit conditions the executor must incorporate (if any).",
    )
    modifications: dict = Field(
        default_factory=dict,
        description="Field overrides or additions to the original request (if any).",
    )


# Typing alias for the review handler callable
ReviewHandler = Callable[[HumanReviewRequest], Awaitable[HumanReviewResponse]]

# Decisions that allow the executor to proceed
_APPROVAL_DECISIONS = {
    "APPROVE",
    "APPROVED",
    "APPROVE_WITH_CONDITIONS",
    "APPROVED_WITH_CONDITIONS",
}


# ── HITLAgent ─────────────────────────────────────────────────────────────────

class HITLAgent(Agent):
    """
    Human-in-the-Loop agent that pauses execution at a mandatory human
    checkpoint before producing a final output.

    Parameters
    ----------
    name : str
        Agent name registered in the runtime.
    analyzer : Agent
        Performs automated pre-processing and risk screening.
        Must return a ``ScreeningResult``-compatible response.
    executor : Agent
        Produces the final output after the human decision is known.
        Receives: original request + screening findings + human decision.
    review_handler : ReviewHandler
        Async callable that receives a ``HumanReviewRequest`` and returns a
        ``HumanReviewResponse``.  This is the pause point.
    submit_type : str
        Message type string that triggers this agent (e.g. "wire.edd.submitted").
    result_type : str
        Message type string used for the outgoing result (e.g. "wire.edd.processed").
    verbose : bool
        Print step-by-step progress to stdout.  Default: False.
    """

    def __init__(
        self,
        name: str,
        *,
        analyzer: Agent,
        executor: Agent,
        review_handler: ReviewHandler,
        submit_type: str,
        result_type: str,
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.analyzer      = analyzer
        self.executor      = executor
        self.review_handler = review_handler
        self.submit_type   = submit_type
        self.result_type   = result_type
        self.verbose       = verbose
        self._ref: Optional[AgentRef] = None

    # ── Runtime registration ──────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        runtime.register(self.analyzer)
        runtime.register(self.executor)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute(message)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    async def _execute(self, message: Message) -> None:
        if self._ref is None:
            raise RuntimeError(
                f"HITLAgent '{self.name}' has not been bound to a runtime."
            )

        payload: Dict[str, Any] = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender

        _say = self._say

        # ── Step 1: Automated analysis / screening ────────────────────────
        _say("▶  Running automated screening…")
        analysis_raw = await self.analyzer.think(self._analysis_prompt(payload))
        screening = self._coerce_screening(analysis_raw)
        _say(f"✓  Screening complete — risk level: {screening.risk_level}")

        # ── Step 2: Build review request ──────────────────────────────────
        review_request = self._build_review_request(payload, screening)
        _say(
            f"\n⏸  PAUSING for human review  [task: {review_request.task_id}]"
            f"\n   Risk: {review_request.risk_level}  |  "
            f"Options: {', '.join(review_request.options)}"
        )

        # ── Step 3: Human decision gate ───────────────────────────────────
        human_response = await self.review_handler(review_request)
        decision_upper = human_response.decision.upper().replace(" ", "_")
        approved = decision_upper in _APPROVAL_DECISIONS

        _say(
            f"\n▶  Human decided: {human_response.decision}"
            + (f"  ({'APPROVED path' if approved else 'REJECTED path'})")
        )
        if human_response.conditions:
            _say(f"   Conditions: {len(human_response.conditions)} item(s)")

        # ── Step 4: Execute based on decision ─────────────────────────────
        _say("▶  Generating final output…")
        result_raw = await self.executor.think(
            self._execution_prompt(payload, screening, human_response, approved)
        )
        final_result = (
            result_raw.model_dump()
            if hasattr(result_raw, "model_dump")
            else (result_raw if isinstance(result_raw, dict) else {"text": result_raw})
        )

        _say(f"■  Decision: {final_result.get('status', '?')}  → sending result to '{caller}'")

        # ── Step 5: Deliver ───────────────────────────────────────────────
        await self._ref.send(
            to       = caller,
            type     = self.result_type,
            payload  = {
                "result":           final_result,
                "human_decision":   human_response.decision,
                "human_conditions": human_response.conditions,
                "risk_level":       screening.risk_level,
                "input":            payload,
            },
            metadata = {"in_reply_to": message.id},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {msg}")

    @staticmethod
    def _analysis_prompt(payload: Dict[str, Any]) -> str:
        return (
            "Perform a thorough automated screening of the following request "
            "and return a JSON object with EXACTLY these keys:\n\n"
            "  summary            – one-sentence description of the request and "
            "its primary risk factors\n"
            "  automated_findings – detailed text: every flag raised, every check "
            "passed, every data point considered; be specific\n"
            "  risk_level         – aggregate risk assessment: one of "
            "LOW | MEDIUM | HIGH | CRITICAL\n"
            "  question           – the exact question the human reviewer must answer\n"
            "  options            – JSON array of decision options, e.g. "
            "[\"APPROVE\", \"APPROVE_WITH_CONDITIONS\", \"REJECT\"]\n\n"
            f"REQUEST PAYLOAD:\n"
            f"{json.dumps(payload, indent=2, default=str)}"
        )

    @staticmethod
    def _execution_prompt(
        payload: Dict[str, Any],
        screening: ScreeningResult,
        human_response: HumanReviewResponse,
        approved: bool,
    ) -> str:
        status_block = (
            "STATUS: APPROVED — proceed with processing and incorporate all "
            "human conditions into the final output."
            if approved
            else
            "STATUS: NOT APPROVED — produce a formal rejection notice with the "
            "human's stated reasons and any required next steps."
        )
        return (
            f"{status_block}\n\n"
            f"ORIGINAL REQUEST:\n"
            f"{json.dumps(payload, indent=2, default=str)}\n\n"
            f"AUTOMATED SCREENING FINDINGS:\n{screening.automated_findings}\n\n"
            f"HUMAN DECISION: {human_response.decision}\n"
            f"HUMAN COMMENTS: {human_response.comments}\n"
            f"CONDITIONS: {json.dumps(human_response.conditions)}\n"
            f"MODIFICATIONS: {json.dumps(human_response.modifications, default=str)}"
        )

    @staticmethod
    def _build_review_request(
        payload: Dict[str, Any],
        screening: ScreeningResult,
    ) -> HumanReviewRequest:
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        task_id = "HITL-" + hashlib.sha1(raw).hexdigest()[:8].upper()
        return HumanReviewRequest(
            task_id=task_id,
            summary=screening.summary,
            automated_findings=screening.automated_findings,
            risk_level=screening.risk_level,
            question=screening.question,
            options=screening.options,
            context_payload=payload,
        )

    def _coerce_screening(self, raw: Any) -> ScreeningResult:
        if isinstance(raw, ScreeningResult):
            return raw
        if isinstance(raw, dict):
            return ScreeningResult.model_validate(raw)
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            return ScreeningResult.model_validate(parsed)
        except Exception as exc:
            raise ValueError(
                f"[{self.name}] analyzer output cannot be coerced to "
                f"ScreeningResult: {raw!r}"
            ) from exc


