"""
HITLAgent — Human-in-the-Loop pause pattern  (choreography model)
==================================================================

Architecture
------------
Classic implementations call ``analyzer.think()`` and ``executor.think()``
from inside a central ``_execute()`` method — an orchestration loop.

This implementation uses **pure message choreography**: each stage is an
autonomous agent that reacts to its own message type, performs its action,
and fires the next message.  ``HITLAgent`` only launches the chain then
steps back completely, just like ``SagaAgent``.

Execution graph
---------------

  HITLAgent ─[__hitl.screen__]─▶ _screener
                                      │
                               [__hitl.review__]─▶ _gate  (human pause)
                                                       │
                                                [__hitl.execute__]─▶ _executor
                                                                          │
                                                                   result_type ─▶ caller

``_HITLScreenerAgent`` (SimpleAgent):
  - Reacts to ``__hitl.screen__``
  - Calls LLM to produce ``ScreeningResult``
  - Builds ``HumanReviewRequest``
  - Emits ``__hitl.review__`` to ``_gate``

``_HITLGateAgent`` (Agent, no LLM):
  - Reacts to ``__hitl.review__``
  - Calls ``review_handler(request)`` — the human pause point
  - Emits ``__hitl.execute__`` to ``_executor`` with the human decision embedded

``_HITLExecutorAgent`` (SimpleAgent):
  - Reacts to ``__hitl.execute__``
  - Calls LLM to produce the final structured output
  - Delivers result to original caller via the bus

HITL state
----------
All intermediate results travel immutably under the ``__hitl__`` key in
the message payload.  No shared mutable state outside the bus.

Public API
----------
::

    from synaptum.patterns.hitl import (
        HITLAgent, HumanReviewRequest, HumanReviewResponse, ScreeningResult
    )

    async def my_review_handler(request: HumanReviewRequest) -> HumanReviewResponse:
        print(request.model_dump_json(indent=2))
        decision = input("Decision: ").strip().upper()
        return HumanReviewResponse(decision=decision, comments="")

    hitl = HITLAgent(
        "wire-edd-review",
        screener_prompt = "bank.hitl.aml_screener.system",
        executor_prompt = "bank.hitl.wire_operations.system",
        executor_model  = WireReleaseInstruction,
        review_handler  = my_review_handler,
        trigger_type    = "wire.edd.submitted",
        result_type     = "wire.edd.processed",
        verbose         = True,
    )
    runtime.register(hitl)   # registers hitl + 3 internal agents automatically
"""

from __future__ import annotations

import hashlib
import json
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ..core.agent import Agent, CompositeAgent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef
from ..agents.simple_agent import SimpleAgent


# ── Internal message types ────────────────────────────────────────────────────
_HITL_SCREEN  = "__hitl.screen__"
_HITL_REVIEW  = "__hitl.review__"
_HITL_EXECUTE = "__hitl.execute__"


# ── Public data models ────────────────────────────────────────────────────────

class ScreeningResult(BaseModel):
    """
    Structured output that the screener LLM must return.
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
        description="The specific question the human reviewer must answer.",
    )
    options: List[str] = Field(
        description=(
            "Decision options available to the reviewer, e.g. "
            "[\"APPROVE\", \"APPROVE_WITH_CONDITIONS\", \"REJECT\"]."
        ),
    )


class HumanReviewRequest(BaseModel):
    """Package delivered to the ``review_handler``. Contains everything a human
    reviewer needs to make an informed decision."""
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
    """The human reviewer's structured response."""
    decision: str = Field(
        description="One of the options offered in HumanReviewRequest.options.",
    )
    comments: str = Field(
        default="",
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


# ── Internal agent: screener ──────────────────────────────────────────────────

class _HITLScreenerAgent(SimpleAgent):
    """
    Autonomous screening node.

    Inherits LLM infrastructure from ``SimpleAgent``.  Reacts to
    ``__hitl.screen__``, calls the LLM to produce a ``ScreeningResult``,
    builds the ``HumanReviewRequest``, and fires ``__hitl.review__`` to
    the gate agent.
    """

    def __init__(
        self,
        name: str,
        *,
        prompt_name:   str,
        review_target: str,
        verbose:       bool = False,
    ) -> None:
        super().__init__(name, prompt_name=prompt_name, output_model=ScreeningResult)
        self.review_target = review_target
        self.verbose       = verbose

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _HITL_SCREEN:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload = message.payload
        hitl    = dict(payload.get("__hitl__", {}))
        clean   = {k: v for k, v in payload.items() if k != "__hitl__"}
        saga_name = self.name.split(".")[0]

        if self.verbose:
            print(f"[{saga_name}] ▶  Running automated screening…")

        try:
            screening: ScreeningResult = await self.think(_analysis_prompt(clean))
        except Exception as exc:
            screening = ScreeningResult(
                summary="Screening failed due to an internal error.",
                automated_findings=f"{exc}\n{traceback.format_exc(limit=2)}",
                risk_level="CRITICAL",
                question="An error occurred. Should this request be rejected?",
                options=["REJECT"],
            )

        if self.verbose:
            print(f"[{saga_name}]    ✓  Screening complete — risk: {screening.risk_level}")

        review_request = _build_review_request(clean, screening)
        new_hitl = {
            **hitl,
            "screening":      screening.model_dump(),
            "review_request": review_request.model_dump(),
        }

        await self._ref.send(
            to       = self.review_target,
            type     = _HITL_REVIEW,
            payload  = {**payload, "__hitl__": new_hitl},
            reply_to = message.reply_to,
        )


# ── Internal agent: gate (human pause) ───────────────────────────────────────

class _HITLGateAgent(Agent):
    """
    Human decision gate — the pause point.

    No LLM.  Reacts to ``__hitl.review__``, calls the application-supplied
    ``review_handler`` coroutine (which may open a web form, send a Slack
    message, block on stdin, etc.), then fires ``__hitl.execute__`` with the
    human decision embedded in the ``__hitl__`` state.
    """

    def __init__(
        self,
        name: str,
        *,
        review_handler: ReviewHandler,
        execute_target: str,
        verbose:        bool = False,
    ) -> None:
        super().__init__(name)
        self.review_handler = review_handler
        self.execute_target = execute_target
        self.verbose        = verbose
        self._ref: Optional[AgentRef] = None

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _HITL_REVIEW:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload        = message.payload
        hitl           = dict(payload.get("__hitl__", {}))
        review_request = HumanReviewRequest.model_validate(hitl["review_request"])
        saga_name      = self.name.split(".")[0]

        if self.verbose:
            print(
                f"[{saga_name}] ⏸  PAUSING for human review"
                f"  [task: {review_request.task_id}]"
                f"\n[{saga_name}]    Risk: {review_request.risk_level}"
                f"  |  Options: {', '.join(review_request.options)}"
            )

        human_response = await self.review_handler(review_request)
        decision_upper = human_response.decision.upper().replace(" ", "_")
        approved       = decision_upper in _APPROVAL_DECISIONS

        if self.verbose:
            path = "APPROVED path" if approved else "REJECTED path"
            print(f"[{saga_name}] ▶  Human decided: {human_response.decision}  ({path})")
            if human_response.conditions:
                print(f"[{saga_name}]    Conditions: {len(human_response.conditions)} item(s)")

        new_hitl = {
            **hitl,
            "human_response": human_response.model_dump(),
            "approved":       approved,
        }

        await self._ref.send(
            to       = self.execute_target,
            type     = _HITL_EXECUTE,
            payload  = {**payload, "__hitl__": new_hitl},
            reply_to = message.reply_to,
        )


# ── Internal agent: executor ──────────────────────────────────────────────────

class _HITLExecutorAgent(SimpleAgent):
    """
    Final output node.

    Inherits LLM infrastructure from ``SimpleAgent``.  Reacts to
    ``__hitl.execute__``, builds the execution prompt from the full
    ``__hitl__`` state (original payload + screening + human decision),
    calls the LLM, and delivers the result to the original caller.
    """

    def __init__(
        self,
        name: str,
        *,
        prompt_name:  str,
        output_model: Optional[Type[BaseModel]],
        result_type:  str,
        verbose:      bool = False,
    ) -> None:
        super().__init__(name, prompt_name=prompt_name, output_model=output_model)
        self.result_type = result_type
        self.verbose     = verbose

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _HITL_EXECUTE:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload        = message.payload
        hitl           = payload.get("__hitl__", {})
        caller         = message.reply_to or hitl.get("caller", "")
        clean          = {k: v for k, v in payload.items() if k != "__hitl__"}
        screening      = ScreeningResult.model_validate(hitl["screening"])
        human_response = HumanReviewResponse.model_validate(hitl["human_response"])
        approved       = hitl.get("approved", False)
        saga_name      = self.name.split(".")[0]

        if self.verbose:
            print(f"[{saga_name}] ▶  Generating final output…")

        result_raw = await self.think(
            _execution_prompt(clean, screening, human_response, approved)
        )

        final_result = (
            result_raw.model_dump()
            if hasattr(result_raw, "model_dump")
            else (result_raw if isinstance(result_raw, dict) else {"text": result_raw})
        )

        if self.verbose:
            status = final_result.get("status", "?")
            print(f"[{saga_name}] ■  {status}  →  delivering '{self.result_type}' to '{caller}'")

        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = {
                "result":           final_result,
                "human_decision":   human_response.decision,
                "human_conditions": human_response.conditions,
                "risk_level":       screening.risk_level,
                "input":            clean,
            },
        )


# ── Public entry point ────────────────────────────────────────────────────────

class HITLAgent(CompositeAgent):
    """
    Entry point and topology hub for a choreographed HITL pipeline.

    Inherits from ``CompositeAgent``: the internal topology
    (``_HITLScreenerAgent``, ``_HITLGateAgent``, ``_HITLExecutorAgent``)
    is built in ``__init__`` and registered automatically by the runtime
    via ``sub_agents()`` — no manual ``runtime.register()`` calls needed.

    At runtime, ``HITLAgent`` acts as the public entry point: it receives
    the trigger message, initialises the ``__hitl__`` state envelope, and
    fires the first ``__hitl.screen__`` message.  After that it steps back
    — the chain is self-driving.

    Parameters
    ----------
    name            : Bus address (e.g. ``"wire-edd-review"``).
    screener_prompt : Prompt key for the AML / risk screening LLM.
    executor_prompt : Prompt key for the final output LLM.
    executor_model  : Pydantic model for the executor's structured output.
    review_handler  : Async callable that delivers ``HumanReviewRequest``
                      and awaits a ``HumanReviewResponse``.
    trigger_type    : Message type that starts the pipeline.
    result_type     : Message type for the delivered result.
    verbose         : Print step-by-step progress to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        screener_prompt: str,
        executor_prompt: str,
        executor_model:  Optional[Type[BaseModel]] = None,
        review_handler:  ReviewHandler,
        trigger_type:    str = "hitl.submitted",
        result_type:     str = "hitl.processed",
        verbose:         bool = False,
    ) -> None:
        super().__init__(name)
        self.screener_prompt = screener_prompt
        self.executor_prompt = executor_prompt
        self.executor_model  = executor_model
        self.review_handler  = review_handler
        self.trigger_type    = trigger_type
        self.result_type     = result_type
        self.verbose         = verbose
        self._ref: Optional[AgentRef] = None

        # Pre-compute internal agent names
        self._screener_name = f"{name}._screener"
        self._gate_name     = f"{name}._gate"
        self._executor_name = f"{name}._executor"

        # Build topology at construction time — runtime registers via sub_agents()
        self._topology: List[Agent] = self._build_topology()

    # ── Topology declaration ──────────────────────────────────────────────────

    def sub_agents(self) -> List[Agent]:
        """Declares the internal agents that form this HITL pipeline."""
        return self._topology

    def _build_topology(self) -> List[Agent]:
        """Constructs all internal agents. Called once in ``__init__``."""
        return [
            _HITLScreenerAgent(
                self._screener_name,
                prompt_name   = self.screener_prompt,
                review_target = self._gate_name,
                verbose       = self.verbose,
            ),
            _HITLGateAgent(
                self._gate_name,
                review_handler = self.review_handler,
                execute_target = self._executor_name,
                verbose        = self.verbose,
            ),
            _HITLExecutorAgent(
                self._executor_name,
                prompt_name  = self.executor_prompt,
                output_model = self.executor_model,
                result_type  = self.result_type,
                verbose      = self.verbose,
            ),
        ]

    # ── Runtime binding ───────────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        """Sets up the bus reference. Sub-agents are registered by the runtime."""
        self._ref = AgentRef(self.name, runtime._bus)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type != self.trigger_type:
            return

        caller = message.reply_to or message.sender

        if self.verbose:
            sep = "━" * 56
            print(f"\n{sep}\n  HITL START — {self.name}\n{sep}")

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )

        hitl_state: Dict[str, Any] = {
            "caller":      caller,
            "result_type": self.result_type,
        }

        # Fire the chain — we step back; the bus does the rest
        await self._ref.send(
            to       = self._screener_name,
            type     = _HITL_SCREEN,
            payload  = {**payload, "__hitl__": hitl_state},
            reply_to = caller,
        )


# ── Prompt builders (pure functions) ─────────────────────────────────────────

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
        f"REQUEST PAYLOAD:\n{json.dumps(payload, indent=2, default=str)}"
    )


def _execution_prompt(
    payload:        Dict[str, Any],
    screening:      ScreeningResult,
    human_response: HumanReviewResponse,
    approved:       bool,
) -> str:
    status_block = (
        "STATUS: APPROVED — proceed with processing and incorporate all "
        "human conditions into the final output."
        if approved else
        "STATUS: NOT APPROVED — produce a formal rejection notice with the "
        "human's stated reasons and any required next steps."
    )
    return (
        f"{status_block}\n\n"
        f"ORIGINAL REQUEST:\n{json.dumps(payload, indent=2, default=str)}\n\n"
        f"AUTOMATED SCREENING FINDINGS:\n{screening.automated_findings}\n\n"
        f"HUMAN DECISION: {human_response.decision}\n"
        f"HUMAN COMMENTS: {human_response.comments}\n"
        f"CONDITIONS: {json.dumps(human_response.conditions)}\n"
        f"MODIFICATIONS: {json.dumps(human_response.modifications, default=str)}"
    )


def _build_review_request(
    payload:   Dict[str, Any],
    screening: ScreeningResult,
) -> HumanReviewRequest:
    raw     = json.dumps(payload, sort_keys=True, default=str).encode()
    task_id = "HITL-" + hashlib.sha1(raw).hexdigest()[:8].upper()
    return HumanReviewRequest(
        task_id            = task_id,
        summary            = screening.summary,
        automated_findings = screening.automated_findings,
        risk_level         = screening.risk_level,
        question           = screening.question,
        options            = screening.options,
        context_payload    = payload,
    )
