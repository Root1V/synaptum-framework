"""
ConsensusAgent - independent multi-panel voting and synthesis pattern  (choreography model)
===========================================================================================

Architecture
------------
Classic implementations call ``panelist.think()`` inside an ``asyncio.gather``
loop, then ``judge.think()`` - imperative orchestration inside a single agent.

This implementation uses **pure message choreography**: each stage is an
autonomous agent that reacts to its own message type, performs its action,
and fires the next message.  ``ConsensusAgent`` only fans out the panel
messages then steps back completely, just like ``HITLAgent`` and ``SagaAgent``.

Execution graph
---------------

  ConsensusAgent --[__consensus.panel__]--> _PanelistAgent[role-A]
                 --[__consensus.panel__]--> _PanelistAgent[role-B]   (concurrent)
                 --[__consensus.panel__]--> _PanelistAgent[role-C]
                                                    |  (each fires independently)
                                          [__consensus.verdict__]--> _AggregatorAgent
                                          (accumulated until count == expected)
                                                    |
                                          [__consensus.judge__]--> _JudgeAgent
                                                                        |
                                                               result_type --> caller

``_PanelistAgent`` (LLMAgent x N):
  - Reacts to ``__consensus.panel__``
  - Calls LLM to produce ``PanelistVerdict``
  - Fires ``__consensus.verdict__`` to the aggregator

``_AggregatorAgent`` (MessageAgent, no LLM):
  - Reacts to ``__consensus.verdict__``
  - Accumulates verdicts per ``correlation_id`` until all N arrive
  - Fires ``__consensus.judge__`` once complete

``_JudgeAgent`` (LLMAgent):
  - Reacts to ``__consensus.judge__``
  - Receives original input + all verdicts as serialised JSON
  - Produces final synthesised decision
  - Delivers result to original caller via the bus

Consensus state
---------------
All intermediate results travel immutably under the ``__consensus__`` key in
the message payload.  No shared mutable state outside the bus  (except the
aggregator's in-flight correlation map, which is ephemeral per-run).

Public API
----------
::

    from synaptum.patterns.consensus import ConsensusAgent, PanelistVerdict

    consensus = ConsensusAgent(
        "loan-committee",
        panelists    = {
            "credit-analyst": "bank.consensus.credit_analyst.system",
            "risk-officer":   "bank.consensus.risk_officer.system",
            "sector-expert":  "bank.consensus.sector_expert.system",
        },
        judge_prompt = "bank.consensus.committee_chair.system",
        judge_model  = CommitteeDecision,
        submit_type  = "loan.committee.submitted",
        result_type  = "loan.committee.decision",
        verbose      = True,
    )
    runtime.register(consensus)   # registers consensus + all internal agents automatically
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ..core.agent import Agent, CompositeAgent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef
from ..agents.llm_agent import LLMAgent
from ..agents.message_agent import MessageAgent


# -- Internal message types ----------------------------------------------------
_CONSENSUS_PANEL   = "__consensus.panel__"
_CONSENSUS_VERDICT = "__consensus.verdict__"
_CONSENSUS_JUDGE   = "__consensus.judge__"


# -- Public data model ---------------------------------------------------------

class PanelistVerdict(BaseModel):
    """
    Structured output that each panelist LLM must return.

    Fields
    ------
    recommendation : str
        Recommended decision, e.g. APPROVE / DECLINE / etc.
    confidence : str
        Confidence level: LOW | MEDIUM | HIGH.
    rationale : str
        Detailed justification (3-5 sentences).
    key_concerns : list[str]
        Top 2-4 concerns or risks (empty if none).
    key_positives : list[str]
        Top 2-4 strengths (empty if none).
    conditions : list[str]
        Conditions attached to a positive recommendation.
        Empty if unconditional or declining.
    """
    recommendation: str = Field(
        description="This panelist's recommended decision.",
    )
    confidence: str = Field(
        description="Confidence level: LOW | MEDIUM | HIGH.",
    )
    rationale: str = Field(
        description="Detailed justification for the recommendation (3-5 sentences).",
    )
    key_concerns: List[str] = Field(
        description="Top 2-4 specific concerns or risks.",
    )
    key_positives: List[str] = Field(
        description="Top 2-4 strengths or positives.",
    )
    conditions: List[str] = Field(
        description=(
            "Conditions attached to a positive recommendation. "
            "Empty if unconditional or declining."
        ),
    )


# -- Internal agent: panelist --------------------------------------------------

class _PanelistAgent(LLMAgent):
    """
    Autonomous panelist node.

    Inherits LLM infrastructure from ``LLMAgent``.  Reacts to
    ``__consensus.panel__``, calls the LLM to produce a ``PanelistVerdict``,
    and fires ``__consensus.verdict__`` to the aggregator agent.
    """

    def __init__(
        self,
        name: str,
        *,
        role:        str,
        prompt_name: str,
        verbose:     bool = False,
    ) -> None:
        super().__init__(name, prompt_name=prompt_name, output_model=PanelistVerdict)
        self.role    = role
        self.verbose = verbose

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _CONSENSUS_PANEL:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload    = message.payload
        state      = dict(payload.get("__consensus__", {}))
        clean      = {k: v for k, v in payload.items() if k != "__consensus__"}
        agent_name = self.name.split(".")[0]

        if self.verbose:
            print(f"[{agent_name}] > [{self.role}] evaluating case...")

        verdict: PanelistVerdict = await self.think(
            json.dumps(clean, indent=2, default=str)
        )

        if self.verbose:
            print(
                f"[{agent_name}]    ok [{self.role}]"
                f"  {verdict.recommendation} | {verdict.confidence}"
            )

        await self._ref.send(
            to      = state["aggregator_target"],
            type    = _CONSENSUS_VERDICT,
            payload = {
                "__consensus__": state,
                "role":          self.role,
                "verdict":       verdict.model_dump(),
            },
            reply_to = message.reply_to,
        )


# -- Internal agent: aggregator ------------------------------------------------

class _AggregatorAgent(MessageAgent):
    """
    Verdict collector - the fan-in point.

    No LLM.  Inherits ``self._ref`` from ``MessageAgent`` - no ``_bind_runtime``
    boilerplate needed.  Reacts to ``__consensus.verdict__``, accumulates verdicts
    by ``correlation_id`` until all N panelists have responded, then fires
    ``__consensus.judge__`` to the judge agent.

    Internal state (ephemeral per-run)
    -----------------------------------
    ``_pending`` maps ``correlation_id`` to a dict containing the accumulated
    verdicts and the consensus state envelope.  Entries are removed once the
    judge message is fired.
    """

    def __init__(self, name: str, *, verbose: bool = False) -> None:
        super().__init__(name)
        self.verbose  = verbose
        self._pending: Dict[str, Dict[str, Any]] = {}
        # self._ref inherited from MessageAgent

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _CONSENSUS_VERDICT:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload        = message.payload
        state          = dict(payload.get("__consensus__", {}))
        role           = payload["role"]
        verdict        = payload["verdict"]
        correlation_id = state["correlation_id"]
        expected       = state["expected"]
        agent_name     = self.name.split(".")[0]

        if correlation_id not in self._pending:
            self._pending[correlation_id] = {"verdicts": {}, "state": state}

        self._pending[correlation_id]["verdicts"][role] = verdict
        collected = len(self._pending[correlation_id]["verdicts"])

        if self.verbose:
            print(
                f"[{agent_name}]    verdict received: {role}"
                f"  ({collected}/{expected})"
            )

        if collected < expected:
            return

        bucket   = self._pending.pop(correlation_id)
        verdicts = bucket["verdicts"]

        if self.verbose:
            recs   = [v["recommendation"] for v in verdicts.values()]
            unique = set(recs)
            if len(unique) == 1:
                print(f"[{agent_name}] -> UNANIMOUS: {recs[0]}")
            else:
                counts  = {r: recs.count(r) for r in unique}
                summary = "  ".join(f"{r}x{c}" for r, c in counts.items())
                print(f"[{agent_name}] -> SPLIT panel: {summary}")
            print(f"[{agent_name}] > forwarding to judge...")

        await self._ref.send(
            to      = state["judge_target"],
            type    = _CONSENSUS_JUDGE,
            payload = {
                "__consensus__":  state,
                "verdicts":       verdicts,
                "original_input": state["original_input"],
            },
            reply_to = message.reply_to,
        )


# -- Internal agent: judge -----------------------------------------------------

class _JudgeAgent(LLMAgent):
    """
    Final synthesis node.

    Inherits LLM infrastructure from ``LLMAgent``.  Reacts to
    ``__consensus.judge__``, receives the original input and all panelist
    verdicts as serialised JSON, calls the LLM, and delivers the result
    to the original caller.
    """

    def __init__(
        self,
        name: str,
        *,
        prompt_name:  str,
        output_model: Optional[Type[BaseModel]],
        verbose:      bool = False,
    ) -> None:
        super().__init__(name, prompt_name=prompt_name, output_model=output_model)
        self.verbose = verbose

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == _CONSENSUS_JUDGE:
            await self._run(message)

    async def _run(self, message: Message) -> None:
        payload        = message.payload
        state          = payload.get("__consensus__", {})
        verdicts       = payload["verdicts"]
        original_input = payload["original_input"]
        caller         = message.reply_to or state.get("caller", "")
        result_type    = state.get("result_type", "consensus.result")
        agent_name     = self.name.split(".")[0]

        if self.verbose:
            print(f"[{agent_name}] > Judge synthesising {len(verdicts)} verdicts...")

        raw_result = await self.think(
            json.dumps({
                "original_input": original_input,
                "verdicts":       verdicts,
            }, indent=2, default=str)
        )

        final_result = (
            raw_result.model_dump()
            if hasattr(raw_result, "model_dump")
            else (raw_result if isinstance(raw_result, dict) else {"text": raw_result})
        )

        if self.verbose:
            decision = final_result.get("decision", final_result.get("recommendation", "?"))
            print(f"[{agent_name}] [] Decision: {decision}  -> delivering '{result_type}' to '{caller}'")

        await self._ref.send(
            to      = caller,
            type    = result_type,
            payload = {
                "result":    final_result,
                "verdicts":  verdicts,
                "panelists": list(verdicts.keys()),
                "input":     original_input,
            },
        )


# -- Public entry point --------------------------------------------------------

class ConsensusAgent(CompositeAgent):
    """
    Entry point and topology hub for a choreographed consensus pipeline.

    Inherits from ``CompositeAgent``: the internal topology
    (``_PanelistAgent`` x N, ``_AggregatorAgent``, ``_JudgeAgent``)
    is built in ``__init__`` and registered automatically by the runtime
    via ``sub_agents()`` - no manual ``runtime.register()`` calls needed.

    At runtime, ``ConsensusAgent`` acts as the public entry point: it receives
    the trigger message, initialises the ``__consensus__`` state envelope, and
    fans out one ``__consensus.panel__`` message per panelist concurrently.
    After that it steps back - the chain is self-driving.

    Parameters
    ----------
    name         : Bus address (e.g. ``"loan-committee"``).
    panelists    : Mapping of role name -> prompt key.
    judge_prompt : Prompt key for the judge/synthesiser LLM.
    judge_model  : Pydantic model for the judge's structured output.
    submit_type  : Message type that starts the pipeline.
    result_type  : Message type for the delivered result.
    verbose      : Print step-by-step progress to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        panelists:    Dict[str, str],
        judge_prompt: str,
        judge_model:  Optional[Type[BaseModel]] = None,
        submit_type:  str = "consensus.submitted",
        result_type:  str = "consensus.result",
        verbose:      bool = False,
    ) -> None:
        super().__init__(name)
        self.panelists    = panelists
        self.judge_prompt = judge_prompt
        self.judge_model  = judge_model
        self.submit_type  = submit_type
        self.result_type  = result_type
        self.verbose      = verbose
        self._ref: Optional[AgentRef] = None

        self._aggregator_name = f"{name}._aggregator"
        self._judge_name      = f"{name}._judge"
        self._panelist_names  = {
            role: f"{name}._panelist.{role}"
            for role in panelists
        }

        self._topology: List[Agent] = self._build_topology()

    # -- Topology declaration --------------------------------------------------

    def sub_agents(self) -> List[Agent]:
        """Declares the internal agents that form this consensus pipeline."""
        return self._topology

    def _build_topology(self) -> List[Agent]:
        """Constructs all internal agents. Called once in ``__init__``."""
        agents: List[Agent] = []

        for role, prompt_name in self.panelists.items():
            agents.append(
                _PanelistAgent(
                    self._panelist_names[role],
                    role        = role,
                    prompt_name = prompt_name,
                    verbose     = self.verbose,
                )
            )

        agents.append(
            _AggregatorAgent(
                self._aggregator_name,
                verbose = self.verbose,
            )
        )

        agents.append(
            _JudgeAgent(
                self._judge_name,
                prompt_name  = self.judge_prompt,
                output_model = self.judge_model,
                verbose      = self.verbose,
            )
        )

        return agents

    # -- Runtime binding -------------------------------------------------------

    def _bind_runtime(self, runtime) -> None:
        """Sets up the bus reference. Sub-agents are registered by the runtime."""
        self._ref = AgentRef(self.name, runtime._bus)

    # -- Message handling ------------------------------------------------------

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type != self.submit_type:
            return

        caller = message.reply_to or message.sender

        if self.verbose:
            sep   = "=" * 56
            roles = list(self.panelists.keys())
            print(f"\n{sep}\n  CONSENSUS START - {self.name}  ({len(roles)} panelists)\n{sep}")
            print(f"  Panel: {', '.join(roles)}")

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )

        correlation_id = str(uuid.uuid4())

        consensus_state: Dict[str, Any] = {
            "correlation_id":    correlation_id,
            "expected":          len(self.panelists),
            "caller":            caller,
            "result_type":       self.result_type,
            "aggregator_target": self._aggregator_name,
            "judge_target":      self._judge_name,
            "original_input":    payload,
        }

        for role, panelist_name in self._panelist_names.items():
            await self._ref.send(
                to       = panelist_name,
                type     = _CONSENSUS_PANEL,
                payload  = {**payload, "__consensus__": consensus_state},
                reply_to = caller,
            )
