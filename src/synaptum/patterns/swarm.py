"""
SwarmAgent — autonomous peer-to-peer handoff coordination pattern  (choreography model).

Design
------
In a Swarm, there is **no central orchestrator**.  Each agent decides for
itself whether it has finished or whether it should pass control to a different
specialist.  The decision comes from *inside* the agent based on what it
discovers.

  1. **Entry**     — control starts at a designated entry agent.
  2. **Turn**      — the coordinator sends a bus message to the current agent,
                     which calls its LLM and replies autonomously.
  3. **Handoff**   — if the agent sets ``handoff_to``, the coordinator forwards
                     control to that peer, carrying the full history forward.
  4. **Terminate** — any agent can end the swarm by setting
                     ``handoff_to = None``.  The result is delivered to the
                     original caller.

Architecture — pure message choreography
-----------------------------------------
``SwarmAgent`` never calls ``participant.think()`` directly.  Every exchange
travels through the message bus:

  Client       ──[submit_type]──►  SwarmAgent
  SwarmAgent   ──[__swarm.turn__]──► current participant (LLMAgent)
  participant  ──[agent.output] ──►  SwarmAgent   (reply_to=swarm)
  SwarmAgent   → parse HandoffDecision
               → next turn OR deliver result_type to Client

In-flight state per run:
  ``_runs``         run_id → _RunState
  ``_pending_msgs`` sent_msg_id → run_id

Participants are **registered independently** — ``SwarmAgent`` only records
their bus names for routing and validation.  No participant is registered
inside ``_bind_runtime``.

Difference from other patterns
-------------------------------
  Router           → an *external* component reads the message and directs it
                     to one of N agents; agents are passive
  Supervisor/Worker→ a central planner dispatches and collects; workers never
                     decide who goes next
  Plan-and-Execute → an explicit plan artefact is created upfront; steps are
                     sequential and assigned by the planner
  Swarm            → agents are **peers**; *any* agent can hand off to *any*
                     other agent based on what it discovers; there is no
                     external authority

Usage
-----
::

    from synaptum.patterns.swarm import SwarmAgent, HandoffDecision

    fraud_analyst  = LLMAgent("fraud-analyst",  prompt_name="...", output_model=HandoffDecision)
    aml_specialist = LLMAgent("aml-specialist", prompt_name="...", output_model=HandoffDecision)
    compliance_mgr = LLMAgent("compliance-officer", prompt_name="...", output_model=HandoffDecision)

    swarm = SwarmAgent(
        "fraud-swarm",
        participants = {
            "fraud-analyst":      fraud_analyst,
            "aml-specialist":     aml_specialist,
            "compliance-officer": compliance_mgr,
        },
        entry                = "fraud-analyst",
        submit_type          = "fraud.alert",
        result_type          = "fraud.decision",
        turn_prompt_name     = "bank.swarm.turn_user_prompt",
        handoff_prompt_name  = "bank.swarm.handoff_user_prompt",
        max_turns            = 10,
        verbose              = True,
    )

    # Register all agents independently — SwarmAgent does NOT auto-register them.
    runtime.register(fraud_analyst)
    runtime.register(aml_specialist)
    runtime.register(compliance_mgr)
    runtime.register(swarm)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..agents.message_agent import MessageAgent
from ..core.context import AgentContext
from ..core.message import Message
from ..prompts.template import PromptTemplate
from ..prompts.provider import PromptProvider
from ..utils.formatting import fmt_dict, fmt_list, fmt_records


# ── Internal message type ─────────────────────────────────────────────────────

_SWARM_TURN = "__swarm.turn__"


# ── Handoff data structure ────────────────────────────────────────────────────

class HandoffDecision(BaseModel):
    """
    The structured response every swarm participant must return.

    Fields
    ------
    findings : str
        What this agent discovered or concluded during its analysis.
        Should be detailed enough for the next agent to act on.
    action : str
        Current recommended disposition: INVESTIGATE_FURTHER, CLEAR,
        BLOCK, ESCALATE, or APPROVE.
    confidence : str
        Agent's confidence in its assessment: LOW, MEDIUM, or HIGH.
    handoff_to : str or None
        Name of the next agent to receive control, drawn from the pool of
        available participants.  Set to null (None) when this agent is
        terminating the swarm — i.e., it has reached a final conclusion.
    handoff_reason : str
        Why control is being passed (or why the swarm is terminating).
        Helps the next agent understand the context of the handoff.
    """
    findings: str = Field(
        description=(
            "Detailed findings from this agent's analysis. Be specific — "
            "this becomes context for the next agent."
        )
    )
    action: str = Field(
        description="Current recommended action: INVESTIGATE_FURTHER | CLEAR | BLOCK | ESCALATE | APPROVE."
    )
    confidence: str = Field(
        description="Confidence in current assessment: LOW | MEDIUM | HIGH."
    )
    handoff_to: Optional[str] = Field(
        description=(
            "Name of the next specialist agent to handle this case, or null "
            "if this agent is making a final decision and terminating the swarm."
        )
    )
    handoff_reason: str = Field(
        description=(
            "Why are you handing off, or why are you terminating? "
            "Explain what expertise is needed next (or why you are done)."
        )
    )


# ── Run state ─────────────────────────────────────────────────────────────────

@dataclass
class _RunState:
    """Per-run mutable context held by SwarmAgent between messages."""
    caller:          str
    original_msg_id: str
    payload:         Dict[str, Any]
    current_agent:   str
    history:         List[Dict[str, Any]] = field(default_factory=list)
    turns:           int   = 0
    t_start:         float = 0.0
    t_phase:         float = 0.0


# ── SwarmAgent ────────────────────────────────────────────────────────────────

class SwarmAgent(MessageAgent):
    """
    Choreographs a pool of peer agents that autonomously pass control to each
    other via the message bus.  No direct ``agent.think()`` calls are made.

    Parameters
    ----------
    name : str
        Bus address for this swarm coordinator.
    participants : dict[str, Any]
        Pool of agents keyed by their bus name.  Only the **keys** are used
        for routing and validation — register each agent independently via
        ``runtime.register()``.
    entry : str
        Bus name of the first agent to receive the initial message.
    submit_type : str
        Message type that triggers a swarm run.
    result_type : str
        Message type for the final result sent back to the caller.
    reply_type : str
        Message type expected back from participant agents (default: ``agent.output``).
    turn_type : str
        Message type sent to a participant for its turn (default: ``__swarm.turn__``).
    max_turns : int
        Safety ceiling on total agent invocations (default: 10).
    turn_prompt_name : str, optional
        YAML key for the first-turn user prompt template.
        Variables: ``{payload}``, ``{agent_name}``, ``{peer_names}``.
    handoff_prompt_name : str, optional
        YAML key for subsequent-turn user prompt template.
        Variables: ``{payload}``, ``{agent_name}``, ``{peer_names}``, ``{history}``.
    verbose : bool
        Print handoff trace to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        participants: Dict[str, Any],
        entry: str,
        submit_type:         str = "swarm.submitted",
        result_type:         str = "swarm.result",
        reply_type:          str = "agent.output",
        turn_type:           str = _SWARM_TURN,
        max_turns:           int = 10,
        turn_prompt_name:    Optional[str] = None,
        handoff_prompt_name: Optional[str] = None,
        verbose:             bool = False,
    ) -> None:
        super().__init__(name)
        self._participant_names: frozenset = frozenset(participants.keys())
        self.entry              = entry
        self.submit_type        = submit_type
        self.result_type        = result_type
        self.reply_type         = reply_type
        self.turn_type          = turn_type
        self.max_turns          = max_turns
        self.verbose            = verbose

        self._turn_prompt_tpl:    Optional[PromptTemplate] = None
        self._handoff_prompt_tpl: Optional[PromptTemplate] = None
        self._pending_turn_prompt_name:    Optional[str] = turn_prompt_name
        self._pending_handoff_prompt_name: Optional[str] = handoff_prompt_name

        self._runs:         Dict[str, _RunState] = {}
        self._pending_msgs: Dict[str, str]       = {}  # sent_msg_id → run_id

    # ── Prompt resolution ─────────────────────────────────────────────────────

    def _inject_prompt_registry(self, provider: PromptProvider) -> None:
        """Called by AgentRuntime.register() to resolve deferred prompt names."""
        if self._pending_turn_prompt_name is not None:
            self._turn_prompt_tpl = provider.get(self._pending_turn_prompt_name)
            self._pending_turn_prompt_name = None
        if self._pending_handoff_prompt_name is not None:
            self._handoff_prompt_tpl = provider.get(self._pending_handoff_prompt_name)
            self._pending_handoff_prompt_name = None

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
            current_agent   = self.entry,
            t_start         = time.perf_counter(),
        )

        if self.verbose:
            peers = sorted(self._participant_names)
            print(f"\n── [{self.name}]  SWARM START  entry={self.entry} ──")
            print(f"   Peers: {', '.join(peers)}")

        await self._send_turn(run_id)

    # ── Reply router ──────────────────────────────────────────────────────────

    async def _handle_reply(self, message: Message) -> None:
        in_reply_to = message.metadata.get("in_reply_to")
        run_id = self._pending_msgs.pop(in_reply_to, None)
        if run_id is None:
            return  # reply not addressed to this swarm

        state = self._runs.get(run_id)
        if state is None:
            return

        await self._on_turn_result(run_id, state, message)

    # ── Process turn result ───────────────────────────────────────────────────

    async def _on_turn_result(
        self, run_id: str, state: _RunState, message: Message
    ) -> None:
        elapsed_s = round(time.perf_counter() - state.t_phase, 2)

        raw      = message.payload.get("answer", message.payload)
        decision = self._coerce_decision(raw)

        state.history.append({
            "turn":       state.turns,
            "agent":      state.current_agent,
            "findings":   decision.findings,
            "action":     decision.action,
            "confidence": decision.confidence,
            "reason":     decision.handoff_reason,
            "handoff_to": decision.handoff_to,
            "elapsed_s":  elapsed_s,
        })

        if self.verbose:
            action_tag = f"[{decision.action}|{decision.confidence}]"
            print(f"     {action_tag} {decision.findings[:100].replace(chr(10), ' ')}…")
            print(f"     ⏱  {elapsed_s:.2f}s")
            if decision.handoff_to:
                print(f"     → handoff to: {decision.handoff_to} — {decision.handoff_reason[:80]}")
            else:
                print(f"     ■ TERMINATE — {decision.handoff_reason[:80]}")

        # Terminate: agent decided, or safety ceiling reached
        if decision.handoff_to is None:
            await self._deliver(run_id, state)
            return

        if state.turns >= self.max_turns:
            if self.verbose:
                print(f"\n   ⚠  max_turns ({self.max_turns}) reached without termination.")
            await self._deliver(run_id, state)
            return

        # Validate handoff target
        if decision.handoff_to not in self._participant_names:
            raise KeyError(
                f"Agent '{state.current_agent}' tried to hand off to "
                f"'{decision.handoff_to}', which is not in the swarm pool. "
                f"Available: {sorted(self._participant_names)}"
            )

        state.current_agent = decision.handoff_to
        await self._send_turn(run_id)

    # ── Send next turn ────────────────────────────────────────────────────────

    async def _send_turn(self, run_id: str) -> None:
        state = self._runs[run_id]
        state.turns += 1

        if self.verbose:
            print(f"\n   ▶ Turn {state.turns} — {state.current_agent}")

        peer_names = sorted(
            n for n in self._participant_names if n != state.current_agent
        )

        if not state.history:
            # First turn — no prior analysis accumulated yet
            tpl = self._turn_prompt_tpl
            if tpl is None:
                raise RuntimeError(
                    f"SwarmAgent '{self.name}': no turn_prompt_name configured. "
                    "Pass 'turn_prompt_name' to SwarmAgent."
                )
            prompt = tpl.render(
                payload    = fmt_dict(state.payload),
                agent_name = state.current_agent,
                peer_names = fmt_list(peer_names),
            )
        else:
            # Handoff turn — include full accumulated history
            tpl = self._handoff_prompt_tpl
            if tpl is None:
                raise RuntimeError(
                    f"SwarmAgent '{self.name}': no handoff_prompt_name configured. "
                    "Pass 'handoff_prompt_name' to SwarmAgent."
                )
            prompt = tpl.render(
                payload    = fmt_dict(state.payload),
                agent_name = state.current_agent,
                peer_names = fmt_list(peer_names),
                history    = fmt_records(
                    state.history,
                    "Turn {turn} — {agent} [{action}|{confidence}]\n"
                    "Findings: {findings}\n"
                    "Reason for handoff: {reason}\n",
                ),
            )

        msg_id = await self._ref.send(
            to       = state.current_agent,
            type     = self.turn_type,
            payload  = {"text": prompt},
            reply_to = self.name,
            metadata = {"run_id": run_id},
        )
        state.t_phase = time.perf_counter()
        self._pending_msgs[msg_id] = run_id

    # ── Deliver result ────────────────────────────────────────────────────────

    async def _deliver(self, run_id: str, state: _RunState) -> None:
        elapsed_total_s = round(time.perf_counter() - state.t_start, 2)
        final_turn   = state.history[-1] if state.history else {}
        final_action = final_turn.get("action", "UNKNOWN")

        if self.verbose:
            print(
                f"\n   ■  Swarm complete — {state.turns} turn(s) — "
                f"{elapsed_total_s:.2f}s total — final action: {final_action}"
            )
            print(f"   ■  Sending '{self.result_type}' to '{state.caller}'")

        await self._ref.send(
            to      = state.caller,
            type    = self.result_type,
            payload = {
                "final_action":    final_action,
                "final_agent":     final_turn.get("agent", "unknown"),
                "turns":           state.turns,
                "elapsed_total_s": elapsed_total_s,
                "history":         state.history,
                "input":           state.payload,
            },
            metadata = {"in_reply_to": state.original_msg_id},
        )
        self._runs.pop(run_id, None)

    # ── Coercion ──────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_decision(raw: Any) -> HandoffDecision:
        if isinstance(raw, HandoffDecision):
            return raw
        if isinstance(raw, dict):
            return HandoffDecision.model_validate(raw)
        if isinstance(raw, str):
            return HandoffDecision.model_validate_json(raw)
        raise TypeError(f"Cannot coerce {type(raw)} to HandoffDecision")
