"""
SwarmAgent — autonomous peer-to-peer handoff coordination pattern.

Design
------
In a Swarm, there is **no central orchestrator**.  Each agent decides for
itself whether it has finished or whether it should pass control to a different
specialist.  The decision comes from *inside* the agent based on what it
discovers.

  1. **Entry**     — control starts at a designated entry agent.
  2. **Think**     — the current agent analyses its input plus the full
                     accumulated history of prior turns.
  3. **Handoff**   — if the agent determines that another specialist is needed
                     it returns ``handoff_to = "<agent-name>"``; the swarm
                     engine transparently passes control to that agent, carrying
                     the full context forward.
  4. **Terminate** — any agent can end the swarm by returning
                     ``handoff_to = None``.  The swarm then delivers the result
                     to the original caller.

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

    from synaptum.patterns.swarm import SwarmAgent

    fraud_analyst   = SimpleAgent("fraud-analyst",  ...)
    aml_specialist  = SimpleAgent("aml-specialist", ...)
    compliance_mgr  = SimpleAgent("compliance-officer", ...)

    swarm = SwarmAgent(
        "fraud-swarm",
        participants = {
            "fraud-analyst":    fraud_analyst,
            "aml-specialist":   aml_specialist,
            "compliance-officer": compliance_mgr,
        },
        entry       = "fraud-analyst",
        submit_type = "fraud.alert",
        result_type = "fraud.decision",
        max_turns   = 10,
        verbose     = True,
    )

    runtime.register(swarm)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef


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


# ── SwarmAgent ────────────────────────────────────────────────────────────────

class SwarmAgent(Agent):
    """
    Manages a pool of peer agents that autonomously pass control to each other.

    Parameters
    ----------
    name : str
        Bus address for this swarm coordinator.
    participants : dict[str, Agent]
        Pool of agents keyed by name.  Any agent can hand off to any other.
    entry : str
        Name of the first agent to receive the initial message.
    submit_type : str
        Message type that triggers a swarm run.
    result_type : str
        Message type for the final result sent back to the caller.
    max_turns : int
        Safety ceiling on total agent invocations (default: 10).
    verbose : bool
        Print handoff trace to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        participants: Dict[str, Agent],
        entry: str,
        submit_type: str = "swarm.submitted",
        result_type: str = "swarm.result",
        max_turns: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.participants = participants
        self.entry        = entry
        self.submit_type  = submit_type
        self.result_type  = result_type
        self.max_turns    = max_turns
        self.verbose      = verbose
        self._ref: Optional[AgentRef] = None

    # ── Runtime binding ───────────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        for agent in self.participants.values():
            runtime.register(agent)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute(message)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute(self, message: Message) -> None:
        if self._ref is None:
            raise RuntimeError(
                f"SwarmAgent '{self.name}' has not been bound to a runtime."
            )

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender

        current_agent_name = self.entry
        history: List[Dict[str, Any]] = []   # [{agent, findings, action, confidence, reason}]
        turns = 0

        if self.verbose:
            peer_names = list(self.participants.keys())
            print(f"\n── [{self.name}]  SWARM START  entry={self.entry} ──")
            print(f"   Peers: {', '.join(peer_names)}")

        while turns < self.max_turns:
            agent = self.participants.get(current_agent_name)
            if agent is None:
                raise KeyError(
                    f"Swarm: no participant named '{current_agent_name}'. "
                    f"Available: {list(self.participants.keys())}"
                )

            turns += 1
            if self.verbose:
                print(f"\n   ▶ Turn {turns} — {current_agent_name}")

            prompt = self._build_prompt(
                payload       = payload,
                agent_name    = current_agent_name,
                history       = history,
                peer_names    = [n for n in self.participants if n != current_agent_name],
            )

            raw    = await agent.think(prompt)
            decision = self._coerce_decision(raw)

            history.append({
                "agent":      current_agent_name,
                "findings":   decision.findings,
                "action":     decision.action,
                "confidence": decision.confidence,
                "reason":     decision.handoff_reason,
                "handoff_to": decision.handoff_to,
            })

            if self.verbose:
                action_tag = f"[{decision.action}|{decision.confidence}]"
                print(f"     {action_tag} {decision.findings[:100].replace(chr(10),' ')}…")
                if decision.handoff_to:
                    print(f"     → handoff to: {decision.handoff_to} — {decision.handoff_reason[:80]}")
                else:
                    print(f"     ■ TERMINATE — {decision.handoff_reason[:80]}")

            if decision.handoff_to is None:
                # Swarm terminates — last agent made final decision
                break

            # Validate handoff target before proceeding
            if decision.handoff_to not in self.participants:
                raise KeyError(
                    f"Agent '{current_agent_name}' tried to hand off to "
                    f"'{decision.handoff_to}', which is not in the swarm pool. "
                    f"Available: {list(self.participants.keys())}"
                )

            current_agent_name = decision.handoff_to

        else:
            # max_turns exceeded — record this
            if self.verbose:
                print(f"\n   ⚠  max_turns ({self.max_turns}) reached without termination.")

        # Final result
        final_turn  = history[-1] if history else {}
        final_action = final_turn.get("action", "UNKNOWN")

        if self.verbose:
            print(f"\n   ■  Swarm complete — {turns} turn(s) — final action: {final_action}")
            print(f"   ■  Sending '{self.result_type}' to '{caller}'")

        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = {
                "final_action": final_action,
                "final_agent":  final_turn.get("agent", "unknown"),
                "turns":        turns,
                "history":      history,
                "input":        payload,
            },
            metadata = {"in_reply_to": message.id},
        )

    # ── Prompt builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        payload:    Dict[str, Any],
        agent_name: str,
        history:    List[Dict[str, Any]],
        peer_names: List[str],
    ) -> str:
        lines: List[str] = []

        # ── Original case ──────────────────────────────────────────────────
        lines.append("── CASE DETAILS ──────────────────────────────────────")
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:400]}")
            else:
                lines.append(f"{k}: {v}")

        # ── Prior turns ────────────────────────────────────────────────────
        if history:
            lines.append("")
            lines.append("── PRIOR ANALYSIS (from other agents) ────────────────")
            for i, turn in enumerate(history, 1):
                lines.append(
                    f"Turn {i} — {turn['agent']} [{turn['action']}|{turn['confidence']}]:"
                )
                lines.append(f"  Findings: {turn['findings']}")
                lines.append(f"  Reason for handoff: {turn['reason']}")
                lines.append("")

        # ── Current agent instructions ─────────────────────────────────────
        lines.append(f"── YOUR TURN: {agent_name} ───────────────────────────")
        lines.append(
            "You are receiving control of this case. Review the case details "
            "and all prior analysis above, then apply your specialist expertise."
        )

        if peer_names:
            lines.append("")
            lines.append(
                "You may hand off to ONE of the following specialists "
                "(use the exact name) if their expertise is required:"
            )
            for n in peer_names:
                lines.append(f"  · {n}")
            lines.append(
                "Set handoff_to to null ONLY when you are making a FINAL decision "
                "that terminates the investigation — no further specialist review needed."
            )
        else:
            lines.append(
                "You are the only remaining specialist. "
                "YOU MUST terminate by setting handoff_to to null."
            )

        lines.append("")
        lines.append(
            "Respond ONLY with a valid JSON object matching the HandoffDecision schema."
        )

        return "\n".join(lines)

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
