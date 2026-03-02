"""
ConsensusAgent — independent multi-panel voting and synthesis pattern.

Design
------
The Consensus pattern solicits *independent* opinions from N specialist agents
and then routes all responses to a judge that synthesises them into a final
authoritative decision.

  1. **Panel**    — all panelist agents receive the SAME input concurrently
                    (via ``asyncio.gather``).  They produce independent verdicts
                    with no knowledge of each other's views.

  2. **Aggregate** — the judge receives every panelist response plus the
                     original input and produces a synthesized final decision.
                     The judge can weigh perspectives, identify consensus vs.
                     dissent, and override minority views with justification.

  3. **Deliver**  — the result includes the final decision, each panelist's
                    individual verdict, and a consensus summary.

Difference from other patterns
-------------------------------
  parallel()       → N different agents process the SAME graph state
                      as part of a larger workflow; there is no judge
  MapReduceAgent   → the SAME agent processes N different data chunks;
                      reducer merges results (not opinions)
  Swarm            → agents hand off control sequentially based on findings
  Reflection       → one agent iteratively refines a single output
  Consensus        → N DIFFERENT agents produce INDEPENDENT verdicts on the
                      SAME input; a judge synthesises them — majority or
                      weighted vote, or expert synthesis

Usage
-----
::

    from synaptum.patterns.consensus import ConsensusAgent

    credit_analyst = SimpleAgent("credit-analyst", ...)
    risk_officer   = SimpleAgent("risk-officer",   ...)
    sector_expert  = SimpleAgent("sector-expert",  ...)
    judge          = SimpleAgent("credit-committee-chair", ...)

    panel = ConsensusAgent(
        "loan-committee",
        panelists   = {
            "credit-analyst": credit_analyst,
            "risk-officer":   risk_officer,
            "sector-expert":  sector_expert,
        },
        judge       = judge,
        submit_type = "loan.committee.submitted",
        result_type = "loan.committee.decision",
        verbose     = True,
    )

    runtime.register(panel)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef


# ── Panelist verdict ──────────────────────────────────────────────────────────

class PanelistVerdict(BaseModel):
    """
    The structured response each panelist must return.

    Fields
    ------
    recommendation : str
        This panelist's recommended decision, e.g. APPROVE / DECLINE / etc.
    confidence : str
        Confidence level: LOW | MEDIUM | HIGH.
    rationale : str
        Detailed justification for the recommendation (3-5 sentences).
    key_concerns : list[str]
        Top 2-4 concerns or risks (empty list if none).
    key_positives : list[str]
        Top 2-4 strengths or positives (empty list if none).
    conditions : list[str]
        Any conditions the panelist would attach to a positive decision.
        Empty list if unconditional or declining.
    """
    recommendation: str = Field(
        description="This panelist's recommended decision."
    )
    confidence: str = Field(
        description="Confidence level: LOW | MEDIUM | HIGH."
    )
    rationale: str = Field(
        description="Detailed justification for the recommendation (3-5 sentences)."
    )
    key_concerns: List[str] = Field(
        description="Top 2-4 specific concerns or risks."
    )
    key_positives: List[str] = Field(
        description="Top 2-4 strengths or positives."
    )
    conditions: List[str] = Field(
        description=(
            "Conditions attached to a positive recommendation. "
            "Empty if unconditional or declining."
        )
    )


# ── ConsensusAgent ────────────────────────────────────────────────────────────

class ConsensusAgent(Agent):
    """
    Runs N panelist agents concurrently on the same input, then routes all
    verdicts to a judge agent for synthesis into a final decision.

    Parameters
    ----------
    name : str
        Bus address for this agent.
    panelists : dict[str, Agent]
        Specialist agents keyed by role name.  All run concurrently.
    judge : Agent
        Synthesises panelist verdicts into the final decision.
    submit_type : str
        Message type that triggers a run.
    result_type : str
        Message type for the final result.
    verbose : bool
        Print panel trace to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        panelists: Dict[str, Agent],
        judge: Agent,
        submit_type: str = "consensus.submitted",
        result_type: str = "consensus.result",
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.panelists   = panelists
        self.judge       = judge
        self.submit_type = submit_type
        self.result_type = result_type
        self.verbose     = verbose
        self._ref: Optional[AgentRef] = None

    # ── Runtime binding ───────────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        for agent in self.panelists.values():
            runtime.register(agent)
        runtime.register(self.judge)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute(message)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute(self, message: Message) -> None:
        if self._ref is None:
            raise RuntimeError(
                f"ConsensusAgent '{self.name}' has not been bound to a runtime."
            )

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender

        if self.verbose:
            names = list(self.panelists.keys())
            print(f"\n── [{self.name}]  CONSENSUS PANEL  ({len(names)} panelists) ──")
            print(f"   Panelists: {', '.join(names)}")

        # ── 1. Concurrent panel ───────────────────────────────────────────────
        panel_prompt = self._panel_prompt(payload)

        raw_verdicts = await asyncio.gather(
            *[agent.think(panel_prompt) for agent in self.panelists.values()]
        )

        verdicts: Dict[str, PanelistVerdict] = {}
        for role, raw in zip(self.panelists.keys(), raw_verdicts):
            verdict = self._coerce_verdict(raw)
            verdicts[role] = verdict
            if self.verbose:
                print(
                    f"   ✦ {role:28s}  [{verdict.recommendation}|{verdict.confidence}]  "
                    f"{verdict.rationale[:60].replace(chr(10),' ')}…"
                )

        # Quick consensus/dissent summary for verbose output
        if self.verbose:
            recs = [v.recommendation for v in verdicts.values()]
            unique = set(recs)
            if len(unique) == 1:
                print(f"\n   → UNANIMOUS: all panelists recommend {recs[0]}")
            else:
                counts = {r: recs.count(r) for r in unique}
                summary = ", ".join(f"{r}×{c}" for r, c in counts.items())
                print(f"\n   → SPLIT panel: {summary}")

        # ── 2. Judge synthesises ──────────────────────────────────────────────
        if self.verbose:
            print(f"\n   ▶ Judge synthesising…")

        judge_prompt = self._judge_prompt(payload, verdicts)
        raw_final    = await self.judge.think(judge_prompt)
        final_result = (
            raw_final.model_dump()
            if hasattr(raw_final, "model_dump")
            else (raw_final if isinstance(raw_final, dict) else {"text": raw_final})
        )

        if self.verbose:
            decision = final_result.get("decision", final_result.get("recommendation", "?"))
            print(f"   ■  Judge decision: {decision}")
            print(f"   ■  Sending '{self.result_type}' to '{caller}'")

        # ── 3. Deliver ────────────────────────────────────────────────────────
        verdicts_dict = {role: v.model_dump() for role, v in verdicts.items()}

        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = {
                "result":    final_result,
                "verdicts":  verdicts_dict,
                "panelists": list(self.panelists.keys()),
                "input":     payload,
            },
            metadata = {"in_reply_to": message.id},
        )

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _panel_prompt(payload: Dict[str, Any]) -> str:
        lines: List[str] = [
            "── CASE FOR REVIEW ─────────────────────────────────────",
        ]
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:500]}")
            else:
                lines.append(f"{k}: {v}")

        lines += [
            "",
            "Review the case above using your specialist expertise.",
            "Your opinion is INDEPENDENT — you have not seen any other panelist's view.",
            "Be specific and rigorous. Your verdict will be presented to a judge.",
            "",
            "Respond ONLY with a valid JSON object matching the PanelistVerdict schema.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _judge_prompt(
        payload: Dict[str, Any],
        verdicts: Dict[str, PanelistVerdict],
    ) -> str:
        lines: List[str] = [
            "── ORIGINAL CASE ───────────────────────────────────────",
        ]
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:400]}")
            else:
                lines.append(f"{k}: {v}")

        lines += ["", "── PANELIST VERDICTS ───────────────────────────────────"]
        for role, verdict in verdicts.items():
            lines.append(f"\n{role}  [{verdict.recommendation} | {verdict.confidence}]")
            lines.append(f"  Rationale: {verdict.rationale}")
            if verdict.key_concerns:
                lines.append(f"  Concerns:  {'; '.join(verdict.key_concerns)}")
            if verdict.key_positives:
                lines.append(f"  Positives: {'; '.join(verdict.key_positives)}")
            if verdict.conditions:
                lines.append(f"  Conditions: {'; '.join(verdict.conditions)}")

        lines += [
            "",
            "── YOUR ROLE: JUDGE / SYNTHESISER ─────────────────────",
            "You have received independent verdicts from all panelists above.",
            "Your job:",
            "  1. Identify where panelists agree (consensus) and disagree (dissent).",
            "  2. Weigh the arguments — not just count votes.",
            "  3. Produce a final authoritative decision with full justification.",
            "  4. If overriding a minority view, explain why explicitly.",
            "  5. Merge all conditions from positive panelists into a final",
            "     conditions list, removing duplicates.",
            "",
            "Respond ONLY with a valid JSON object matching the required schema.",
        ]
        return "\n".join(lines)

    # ── Coercion ──────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_verdict(raw: Any) -> PanelistVerdict:
        if isinstance(raw, PanelistVerdict):
            return raw
        if isinstance(raw, dict):
            return PanelistVerdict.model_validate(raw)
        if isinstance(raw, str):
            return PanelistVerdict.model_validate_json(raw)
        raise TypeError(f"Cannot coerce {type(raw)} to PanelistVerdict")
