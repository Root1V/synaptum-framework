"""
Pattern: Swarm / Handoff — Suspicious Wire Transfer Investigation (Banking)
===========================================================================
Demonstrates the ``SwarmAgent`` pattern: each specialist agent autonomously
decides whether to handle the case to completion or hand control off to a
different peer — with no external orchestrator making that choice.

Swarm vs other multi-agent patterns:

  Router            → an *external* component decides which agent handles the
                       message; the agent never chooses the next one
  Supervisor/Worker → a central supervisor dispatches tasks; workers are
                       passive and never redirect the workflow
  Plan-and-Execute  → an upfront plan assigns who does what; a planner (not
                       an executor) decides sequence
  Swarm             → agents are PEERS; any agent can hand off to any other
                       based on what IT discovers; the decision is made from
                       INSIDE the agent, not from a central authority

Use-case — cross-department fraud / AML / sanctions investigation:
  The bank's automated monitoring system flags a suspicious wire transfer.
  Multiple specialist departments may need to be involved, but which ones —
  and in what order — depends on what each inspection reveals.  This is
  exactly the swarm use case: nobody plans the path upfront.

  · ``fraud-analyst``       — first responder; detects fraud patterns,
                               velocity anomalies, and structuring signals
  · ``aml-specialist``      — investigates money-laundering typologies
                               (placement / layering / integration)
  · ``sanctions-screener``  — screens counterparties against OFAC, EU, UN
                               lists; identifies PEP connections
  · ``compliance-officer``  — FINAL decision-maker; synthesises all findings
                               and issues CLEAR / BLOCK / ESCALATE verdict

Architecture — pure message choreography (no direct think() calls):

  client ──[alert.triggered]──▶ fraud-investigation-swarm (SwarmAgent)
               │
        [__swarm.turn__]──▶ fraud-analyst (LLMAgent)
                                  │
                           [agent.output]──▶ fraud-investigation-swarm
                                  │  "structuring pattern + offshore → AML"
                           [__swarm.turn__]──▶ aml-specialist (LLMAgent)
                                                     │
                                              [agent.output]──▶ fraud-investigation-swarm
                                                     │  "BVI shell + layering → sanctions"
                                              [__swarm.turn__]──▶ sanctions-screener (LLMAgent)
                                                                         │
                                                                  [agent.output]──▶ fraud-investigation-swarm
                                                                         │  "PEP match → compliance must decide"
                                                                  [__swarm.turn__]──▶ compliance-officer (LLMAgent)
                                                                                            │
                                                                                     [agent.output]──▶ fraud-investigation-swarm
                                                                                            │  handoff_to=null → TERMINATE
                                                                               [alert.decision]──▶ client
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents import LLMAgent
from synaptum.agents.message_agent import MessageAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.swarm import HandoffDecision, SwarmAgent
from synaptum.prompts import FilePromptProvider


# ── Client handler ─────────────────────────────────────────────────────────────

async def client_handler(agent: MessageAgent, msg: Message, ctx):
    if msg.type != "alert.decision":
        return

    p              = msg.payload
    history        = p["history"]
    inp            = p["input"]
    elapsed_total  = p.get("elapsed_total_s", 0.0)

    action_colors = {
        "CLEAR":    "✅",
        "APPROVE":  "✅",
        "BLOCK":    "🔴",
        "ESCALATE": "🚨",
    }

    W = 62
    print(f"\n{'═' * W}")
    print(f"  FRAUD / AML INVESTIGATION DECISION")
    print(f"  Alert:          {inp.get('alert_id', '—')}")
    print(f"  Transaction:    {inp.get('transaction_id', '—')}")
    amount = inp.get("amount_usd")
    if amount:
        print(f"  Amount:         ${amount:,.2f}")
    print(f"  Turns:          {p['turns']}")
    print(f"  Final agent:    {p['final_agent']}")
    print(f"  Total time:     {elapsed_total:.2f}s")

    icon = action_colors.get(p["final_action"], "❓")
    print(f"\n  {icon}  DECISION: {p['final_action']}")
    print(f"{'─' * W}")

    # Print handoff chain
    print(f"\n  Investigation trail:")
    for i, turn in enumerate(history, 1):
        elapsed = turn.get("elapsed_s", 0.0)
        arrow = "→" if turn.get("handoff_to") else "■"
        print(f"\n  [{i}] {turn['agent']}  [{turn['action']}|{turn['confidence']}]  ⏱ {elapsed:.2f}s")
        # Print findings wrapped
        findings = turn["findings"]
        for line in findings.splitlines():
            print(f"      {line}")
        if turn.get("handoff_to"):
            print(f"      {arrow} handed off to: {turn['handoff_to']}")
            print(f"        reason: {turn['reason']}")
        else:
            print(f"      ■ TERMINATED — {turn['reason']}")

    # Timing summary table
    print(f"\n{'─' * W}")
    print(f"  Timing breakdown:")
    for turn in history:
        elapsed = turn.get("elapsed_s", 0.0)
        bar_len = min(int(elapsed * 2), 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        pct     = (elapsed / elapsed_total * 100) if elapsed_total else 0
        print(f"    {turn['agent']:22s}  {elapsed:5.2f}s  {pct:4.0f}%  {bar}")
    print(f"    {'─' * 22}  {'─' * 5}")
    print(f"    {'TOTAL':22s}  {elapsed_total:5.2f}s")

    print(f"\n{'═' * W}\n")


# ── Bootstrap ──────────────────────────────────────────────────────────────────

async def main():
    bus             = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/swarm.yaml")
    runtime         = AgentRuntime(bus, prompts=prompt_provider)

    # ── Swarm participants ─────────────────────────────────────────────────────
    fraud_analyst = LLMAgent(
        "fraud-analyst",
        prompt_name  = "bank.swarm.fraud_analyst.system",
        output_model = HandoffDecision,
    )
    aml_specialist = LLMAgent(
        "aml-specialist",
        prompt_name  = "bank.swarm.aml_specialist.system",
        output_model = HandoffDecision,
    )
    sanctions_screener = LLMAgent(
        "sanctions-screener",
        prompt_name  = "bank.swarm.sanctions_screener.system",
        output_model = HandoffDecision,
    )
    compliance_officer = LLMAgent(
        "compliance-officer",
        prompt_name  = "bank.swarm.compliance_officer.system",
        output_model = HandoffDecision,
    )

    # ── SwarmAgent ─────────────────────────────────────────────────────────────
    investigation_swarm = SwarmAgent(
        "fraud-investigation-swarm",
        participants = {
            "fraud-analyst":      fraud_analyst,
            "aml-specialist":     aml_specialist,
            "sanctions-screener": sanctions_screener,
            "compliance-officer": compliance_officer,
        },
        entry                = "fraud-analyst",
        submit_type          = "alert.triggered",
        result_type          = "alert.decision",
        turn_prompt_name     = "bank.swarm.turn_user_prompt",
        handoff_prompt_name  = "bank.swarm.handoff_user_prompt",
        max_turns            = 8,
        verbose              = True,
    )

    client = MessageAgent("client", handler=client_handler)

    # Participants are registered independently — SwarmAgent does NOT auto-register them.
    runtime.register(fraud_analyst)
    runtime.register(aml_specialist)
    runtime.register(sanctions_screener)
    runtime.register(compliance_officer)
    runtime.register(investigation_swarm)
    runtime.register(client)

    await runtime.start(run_id="run-swarm-alert-2026")

    # ── Case: Alerta AML-2026-0312 ─────────────────────────────────────────────
    # Constructora Río Verde S.A.S. initiates three same-day wire transfers
    # totalling $2.3M to an offshore shell in the BVI.  The bank's monitoring
    # system flags them.  The swarm investigates autonomously.
    await client._ref.send(
        to      = "fraud-investigation-swarm",
        type    = "alert.triggered",
        payload = {
            "alert_id":       "AML-2026-0312",
            "alert_type":     "Suspicious outbound wire — possible structuring",
            "transaction_id": "TXN-2026-031209",
            "customer": {
                "id":             "CORP-4471",
                "name":           "Constructora Río Verde S.A.S.",
                "type":           "Corporate",
                "country":        "Colombia",
                "industry":       "Construction",
                "years_as_client": 3,
                "avg_monthly_outbound_usd": 180_000,
                "last_wire_amount_usd": 95_000,
                "recent_profile_changes": "Email and phone updated 11 days ago",
            },
            "transactions": [
                {
                    "id":          "TXN-2026-031209-A",
                    "date":        "2026-03-12 09:14",
                    "amount_usd":  980_000,
                    "beneficiary": "Astoria Holdings Ltd.",
                    "bene_bank":   "First BVI Bank",
                    "bene_country":"British Virgin Islands",
                    "purpose":     "Construction equipment purchase",
                },
                {
                    "id":          "TXN-2026-031209-B",
                    "date":        "2026-03-12 11:47",
                    "amount_usd":  820_000,
                    "beneficiary": "Astoria Holdings Ltd.",
                    "bene_bank":   "First BVI Bank",
                    "bene_country":"British Virgin Islands",
                    "purpose":     "Construction equipment purchase",
                },
                {
                    "id":          "TXN-2026-031209-C",
                    "date":        "2026-03-12 14:22",
                    "amount_usd":  500_000,
                    "beneficiary": "Astoria Holdings Ltd.",
                    "bene_bank":   "First BVI Bank",
                    "bene_country":"British Virgin Islands",
                    "purpose":     "Consulting services",
                },
            ],
            "amount_usd": 2_300_000,
            "intelligence_notes": (
                "Astoria Holdings Ltd. was incorporated in the BVI 6 weeks ago. "
                "No known web presence. UBO disclosed as 'Carlos Mendoza Rivas'. "
                "OSINT shows a 'Carlos A. Mendoza Rivas' is the brother of a "
                "serving deputy minister of public works in Venezuela. "
                "Customer called branch twice this week asking to expedite transfers."
            ),
        },
        reply_to = "client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
