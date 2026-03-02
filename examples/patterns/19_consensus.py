"""
Pattern: Consensus / Voting — Loan Committee Decision (Banking)
===============================================================
Demonstrates the ``ConsensusAgent`` pattern: N specialist agents produce
independent verdicts concurrently on the same input, then a judge synthesises
all opinions into a final authoritative decision.

Consensus vs other multi-agent patterns:

  parallel()        → N different agents process the same graph state as part
                       of a larger workflow; results feed into the next node
  MapReduceAgent    → the SAME agent processes N different data CHUNKS concurrently;
                       reducer merges results (not opinions)
  Swarm             → agents hand off control sequentially based on findings;
                       only one agent is active at a time
  Reflection        → one agent iteratively refines the SAME output via a critic
  Consensus         → N DIFFERENT agents give INDEPENDENT opinions on the SAME
                       input simultaneously; a judge then weighs and decides

Use-case — multi-member credit committee:
  The bank's lending policy requires all loans above $2M to be approved by
  a three-member credit committee, each member voting independently from their
  specialist perspective.  No panelist knows the others' views when voting.
  The committee chair then synthesises the verdicts.

  · ``credit-analyst``    — financial ratios, repayment capacity, leverage
  · ``risk-officer``      — risk rating, collateral/LTV, portfolio exposure
  · ``sector-expert``     — industry outlook, management, strategic fit
  · ``committee-chair``   — judge: weighs all opinions, final decision

Execution flow:

  submit("loan.committee.submitted")
         │
         ├──────────────────────────────────────────┐
         │  (concurrent, independent)               │
         ▼                       ▼                   ▼
  credit-analyst.think()  risk-officer.think()  sector-expert.think()
         │                       │                   │
         └──────────── asyncio.gather() ─────────────┘
                                 │
                                 ▼
                    committee-chair.think(all 3 verdicts)
                                 │
                                 ▼
                    send("loan.committee.decision") → client
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.consensus import ConsensusAgent, PanelistVerdict
from synaptum.prompts import FilePromptProvider


# ── Judge output schema ────────────────────────────────────────────────────────

class CommitteeDecision(BaseModel):
    decision: str = Field(
        description="APPROVE | APPROVE_WITH_CONDITIONS | DECLINE"
    )
    consensus_level: str = Field(
        description="UNANIMOUS | MAJORITY | SPLIT"
    )
    vote_tally: dict = Field(
        description="Count of each recommendation, e.g. {'APPROVE': 1, 'APPROVE_WITH_CONDITIONS': 2}."
    )
    deciding_factor: str = Field(
        description="The single most important factor that determined the outcome."
    )
    rationale: str = Field(
        description="Full justification for the decision (3-5 sentences)."
    )
    conditions: list[str] = Field(
        description="Consolidated conditions if approved. Empty if declining or unconditional."
    )
    dissent_note: str = Field(
        description="Explanation of any dissenting view and why it was or was not adopted."
    )
    risk_rating: str = Field(
        description="Overall credit risk: LOW | MEDIUM | HIGH | CRITICAL"
    )
    monitoring_requirements: list[str] = Field(
        description="Post-approval monitoring and reporting requirements."
    )


# ── Client handler ─────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type != "loan.committee.decision":
        return

    p         = msg.payload
    decision  = p["result"]
    verdicts  = p["verdicts"]
    panelists = p["panelists"]
    inp       = p["input"]

    decision_icon = {
        "APPROVE":                  "✅",
        "APPROVE_WITH_CONDITIONS":  "✅⚠️",
        "DECLINE":                  "🔴",
    }.get(decision.get("decision", ""), "❓")

    W = 64
    print(f"\n{'═' * W}")
    print(f"  LOAN COMMITTEE DECISION")
    print(f"  Applicant:      {inp.get('applicant_name','—')}")
    print(f"  Loan:           ${inp.get('loan_amount_usd', 0):,.0f}  ·  {inp.get('loan_purpose','—')[:40]}")
    print(f"  Panel:          {', '.join(panelists)}")
    print(f"{'─' * W}")

    # Individual verdicts
    print(f"\n  Individual verdicts:")
    for role, v in verdicts.items():
        conf_dot = {"LOW": "○", "MEDIUM": "◑", "HIGH": "●"}.get(v["confidence"], "?")
        rec_icon = {"APPROVE": "✅", "APPROVE_WITH_CONDITIONS": "⚠️ ", "DECLINE": "🔴"}.get(
            v["recommendation"], "?"
        )
        print(f"\n    {role}")
        print(f"      {rec_icon} {v['recommendation']}  {conf_dot} {v['confidence']}")
        print(f"      {v['rationale'][:110].replace(chr(10), ' ')}…")
        if v["key_concerns"]:
            print(f"      ✗ {v['key_concerns'][0]}")
        if v["conditions"]:
            for c in v["conditions"][:2]:
                print(f"      ∘ Condition: {c}")

    # Committee verdict
    print(f"\n{'─' * W}")
    tally = decision.get("vote_tally", {})
    tally_str = "  ".join(f"{k}: {v}" for k, v in tally.items())
    print(f"  Vote tally:     {tally_str}")
    print(f"  Consensus:      {decision.get('consensus_level','—')}")
    print(f"  Risk rating:    {decision.get('risk_rating','—')}")
    print(f"\n  {decision_icon}  FINAL DECISION: {decision.get('decision','—')}")
    print(f"\n  Deciding factor:")
    print(f"    {decision.get('deciding_factor','')}")
    print(f"\n  Rationale:")
    for line in decision.get("rationale", "").splitlines():
        print(f"    {line}")

    if decision.get("conditions"):
        print(f"\n  Conditions:")
        for c in decision["conditions"]:
            print(f"    · {c}")

    if decision.get("dissent_note"):
        print(f"\n  Dissent note:")
        print(f"    {decision['dissent_note']}")

    if decision.get("monitoring_requirements"):
        print(f"\n  Monitoring requirements:")
        for m in decision["monitoring_requirements"]:
            print(f"    · {m}")

    print(f"{'═' * W}\n")


# ── Bootstrap ──────────────────────────────────────────────────────────────────

async def main():
    bus             = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/consensus.yaml")
    runtime         = AgentRuntime(bus, prompts=prompt_provider)

    # ── Panel agents (run concurrently, independently) ────────────────────────
    credit_analyst = SimpleAgent(
        "credit-analyst",
        prompt_name  = "bank.consensus.credit_analyst.system",
        output_model = PanelistVerdict,
    )
    risk_officer = SimpleAgent(
        "risk-officer",
        prompt_name  = "bank.consensus.risk_officer.system",
        output_model = PanelistVerdict,
    )
    sector_expert = SimpleAgent(
        "sector-expert",
        prompt_name  = "bank.consensus.sector_expert.system",
        output_model = PanelistVerdict,
    )

    # ── Judge ─────────────────────────────────────────────────────────────────
    committee_chair = SimpleAgent(
        "committee-chair",
        prompt_name  = "bank.consensus.committee_chair.system",
        output_model = CommitteeDecision,
    )

    # ── ConsensusAgent ────────────────────────────────────────────────────────
    loan_committee = ConsensusAgent(
        "loan-committee",
        panelists = {
            "credit-analyst": credit_analyst,
            "risk-officer":   risk_officer,
            "sector-expert":  sector_expert,
        },
        judge       = committee_chair,
        submit_type = "loan.committee.submitted",
        result_type = "loan.committee.decision",
        verbose     = True,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(loan_committee)
    runtime.register(client)

    await runtime.start(run_id="run-committee-2026")

    # ── Case: Agroindustrias del Meta S.A.S. ──────────────────────────────────
    # A mid-market palm-oil processor seeking $2.8M for a refinery upgrade.
    # Revenue growing but customer concentration is high and leverage elevated.
    await client._ref.send(
        to      = "loan-committee",
        type    = "loan.committee.submitted",
        payload = {
            "applicant_name":   "Agroindustrias del Meta S.A.S.",
            "loan_amount_usd":  2_800_000,
            "loan_purpose":     "Modernisation of palm-oil refinery — new fractional distillation unit",
            "loan_tenor_years": 8,
            "country":          "Colombia",
            "industry":         "Agro-industrial processing (palm oil)",
            "financials": {
                "annual_revenue_usd":           11_200_000,
                "revenue_prior_year_usd":        9_600_000,
                "ebitda_usd":                    2_016_000,
                "ebit_usd":                      1_344_000,
                "net_income_usd":                680_000,
                "total_assets_usd":              18_500_000,
                "total_debt_usd":                7_400_000,
                "total_equity_usd":              7_100_000,
                "cash_usd":                      420_000,
                "annual_interest_expense_usd":   680_000,
                "annual_principal_repayment_usd": 560_000,
                "capex_usd":                     310_000,
                "accounts_receivable_days":      48,
                "inventory_days":                35,
            },
            "collateral": {
                "type":              "Industrial real estate + processing equipment",
                "description":       "Refinery plant in Villavicencio (5,500 m²) + existing machinery",
                "appraised_value_usd": 3_800_000,
                "first_charge":      True,
                "ltv_pct":           73.7,
            },
            "company_background": (
                "Founded 2006. 140 employees. Family-owned (Ospina family, 3rd generation). "
                "Revenue: 16.7% YoY growth driven by new export contracts to Panama and Costa Rica. "
                "Customer concentration: two buyers account for 71% of revenue (Grupo Alianza 44%, "
                "Exportadora Andina 27%). "
                "No prior loan defaults. Existing $1.2M equipment line with the bank (current). "
                "Planned refinery upgrade will increase capacity by 35% and reduce energy costs by 20%."
            ),
        },
        reply_to = "client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
