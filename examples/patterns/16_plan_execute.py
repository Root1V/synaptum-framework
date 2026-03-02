"""
Pattern: Plan-and-Execute — Corporate Debt Restructuring (Banking)
==================================================================
Demonstrates the ``PlanAndExecuteAgent`` pattern: a planner agent creates
an explicit, ordered analytical plan; each step is executed by a designated
specialist; after each step the planner reviews the result and may adaptively
revise the remaining plan before proceeding.

Plan-and-Execute vs other multi-agent patterns:

  Supervisor/Worker    → planner dispatches tasks dynamically with no upfront
                         plan; workers are unaware of each other
  GraphBuilder         → structure is fully declared at build-time; transitions
                         are deterministic (or conditional on a fixed rule)
  Plan-and-Execute     → EXPLICIT upfront plan artefact; steps are ordered and
                         assigned to named executors; plan can be revised
                         mid-execution if earlier results reveal new information

Use-case — corporate debt restructuring:
  A commercial bank is asked to restructure $8.5 M of corporate debt for a
  conglomerate (real estate + retail).  The complexity requires multiple
  specialists.  The plan-and-execute pattern is ideal because:
    1. The planner decides which analyses are needed and in what order.
    2. A financial audit may reveal insolvency → triggers a replan to add
       a specialised distress-resolution step before the legal review.
    3. A final Credit Committee Memorandum synthesises all findings.

Execution flow:

  submit("restructuring.submitted")
         │
         ▼
  planner.think(goal)  →  Plan { steps: [step-1 … step-N] }
         │
         ├─ step-1 ─► financial-analyst.think(prompt + context)  ─┐
         │                                                          │ check → replan?
         ├─ step-2 ─► market-analyst.think(prompt + context)    ─-┤
         │                                                          │ check → replan?
         ├─ step-3 ─► legal-advisor.think(prompt + context)     ─-┤
         │                                                          │ check → replan?
         └─ step-4 ─► restructuring-specialist.think(…context)  ──┘
                │
                ▼
        finalizer.think(all results) → CreditMemo
                │
                ▼
        send("restructuring.memo") → client
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.plan_execute import (
    Plan,
    PlanAndExecuteAgent,
    ReplanDecision,
)
from synaptum.prompts import FilePromptProvider


# ── Output schemas ─────────────────────────────────────────────────────────────

class FinancialAnalysis(BaseModel):
    financial_risk: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    viability: str = Field(description="VIABLE | MARGINAL | DISTRESSED | INSOLVENT")
    interest_coverage_ratio: float | None = Field(
        description="EBIT / interest expense, null if not computable."
    )
    net_debt_ebitda: float | None = Field(
        description="Net debt / EBITDA multiple, null if not computable."
    )
    altman_flag: str = Field(description="SAFE | GREY | DISTRESS")
    maximum_sustainable_debt_usd: float | None = Field(
        description="Maximum debt the company can service at current earnings."
    )
    key_findings: list[str] = Field(description="3-5 key analytical findings.")
    summary: str = Field(description="2-3 sentence narrative.")


class MarketAnalysis(BaseModel):
    market_risk: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    business_viability: str = Field(
        description="SOUND | CHALLENGED | UNCERTAIN | FAILING"
    )
    industry_outlook: str = Field(
        description="GROWING | STABLE | DECLINING | CONTRACTING"
    )
    asset_quality: str = Field(description="HIGH | MEDIUM | LOW")
    collateral_recovery_pct: float | None = Field(
        description="Estimated recovery as % of book value under forced sale."
    )
    strategic_risks: list[str] = Field(description="2-4 strategic risk items.")
    summary: str = Field(description="2-3 sentence narrative.")


class LegalAnalysis(BaseModel):
    legal_complexity: str = Field(
        description="LOW | MEDIUM | HIGH | VERY_HIGH"
    )
    recommended_mechanism: str = Field(
        description="Recommended legal restructuring vehicle."
    )
    covenant_status: str = Field(
        description="COMPLIANT | BREACHED | AT_RISK"
    )
    security_quality: str = Field(
        description="STRONG | ADEQUATE | WEAK | NONE"
    )
    creditor_consent_required: bool = Field(
        description="True if multi-creditor consent is needed."
    )
    estimated_timeline_weeks: int | None = Field(
        description="Estimated weeks to complete the restructuring."
    )
    legal_risks: list[str] = Field(description="2-4 legal risk items.")
    summary: str = Field(description="2-3 sentence narrative.")


class RestructuringProposal(BaseModel):
    recommendation: str = Field(
        description="APPROVE | APPROVE_WITH_CONDITIONS | DECLINE"
    )
    principal_haircut_pct: float = Field(
        description="Principal write-down as % of outstanding (0 if none)."
    )
    new_interest_rate_pct: float = Field(description="Proposed new annual interest rate.")
    tenor_extension_months: int = Field(
        description="Additional months added to loan tenor."
    )
    grace_period_months: int = Field(
        description="Months of principal grace period (0 if none)."
    )
    equity_component: str | None = Field(
        description="Description of any equity/quasi-equity element, or null."
    )
    recovery_restructured_pct: float = Field(
        description="Expected bank recovery % under restructuring."
    )
    recovery_liquidation_pct: float = Field(
        description="Expected bank recovery % under liquidation (for comparison)."
    )
    additional_provisions_pct: float = Field(
        description="Additional loan-loss provisions required as % of outstanding."
    )
    success_probability: str = Field(description="HIGH | MEDIUM | LOW")
    conditions_precedent: list[str] = Field(
        description="Required conditions before restructuring takes effect."
    )
    key_risks: list[str] = Field(description="Main risks to the proposed plan.")
    summary: str = Field(description="3-4 sentence narrative.")


class CreditMemo(BaseModel):
    executive_summary: str = Field(
        description="3-5 sentence summary for the Credit Committee."
    )
    client_risk_profile: dict = Field(
        description="Dict with 'rating' (LOW/MEDIUM/HIGH/CRITICAL) and 'narrative'."
    )
    restructuring_terms: str = Field(
        description="Concise summary of proposed restructuring terms."
    )
    bank_position: str = Field(
        description="Expected recovery under deal vs. liquidation; provisions impact."
    )
    conditions: list[str] = Field(
        description="Key conditions precedent and post-deal monitoring requirements."
    )
    recommendation: str = Field(
        description="Final credit decision: APPROVE | APPROVE_WITH_CONDITIONS | DECLINE"
    )
    justification: str = Field(description="2-3 sentence justification.")


# ── Client handler ─────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type != "restructuring.memo":
        return

    p       = msg.payload
    memo    = p["result"]
    plan    = p["plan"]
    steps   = p["steps"]
    replans = p["replans"]
    inp     = p["input"]
    client  = inp.get("client_name", "—")

    W = 60
    print(f"\n{'═' * W}")
    print(f"  CREDIT COMMITTEE MEMORANDUM")
    print(f"  Client:          {client}")
    print(f"  Debt:            ${inp.get('total_debt_usd', 0):,.0f}")
    print(f"  Plan steps:      {len(steps)}  (replans: {replans})")
    print(f"{'─' * W}")

    # Executive Summary
    print(f"\n  {memo.get('executive_summary','')}")

    # Risk Profile
    profile = memo.get("client_risk_profile", {})
    print(f"\n  Risk Rating:  {profile.get('rating','—')}")
    if profile.get("narrative"):
        print(f"  {profile['narrative']}")

    # Restructuring Terms
    print(f"\n  Terms:  {memo.get('restructuring_terms','—')}")
    print(f"  Bank position:  {memo.get('bank_position','—')}")

    if memo.get("conditions"):
        print(f"\n  Conditions:")
        for c in memo["conditions"]:
            print(f"    · {c}")

    print(f"\n{'─' * W}")
    print(f"  RECOMMENDATION:  {memo.get('recommendation','—')}")
    print(f"  {memo.get('justification','')}")

    # Step-by-step summary
    print(f"\n{'─' * W}")
    print(f"  Analytical steps executed:")
    for entry in steps:
        s = entry["step"]
        print(f"    [{s['id']:6s}] {s['executor']:28s} – {s['description'][:38]}…")

    print(f"{'═' * W}\n")


# ── Bootstrap ──────────────────────────────────────────────────────────────────

async def main():
    bus             = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/plan_execute.yaml")
    runtime         = AgentRuntime(bus, prompts=prompt_provider)

    # ── Planner agents ────────────────────────────────────────────────────────
    planner = SimpleAgent(
        "deal-planner",
        prompt_name  = "bank.plan_execute.planner.system",
        output_model = Plan,
    )
    replanner = SimpleAgent(
        "deal-replanner",
        prompt_name  = "bank.plan_execute.replan.system",
        output_model = ReplanDecision,
    )
    finalizer = SimpleAgent(
        "deal-writer",
        prompt_name  = "bank.plan_execute.finalizer.system",
        output_model = CreditMemo,
    )

    # ── Specialist executors ──────────────────────────────────────────────────
    financial_analyst = SimpleAgent(
        "financial-analyst",
        prompt_name  = "bank.plan_execute.financial_analyst.system",
        output_model = FinancialAnalysis,
    )
    market_analyst = SimpleAgent(
        "market-analyst",
        prompt_name  = "bank.plan_execute.market_analyst.system",
        output_model = MarketAnalysis,
    )
    legal_advisor = SimpleAgent(
        "legal-advisor",
        prompt_name  = "bank.plan_execute.legal_advisor.system",
        output_model = LegalAnalysis,
    )
    restructuring_specialist = SimpleAgent(
        "restructuring-specialist",
        prompt_name  = "bank.plan_execute.restructuring_specialist.system",
        output_model = RestructuringProposal,
    )

    # ── PlanAndExecuteAgent ───────────────────────────────────────────────────
    processor = PlanAndExecuteAgent(
        "debt-restructuring",
        planner   = planner,
        replanner = replanner,
        finalizer = finalizer,
        executors = {
            "financial-analyst":        financial_analyst,
            "market-analyst":           market_analyst,
            "legal-advisor":            legal_advisor,
            "restructuring-specialist": restructuring_specialist,
        },
        submit_type = "restructuring.submitted",
        result_type = "restructuring.memo",
        max_replans = 2,
        verbose     = True,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(processor)  # also registers all child agents
    runtime.register(client)

    await runtime.start(run_id="run-hec-restructuring-2026")

    # ── Case: Holding Empresarial del Caribe S.A.S. ───────────────────────────
    # A mid-sized conglomerate (commercial real estate 60% + retail chain 40%)
    # requesting restructuring of $8.5 M in total debt after a revenue drop.
    await client._ref.send(
        to      = "debt-restructuring",
        type    = "restructuring.submitted",
        payload = {
            "client_name":         "Holding Empresarial del Caribe S.A.S.",
            "industry":            "Commercial real estate (60%) + retail chain (40%)",
            "country":             "Colombia",
            "total_debt_usd":      8_500_000,
            "debt_facilities": [
                {
                    "name":            "Senior Secured Term Loan",
                    "outstanding_usd": 5_000_000,
                    "rate_pct":        9.5,
                    "maturity":        "2025-06-30",
                    "collateral":      "Three commercial office buildings in Barranquilla",
                    "status":          "Current — maturity in 3 months, refinancing not secured",
                },
                {
                    "name":            "Revolving Working Capital Line",
                    "outstanding_usd": 2_200_000,
                    "rate_pct":        12.0,
                    "maturity":        "2024-12-31",
                    "collateral":      "Accounts receivable pledge",
                    "status":          "Already past due 45 days — lender has issued default notice",
                },
                {
                    "name":            "Subordinated Mezzanine Note",
                    "outstanding_usd": 1_300_000,
                    "rate_pct":        15.5,
                    "maturity":        "2026-12-31",
                    "collateral":      "None (unsecured)",
                    "status":          "Current — but mezz holder has cross-default rights",
                },
            ],
            "financials": {
                "annual_revenue_usd":        12_400_000,
                "revenue_change_yoy_pct":    -28,
                "ebitda_usd":                1_820_000,
                "ebit_usd":                  980_000,
                "annual_interest_expense_usd": 1_050_000,
                "net_income_usd":            -210_000,
                "total_assets_usd":          22_000_000,
                "total_equity_usd":          8_700_000,
                "cash_usd":                  340_000,
                "notes": (
                    "Revenue decline driven by pandemic-era retail lease non-renewals "
                    "and a 35% occupancy drop in office buildings. "
                    "EBITDA positive but insufficient to cover full interest burden. "
                    "Net loss for second consecutive year."
                ),
            },
            "restructuring_request": (
                "Consolidate all three facilities into a single 7-year senior secured "
                "term loan at a rate the company can service.  The owner proposes a "
                "12-month principal grace period, partial write-down of the mezz "
                "principal, and pledging two additional residential properties as "
                "supplemental collateral."
            ),
        },
        reply_to = "client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
