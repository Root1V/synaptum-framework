"""
Pattern: Map-Reduce — Annual Portfolio Credit Review (Banking)
==============================================================
Demonstrates the ``MapReduceAgent`` pattern: the same analyst agent is applied
independently and concurrently to each loan in a portfolio (map phase), then a
portfolio manager synthesises all individual assessments into a single portfolio
health report (reduce phase).

Map-Reduce vs parallel():

  parallel()        → N *different* agents process the *same* input simultaneously
  MapReduceAgent    → the *same* agent processes N *different* chunks simultaneously

Use-case — annual credit review:
  A bank must reassess the risk of every active loan in a portfolio once a year.
  Each loan is independent, so the analyst runs concurrently across all loans.
  The portfolio manager then reviews all assessments to flag concentration risk,
  assign an overall health rating, and recommend actions.

Graph of execution (not a GraphBuilder graph — a standalone agent pattern):

  submit("portfolio.submitted")
        │
        ▼
  splitter(payload) → [loan_1, loan_2, loan_3, loan_4, loan_5]
        │
        ├─ loan_analyst.think(loan_1) ─┐
        ├─ loan_analyst.think(loan_2) ─┤
        ├─ loan_analyst.think(loan_3) ─┤ asyncio.gather()
        ├─ loan_analyst.think(loan_4) ─┤
        └─ loan_analyst.think(loan_5) ─┘
                                       │
                                       ▼
                        portfolio_manager.think(all_5_results)
                                       │
                                       ▼
                        send("portfolio.report") → client
"""

import asyncio
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.map_reduce import MapReduceAgent
from synaptum.prompts import FilePromptProvider


# ── Output schemas ────────────────────────────────────────────────────────────

class LoanAssessment(BaseModel):
    loan_id: str = Field(
        description="Loan identifier, copied verbatim from the input."
    )
    risk_rating: str = Field(
        description=(
            "Current risk rating: PERFORMING, WATCH, SUBSTANDARD, DOUBTFUL, or LOSS."
        )
    )
    risk_change: str = Field(
        description=(
            "Change from last rating: IMPROVED, STABLE, or DETERIORATED."
        )
    )
    dti_percent: float = Field(
        description="Current debt-to-income ratio as a percentage."
    )
    key_risks: list[str] = Field(
        description="2-3 main risk factors for this loan."
    )
    recommended_action: str = Field(
        description=(
            "Recommended action: MAINTAIN, MONITOR, RESTRUCTURE, or PROVISION."
        )
    )
    notes: str = Field(
        description="One-sentence narrative for the credit file."
    )


class PortfolioReport(BaseModel):
    portfolio_id: str = Field(
        description="Portfolio identifier, copied from the input."
    )
    review_date: str = Field(
        description="Date of this review in YYYY-MM-DD format."
    )
    health_rating: str = Field(
        description=(
            "Overall portfolio health: STRONG, SATISFACTORY, FAIR, or WEAK."
        )
    )
    total_loans: int = Field(
        description="Total number of loans reviewed."
    )
    performing_count: int = Field(
        description="Number of loans rated PERFORMING."
    )
    watch_or_worse_count: int = Field(
        description="Number of loans rated WATCH, SUBSTANDARD, DOUBTFUL, or LOSS."
    )
    concentration_risks: list[str] = Field(
        description="Portfolio-level concentration risks identified (sector, geography, etc.)."
    )
    top_concerns: list[str] = Field(
        description="2-3 most urgent issues requiring management attention."
    )
    recommended_provisions_usd: float = Field(
        description=(
            "Estimated additional provisions recommended in USD. 0.0 if none needed."
        )
    )
    executive_summary: str = Field(
        description="3-4 sentence executive summary suitable for the credit committee."
    )


# ── Run state type (informational) ────────────────────────────────────────────

class PortfolioState(TypedDict, total=False):
    portfolio_id: str
    review_date:  str
    loans:        list   # list[dict] — individual loan records
    # result fields populated by MapReduceAgent
    map_results:  list   # list[LoanAssessment]
    result:       dict   # PortfolioReport


# ── Client ────────────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "portfolio.report":
        r    = msg.payload["result"]
        maps = msg.payload["map_results"]
        pid  = msg.payload["input"].get("portfolio_id", "?")

        print(f"\n{'─' * 55}")
        print(f"  PORTFOLIO REVIEW  —  {pid}")
        print(f"  Date:             {r.get('review_date', '—')}")
        print(f"  Health:           {r.get('health_rating')}")
        print(f"  Loans reviewed:   {r.get('total_loans')}")
        print(f"  Performing:       {r.get('performing_count')}")
        print(f"  Watch or worse:   {r.get('watch_or_worse_count')}")
        if r.get("recommended_provisions_usd", 0) > 0:
            print(f"  Est. provisions:  ${r['recommended_provisions_usd']:,.0f}")
        if r.get("concentration_risks"):
            print("  Concentration risks:")
            for c in r["concentration_risks"]:
                print(f"    · {c}")
        if r.get("top_concerns"):
            print("  Top concerns:")
            for c in r["top_concerns"]:
                print(f"    · {c}")
        print(f"\n  Summary: {r.get('executive_summary', '')}")

        print(f"\n  {'─' * 50}")
        print("  Individual loan assessments:")
        for a in maps:
            change_sym = {"IMPROVED": "↑", "STABLE": "→", "DETERIORATED": "↓"}.get(
                a.get("risk_change", ""), "?"
            )
            print(
                f"    {a.get('loan_id', '?'):20s}  "
                f"{a.get('risk_rating', '?'):14s}  "
                f"{change_sym}  "
                f"DTI {a.get('dti_percent', 0):.1f}%  "
                f"→ {a.get('recommended_action', '?')}"
            )
        print(f"{'─' * 55}\n")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

async def main():
    bus             = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/map_reduce.yaml")
    runtime         = AgentRuntime(bus, prompts=prompt_provider)

    # ── Agents ────────────────────────────────────────────────────────────────
    loan_analyst = SimpleAgent(
        "loan-analyst",
        prompt_name="bank.map_reduce.loan_analyst.system",
        output_model=LoanAssessment,
    )
    portfolio_manager = SimpleAgent(
        "portfolio-manager",
        prompt_name="bank.map_reduce.portfolio_manager.system",
        output_model=PortfolioReport,
    )

    # ── MapReduce processor ───────────────────────────────────────────────────
    processor = MapReduceAgent(
        "portfolio-processor",
        mapper      = loan_analyst,
        reducer     = portfolio_manager,
        splitter    = lambda payload: payload["loans"],
        submit_type = "portfolio.submitted",
        result_type = "portfolio.report",
        verbose     = True,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(processor)   # also registers loan_analyst and portfolio_manager
    runtime.register(client)

    await runtime.start(run_id="run-portfolio-review")

    await client._ref.send(
        to="portfolio-processor",
        type="portfolio.submitted",
        payload={
            "portfolio_id": "LATAM-SME-2026-Q1",
            "review_date":  "2026-03-01",
            "loans": [
                {
                    "loan_id":                  "SME-001",
                    "borrower":                 "Distribuidora Andina S.A.",
                    "sector":                   "Wholesale trade",
                    "outstanding_usd":          480_000,
                    "original_amount_usd":      600_000,
                    "monthly_payment_usd":      11_200,
                    "gross_monthly_revenue_usd": 95_000,
                    "monthly_debt_obligations_usd": 38_000,
                    "days_past_due":            0,
                    "collateral":               "Warehouse inventory (valued $520k)",
                    "last_rating":              "PERFORMING",
                    "notes":                    "Revenue down 12% YoY due to supply chain disruptions.",
                },
                {
                    "loan_id":                  "SME-002",
                    "borrower":                 "Constructora Pacífico Ltda.",
                    "sector":                   "Construction",
                    "outstanding_usd":          1_200_000,
                    "original_amount_usd":      1_500_000,
                    "monthly_payment_usd":      28_000,
                    "gross_monthly_revenue_usd": 210_000,
                    "monthly_debt_obligations_usd": 95_000,
                    "days_past_due":            45,
                    "collateral":               "Residential project plots (valued $900k)",
                    "last_rating":              "WATCH",
                    "notes":                    "Two delayed payments. Project completion delayed 6 months. Collateral value under review.",
                },
                {
                    "loan_id":                  "SME-003",
                    "borrower":                 "Café Origen Colombia SAS",
                    "sector":                   "Agriculture / Export",
                    "outstanding_usd":          320_000,
                    "original_amount_usd":      400_000,
                    "monthly_payment_usd":      7_500,
                    "gross_monthly_revenue_usd": 88_000,
                    "monthly_debt_obligations_usd": 18_000,
                    "days_past_due":            0,
                    "collateral":               "Coffee export contracts (forward sales)",
                    "last_rating":              "PERFORMING",
                    "notes":                    "Export volumes up 20%. New buyer in Germany signed. Strong repayment history.",
                },
                {
                    "loan_id":                  "SME-004",
                    "borrower":                 "Logística Express del Norte",
                    "sector":                   "Transportation",
                    "outstanding_usd":          680_000,
                    "original_amount_usd":      750_000,
                    "monthly_payment_usd":      15_800,
                    "gross_monthly_revenue_usd": 72_000,
                    "monthly_debt_obligations_usd": 51_000,
                    "days_past_due":            0,
                    "collateral":               "Fleet of 12 trucks (valued $420k, depreciating)",
                    "last_rating":              "WATCH",
                    "notes":                    "High DTI. Fuel cost increases squeezing margins. Owner considering fleet reduction.",
                },
                {
                    "loan_id":                  "SME-005",
                    "borrower":                 "Textiles Medellín SA",
                    "sector":                   "Manufacturing",
                    "outstanding_usd":          290_000,
                    "original_amount_usd":      350_000,
                    "monthly_payment_usd":      6_800,
                    "gross_monthly_revenue_usd": 115_000,
                    "monthly_debt_obligations_usd": 22_000,
                    "days_past_due":            0,
                    "collateral":               "Industrial machinery + real estate ($480k combined)",
                    "last_rating":              "PERFORMING",
                    "notes":                    "Strong domestic orders. Recently signed government supply contract.",
                },
            ],
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
