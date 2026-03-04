"""
Pattern: Reflection Loop — Credit Assessment Report (Banking)
=============================================================
Demonstrates the ``ReflectionAgent`` pattern: a credit analyst generates a
structured loan assessment report, a senior credit officer critiques it against
explicit quality criteria, and the analyst iterates until the report meets
the required standard — or the iteration budget runs out.

Reflection Loop vs other multi-agent patterns:

  Supervisor/Worker → the supervisor dispatches tasks and collects results;
                       there is no quality gate or iteration
  Plan-and-Execute  → a planner revises the sequence of steps mid-run based on
                       emerging results; it does not re-evaluate quality
  Swarm             → agents hand off control to peers based on their findings;
                       no single output is iteratively refined
  Reflection        → the SAME output is generated, scored, and REVISED
                       repeatedly; quality improves through explicit feedback
                       until a threshold is met

Use-case — pre-committee credit assessment:
  A loan application arrives for Industrias Llanero S.A.S. — a mid-sized
  agricultural-equipment manufacturer requesting a $3.2M expansion loan.
  Before the case goes to the Credit Committee, the bank's policy requires
  the credit memorandum to score at least 7.5/10 across completeness,
  analytical rigour, risk coverage, and regulatory compliance.

  · ``credit-analyst``  — writes the report; revises based on feedback
  · ``credit-reviewer`` — scores the report; returns structured critique

Execution flow:

  submit("credit.assessment.requested")
         │
         ▼
  credit-analyst.think(task)  →  CreditReport (draft)
         │
         ▼
  credit-reviewer.think(report)  →  Critique (score, weaknesses, instructions)
         │
         ├─ score >= 7.5?  →  deliver report               ← passes in iteration N
         │
         └─ score < 7.5?   →  credit-analyst.think(task + critique)  → revised report
                                      │
                                      ▼  (repeat up to max_iterations)
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents import LLMAgent
from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.reflection import Critique, ReflectionAgent
from synaptum.prompts import FilePromptProvider


# ── Output schema  ─────────────────────────────────────────────────────────────

class CreditReport(BaseModel):
    executive_summary:   str = Field(description="3-4 sentence summary with loan amount, purpose, and recommendation.")
    applicant_profile:   str = Field(description="Business description, years in operation, industry, ownership.")
    financial_analysis:  str = Field(description="Revenue trend, EBITDA margin, D/E ratio, ICR with workings, net debt/EBITDA, liquidity.")
    risk_assessment:     str = Field(description="Credit risk rating (LOW/MEDIUM/HIGH/CRITICAL) + top 3-5 risks each with mitigant.")
    collateral_analysis: str = Field(description="Description, estimated value, LTV ratio, enforceability.")
    recommendation:      str = Field(description="APPROVE | APPROVE_WITH_CONDITIONS | DECLINE")
    proposed_terms:      str = Field(description="Loan amount, tenor, rate, covenants if approved.")
    regulatory_flags:    str = Field(description="IFRS 9 stage, provisioning estimate, Basel III / local regulatory notes.")


# ── Client handler ─────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type != "credit.assessment.final":
        return

    p        = msg.payload
    report   = p["result"]
    score    = p["score"]
    passed   = p["passed"]
    iters    = p["iterations_used"]
    history  = p["history"]
    inp      = p["input"]

    W = 64
    status = "✅ PASSED" if passed else "⚠️  BEST AVAILABLE"
    print(f"\n{'═' * W}")
    print(f"  CREDIT ASSESSMENT REPORT — FINAL")
    print(f"  Applicant:       {inp.get('applicant_name','—')}")
    print(f"  Loan requested:  ${inp.get('loan_amount_usd', 0):,.0f}")
    print(f"  Iterations:      {iters}")
    print(f"  Final score:     {score:.1f}/10  {status}")
    print(f"{'─' * W}")

    # Score history
    if len(history) > 1:
        print("\n  Score progression:")
        for h in history:
            bar_len  = int(h["critique"]["score"] * 4)
            bar      = "█" * bar_len + "░" * (40 - bar_len)
            print(f"    Iter {h['iteration']}: {h['critique']['score']:4.1f}  {bar}")

    print(f"\n  RECOMMENDATION:  {report.get('recommendation','—')}")
    print(f"\n  Executive Summary:")
    for line in report.get("executive_summary", "").splitlines():
        print(f"    {line}")

    print(f"\n  Proposed Terms:")
    for line in report.get("proposed_terms", "").splitlines():
        print(f"    {line}")

    print(f"\n  Regulatory Flags:")
    for line in report.get("regulatory_flags", "").splitlines():
        print(f"    {line}")

    # Final critique breakdown
    final_critique = history[-1]["critique"]
    dim = final_critique.get("dimension_scores", {})
    print(f"\n{'─' * W}")
    print(f"  Final quality score breakdown:")
    for dim_name, dim_score in dim.items():
        bar_len = int(dim_score * 3)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {dim_name:25s}  {dim_score:4.1f}  {bar}")

    if final_critique.get("strengths"):
        print(f"\n  Strengths:")
        for s in final_critique["strengths"]:
            print(f"    + {s}")

    if not passed and final_critique.get("weaknesses"):
        print(f"\n  Remaining issues:")
        for w in final_critique["weaknesses"]:
            print(f"    - {w}")

    print(f"{'═' * W}\n")


# ── Bootstrap ──────────────────────────────────────────────────────────────────

async def main():
    bus             = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/reflection.yaml")
    runtime         = AgentRuntime(bus, prompts=prompt_provider)

    # ── Agents ────────────────────────────────────────────────────────────────
    credit_analyst = LLMAgent(
        "credit-analyst",
        prompt_name  = "bank.reflection.credit_analyst.system",
        output_model = CreditReport,
    )
    credit_reviewer = LLMAgent(
        "credit-reviewer",
        prompt_name  = "bank.reflection.credit_reviewer.system",
        output_model = Critique,
    )

    # ── ReflectionAgent ───────────────────────────────────────────────────────
    reflection_loop = ReflectionAgent(
        "credit-assessment-loop",
        generator      = credit_analyst,
        critic         = credit_reviewer,
        pass_threshold = 7.5,
        max_iterations = 3,
        submit_type    = "credit.assessment.requested",
        result_type    = "credit.assessment.final",
        verbose        = True,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(credit_analyst)
    runtime.register(credit_reviewer)
    runtime.register(reflection_loop)
    runtime.register(client)

    await runtime.start(run_id="run-credit-assessment-2026")

    # ── Case: Industrias Llanero S.A.S. ──────────────────────────────────────
    # Mid-sized agricultural-equipment manufacturer seeking $3.2M to expand
    # production capacity. Revenue growing but leverage is elevated.
    await client._ref.send(
        to      = "credit-assessment-loop",
        type    = "credit.assessment.requested",
        payload = {
            "applicant_name":    "Industrias Llanero S.A.S.",
            "loan_amount_usd":   3_200_000,
            "loan_purpose":      "Expansion of manufacturing plant — new production line for precision agroindustrial components",
            "loan_tenor_years":  7,
            "country":           "Colombia",
            "industry":          "Agricultural equipment manufacturing",
            "financials": {
                "year":                         2025,
                "annual_revenue_usd":           8_400_000,
                "revenue_prior_year_usd":       7_200_000,
                "ebitda_usd":                   1_680_000,
                "ebit_usd":                     1_120_000,
                "net_income_usd":               510_000,
                "total_assets_usd":             14_500_000,
                "total_debt_usd":               5_800_000,
                "total_equity_usd":             5_200_000,
                "cash_usd":                     620_000,
                "annual_interest_expense_usd":  490_000,
                "capex_usd":                    380_000,
                "accounts_receivable_days":     62,
                "inventory_days":               90,
            },
            "collateral": {
                "type":          "Industrial real estate + machinery",
                "description":   "Manufacturing plant in Villavicencio (3,200 m²) + 4 CNC lathes",
                "appraised_value_usd": 4_800_000,
                "first_charge":  True,
            },
            "company_background": (
                "Founded in 2009 by the Vargas family. 85 employees. "
                "Major clients: 3 agro-industrial conglomerates (60% of revenue). "
                "Export to Ecuador and Peru started in 2024 (12% of revenue). "
                "No prior defaults. CEO has 20 years of industry experience."
            ),
        },
        reply_to = "client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
