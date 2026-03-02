"""
Pattern: Parallel Graph — Personal Loan Risk Assessment (Banking)
=================================================================
Demonstrates the ``ParallelNode`` / ``parallel()`` API: a fork/join stage
where multiple independent agents run concurrently with ``asyncio.gather()``,
then converge back into a single sequential flow.

Graph topology:

  intake ──────▶  ┌─ credit-check ─────┐
                  ├─ employment-verify ─┤ ──▶  decision
                  └─ fraud-scan ────────┘

  • intake            — pre-processes raw application; calculates DTI/LTV ratios
  • credit-check      — assesses creditworthiness (runs in parallel)
  • employment-verify — validates employment stability (runs in parallel)
  • fraud-scan        — flags potential fraud indicators (runs in parallel)
  • decision          — aggregates all parallel findings and issues final ruling

The three parallel agents all receive the same accumulated state (input
payload + intake result) and write their outputs under their own state keys.
The decision agent then has the full state at its disposal.

Key API shown:

    from synaptum.patterns.graph_builder import parallel

    risk_checks = parallel(
        "risk-checks",
        credit_check,
        employment_verify,
        fraud_scan,
    )

    GraphBuilder(...)
        .add_node(intake)
        .add_node(risk_checks)   # ParallelNode accepted just like a SimpleAgent
        .add_node(decision)
        .set_entry(intake)
        .add_edge(intake,       risk_checks)
        .add_edge(risk_checks,  decision)
        .add_edge(decision,     END)
        .build()
"""

import asyncio
from typing import Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.graph_builder import END, GraphBuilder, parallel
from synaptum.prompts import FilePromptProvider


# ── Output schemas ────────────────────────────────────────────────────────────

class IntakeResult(BaseModel):
    debt_to_income_percent: float = Field(
        description=(
            "DTI ratio: (monthly_debt_usd / gross_monthly_income_usd) × 100, "
            "rounded to 1 decimal."
        )
    )
    loan_to_value_percent: float = Field(
        description=(
            "LTV ratio: (requested_amount_usd / collateral_value_usd) × 100. "
            "Use 100.0 when no collateral is supplied."
        )
    )
    monthly_payment_estimate: float = Field(
        description="Estimated monthly payment in USD (4-year term, 12 % annual rate)."
    )
    profile_summary: str = Field(
        description="2-3 sentences summarising the applicant's financial profile."
    )


class CreditAssessment(BaseModel):
    score_band: str = Field(
        description="Credit band: EXCELLENT, GOOD, FAIR, or POOR."
    )
    risk_level: str = Field(
        description="Risk level: LOW, MEDIUM, or HIGH."
    )
    findings: list[str] = Field(
        description="2-4 key observations supporting the score band."
    )


class EmploymentAssessment(BaseModel):
    stability: str = Field(
        description="Employment stability: STABLE, MODERATE, or UNSTABLE."
    )
    income_verified: bool = Field(
        description="True if stated income appears credible with no red flags."
    )
    findings: list[str] = Field(
        description="2-3 observations about employment and income."
    )


class FraudAssessment(BaseModel):
    risk_level: str = Field(
        description="Fraud risk level: NONE, LOW, MEDIUM, or HIGH."
    )
    flags: list[str] = Field(
        description="Specific fraud indicators found. Empty list if none."
    )
    recommendation: str = Field(
        description="Action: PROCEED, REVIEW, or REJECT."
    )


class LoanDecision(BaseModel):
    outcome: str = Field(
        description="Lending outcome: APPROVED, APPROVED_MODIFIED, or DECLINED."
    )
    amount_approved: float = Field(
        description="Approved loan amount in USD. 0.0 if declined."
    )
    interest_rate: float = Field(
        description="Indicative annual interest rate as a percentage. 0.0 if declined."
    )
    conditions: list[str] = Field(
        description="Conditions or requirements attached to the decision."
    )
    reason: str = Field(
        description="Clear explanation suitable for the applicant."
    )


# ── Run state ─────────────────────────────────────────────────────────────────

class LoanState(TypedDict, total=False):
    """Typed run-state that flows through every stage."""
    # Input fields
    loan_id:     str
    application: dict
    # Stage outputs (agent name with hyphens replaced by underscores)
    intake:             dict   # IntakeResult
    risk_checks:        dict   # merged ParallelNode aggregate
    credit_check:       dict   # CreditAssessment    (written directly by child)
    employment_verify:  dict   # EmploymentAssessment (written directly by child)
    fraud_scan:         dict   # FraudAssessment      (written directly by child)
    decision:           dict   # LoanDecision


# ── Client ────────────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "loan.decision":
        d       = msg.payload["result"]
        s       = msg.payload["state"]
        loan_id = msg.payload["input"].get("loan_id", "?")

        print(f"\n── LOAN DECISION  (id: {loan_id}) " + "─" * 30)
        print(f"  Outcome:    {d.get('outcome', 'N/A')}")
        if d.get("amount_approved", 0) > 0:
            print(f"  Amount:     ${d['amount_approved']:,.0f}")
            print(f"  Rate:       {d['interest_rate']} % p.a.")
        if d.get("conditions"):
            print("  Conditions:")
            for c in d["conditions"]:
                print(f"    · {c}")
        print(f"  Reason:     {d.get('reason', '')}")

        # Show the parallel checks summary
        print("\n  ── Parallel checks ──────────────────────")
        if cc := s.get("credit_check"):
            print(f"  Credit:     {cc.get('score_band')}  ({cc.get('risk_level')})")
        if ev := s.get("employment_verify"):
            stable = ev.get("stability")
            iv     = "✓" if ev.get("income_verified") else "✗"
            print(f"  Employment: {stable}  income_verified={iv}")
        if fs := s.get("fraud_scan"):
            flags = fs.get("flags") or []
            print(f"  Fraud:      {fs.get('risk_level')}  → {fs.get('recommendation')}"
                  + (f"  flags={flags}" if flags else ""))
        print("─" * 50 + "\n")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

async def main():
    bus             = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/parallel.yaml")
    runtime         = AgentRuntime(bus, prompts=prompt_provider)

    # ── Sequential nodes ──────────────────────────────────────────────────────
    intake = SimpleAgent(
        "intake",
        prompt_name="bank.parallel.intake.system",
        output_model=IntakeResult,
    )
    decision = SimpleAgent(
        "decision",
        prompt_name="bank.parallel.decision.system",
        output_model=LoanDecision,
    )

    # ── Parallel nodes (fork/join) ────────────────────────────────────────────
    credit_check = SimpleAgent(
        "credit-check",
        prompt_name="bank.parallel.credit_check.system",
        output_model=CreditAssessment,
    )
    employment_verify = SimpleAgent(
        "employment-verify",
        prompt_name="bank.parallel.employment_verify.system",
        output_model=EmploymentAssessment,
    )
    fraud_scan = SimpleAgent(
        "fraud-scan",
        prompt_name="bank.parallel.fraud_scan.system",
        output_model=FraudAssessment,
    )

    risk_checks = parallel("risk-checks", credit_check, employment_verify, fraud_scan)

    # ── Graph ─────────────────────────────────────────────────────────────────
    processor = (
        GraphBuilder("loan-processor", state=LoanState)
        .submit("loan.submitted")
        .result("loan.decision")

        .add_node(intake)
        .add_node(risk_checks)   # ParallelNode — adds all three children
        .add_node(decision)

        .set_entry(intake)
        .add_edge(intake,      risk_checks)
        .add_edge(risk_checks, decision)
        .add_edge(decision,    END)
        .verbose()
        .build()
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(processor)   # cascades: intake, credit-check, employment-verify, fraud-scan, decision
    runtime.register(client)

    await runtime.start(run_id="run-loan-parallel")

    # Applicant with mixed signals: stable income but some credit concerns
    await client._ref.send(
        to="loan-processor",
        type="loan.submitted",
        payload={
            "loan_id": "PL-2026-00891",
            "application": {
                "applicant":                  "Carlos Mendoza Vega",
                "national_id":                "CC-72914830",
                "date_of_birth":              "1985-07-22",
                "employment_type":            "Salaried",
                "employer":                   "TechParts Colombia S.A.S.",
                "job_title":                  "Senior Engineer",
                "years_employed":             6.0,
                "gross_monthly_income_usd":   4200,
                "monthly_debt_usd":           1100,
                "requested_amount_usd":       18000,
                "collateral_value_usd":       0,
                "loan_purpose":               "Home renovation",
                "credit_score_self_reported": 680,
                "payment_history_notes": (
                    "Two late payments 18 months ago during a medical emergency. "
                    "All current accounts up to date. One active credit card at 65 % utilisation."
                ),
                "additional_notes": (
                    "Applicant has been with the same employer for 6 years and recently "
                    "received a 10 % merit increase. No outstanding defaults."
                ),
            },
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
