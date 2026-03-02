"""
Pattern: Graph — Mortgage Application State Machine (Banking)
=============================================================
A directed graph where each node is a specialist agent and edges are
conditional transitions driven by structured Pydantic outputs.  Unlike a
pipeline (fixed linear steps) or a router (one-shot dispatch), the graph
accumulates context across stages and branches based on the actual model
output at each step.

Graph topology:

  intake (pre-processing) ──always──▶ credit-check ──EXCELLENT / GOOD ──▶ decision
                                           │
                                           └──── FAIR ──▶ underwriting ──▶ decision
                                           │
                                           └──── POOR ──▶ decision

State machine rules:
  intake       → always      → credit-check
  credit-check → EXCELLENT/GOOD → decision
               → FAIR          → underwriting
               → POOR          → decision
  underwriting → any          → decision
  decision     → (terminal)

The `GraphBuilder` assembles the graph: nodes are named ``SimpleAgent``
instances, edges declare the topology, and ``build()`` returns a
``GraphAgent`` that drives the run.  A typed ``MortgageState`` dict flows
through every stage so each node has full context from prior steps.
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
from synaptum.patterns.graph_builder import END, GraphBuilder
from synaptum.prompts import FilePromptProvider


# ── Output schemas ────────────────────────────────────────────────────────────

class IntakeResult(BaseModel):
    debt_to_income_percent: float = Field(
        description="DTI ratio: (monthly_debt_obligations / gross_monthly_income) * 100, rounded to 1 decimal."
    )
    loan_to_value_percent: float = Field(
        description="LTV ratio: (requested_amount / (requested_amount + down_payment)) * 100, rounded to 1 decimal."
    )
    profile_summary: str = Field(
        description="2-3 sentence summary of the applicant's financial profile and notable risk/compensating factors."
    )


class CreditResult(BaseModel):
    dti_percent: float = Field(
        description="Calculated debt-to-income ratio as a percentage (e.g. 34.5)."
    )
    band: str = Field(
        description="Credit band: EXCELLENT, GOOD, FAIR, or POOR."
    )
    rationale: str = Field(description="Explanation supporting the assigned band.")


class UnderwritingResult(BaseModel):
    recommendation: str = Field(
        description="APPROVE, COUNTER_OFFER, or DECLINE."
    )
    compensating_factors: list[str] = Field(
        description="Factors that reduce risk."
    )
    risk_factors: list[str] = Field(
        description="Factors that increase risk."
    )
    notes: str = Field(description="Underwriter narrative.")


class LoanDecision(BaseModel):
    outcome: str = Field(
        description="APPROVED, APPROVED_MODIFIED, or DECLINED."
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
    reason: str = Field(description="Clear explanation of the decision for the applicant.")


# ── Run state ─────────────────────────────────────────────────────────────────

class MortgageState(TypedDict, total=False):
    """Typed state dict that flows through every stage of the graph."""
    # Input payload fields
    loan_id:     str
    application: dict
    # Stage outputs (keyed by normalised agent name: hyphens → underscores)
    intake:       dict   # IntakeResult
    credit_check: dict   # CreditResult
    underwriting: dict   # UnderwritingResult
    decision:     dict   # LoanDecision

# ── Client ────────────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "mortgage.decision":
        d = msg.payload["result"]
        loan_id = msg.payload["input"].get("loan_id", "?")
        print(f"\n── MORTGAGE DECISION  (loan: {loan_id}) ──")
        print(f"  Outcome:       {d.get('outcome', 'N/A')}")
        if d.get("amount_approved", 0) > 0:
            print(f"  Amount:        ${d['amount_approved']:,.0f}")
            print(f"  Interest rate: {d['interest_rate']}%")
        if d.get("conditions"):
            print("  Conditions:")
            for c in d["conditions"]:
                print(f"    · {c}")
        print(f"  Reason: {d.get('reason', d.get('summary', ''))}")
        print("─" * 60 + "\n")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/graph.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    intake = SimpleAgent(
        "intake",
        prompt_name="bank.graph.intake.system",
        output_model=IntakeResult,
    )
    credit_check = SimpleAgent(
        "credit-check",
        prompt_name="bank.graph.credit_check.system",
        output_model=CreditResult,
    )
    underwriting = SimpleAgent(
        "underwriting",
        prompt_name="bank.graph.underwriting.system",
        output_model=UnderwritingResult,
    )
    decision = SimpleAgent(
        "decision",
        prompt_name="bank.graph.decision.system",
        output_model=LoanDecision,
    )

    # ── Graph ───────────────────────────────────────────────────────────────────
    processor = (
        GraphBuilder("mortgage-processor", state=MortgageState)
        .submit("mortgage.submitted")
        .result("mortgage.decision")
        
        .add_node(intake)
        .add_node(credit_check)
        .add_node(underwriting)
        .add_node(decision)
        
        .set_entry(intake)
        .add_edge(intake, credit_check)
        .add_conditional_edges(
            credit_check,
            lambda state: state["credit_check"]["band"].upper(),
            {
                "EXCELLENT": decision,
                "GOOD":      decision,
                "FAIR":      underwriting,
                "POOR":      decision,
            },
        )
        .add_edge(underwriting, decision)
        .add_edge(decision, END)
        .verbose()
        .build()
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(processor)  # also registers intake, credit_check, underwriting, decision
    runtime.register(client)

    await runtime.start(run_id="run-mortgage-graph")

    # Borderline applicant: FAIR credit band expected → underwriting path
    await client._ref.send(
        to="mortgage-processor",
        type="mortgage.submitted",
        payload={
            "loan_id": "MTG-2026-00447",
            "application": {
                "applicant":           "Isabel Fuentes Restrepo",
                "national_id":         "CC-49271834",
                "date_of_birth":       "1988-03-14",
                "marital_status":      "Married",
                "employment_type":     "Salaried",
                "employer":            "Andina Logistics S.A.",
                "years_employed":      1.5,
                "gross_monthly_income_usd": 5800,
                "monthly_debt_obligations_usd": 2350,
                "requested_amount_usd": 195000,
                "property_address":    "Calle 72 #15-30, Bogotá",
                "down_payment_usd":    35000,
                "down_payment_source": "Personal savings",
                "credit_consent_signed": True,
                "notes": (
                    "Applicant recently changed jobs (8 months ago) but received a "
                    "15% salary increase. Previous employer: 4 years. No missed payments "
                    "in the last 24 months. One late payment 3 years ago."
                ),
            },
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
