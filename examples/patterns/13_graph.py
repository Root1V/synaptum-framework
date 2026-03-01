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

The `controller` agent (passive) drives transitions: it stores the growing
application context in agent.state, routes stage.input messages, and
delivers the final letter to the client.
"""

import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
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


# ── Transition table ──────────────────────────────────────────────────────────

def _next_stage(stage: str, result: dict[str, Any]) -> str | None:
    """Pure function: given the completed stage and its result, return the next stage."""
    if stage == "stage-intake":
        return "stage-credit-check"   # always proceed; intake is pre-processing only

    if stage == "stage-credit-check":
        band = result.get("band", "").upper()
        if band in ("EXCELLENT", "GOOD"):
            return "stage-decision"
        if band == "FAIR":
            return "stage-underwriting"
        return "stage-decision"          # POOR → decline path

    if stage == "stage-underwriting":
        return "stage-decision"

    return None                          # stage-decision is terminal


# ── Stage handler (shared) ────────────────────────────────────────────────────

async def stage_handler(agent: SimpleAgent, msg: Message, ctx):
    """Run the stage LLM and report the structured result back to the controller."""
    if msg.type != "stage.input":
        return

    context_text = msg.payload.get("context", "")
    result = await agent.think(
        f"Application context:\n{context_text}\n\nProcess this stage of the application."
    )

    await agent._ref.send(
        to="controller",
        type="stage.output",
        payload={"stage": agent.name, "result": result.model_dump()},
        metadata=msg.metadata,
    )


# ── Controller (passive — no LLM) ─────────────────────────────────────────────

async def controller_handler(agent: SimpleAgent, msg: Message, ctx):
    """
    Drives the graph: routes stage.input, collects stage.output,
    applies the transition table, and delivers the final decision.
    """
    if msg.type == "mortgage.submitted":
        loan_id = msg.payload["loan_id"]
        application = msg.payload["application"]

        agent.state[loan_id] = {
            "caller":      msg.sender,
            "msg_id":      msg.id,
            "application": application,
            "history":     [],           # list of {stage, result} dicts
        }

        app_text = "\n".join(f"  {k}: {v}" for k, v in application.items())
        print(f"\n── MORTGAGE {loan_id}: entering graph at stage-intake ──")
        await _dispatch(agent, loan_id, "stage-intake", app_text)
        return

    if msg.type == "stage.output":
        loan_id = msg.metadata.get("loan_id")
        if not loan_id or loan_id not in agent.state:
            return

        state = agent.state[loan_id]
        stage = msg.payload["stage"]
        result = msg.payload["result"]

        # Accumulate context on the blackboard-like state
        state["history"].append({"stage": stage, "result": result})
        print(f"  {stage}: {_short(result)}")

        next_stage = _next_stage(stage, result)

        if next_stage is None:
            # Terminal — deliver decision to caller
            final = next(
                (e["result"] for e in reversed(state["history"])
                 if e["stage"] == "stage-decision"),
                state["history"][-1]["result"],
            )
            caller = state["caller"]
            caller_msg_id = state["msg_id"]
            del agent.state[loan_id]

            await agent._ref.send(
                to=caller,
                type="mortgage.decision",
                payload={"loan_id": loan_id, "decision": final},
                metadata={"in_reply_to": caller_msg_id},
            )
            return

        # Build accumulated context for the next stage
        context_lines = [
            "Original application:",
            *[f"  {k}: {v}" for k, v in state["application"].items()],
        ]
        for entry in state["history"]:
            context_lines.append(f"\n[{entry['stage']} output]")
            context_lines += [f"  {k}: {v}" for k, v in entry["result"].items()]

        print(f"  → routing to {next_stage}")
        await _dispatch(agent, loan_id, next_stage, "\n".join(context_lines))


async def _dispatch(agent: SimpleAgent, loan_id: str, stage: str, context: str):
    """Route to the next graph stage."""
    await agent._ref.send(
        to=stage,
        type="stage.input",
        payload={"context": context},
        reply_to="controller",
        metadata={"loan_id": loan_id},
    )


def _short(result: dict) -> str:
    """One-line summary of a stage result for the progress log."""
    for key in ("band", "recommendation", "outcome", "status"):
        if key in result:
            return f"{key}={result[key]}"
    if "debt_to_income_percent" in result:
        return f"DTI={result['debt_to_income_percent']}%  LTV={result.get('loan_to_value_percent')}%"
    return str(list(result.items())[:1])


# ── Client ────────────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "mortgage.decision":
        d = msg.payload["decision"]
        print(f"\n── MORTGAGE DECISION  (loan: {msg.payload['loan_id']}) ──")
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

    controller = SimpleAgent("controller", handler=controller_handler)

    stages = [
        SimpleAgent(
            "stage-intake",
            prompt_name="bank.graph.intake.system",
            output_model=IntakeResult,
            handler=stage_handler,
        ),
        SimpleAgent(
            "stage-credit-check",
            prompt_name="bank.graph.credit_check.system",
            output_model=CreditResult,
            handler=stage_handler,
        ),
        SimpleAgent(
            "stage-underwriting",
            prompt_name="bank.graph.underwriting.system",
            output_model=UnderwritingResult,
            handler=stage_handler,
        ),
        SimpleAgent(
            "stage-decision",
            prompt_name="bank.graph.decision.system",
            output_model=LoanDecision,
            handler=stage_handler,
        ),
    ]

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(controller)
    for stage in stages:
        runtime.register(stage)
    runtime.register(client)

    await runtime.start(run_id="run-mortgage-graph")

    # Borderline applicant: FAIR credit band expected → underwriting path
    await client._ref.send(
        to="controller",
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
