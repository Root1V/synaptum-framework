"""
Pattern: Blackboard — AML Suspicious Activity Report (Banking)
==============================================================
Multiple specialist agents independently analyse different dimensions of a
suspicious case and write their structured findings to a shared blackboard
(InMemoryMemoryStore).  Once every analyst has contributed, the supervisor
reads the complete picture and issues the final SAR determination.

The blackboard is the single source of truth for the investigation:
  - Analysts READ the initial case data written there by the coordinator.
  - Analysts WRITE their AnalystFinding back to the blackboard.
  - The supervisor READS all entries and synthesises the final SARDecision.

Flow:
  client ──[sar.opened]──▶ coordinator
                            coordinator writes case → blackboard
                            coordinator ──[sar.analyze]──▶ analyst-* (fan-out)
                            each analyst READS blackboard, WRITES finding,
                            analyst ──[sar.finding]──▶ supervisor
                            supervisor (waits for all)
                            supervisor READS blackboard → SARDecision
  client ◀──[sar.decision]── supervisor
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.memory.in_memory import InMemoryMemoryStore
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import FilePromptProvider


# ── Output schemas ────────────────────────────────────────────────────────────

class AnalystFinding(BaseModel):
    risk_score: float = Field(
        description=(
            "Risk score as a decimal between 0.0 (no concern) and 1.0 (critical). "
            "Use 0.0-1.0 scale only — do NOT use a 0-10 scale."
        ),
    )
    flags: list[str] = Field(
        description="Specific red flags identified in this dimension."
    )
    summary: str = Field(
        description="Concise analyst summary of findings."
    )

    @field_validator("risk_score", mode="before")
    @classmethod
    def normalise_risk_score(cls, v: float) -> float:
        """Accept 0-10 scale from models that ignore the 0-1 instruction."""
        v = float(v)
        if v > 1.0:
            v = v / 10.0
        return max(0.0, min(1.0, v))


class SARDecision(BaseModel):
    verdict: str = Field(
        description=(
            "Final determination: ESCALATE_TO_REGULATOR, INTERNAL_HOLD, "
            "MONITOR, or DISMISS."
        )
    )
    risk_level: str = Field(
        description="Overall risk level: CRITICAL, HIGH, MEDIUM, or LOW."
    )
    summary: str = Field(
        description="Combined findings summary supporting the verdict."
    )
    recommended_actions: list[str] = Field(
        description="Ordered list of recommended follow-up actions."
    )


# ── Shared blackboard (created once, captured by all handlers via closure) ────

blackboard = InMemoryMemoryStore()


# ── Coordinator (passive — no LLM) ────────────────────────────────────────────

async def coordinator_handler(agent: SimpleAgent, msg: Message, ctx):
    """Write initial case to blackboard, then fan-out to all analysts."""
    if msg.type != "sar.opened":
        return

    case_id = msg.payload["case_id"]
    case = msg.payload["case"]

    # Seed the blackboard with the raw case data
    await blackboard.append(f"sar:{case_id}", {"source": "coordinator", "case": case})

    analyst_names = ctx.agent_names(prefix="analyst-")
    agent.state[case_id] = {
        "caller":   msg.sender,
        "msg_id":   msg.id,
        "expected": len(analyst_names),
        "received": 0,
    }

    print(f"\n── SAR {case_id}: dispatching to {len(analyst_names)} analysts ──")

    for analyst in analyst_names:
        await agent._ref.send(
            to=analyst,
            type="sar.analyze",
            payload={"case_id": case_id},
            reply_to="supervisor",
            metadata={"case_id": case_id, "caller": msg.sender, "caller_msg_id": msg.id},
        )


# ── Analyst (shared handler, each agent has its own LLM + prompt + output_model) ─

async def analyst_handler(agent: SimpleAgent, msg: Message, ctx):
    """Read case from blackboard, produce structured finding, write it back."""
    if msg.type != "sar.analyze":
        return

    case_id = msg.payload["case_id"]
    entries = await blackboard.read(f"sar:{case_id}")
    case = next(e["case"] for e in entries if e.get("source") == "coordinator")
    case_text = "\n".join(f"  {k}: {v}" for k, v in case.items())

    finding: AnalystFinding = await agent.think(
        f"Case for AML analysis:\n{case_text}\n\nProvide your specialist finding."
    )

    await blackboard.append(
        f"sar:{case_id}",
        {"source": agent.name, "finding": finding.model_dump()},
    )

    print(f"  {agent.name}: risk_score={finding.risk_score:.2f}  flags={finding.flags}")

    await agent._ref.send(
        to=msg.reply_to,
        type="sar.finding",
        payload={"case_id": case_id},
        metadata=msg.metadata,
    )


# ── Supervisor ────────────────────────────────────────────────────────────────

async def supervisor_handler(agent: SimpleAgent, msg: Message, ctx):
    """Collect findings; once all analysts have reported, issue SAR decision."""
    if msg.type != "sar.finding":
        return

    case_id = msg.metadata.get("case_id")
    if not case_id:
        return

    # Initialise state on first finding for this case
    if case_id not in agent.state:
        agent.state[case_id] = {
            "caller":        msg.metadata.get("caller"),
            "caller_msg_id": msg.metadata.get("caller_msg_id"),
            "received":      0,
        }

    agent.state[case_id]["received"] += 1

    # Count how many analyst findings are in the blackboard
    entries = await blackboard.read(f"sar:{case_id}")
    analyst_entries = [e for e in entries if e["source"].startswith("analyst-")]
    expected = len(ctx.agent_names(prefix="analyst-"))

    if agent.state[case_id]["received"] < expected:
        return   # still waiting for remaining analysts

    # All findings are in — synthesise
    findings_text = "\n\n".join(
        f"[{e['source']}]\n"
        + "\n".join(f"  {k}: {v}" for k, v in e["finding"].items())
        for e in analyst_entries
    )
    case = next(e["case"] for e in entries if e.get("source") == "coordinator")
    case_text = "\n".join(f"  {k}: {v}" for k, v in case.items())

    decision: SARDecision = await agent.think(
        f"Original case:\n{case_text}\n\n"
        f"Analyst findings:\n{findings_text}\n\n"
        "Issue the final SAR determination."
    )

    caller = agent.state[case_id]["caller"]
    caller_msg_id = agent.state[case_id]["caller_msg_id"]
    del agent.state[case_id]

    await agent._ref.send(
        to=caller,
        type="sar.decision",
        payload={"case_id": case_id, "decision": decision.model_dump()},
        metadata={"in_reply_to": caller_msg_id},
    )


# ── Client ────────────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "sar.decision":
        d = msg.payload["decision"]
        print(f"\n── SAR DECISION  (case: {msg.payload['case_id']}) ──")
        print(f"  Verdict:     {d['verdict']}")
        print(f"  Risk level:  {d['risk_level']}")
        print(f"  Summary:     {d['summary']}")
        print("  Actions:")
        for action in d["recommended_actions"]:
            print(f"    · {action}")
        print("─" * 60 + "\n")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/blackboard.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    coordinator = SimpleAgent("coordinator", handler=coordinator_handler)

    analysts = {
        "analyst-transactions": "bank.aml.analyst.transactions.system",
        "analyst-kyc":          "bank.aml.analyst.kyc.system",
        "analyst-network":      "bank.aml.analyst.network.system",
    }

    supervisor = SimpleAgent(
        "supervisor",
        prompt_name="bank.aml.supervisor.system",
        output_model=SARDecision,
        handler=supervisor_handler,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(coordinator)
    for name, prompt_name in analysts.items():
        runtime.register(
            SimpleAgent(
                name,
                prompt_name=prompt_name,
                output_model=AnalystFinding,
                handler=analyst_handler,
            )
        )
    runtime.register(supervisor)
    runtime.register(client)

    await runtime.start(run_id="run-aml-blackboard")

    # Suspected structuring case: series of cash deposits just below the $10,000
    # reporting threshold, customer is a PEP, funds flow to a high-risk jurisdiction.
    await client._ref.send(
        to="coordinator",
        type="sar.opened",
        payload={
            "case_id": "SAR-2026-00291",
            "case": {
                "customer":          "Rodrigo Valenzuela Mora",
                "account":           "****3847",
                "occupation":        "Government procurement official",
                "country_of_origin": "Venezuela",
                "transactions": (
                    "7 cash deposits in 10 days: $9,800 / $9,750 / $9,900 / "
                    "$9,600 / $9,850 / $9,700 / $9,950 — total $68,550"
                ),
                "destination":       "Wire transfer to Horizon Capital Ltd, Belize",
                "counterparty_note": "Horizon Capital Ltd — no web presence, incorporated 6 months ago",
                "account_history":   "Average monthly activity prior to this: $1,200",
            },
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
