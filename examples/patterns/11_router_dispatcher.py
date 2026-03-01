"""
Pattern: Router / Dispatcher — Customer Support Triage (Banking)
================================================================
An intelligent router agent uses LLM classification to analyse incoming
customer support cases and dispatch each one to the correct specialist
department.  The specialist resolves the case and the router forwards
the resolution back to the original caller.

Flow:
  client ──[case.submitted]──▶ router
                               router (LLM classifies) ──▶ specialist-X
                               router ◀──[case.resolved]── specialist-X
  client ◀──[case.resolved]── router

The router stores pending-case context in agent.state (keyed by case id)
so responses are always returned to the right caller, even under
concurrent submissions.
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import FilePromptProvider


# ── Output schema ────────────────────────────────────────────────────────────

class RouteDecision(BaseModel):
    department: str = Field(
        description="Exact name of the specialist department to handle this case."
    )


# ── Router ────────────────────────────────────────────────────────────────────

async def router_handler(agent: SimpleAgent, msg: Message, ctx):
    """
    On case.submitted: ask the LLM which department should handle it,
    then dispatch to that specialist.
    On case.resolved: forward the resolution to the original caller.
    """
    if msg.type == "case.submitted":
        case = msg.payload["case"]
        case_id = case.get("id", msg.id)
        case_text = "\n".join(f"  {k}: {v}" for k, v in case.items())

        # LLM classifies the case into one department
        decision: RouteDecision = await agent.think(
            f"Incoming customer support case:\n{case_text}\n\nClassify and route this case."
        )
        department = decision.department

        # Persist caller context so we can route the reply back
        agent.state[case_id] = {
            "caller": msg.sender,
            "msg_id": msg.id,
        }

        print(f"\n── ROUTER: case {case_id} → {department} ──")

        await agent._ref.send(
            to=department,
            type="case.assigned",
            payload={"case": case, "case_id": case_id},
            reply_to=agent.name,
            metadata={"case_id": case_id},
        )

    elif msg.type == "case.resolved":
        case_id = msg.metadata.get("case_id")
        if not case_id or case_id not in agent.state:
            return

        pending = agent.state[case_id]
        del agent.state[case_id]

        await agent._ref.send(
            to=pending["caller"],
            type="case.resolved",
            payload={
                "case_id":    case_id,
                "handled_by": msg.sender,
                "resolution": msg.payload["resolution"],
            },
            metadata={"in_reply_to": pending["msg_id"]},
        )


# ── Specialist (shared handler) ───────────────────────────────────────────────

async def specialist_handler(agent: SimpleAgent, msg: Message, ctx):
    """Resolve the assigned case using domain-specific LLM reasoning."""
    if msg.type != "case.assigned":
        return

    case = msg.payload["case"]
    case_id = msg.payload["case_id"]
    case_text = "\n".join(f"  {k}: {v}" for k, v in case.items())

    resolution = await agent.think(
        f"Customer case assigned to your department:\n{case_text}\n\nProvide your resolution."
    )

    await agent._ref.send(
        to=msg.reply_to,
        type="case.resolved",
        payload={"resolution": resolution},
        metadata={"case_id": case_id},
    )


# ── Client ────────────────────────────────────────────────────────────────────

async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "case.resolved":
        print(
            f"\n── RESOLUTION  (case: {msg.payload['case_id']}"
            f"  ·  handled by: {msg.payload['handled_by']}) ──"
        )
        print(msg.payload["resolution"])
        print("─" * 60 + "\n")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/router.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    router = SimpleAgent(
        name="router",
        prompt_name="bank.router.system",
        output_model=RouteDecision,
        handler=router_handler,
    )

    specialists = {
        "specialist-account": "bank.specialist.account.system",
        "specialist-loans":   "bank.specialist.loans.system",
        "specialist-fraud":   "bank.specialist.fraud.system",
        "specialist-tech":    "bank.specialist.tech.system",
    }

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(router)
    for name, prompt_name in specialists.items():
        runtime.register(SimpleAgent(name, prompt_name=prompt_name, handler=specialist_handler))
    runtime.register(client)

    await runtime.start(run_id="run-support-triage")

    # A customer reports both a technical lockout and suspicious transactions —
    # the router must decide which department owns this case.
    await client._ref.send(
        to="router",
        type="case.submitted",
        payload={
            "case": {
                "id":          "CASE-2026-00834",
                "customer":    "Carlos Andrade",
                "account":     "****7731",
                "channel":     "mobile-app",
                "priority":    "HIGH",
                "description": (
                    "I traveled to Colombia last week and since returning I cannot log in "
                    "to the mobile app — it says my credentials are invalid. I also noticed "
                    "two transactions I do not recognise: $312 at a fuel station and $87 "
                    "at an ATM, both in Bogotá, on the day I was flying back. I never used "
                    "my card that day."
                ),
            }
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
