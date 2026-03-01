"""
Pattern: Publish / Subscribe — High-Value Transaction Event (Banking)
=====================================================================
A transaction monitor publishes a single high-value transaction event to a
shared topic. Four independent LLM-powered subscribers react simultaneously,
each with a different responsibility — without knowing about each other.

Flow:
  transaction-monitor  ──[transaction.high_value]──▶  topic:transaction.events
                                                              │
                              ┌───────────────────────────────┼──────────────────────┐
                              ▼                               ▼                      ▼ ...
                    notification-agent           compliance-agent          risk-agent   audit-agent

Key property: publishers and subscribers are fully decoupled.
Adding a new subscriber requires zero changes to the publisher or other subscribers.
"""

import asyncio

from dotenv import load_dotenv
load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import FilePromptProvider


async def notification_handler(agent: SimpleAgent, msg: Message, ctx):
    """Draft a customer-facing SMS/push alert for the transaction."""
    if msg.type != "transaction.high_value":
        return
    txn = msg.payload["transaction"]
    txn_text = "\n".join(f"  {k}: {v}" for k, v in txn.items())
    response = await agent.think(f"Transaction event:\n{txn_text}")
    print(f"\n[NOTIFICATION AGENT]\n{response}")


async def compliance_handler(agent: SimpleAgent, msg: Message, ctx):
    """Evaluate whether a SAR or CTR must be filed."""
    if msg.type != "transaction.high_value":
        return
    txn = msg.payload["transaction"]
    txn_text = "\n".join(f"  {k}: {v}" for k, v in txn.items())
    response = await agent.think(f"Transaction event:\n{txn_text}")
    print(f"\n[COMPLIANCE AGENT]\n{response}")


async def risk_handler(agent: SimpleAgent, msg: Message, ctx):
    """Score real-time risk and recommend hold or verification."""
    if msg.type != "transaction.high_value":
        return
    txn = msg.payload["transaction"]
    txn_text = "\n".join(f"  {k}: {v}" for k, v in txn.items())
    response = await agent.think(f"Transaction event:\n{txn_text}")
    print(f"\n[RISK AGENT]\n{response}")


async def audit_handler(agent: SimpleAgent, msg: Message, ctx):
    """Write a structured audit log entry for the transaction."""
    if msg.type != "transaction.high_value":
        return
    txn = msg.payload["transaction"]
    txn_text = "\n".join(f"  {k}: {v}" for k, v in txn.items())
    response = await agent.think(f"Transaction event:\n{txn_text}")
    print(f"\n[AUDIT AGENT]\n{response}")


async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/events.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    # publisher — passive agent, no LLM needed
    monitor = SimpleAgent("transaction-monitor")

    # subscribers — all listen on the same topic
    notification = SimpleAgent("topic:transaction.events", prompt_name="bank.events.notification.system", handler=notification_handler)
    compliance   = SimpleAgent("topic:transaction.events", prompt_name="bank.events.compliance.system",   handler=compliance_handler)
    risk         = SimpleAgent("topic:transaction.events", prompt_name="bank.events.risk.system",         handler=risk_handler)
    audit        = SimpleAgent("topic:transaction.events", prompt_name="bank.events.audit.system",        handler=audit_handler)

    runtime.register(monitor)
    runtime.register(notification)
    runtime.register(compliance)
    runtime.register(risk)
    runtime.register(audit)

    await runtime.start(run_id="run-transaction-events")

    # publish a single high-value transaction event to the topic
    await monitor._ref.send(
        to="topic:transaction.events",
        type="transaction.high_value",
        payload={
            "transaction": {
                "id":               "TXN-2026-0091847",
                "account":          "****3371",
                "account_holder":   "Emeric Espiritu Santiago",
                "amount_usd":       18500.00,
                "type":             "cash deposit",
                "channel":          "branch teller",
                "branch":           "San José Central, CR",
                "time":             "2026-02-28 09:14 AM",
                "teller_id":        "T-0042",
                "avg_monthly_cash": 1200.00,
            }
        },
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())

