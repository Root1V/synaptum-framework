"""
Pattern: Request / Reply — Fraud Transaction Triage (Banking)
=============================================================
The simplest multi-agent interaction: one agent sends a request,
one agent processes it and sends a single reply back.

Flow:
  client  ──[transaction data]──▶  fraud-triage
  client  ◀──[SAFE|REVIEW|BLOCK]──  fraud-triage

The fraud-triage agent uses an LLM with a specialized system prompt to
evaluate each incoming transaction and return a structured verdict.
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import FilePromptProvider


async def triage_handler(agent: SimpleAgent, msg: Message, ctx):
    """Analyze the incoming transaction and reply with a fraud verdict."""
    if msg.type != "transaction.submitted":
        return

    txn = msg.payload["transaction"]
    txn_text = "\n".join(f"  {k}: {v}" for k, v in txn.items())
    prompt = f"Transaction submitted for review:\n{txn_text}\n\nProvide your fraud triage assessment."

    verdict = await agent.think(prompt)

    await agent._ref.send(
        to=msg.reply_to,
        type="transaction.verdict",
        payload={"verdict": verdict, "transaction_id": txn.get("id")},
        metadata={"in_reply_to": msg.id},
    )


async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "transaction.verdict":
        print(f"\n── FRAUD TRIAGE VERDICT (txn: {msg.payload['transaction_id']}) ──")
        print(msg.payload["verdict"])
        print("─" * 55 + "\n")


async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/fraud.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    fraud_triage = SimpleAgent(
        "fraud-triage",
        prompt_name="bank.fraud_triage.system",
        handler=triage_handler,
    )
    client = SimpleAgent("client", handler=client_handler)

    runtime.register(fraud_triage)
    runtime.register(client)

    await runtime.start(run_id="run-fraud-triage")

    # sample transaction to evaluate
    await client._ref.send(
        to="fraud-triage",
        type="transaction.submitted",
        payload={
            "transaction": {
                "id":               "TXN-2026-0487231",
                "account":          "****4892",
                "amount_usd":       4750.00,
                "merchant":         "Electronics Superstore",
                "merchant_category":"electronics",
                "location":         "Miami, FL (account holder is in San José, CR)",
                "channel":          "card-present",
                "time":             "02:47 AM local time",
                "device":           "new POS terminal, first use",
                "avg_monthly_spend": 620.00,
                "last_transaction":  "6 hours ago — grocery store, $43",
            }
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
