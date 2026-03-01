"""
Pattern: Pipeline — Wire Transfer Processing (Banking)
=======================================================
A wire transfer request flows sequentially through three LLM-powered stages.
Each stage enriches the data and passes it to the next agent in the pipeline.

Flow:
  client
    ──[raw transfer]──▶  parser       (normalize & extract fields)
                          ──▶  screener    (AML & sanctions screening)
                                ──▶  approver   (final decision letter)
                                      ──▶  client [decision]

Each stage has its own specialized system prompt (pipeline.yaml).
"""

import asyncio

from dotenv import load_dotenv
load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import FilePromptProvider


async def parser_handler(agent: SimpleAgent, msg: Message, ctx):
    """Stage 1 — normalize the raw wire transfer request."""
    if msg.type != "transfer.submitted":
        return

    raw = msg.payload["raw_transfer"]
    raw_text = "\n".join(f"  {k}: {v}" for k, v in raw.items())
    prompt = f"Raw wire transfer received:\n{raw_text}\n\nNormalize and extract the key fields."

    parsed = await agent.think(prompt)
    print(f"\n[PARSER]\n{parsed}")

    await agent._ref.send(
        to="screener",
        type="transfer.parsed",
        payload={"parsed": parsed, "raw": raw},
        reply_to=msg.reply_to,
        metadata={"origin_msg_id": msg.id},
    )


async def screener_handler(agent: SimpleAgent, msg: Message, ctx):
    """Stage 2 — screen the parsed transfer for AML and sanctions risk."""
    if msg.type != "transfer.parsed":
        return

    prompt = (
        f"Normalized transfer summary:\n{msg.payload['parsed']}\n\n"
        "Perform AML and sanctions screening."
    )

    screening = await agent.think(prompt)
    print(f"\n[SCREENER]\n{screening}")

    await agent._ref.send(
        to="approver",
        type="transfer.screened",
        payload={"parsed": msg.payload["parsed"], "screening": screening},
        reply_to=msg.reply_to,
        metadata=msg.metadata,
    )


async def approver_handler(agent: SimpleAgent, msg: Message, ctx):
    """Stage 3 — issue the final processing decision."""
    if msg.type != "transfer.screened":
        return

    prompt = (
        f"Transfer summary:\n{msg.payload['parsed']}\n\n"
        f"Risk screening report:\n{msg.payload['screening']}\n\n"
        "Write the final approval decision."
    )

    decision = await agent.think(prompt)
    print(f"\n[APPROVER]\n{decision}")

    await agent._ref.send(
        to=msg.reply_to,
        type="transfer.decision",
        payload={"decision": decision},
        metadata=msg.metadata,
    )


async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "transfer.decision":
        print("\n── FINAL WIRE TRANSFER DECISION ────────────────────────────────")
        print(msg.payload["decision"])
        print("────────────────────────────────────────────────────────────────\n")


async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/transfer.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    parser   = SimpleAgent("parser",   prompt_name="bank.pipeline.parser.system",   handler=parser_handler)
    screener = SimpleAgent("screener", prompt_name="bank.pipeline.screener.system", handler=screener_handler)
    approver = SimpleAgent("approver", prompt_name="bank.pipeline.approver.system", handler=approver_handler)
    client   = SimpleAgent("client",   handler=client_handler)

    runtime.register(parser)
    runtime.register(screener)
    runtime.register(approver)
    runtime.register(client)

    await runtime.start(run_id="run-wire-transfer")

    # sample international wire transfer
    await client._ref.send(
        to="parser",
        type="transfer.submitted",
        payload={
            "raw_transfer": {
                "origin_account":   "CR21015201001026284066",
                "origin_name":      "Constructora Pacífico S.A.",
                "destination_bank": "Santander, Madrid Spain",
                "destination_iban":  "ES9121000418450200051332",
                "destination_name": "StroyCom LLC",
                "amount":           "250000",
                "currency":         "USD",
                "purpose":          "services",
                "transfer_date":    "2026-02-28",
                "urgency":          "same-day",
            }
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())

