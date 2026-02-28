import asyncio

from dotenv import load_dotenv

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import InMemoryPromptProvider, PromptTemplate


# --- Definición de prompts ---------------------------------------------------
# En producción esto vendría de un archivo YAML o proveedor remoto.
prompt_provider = InMemoryPromptProvider({
    "calculator.system": PromptTemplate(
        content=(
            "You are an arithmetic calculator agent. "
            "You receive messages with a sum expression like '3 + 6 + 2' "
            "and reply with only the numeric result."
        ),
        version="1.0",
        description="System prompt for the arithmetic calculator agent.",
    ),
})


async def client_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "agent.output":
        print("CLIENT GOT result:", msg.payload["answer"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    calculator = SimpleAgent(
        "calculator",
        prompt_name="calculator.system",
        prompt_provider=prompt_provider,
    )
    client = SimpleAgent("client", handler=client_handler)

    rt.register(calculator)
    rt.register(client)

    await rt.start(run_id="run-calculator")

    await client._ref.send(
        to="calculator",
        payload={"text": "37 + 13 + 15"},
        type="request",
        reply_to="client",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
