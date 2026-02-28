import asyncio

from dotenv import load_dotenv

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import FilePromptProvider, InMemoryPromptProvider, PromptTemplate


# --- Prompt provider: se configura UNA vez a nivel de aplicación -------------
#
# Opción A — desde un archivo YAML (recomendado para producción):
#
# Opción B — en memoria (útil para tests y prototipos):
# prompt_provider = InMemoryPromptProvider({
#     "calculator.system": PromptTemplate(
#         content=(
#             "You are an arithmetic calculator agent. "
#             "You receive messages with a sum expression like '3 + 6 + 2' "
#             "and reply with only the numeric result."
#         ),
#         version="1.0",
#         description="System prompt for the arithmetic calculator agent.",
#     ),
# })


async def client_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "agent.output":
        print("CLIENT GOT result:", msg.payload["answer"])

async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/calculator.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    calculator = SimpleAgent(name="calculator", prompt_name="calculator.system")
    client = SimpleAgent(name="client", handler=client_handler)

    runtime.register(calculator)
    runtime.register(client)

    await runtime.start(run_id="run-calculator")

    await client._ref.send(
        to="calculator",
        payload={"text": "37 + 13 + 18 + 67 + 434 -3"},
        type="request",
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()

asyncio.run(main())
