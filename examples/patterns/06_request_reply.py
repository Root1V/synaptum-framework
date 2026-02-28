import asyncio

from dotenv import load_dotenv

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus


async def client_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "agent.output":
        print("CLIENT GOT result:", msg.payload["answer"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    calculator = SimpleAgent("calculator", system_prompt="I am a calculator agent that sums three numbers. I receive messages with payloads like '3 +6 + 2' and I reply with messages with payloads like result '11' ")
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
