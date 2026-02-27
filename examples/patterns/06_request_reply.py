import asyncio

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus


async def calculator_handler(agent: SimpleAgent, msg, ctx):
    if msg.type != "request":
        return
    data = msg.payload
    result = float(data["a"]) + float(data["b"])
    
    await agent._ref.send(
        to=msg.reply_to,
        type="reply",
        payload={"result": result},
        metadata={"in_reply_to": msg.id},
    )


async def client_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "reply":
        print("CLIENT GOT result:", msg.payload["result"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    calculator = SimpleAgent("calculator", handler=calculator_handler)
    client = SimpleAgent("client", handler=client_handler)
    
    rt.register(calculator)
    rt.register(client)
    
    await rt.start(run_id="run-calculator")

    await client._ref.send(
        to="calculator",
        payload={"a": 3, "b": 6},
        type="request",
        reply_to="client",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
