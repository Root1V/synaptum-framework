import asyncio

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus


async def router_handler(agent, msg, ctx):
   if msg.type != "event":
       return
   t = msg.payload.get("type")
   if t == "risk":
       await agent._ref.send("risk-agent", msg.payload, type="event")
   else:
       await agent._ref.send("ops-agent", msg.payload, type="event")


async def risk_handler(agent, msg, ctx):
    print("RISK AGENT GOT:", msg.payload)

async def ops_handler(agent, msg, ctx):
    print("OPS AGENT GOT:", msg.payload)


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    router = SimpleAgent("router", handler=router_handler)
    risk = SimpleAgent("risk-agent", handler=risk_handler)
    ops = SimpleAgent("ops-agent", handler=ops_handler)

    client = SimpleAgent("client", handler=None)
    
    rt.register(router)
    rt.register(risk)
    rt.register(ops)
    rt.register(client)
    
    await rt.start(run_id="run-calculator")

    await client._ref.send(
        to="router",
        payload={"type": "other", "data": {"a": 3, "b": 6}},
        type="event",
        reply_to="client",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
