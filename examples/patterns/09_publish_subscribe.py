import asyncio

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus


async def subscriber_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "pub":
        print("SUBSCRIBER GOT event:", msg.payload["event"], agent.id)


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    publisher = SimpleAgent("publisher", handler=None) # No handler needed for publisher
        
    sub1 = SimpleAgent("topic:events", handler=subscriber_handler)
    sub2 = SimpleAgent("topic:events", handler=subscriber_handler)
    sub3 = SimpleAgent("topic:events", handler=subscriber_handler)
    
    rt.register(publisher)
    rt.register(sub1)
    rt.register(sub2)
    rt.register(sub3)
    
    await rt.start(run_id="run-pub-sub")

    await publisher._ref.send(
        to="topic:events",
        payload={"event": "order_created"},
        type="pub",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
