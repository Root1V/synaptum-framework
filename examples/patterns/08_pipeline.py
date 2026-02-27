import asyncio

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus


async def a_handler(agent, msg, ctx):
   if msg.type != "raw":
       return
   text = msg.payload["text"].upper() + " processed by A,"
   await agent._ref.send("agent-b", {"text": text}, type="stage1", reply_to=msg.reply_to)

async def b_handler(agent, msg, ctx):
   if msg.type != "stage1":
       return
   text = msg.payload["text"].upper() + " processed by B,"
   await agent._ref.send("agent-c", {"text": text}, type="stage2", reply_to=msg.reply_to)

async def c_handler(agent, msg, ctx):
   if msg.type != "stage2":
       return
   text = msg.payload["text"].upper() + " and processed by C"
   await agent._ref.send(msg.reply_to, {"text": text}, type="reply")



async def client_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "reply":
        print("CLIENT GOT result:", msg.payload["text"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    a = SimpleAgent("agent-a", handler=a_handler)
    b = SimpleAgent("agent-b", handler=b_handler)
    c = SimpleAgent("agent-c", handler=c_handler)
    client = SimpleAgent("client", handler=client_handler)

    rt.register(a)
    rt.register(b)
    rt.register(c)
    rt.register(client)
    
    await rt.start(run_id="run-pipeline")

    await client._ref.send(
        to="agent-a",
        payload={"text": "hello world"},
        type="raw",
        reply_to="client",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
