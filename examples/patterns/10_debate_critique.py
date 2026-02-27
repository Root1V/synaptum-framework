import asyncio
from typing import Any, Dict, List

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.runtime import AgentRuntime
from synaptum.llm.client import LLMClient, LLMResponse
from synaptum.messaging.in_memory_bus import InMemoryMessageBus

class FakeGeneralLLM(LLMClient):
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        return LLMResponse(content= f"Respuesta general de LLM Fake: {messages[-1]['content']}")

async def moderator_handler(agent:SimpleAgent, msg, ctx):
   if msg.type == "question":
       q = msg.payload["q"]
       await agent._ref.send("expert-1", {"text": q}, type="user.input", reply_to="moderator")
       await agent._ref.send("expert-2", {"text": q}, type="user.input", reply_to="moderator")
       agent.state.set("waiting", 2)
       agent.state.set("answers", [])
       agent.state.set("caller", msg.sender)
       return

   if msg.type == "agent.output" and msg.sender in ("expert-1", "expert-2"):
       answers = agent.state.get("answers", [])
       answers.append({msg.sender: msg.payload["answer"]})
       agent.state.set("answers", answers)

       left = int(agent.state.get("waiting", 0)) - 1
       agent.state.set("waiting", left)

       if left == 0:
           # síntesis mínima (puedes llamar LLM si quieres)
           final = f"Debate:\n{answers}"
           caller = agent.state.get("caller")
           await agent._ref.send(caller, {"text": final}, type="agent.output")


async def client_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "agent.output":
        print("CLIENT GOT result:", msg.payload["text"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    moderator = SimpleAgent("moderator", handler=moderator_handler)
    expert1 = SimpleAgent("expert-1", llm=FakeGeneralLLM(), system_prompt="Eres experto en economia. Responde breve.")
    expert2 = SimpleAgent("expert-2", llm=FakeGeneralLLM(), system_prompt="Eres experto en marketing. Responde breve.")
    client = SimpleAgent("client", handler=client_handler)

    rt.register(moderator)
    rt.register(expert1)
    rt.register(expert2)
    rt.register(client)
    
    await rt.start(run_id="run-debate")

    await client._ref.send(
        to="moderator",
        payload={"q": "¿Cuál es la mejor estrategia para aumentar las ventas?"},
        type="question",
        reply_to="client",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
