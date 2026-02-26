import asyncio
import json
from typing import Any, Dict, List

from agentium.core.message import Message
from agentium.core.agent import Agent
from agentium.core.context import AgentContext
from agentium.core.runtime import AgentRuntime
from agentium.messaging.in_memory_bus import InMemoryMessageBus

from agentium.llm.client import LLMClient, LLMResponse
from agentium.tools.registry import ToolRegistry
from agentium.agents.llm_tool_agent import LLMToolAgent, LLMToolAgentConfig
from agentium.patterns.supervisor import SupervisorPattern, SupervisorPatternConfig


class FakeSupervisorLLM(LLMClient):
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = str(last_user).lower()

        if "crea un plan de subtareas" in low:
            plan = {"tasks": [
                {"worker": "research", "input": "Dame 3 bullets sobre qué es un router de agentes."},
                {"worker": "writer", "input": "Redacta 2 frases claras sobre router de agentes."},
            ]}
            return LLMResponse(content=json.dumps({"final": "PLAN_JSON:" + json.dumps(plan, ensure_ascii=False)}, ensure_ascii=False))

        if "sintetiza una respuesta final" in low:
            return LLMResponse(content=json.dumps({"final": "Síntesis final (demo) por supervisor."}, ensure_ascii=False))

        return LLMResponse(content=json.dumps({"final": "OK"}, ensure_ascii=False))


class FakeWorkerLLM(LLMClient):
    def __init__(self, label: str):
        self.label = label

    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        return LLMResponse(content=json.dumps({"final": f"{self.label} output (demo)."}, ensure_ascii=False))


class ClientAgent(Agent):
    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "agent.output":
            print("CLIENT GOT:", message.payload["text"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    tools = ToolRegistry()

    supervisor_llm = LLMToolAgent(
        LLMToolAgentConfig(agent_id="supervisor_llm", system_prompt="Eres supervisor."),
        llm=FakeSupervisorLLM(),
        tools=tools,
    )
    research = LLMToolAgent(
        LLMToolAgentConfig(agent_id="research", system_prompt="Eres research."),
        llm=FakeWorkerLLM("research"),
        tools=tools,
    )
    writer = LLMToolAgent(
        LLMToolAgentConfig(agent_id="writer", system_prompt="Eres writer."),
        llm=FakeWorkerLLM("writer"),
        tools=tools,
    )

    supervisor_pattern = SupervisorPattern(SupervisorPatternConfig(
        agent_id="supervisor_pattern",
        supervisor_llm_agent_id="supervisor_llm",
        workers=["research", "writer"],
    ))

    client = ClientAgent("client")

    for a in [supervisor_llm, research, writer, supervisor_pattern, client]:
        rt.register(a)

    await rt.start(run_id="run-supervisor")

    await rt.publish(Message(
        sender="client",
        recipient="supervisor_pattern",
        type="user.input",
        payload={"text": "Explica qué es un router de agentes."},
        metadata={"reply_to": "client"},
    ))

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
