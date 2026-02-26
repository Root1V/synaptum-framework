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
from agentium.patterns.graph import GraphPattern, GraphNode


class FakeLLM(LLMClient):
    def __init__(self, label: str):
        self.label = label

    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return LLMResponse(content=json.dumps({"final": f"{self.label}: {last_user}"}, ensure_ascii=False))


class ClientAgent(Agent):
    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "agent.output":
            print("CLIENT GOT:", message.payload["text"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)
    tools = ToolRegistry()

    a1 = LLMToolAgent(LLMToolAgentConfig("node1", "Nodo 1"), llm=FakeLLM("N1"), tools=tools)
    a2 = LLMToolAgent(LLMToolAgentConfig("node2", "Nodo 2"), llm=FakeLLM("N2"), tools=tools)

    nodes = {
        "node1": GraphNode(
            agent_id="node1",
            prompt_builder=lambda task: f"Paso1: {task}",
            next_node=lambda out: "node2",
        ),
        "node2": GraphNode(
            agent_id="node2",
            prompt_builder=lambda task: f"Paso2: usa lo anterior -> {task}",
            next_node=lambda out: None,
        )
    }

    graph = GraphPattern(agent_id="graph", start_agent_id="node1", nodes=nodes)
    client = ClientAgent("client")

    for a in [a1, a2, graph, client]:
        rt.register(a)

    await rt.start(run_id="run-graph")

    await rt.publish(Message(
        sender="client",
        recipient="graph",
        type="user.input",
        payload={"text": "Construye una respuesta en 2 pasos"},
        metadata={"reply_to": "client"},
    ))

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
