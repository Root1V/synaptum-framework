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
from agentium.patterns.router import RouterPattern, RouterPatternConfig


class FakeRouterLLM(LLMClient):
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        # el prompt pide {"final":"<agent_id>"}
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = str(last_user).lower()
        if "calcula" in low or "cuánto" in low or "cuanto" in low:
            return LLMResponse(content=json.dumps({"final": "math"}, ensure_ascii=False))
        return LLMResponse(content=json.dumps({"final": "general"}, ensure_ascii=False))


class FakeGeneralLLM(LLMClient):
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        return LLMResponse(content=json.dumps({"final": "Soy general."}, ensure_ascii=False))


class FakeMathLLM(LLMClient):
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        # responde final directo para demo
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return LLMResponse(content=json.dumps({"final": f"Math dice: {last_user}"}, ensure_ascii=False))


class ClientAgent(Agent):
    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "agent.output":
            print("CLIENT GOT:", message.payload["text"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    tools = ToolRegistry()  # vacío (demo)

    router_llm_agent = LLMToolAgent(
        LLMToolAgentConfig(agent_id="router_llm", system_prompt="Eres router."),
        llm=FakeRouterLLM(),
        tools=tools,
    )
    math_agent = LLMToolAgent(
        LLMToolAgentConfig(agent_id="math", system_prompt="Eres math."),
        llm=FakeMathLLM(),
        tools=tools,
    )
    general_agent = LLMToolAgent(
        LLMToolAgentConfig(agent_id="general", system_prompt="Eres general."),
        llm=FakeGeneralLLM(),
        tools=tools,
    )

    router_pattern = RouterPattern(RouterPatternConfig(
        agent_id="router_pattern",
        router_llm_agent_id="router_llm",
        specialists=["math", "general"],
    ))

    client = ClientAgent("client")

    for a in [router_llm_agent, math_agent, general_agent, router_pattern, client]:
        rt.register(a)

    await rt.start(run_id="run-router")

    await rt.publish(Message(
        sender="client",
        recipient="router_pattern",
        type="user.input",
        payload={"text": "Calcula 2+2"},
        metadata={"reply_to": "client"},
    ))

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
