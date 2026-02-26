import asyncio
import json
from typing import Any, Dict, List

from agentium.core.message import Message
from agentium.core.context import AgentContext
from agentium.core.agent import Agent
from agentium.core.runtime import AgentRuntime
from agentium.messaging.in_memory_bus import InMemoryMessageBus

from agentium.llm.client import LLMClient, LLMResponse
from agentium.tools.base import Tool
from agentium.tools.registry import ToolRegistry
from agentium.agents.llm_tool_agent import LLMToolAgent, LLMToolAgentConfig


class FakeLLMClient(LLMClient):
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        low = last_user.lower()

        if "tool_observation:" in low:
            # extraer número si existe
            i = last_user.find("{")
            j = last_user.rfind("}")
            if i != -1 and j != -1 and j > i:
                try:
                    payload = json.loads(last_user[i:j+1])
                    return LLMResponse(content=json.dumps({"final": f"Resultado: {payload['result']}"}, ensure_ascii=False))
                except Exception:
                    return LLMResponse(content=json.dumps({"final": "No pude parsear la observación."}, ensure_ascii=False))

        # detectar "cuánto es X"
        expr = ""
        if "cuánto es" in low or "cuanto es" in low:
            parts = last_user.split("es", 1)
            expr = parts[1].strip().rstrip("?").strip() if len(parts) == 2 else ""

        if expr:
            return LLMResponse(content=json.dumps({"action": {"tool": "calc", "args": {"expression": expr}}}, ensure_ascii=False))

        return LLMResponse(content=json.dumps({"final": "No necesito tools."}, ensure_ascii=False))


class CalcTool(Tool):
    name = "calc"
    description = "Evalúa una expresión matemática simple."

    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}

    async def execute(self, input: Dict[str, Any], context: AgentContext, state) -> Any:
        expr = str(input.get("expression", "")).strip()
        allowed = set("0123456789+-*/(). %")
        if not expr or any(ch not in allowed for ch in expr):
            return "ERROR: invalid expression"
        try:
            return eval(expr, {"__builtins__": {}}, {})
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"


class ClientAgent(Agent):
    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "agent.output":
            print("CLIENT GOT:", message.payload["text"])


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    tools = ToolRegistry()
    tools.register(CalcTool())

    llm = FakeLLMClient()
    agent = LLMToolAgent(
        config=LLMToolAgentConfig(
            agent_id="llm_agent",
            system_prompt="Eres un agente que usa tools cuando sea útil.",
        ),
        llm=llm,
        tools=tools,
    )

    client = ClientAgent("client")

    rt.register(agent)
    rt.register(client)

    await rt.start(run_id="run-llm-tool")

    await rt.publish(Message(
        sender="client",
        recipient="llm_agent",
        type="user.input",
        payload={"text": "¿Cuánto es 12.5*(3+1)?"},
        metadata={"reply_to": "client"},
    ))

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
