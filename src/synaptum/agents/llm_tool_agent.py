import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.agent import Agent
from ..core.message import Message
from ..core.context import AgentContext
from ..core.errors import UnknownToolError

from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry
from ..agents.decision import DecisionParser


@dataclass
class LLMToolAgentConfig:
    name: str
    system_prompt: str
    max_steps: int = 6
    temperature: float = 0.2
    max_history: int = 20


class LLMToolAgent(Agent):
    """
    Building block: agente que usa LLM + tools pero sigue el contrato:
    - no retorna
    - publica mensajes

    Convención de mensajes:
    - Recibe: type="user.input" payload={"text": "..."} reply_to="some_agent"
    - Responde: type="agent.output" payload={"text": "..."} -> recipient = reply_to
    """
    def __init__(
        self,
        config: LLMToolAgentConfig,
        llm: LLMClient,
        tools: ToolRegistry,
        parser: Optional[DecisionParser] = None,
    ):
        super().__init__(config.name)
        self.cfg = config
        self.llm = llm
        self.tools = tools
        self.parser = parser or DecisionParser()

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type != "user.input":
            return

        text = ""
        if isinstance(message.payload, dict):
            text = str(message.payload.get("text", ""))
        else:
            text = str(message.payload)

        reply_to = message.reply_to or message.sender

        output = await self._run_tool_loop(text, context=context)

        # Propagar toda la metadata del mensaje entrante opáqamente.
        # reply_to es campo propio de Message, no está en metadata,
        # por lo que no hay nada que filtrar.
        out_meta: Dict[str, Any] = dict(message.metadata)
        out_meta["in_reply_to"] = message.id

        await self.runtime.publish(Message(
            sender=self.name,
            recipient=reply_to,
            type="agent.output",
            payload={"text": output},
            metadata=out_meta,
        ))

    async def _run_tool_loop(self, user_text: str, context: AgentContext) -> str:
        # historial desde state (por agente) usando key estable
        hist_key = f"history:{self.name}"
        history: List[Dict[str, str]] = self.state.get(hist_key, [])
        if not isinstance(history, list):
            history = []

        # construir system prompt con tools
        system = self._build_system_prompt()

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(history[-self.cfg.max_history:])
        messages.append({"role": "user", "content": user_text})

        # update history con entrada de usuario
        history.append({"role": "user", "content": user_text})

        for _step in range(self.cfg.max_steps):
            resp = await self.llm.chat(messages, temperature=self.cfg.temperature)
            raw_text = resp.content.strip()
            history.append({"role": "assistant", "content": raw_text})

            decision = self.parser.parse(raw_text)

            if decision.final is not None:
                self.state.set(hist_key, history)
                return decision.final

            if decision.action is None:
                self.state.set(hist_key, history)
                return "ERROR: model did not return action or final"

            tool_name = decision.action.tool
            tool_args = decision.action.args

            tool = None
            for t in self.tools.list():
                if t.name == tool_name:
                    tool = t
                    break
            if tool is None:
                raise UnknownToolError(f"Unknown tool: {tool_name}")

            tool_out = await tool.execute(tool_args, context, self.state)
            observation = "TOOL_OBSERVATION: " + json.dumps({
                "tool": tool_name,
                "result": tool_out,
            }, ensure_ascii=False)

            # meter observation como user message (simple y portable)
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": observation})
            history.append({"role": "user", "content": observation})

        self.state.set(hist_key, history)
        return "ERROR: max_steps reached"

    def _build_system_prompt(self) -> str:
        tools_desc = self.tools.describe_for_llm()
        tools_json = json.dumps(tools_desc, ensure_ascii=False)

        protocol = (
            "Responde SIEMPRE en JSON estricto, sin texto extra.\n"
            "Formato válido:\n"
            '  {"action":{"tool":"NOMBRE","args":{...}}}\n'
            '  {"final":"RESPUESTA"}\n'
        )

        return f"{self.cfg.system_prompt}\n\n{protocol}\nTOOLS: {tools_json}"
