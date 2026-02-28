# simple/simple_agent.py
from typing import Awaitable, Callable, Optional

from ..llm.client import LLMClient
from ..llm.llama_client import LlamaClient
from ..core.context import AgentContext
from ..core.agent import Agent
from ..core.message import Message
from .agent_ref import AgentRef


Handler = Callable[["SimpleAgent", Message, AgentContext], Awaitable[None] ]


class SimpleAgent(Agent):

    def __init__(
        self,
        agent_id: str,
        *,
        llm: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
        handler: Optional[Handler] = None
    ):
        super().__init__(agent_id)

        self._llm = llm or LlamaClient()
        self._system_prompt = system_prompt
        self._custom_handler = handler
        self._ref: Optional[AgentRef] = None

    # inyectado por el runtime
    def _bind_runtime(self, bus):
        self._ref = AgentRef(self.agent_id, bus)

    async def on_message(self, message: Message, context: AgentContext) -> None:

        # 👉 si hay handler especial, úsalo
        if self._custom_handler is not None:
            await self._custom_handler(self, message, context)
            return

        # 👉 comportamiento por defecto
        await self._default_handler(message, context)

    async def _default_handler(self, message: Message, context: AgentContext) -> None:

        # agente pasivo si no tiene LLM
        if self._llm is None:
            return
        
        if self._ref is None:
            raise RuntimeError("AgentRef no inyectado en SimpleAgent")
        
        messages = []
        if self._system_prompt:
            messages.append(
                {"role": "system", "content": self._system_prompt}
            )

        user_text = ""
        if isinstance(message.payload, dict) and "text" in message.payload:
            user_text = str(message.payload["text"])
        else:
            user_text = str(message.payload)
        
        messages.append(
            {"role": "user", "content": user_text}
        )

        result = await self._llm.chat(messages)
        answer = result.content

        # 👉 por defecto responde al sender
        await self._ref.send(
            to=message.reply_to,
            payload={
                "answer": answer
            },
            type="agent.output",
            metadata={"in_reply_to": message.id}
        )