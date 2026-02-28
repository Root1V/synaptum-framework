# simple/simple_agent.py
import warnings
from typing import Awaitable, Callable, Optional

from ..llm.client import LLMClient
from ..llm.llama_client import LlamaClient
from ..core.context import AgentContext
from ..core.agent import Agent
from ..core.message import Message
from ..prompts.template import PromptTemplate
from ..prompts.provider import PromptProvider
from .agent_ref import AgentRef


Handler = Callable[["SimpleAgent", Message, AgentContext], Awaitable[None]]


class SimpleAgent(Agent):
    """
    Agente simple basado en LLM.

    Formas de configurar el system prompt (en orden de prioridad):

    1. ``prompt`` — PromptTemplate directa (recomendado para código).
    2. ``prompt_name`` + ``prompt_provider`` — resolución por nombre
       desde un proveedor externo (recomendado para producción).
    3. ``system_prompt`` (str) — compatibilidad hacia atrás, deprecated.

    Examples::

        # Forma 1 — template directa
        agent = SimpleAgent(
            "calculator",
            prompt=PromptTemplate(
                content="Eres una calculadora. Suma los números.",
                version="1.1",
                description="Calculadora aritmética",
            ),
        )

        # Forma 2 — provider externo
        provider = FilePromptProvider("prompts/agents.yaml")
        agent = SimpleAgent(
            "calculator",
            prompt_name="calculator.system",
            prompt_provider=provider,
        )
    """

    def __init__(
        self,
        name: str,
        *,
        llm: Optional[LLMClient] = None,
        prompt: Optional[PromptTemplate] = None,
        prompt_name: Optional[str] = None,
        prompt_provider: Optional[PromptProvider] = None,
        # --- backward compat ---
        system_prompt: Optional[str] = None,
        handler: Optional[Handler] = None,
    ):
        super().__init__(name)

        self._custom_handler = handler
        self._ref: Optional[AgentRef] = None
        self._pending_prompt_name: Optional[str] = None

        # Resolver el PromptTemplate efectivo
        if prompt is not None:
            self._prompt: Optional[PromptTemplate] = prompt
        elif prompt_name is not None and prompt_provider is not None:
            # Resolución inmediata con provider explícito
            self._prompt = prompt_provider.get(prompt_name)
        elif prompt_name is not None:
            # Resolución diferida: el runtime inyectará el provider en register()
            self._prompt = None
            self._pending_prompt_name = prompt_name
        elif system_prompt is not None:
            warnings.warn(
                "'system_prompt' está deprecated. Usa 'prompt=PromptTemplate(content=...)' "
                "o 'prompt_name' con el PromptProvider configurado en el AgentRuntime.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._prompt = PromptTemplate(content=system_prompt)
        else:
            self._prompt = None

        # Determinar el cliente LLM:
        # - Si se pasó explícitamente, úsalo.
        # - Si hay un prompt configurado (o pendiente), se asume uso de LLM → LlamaClient().
        # - Si no hay prompt, el agente es pasivo y no necesita LLM.
        has_prompt = self._prompt is not None or self._pending_prompt_name is not None
        self._llm: Optional[LLMClient] = llm or (LlamaClient() if has_prompt else None)

    # inyectado por el runtime
    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)

    def _inject_prompt_registry(self, provider: "PromptProvider") -> None:
        """Llamado por AgentRuntime.register() para resolver prompts diferidos."""
        if self._pending_prompt_name is not None:
            self._prompt = provider.get(self._pending_prompt_name)
            self._pending_prompt_name = None

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
        if self._prompt is not None:
            messages.append(
                {"role": "system", "content": self._prompt.render()}
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