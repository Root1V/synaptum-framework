import warnings
from typing import Awaitable, Callable, Optional, Type, Union

from pydantic import BaseModel

from ..core.state import AgentState
from ..llm.client import LLMClient
from ..llm.llama_client import LlamaClient
from ..core.context import AgentContext
from ..core.message import Message
from ..prompts.template import PromptTemplate
from ..prompts.provider import PromptProvider
from .message_agent import MessageAgent
from .agent_ref import AgentRef


Handler = Callable[["LLMAgent", "Message", "AgentContext"], Awaitable[None]]


class LLMAgent(MessageAgent):
    """
    Agent backed by a language model.

    Inherits bus messaging from ``MessageAgent`` (``self._ref`` + ``_bind_runtime``
    are provided automatically — no boilerplate needed in subclasses).

    Adds ``think(user_message)`` which builds the system + user turn,
    calls the configured LLM, and returns either a plain ``str`` or a
    validated Pydantic model instance when ``output_model`` is set.

    Formas de configurar el system prompt (en orden de prioridad):

    1. ``prompt`` — PromptTemplate directa (recomendado para código).
    2. ``prompt_name`` + ``prompt_provider`` — resolución por nombre
       desde un proveedor externo (recomendado para producción).
    3. ``system_prompt`` (str) — compatibilidad hacia atrás, deprecated.

    Structured output::

        class RouteDecision(BaseModel):
            department: str = Field(description="Target specialist department")

        agent = LLMAgent(
            "router",
            prompt_name="bank.router.system",
            output_model=RouteDecision,
        )
        # think() returns a RouteDecision instance, never a raw string
        result: RouteDecision = await agent.think(case_text)

    Examples::

        # Forma 1 — template directa
        agent = LLMAgent(
            "calculator",
            prompt=PromptTemplate(
                content="Eres una calculadora. Suma los números.",
                version="1.1",
                description="Calculadora aritmética",
            ),
        )

        # Forma 2 — provider externo
        provider = FilePromptProvider("prompts/agents.yaml")
        agent = LLMAgent(
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
        output_model: Optional[Type[BaseModel]] = None,
        # --- backward compat ---
        system_prompt: Optional[str] = None,
        handler: Optional[Handler] = None,
    ):
        super().__init__(name)

        self._custom_handler = handler
        self._pending_prompt_name: Optional[str] = None
        self._output_model: Optional[Type[BaseModel]] = output_model
        self.state = AgentState()  # per-instance state store for handlers

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

    # _ref and _bind_runtime are inherited from MessageAgent

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

    async def think(
        self, user_message: str, **kwargs
    ) -> Union[str, BaseModel]:
        """
        Reason about a user message using the agent's LLM.

        Builds the message array internally (system prompt + user turn) so
        callers never need to access ``_llm`` or ``_prompt`` directly.

        If the agent was configured with ``output_model``, the LLM is asked to
        respond as a JSON object matching that Pydantic schema and the method
        returns a validated model instance instead of a plain string.

        Args:
            user_message: The input text for the agent to reason about.
            **kwargs: Extra keyword arguments forwarded to the LLM client
                      (e.g. ``temperature``, ``max_tokens``).

        Returns:
            A plain ``str`` when no ``output_model`` is configured, or a
            validated ``BaseModel`` instance when one is set.

        Raises:
            RuntimeError: If the agent has no LLM configured (passive agent).
        """
        if self._llm is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no LLM configured. "
                "Pass a 'prompt' or 'prompt_name' to enable LLM support."
            )

        messages = []
        if self._prompt is not None:
            messages.append({"role": "system", "content": self._prompt.render()})
        messages.append({"role": "user", "content": user_message})

        if self._output_model is not None:
            kwargs.setdefault(
                "response_format",
                {
                    "type": "json_object",
                    "schema": self._output_model.model_json_schema(),
                },
            )

        result = await self._llm.chat(messages, **kwargs)

        if self._output_model is not None:
            return self._output_model.model_validate_json(result.content)

        return result.content

    async def _default_handler(self, message: Message, context: AgentContext) -> None:

        # agente pasivo si no tiene LLM
        if self._llm is None:
            return

        if self._ref is None:
            raise RuntimeError("AgentRef no inyectado en LLMAgent")

        if isinstance(message.payload, dict) and "text" in message.payload:
            user_text = str(message.payload["text"])
        else:
            user_text = str(message.payload)

        answer = await self.think(user_text)

        # 👉 por defecto responde al sender
        await self._ref.send(
            to=message.reply_to,
            payload={"answer": answer},
            type="agent.output",
            metadata={"in_reply_to": message.id}
        )
