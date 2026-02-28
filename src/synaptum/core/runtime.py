import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

from .agent import Agent
from .context import AgentContext
from .message import Message
from ..messaging.bus import MessageBus
from ..prompts.provider import PromptProvider


@dataclass
class RuntimeConfig:
    """
    - max_messages: evita loops infinitos en demos/tests
    - idle_grace_ms: cuánto esperar sin mensajes antes de considerar idle
    """
    max_messages: int = 10_000
    idle_grace_ms: int = 50


class AgentRuntime:
    """
    Runtime mínimo:
    - Registra agentes
    - Suscribe handlers
    - Encola y entrega mensajes
    - No crea threads
    - No crea event loop oculto
    """
    def __init__(
        self,
        bus: MessageBus,
        config: Optional[RuntimeConfig] = None,
        prompts: Optional[PromptProvider] = None,
    ):
        self._bus = bus
        self._config = config or RuntimeConfig()
        self._prompts = prompts
        self._agents: Dict[str, Agent] = {}
        self._context: Optional[AgentContext] = None

    @property
    def context(self) -> AgentContext:
        if self._context is None:
            raise RuntimeError("Runtime not started. Call start(run_id=...) first.")
        return self._context

    def register(self, agent: Agent) -> None:
        self._agents[agent.name] = agent

        if hasattr(agent, "_bind_runtime"):
            agent._bind_runtime(self)
        else:
            agent._attach_runtime(self)

        # Inyecta el PromptProvider del runtime si el agente lo necesita
        if self._prompts is not None and hasattr(agent, "_inject_prompt_registry"):
            agent._inject_prompt_registry(self._prompts)

        async def handler(msg: Message) -> None:
            await agent.on_message(msg, self._context)

        self._bus.subscribe(agent.name, handler)

    async def publish(self, message: Message) -> None:
        await self._bus.publish(message)

    async def start(self, run_id: str, metadata: Optional[dict] = None) -> None:
        self._context = AgentContext(run_id=run_id, metadata=metadata or {}, agent_registry=self._agents)
        for a in self._agents.values():
            await a.on_start(self._context)

    async def stop(self) -> None:
        if self._context is None:
            return
        for a in self._agents.values():
            await a.on_stop(self._context)

    async def run_until_idle(self) -> int:
        """
        Consume mensajes hasta que el bus quede idle.
        Devuelve # de mensajes procesados.
        """
        processed = 0
        idle_grace = self._config.idle_grace_ms / 1000.0

        while processed < self._config.max_messages:
            msg = await self._bus.next_message(timeout_s=idle_grace)
            if msg is None:
                break
            processed += 1
            await self._bus.deliver(msg)

        return processed
