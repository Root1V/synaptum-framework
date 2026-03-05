from typing import Awaitable, Callable, Optional

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from .agent_ref import AgentRef

Handler = Callable[["MessageAgent", Message, AgentContext], Awaitable[None]]


class MessageAgent(Agent):
    """
    Agent with bus messaging capability but no LLM.

    Provides ``self._ref`` (AgentRef) automatically when registered
    in a runtime via ``_bind_runtime``.  No prompt, no ``think()``.

    Use this as the base class for deterministic, routing, or
    human-in-the-loop agents that send/receive messages on the bus
    but never call a language model.

    Can be instantiated directly with a ``handler`` callable::

        agent = MessageAgent("client", handler=my_handler)

    Or subclassed for more complex logic::

        class _MyGateAgent(MessageAgent):
            async def on_message(self, message, context):
                await self._ref.send(to="next-agent", type="event", payload={...})
    """

    def __init__(self, name: str, *, handler: Optional[Handler] = None) -> None:
        super().__init__(name)
        self._ref: Optional[AgentRef] = None
        self._custom_handler = handler

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if self._custom_handler is not None:
            await self._custom_handler(self, message, context)
            return
        raise NotImplementedError
