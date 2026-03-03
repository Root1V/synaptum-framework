from abc import ABC, abstractmethod
from typing import List
from uuid import uuid4
from .message import Message
from .context import AgentContext
from .state import AgentState


class Agent(ABC):
    """
    Contrato central:
    - El agente NO retorna output.
    - El agente PUBLICA mensajes (vía runtime/bus).
    """
    def __init__(self, name: str):
        self._id: str = str(uuid4())
        self.name = name
        self.state = AgentState()
        self._runtime = None  # seteado por runtime.register()

    @property
    def id(self) -> str:
        return self._id

    def _attach_runtime(self, runtime) -> None:
        self._runtime = runtime

    @property
    def runtime(self):
        if self._runtime is None:
            raise RuntimeError("Agent is not registered in a runtime.")
        return self._runtime

    async def on_start(self, context: AgentContext) -> None:
        return None

    async def on_stop(self, context: AgentContext) -> None:
        return None

    @abstractmethod
    async def on_message(self, message: Message, context: AgentContext) -> None:
        raise NotImplementedError


class CompositeAgent(Agent):
    """
    An Agent that composes a topology of sub-agents.

    ``sub_agents()`` declares the internal agents that form this topology.
    ``AgentRuntime.register()`` automatically registers all of them when
    this composite agent is registered — no manual ``runtime.register()``
    calls needed inside ``_bind_runtime``.

    Use this as the base class whenever an agent's primary job is to wire
    a group of simpler agents together (e.g. SagaAgent, Pipeline, Swarm).
    The composite itself is also a full Agent: it can receive messages via
    ``on_message`` and act as the public entry point for the topology.
    """

    def sub_agents(self) -> List["Agent"]:
        """
        Return the list of agents that make up this topology.
        Called once by the runtime during registration.
        Override in subclasses to declare the internal topology.
        """
        return []
