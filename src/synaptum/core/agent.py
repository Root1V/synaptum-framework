from abc import ABC, abstractmethod
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
    def __init__(self, agent_id: str):
        self._id: str = str(uuid4())
        self.agent_id = agent_id
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
