from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

from ..core.message import Message


MessageHandler = Callable[[Message], Awaitable[None]]


class MessageBus(ABC):
    """
    Bus mínimo:
    - publish: encola
    - subscribe: registra handler por recipient
    - next_message: obtiene siguiente mensaje (o None si timeout)
    - deliver: ejecuta handlers del recipient
    """

    @abstractmethod
    def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        raise NotImplementedError

    @abstractmethod
    async def publish(self, message: Message) -> None:
        raise NotImplementedError

    @abstractmethod
    async def next_message(self, timeout_s: float) -> Optional[Message]:
        raise NotImplementedError

    @abstractmethod
    async def deliver(self, message: Message) -> None:
        raise NotImplementedError
