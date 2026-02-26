import asyncio
from collections import defaultdict
from typing import DefaultDict, List, Optional

from ..core.message import Message
from ..core.errors import UnknownRecipientError
from .bus import MessageBus, MessageHandler


class InMemoryMessageBus(MessageBus):
    def __init__(self):
        self._subscribers: DefaultDict[str, List[MessageHandler]] = defaultdict(list)
        self._queue: asyncio.Queue[Message] = asyncio.Queue()

    def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        self._subscribers[agent_id].append(handler)

    async def publish(self, message: Message) -> None:
        await self._queue.put(message)

    async def next_message(self, timeout_s: float) -> Optional[Message]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    async def deliver(self, message: Message) -> None:
        handlers = self._subscribers.get(message.recipient, [])
        if not handlers:
            raise UnknownRecipientError(f"No subscribers for recipient: {message.recipient}")

        for h in handlers:
            await h(message)

