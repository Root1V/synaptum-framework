# simple/agent_ref.py
from typing import Any, Dict, Optional

from ..messaging.bus import MessageBus

from ..core.message import Message


class AgentRef:

    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id = agent_id
        self._bus = bus

    async def send(self, to: str, payload: Any, type: str = "event", reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        msg = Message(
            sender=self.agent_id,
            recipient=to,
            type=type,
            payload=payload,
            reply_to=reply_to,
            metadata=metadata or {},
        )   
        
        await self._bus.publish(msg)
        return msg.id