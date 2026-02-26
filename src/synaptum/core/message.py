from dataclasses import dataclass, field
from typing import Any, Dict
from uuid import uuid4


@dataclass
class Message:
    sender: str
    recipient: str
    type: str
    payload: Any
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
