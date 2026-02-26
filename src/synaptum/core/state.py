from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AgentState:
    """
    Estado mutable por agente (no por runtime).
    El runtime no impone persistencia.
    """
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
