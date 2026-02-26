from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.context import AgentContext
from ..core.state import AgentState


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, input: Dict[str, Any], context: AgentContext, state: AgentState) -> Any:
        raise NotImplementedError
