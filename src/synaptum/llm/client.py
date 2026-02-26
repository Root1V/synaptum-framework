from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    content: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """
    El framework no sabe nada de llama-server.
    Tu adapter (fuera o dentro de tu repo) implementa esto usando tu SDK.
    """
    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        raise NotImplementedError
