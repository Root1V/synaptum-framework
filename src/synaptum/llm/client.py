from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class LLMResponse:
    """Simple response wrapper used by stub/test LLM clients."""
    content: str


class LLMClient(ABC):
    """
    El framework no sabe nada de llama-server.
    Tu adapter (fuera o dentro de tu repo) implementa esto usando tu SDK.
    """

    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        raise NotImplementedError("LLMClient es una interfaz abstracta. Implementa el método chat().")  
    