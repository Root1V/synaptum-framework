from __future__ import annotations

from .provider import PromptProvider
from .template import PromptTemplate


class InMemoryPromptProvider(PromptProvider):
    """
    PromptProvider respaldado por un diccionario en memoria.

    Ideal para tests, desarrollo local y demos.

    Example:
        provider = InMemoryPromptProvider({
            "calculator.system": PromptTemplate(
                content="Eres una calculadora. Suma los números que recibes.",
                description="Calculadora aritmética simple",
            ),
        })
    """

    def __init__(self, prompts: dict[str, PromptTemplate] | None = None) -> None:
        self._prompts: dict[str, PromptTemplate] = dict(prompts or {})

    def register(self, name: str, template: PromptTemplate) -> None:
        """Registra o sobreescribe un template por nombre."""
        self._prompts[name] = template

    def get(self, name: str) -> PromptTemplate:
        if name not in self._prompts:
            raise KeyError(
                f"Prompt '{name}' no encontrado en InMemoryPromptProvider. "
                f"Prompts disponibles: {sorted(self._prompts)}"
            )
        return self._prompts[name]

    def exists(self, name: str) -> bool:
        return name in self._prompts
