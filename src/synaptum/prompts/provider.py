from __future__ import annotations

from abc import ABC, abstractmethod

from .template import PromptTemplate


class PromptProvider(ABC):
    """
    Interfaz para cualquier fuente de PromptTemplates.

    Implementaciones concretas pueden cargar prompts desde memoria,
    archivos YAML/JSON, bases de datos, o servicios remotos
    (ej. Langfuse, Langsmith, Promptlayer, AWS).
    """

    @abstractmethod
    def get(self, name: str) -> PromptTemplate:
        """
        Retorna el PromptTemplate registrado bajo `name`.

        Raises:
            KeyError: si el nombre no existe en este provider.
        """
        ...

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Retorna True si `name` está disponible en este provider."""
        ...
