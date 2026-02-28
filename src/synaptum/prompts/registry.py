from __future__ import annotations

from .provider import PromptProvider
from .template import PromptTemplate
from .in_memory import InMemoryPromptProvider


class PromptRegistry(PromptProvider):
    """
    Registro central de prompts para el runtime.

    Agrega uno o más PromptProviders con orden de prioridad:
    el primero registrado tiene la máxima prioridad (similar a PATH).
    Si ningún provider tiene el prompt solicitado, lanza KeyError.

    Usage::

        registry = PromptRegistry()
        registry.add_provider(FilePromptProvider("prompts/agents.yaml"))
        registry.add_provider(InMemoryPromptProvider({...}))  # fallback

        template = registry.get("calculator.system")

    El AgentRuntime puede mantener un PromptRegistry global que se inyecta
    automáticamente a los agentes en el momento de registro.
    """

    def __init__(self) -> None:
        self._providers: list[PromptProvider] = []

    def add_provider(self, provider: PromptProvider) -> "PromptRegistry":
        """Agrega un provider al final de la cadena de resolución."""
        self._providers.append(provider)
        return self  # fluent API

    def get(self, name: str) -> PromptTemplate:
        """
        Busca `name` en cada provider en orden de registro.

        Raises:
            KeyError: si ningún provider tiene el prompt.
        """
        for provider in self._providers:
            if provider.exists(name):
                return provider.get(name)

        providers_info = [type(p).__name__ for p in self._providers]
        raise KeyError(
            f"Prompt '{name}' no encontrado en ningún provider. "
            f"Providers registrados: {providers_info}"
        )

    def exists(self, name: str) -> bool:
        return any(p.exists(name) for p in self._providers)

    def register(self, name: str, template: PromptTemplate) -> None:
        """
        Atajo para registrar un prompt directamente en el registry,
        sin necesidad de crear un InMemoryPromptProvider explícito.
        Crea uno interno si no existe.
        """

        for provider in self._providers:
            if isinstance(provider, InMemoryPromptProvider):
                provider.register(name, template)
                return

        # Si no hay ninguno, crea el primero y lo agrega al inicio
        memory_provider = InMemoryPromptProvider({name: template})
        self._providers.insert(0, memory_provider)
