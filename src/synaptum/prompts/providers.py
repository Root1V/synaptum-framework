"""
SYN-28 · De dónde salen los prompts.

``PromptProvider`` es un ``Protocol``: una base de datos, un servicio remoto o un
fichero cumplen el mismo contrato sin heredar nada.

``PromptRegistry`` encadena varios **por prioridad, como un PATH**. Sirve para lo
que uno acaba necesitando siempre: los prompts de producción en un sitio, y unos
cuantos sobrescritos en local sin tocar el fichero compartido.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from ..core.errors import ConfigurationError
from .template import PromptTemplate

__all__ = [
    "PromptProvider",
    "InMemoryPrompts",
    "FilePrompts",
    "PromptRegistry",
]


@runtime_checkable
class PromptProvider(Protocol):
    """Cualquier fuente de prompts."""

    def get(self, name: str) -> PromptTemplate:
        """Raises: ``KeyError`` si no existe."""
        ...

    def exists(self, name: str) -> bool: ...


class InMemoryPrompts:
    """Prompts en un diccionario.  Para tests y para sobrescribir en local."""

    def __init__(self, prompts: Mapping[str, PromptTemplate | str] | None = None) -> None:
        self._prompts: dict[str, PromptTemplate] = {
            name: _coerce(value) for name, value in (prompts or {}).items()
        }

    def register(self, name: str, template: PromptTemplate | str) -> None:
        self._prompts[name] = _coerce(template)

    def get(self, name: str) -> PromptTemplate:
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' no está. Hay: {sorted(self._prompts)}.")
        return self._prompts[name]

    def exists(self, name: str) -> bool:
        return name in self._prompts


class FilePrompts:
    """Prompts en un fichero JSON o YAML.

    **JSON funciona sin dependencias.** YAML necesita PyYAML, y por eso no es el
    camino por defecto: en la v0.4 lo era, y arrastraba una dependencia para todo
    el mundo por una comodidad de sintaxis.

    Carga perezosa y cacheada. ``reload()`` la invalida, que es lo que hace
    utilizable editar prompts sin reiniciar.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cache: dict[str, PromptTemplate] | None = None

    def get(self, name: str) -> PromptTemplate:
        prompts = self._load()
        if name not in prompts:
            raise KeyError(
                f"Prompt '{name}' no está en {self.path}. Hay: {sorted(prompts)}."
            )
        return prompts[name]

    def exists(self, name: str) -> bool:
        return name in self._load()

    def reload(self) -> None:
        self._cache = None

    def _load(self) -> dict[str, PromptTemplate]:
        if self._cache is not None:
            return self._cache

        if not self.path.exists():
            raise ConfigurationError(f"No existe el fichero de prompts {self.path}.")

        suffix = self.path.suffix.lower()
        text = self.path.read_text(encoding="utf-8")

        if suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            data = _load_yaml(text, self.path)
        else:
            raise ConfigurationError(
                f"Formato no soportado: '{suffix}'. Usa .json (sin dependencias) "
                "o .yaml con el extra correspondiente."
            )

        self._cache = {name: _coerce(value) for name, value in (data or {}).items()}
        return self._cache


class PromptRegistry:
    """Encadena proveedores por prioridad: el primero que tenga el prompt, gana.

    El orden importa y es el de registro, como un PATH. Así se sobrescribe un
    prompt en local anteponiendo un proveedor, sin editar el fichero que
    comparte todo el mundo.
    """

    def __init__(self, *providers: PromptProvider) -> None:
        self._providers: list[PromptProvider] = list(providers)

    def add(self, provider: PromptProvider) -> "PromptRegistry":
        self._providers.append(provider)
        return self

    def prepend(self, provider: PromptProvider) -> "PromptRegistry":
        """Lo pone delante de todo: es cómo se sobrescribe sin borrar."""
        self._providers.insert(0, provider)
        return self

    def get(self, name: str) -> PromptTemplate:
        for provider in self._providers:
            if provider.exists(name):
                return provider.get(name)
        raise KeyError(
            f"Prompt '{name}' no está en ningún proveedor "
            f"({[type(p).__name__ for p in self._providers]})."
        )

    def exists(self, name: str) -> bool:
        return any(provider.exists(name) for provider in self._providers)


# ── Piezas ────────────────────────────────────────────────────────────────────

def _coerce(value: PromptTemplate | str | Mapping) -> PromptTemplate:
    """Acepta la forma corta —solo el texto— además de la completa."""
    if isinstance(value, PromptTemplate):
        return value
    if isinstance(value, str):
        return PromptTemplate(content=value)
    return PromptTemplate(
        content=value["content"],
        version=str(value.get("version", "1.0")),
        description=str(value.get("description", "")),
        variables=dict(value.get("variables", {})),
    )


def _load_yaml(text: str, path: Path):
    try:
        import yaml
    except ImportError as missing:
        raise ConfigurationError(
            f"{path} es YAML y PyYAML no está instalado. Usa JSON, que no "
            "necesita nada, o instala el extra de YAML."
        ) from missing
    return yaml.safe_load(text)
