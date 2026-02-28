from __future__ import annotations

import json
from pathlib import Path

from .provider import PromptProvider
from .template import PromptTemplate


class FilePromptProvider(PromptProvider):
    """
    PromptProvider que carga templates desde un archivo YAML o JSON.

    El archivo debe seguir la estructura:

    YAML::

        calculator.system:
          content: "Eres una calculadora. Suma los números que recibes."
          version: "1.0"
          description: "Calculadora aritmética"

    JSON::

        {
          "calculator.system": {
            "content": "Eres una calculadora. Suma los números que recibes.",
            "version": "1.0",
            "description": "Calculadora aritmética"
          }
        }

    Los prompts se cargan de forma lazy al primer acceso y se cachean.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cache: dict[str, PromptTemplate] | None = None

    # ------------------------------------------------------------------
    # PromptProvider interface
    # ------------------------------------------------------------------

    def get(self, name: str) -> PromptTemplate:
        prompts = self._load()
        if name not in prompts:
            raise KeyError(
                f"Prompt '{name}' no encontrado en {self._path}. "
                f"Prompts disponibles: {sorted(prompts)}"
            )
        return prompts[name]

    def exists(self, name: str) -> bool:
        return name in self._load()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, PromptTemplate]:
        if self._cache is not None:
            return self._cache

        suffix = self._path.suffix.lower()

        if suffix in {".yaml", ".yml"}:
            self._cache = self._load_yaml()
        elif suffix == ".json":
            self._cache = self._load_json()
        else:
            raise ValueError(
                f"Formato no soportado: '{suffix}'. Usa .yaml, .yml o .json"
            )

        return self._cache

    def _load_yaml(self) -> dict[str, PromptTemplate]:
        try:
            import yaml  # optional dependency
        except ImportError as exc:
            raise ImportError(
                "PyYAML es necesario para FilePromptProvider con YAML. "
                "Instálalo con: uv add pyyaml"
            ) from exc

        with self._path.open("r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}

        return self._parse(data)

    def _load_json(self) -> dict[str, PromptTemplate]:
        with self._path.open("r", encoding="utf-8") as f:
            data: dict = json.load(f)

        return self._parse(data)

    @staticmethod
    def _parse(data: dict) -> dict[str, PromptTemplate]:
        result: dict[str, PromptTemplate] = {}
        for name, raw in data.items():
            if isinstance(raw, str):
                # formato corto: solo content
                result[name] = PromptTemplate(content=raw)
            else:
                result[name] = PromptTemplate(
                    content=raw["content"],
                    version=raw.get("version", "1.0"),
                    description=raw.get("description", ""),
                    variables=raw.get("variables", {}),
                )
        return result

    def reload(self) -> None:
        """Invalida el cache y fuerza una recarga desde disco."""
        self._cache = None
