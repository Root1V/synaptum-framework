"""
SYN-28 · Prompts como configuración versionada, no como literales en el código.

Es la pieza que mejor envejeció del diseño anterior y vuelve casi intacta. La
idea sigue siendo la misma: un prompt es **contenido con versión**, no una
cadena incrustada en una función.

Lo que cambia respecto a la v0.4
---------------------------------
* ``PromptProvider`` pasa de clase base abstracta a ``typing.Protocol``, como
  todo lo demás: quien lo implemente no hereda ni importa nada.
* El cargador de ficheros lee **JSON sin dependencias** y YAML si hay PyYAML.
  Antes el YAML era el camino principal y arrastraba una dependencia para todo
  el mundo.
* Una variable sin valor **falla al renderizar**, en vez de dejar la llave en el
  prompt. Un ``{cliente}`` literal llegando al modelo no produce un error: da
  una respuesta peor y nadie se entera.

La disciplina que se conserva, y es lo que hacía valer esto: **todo el texto
estructural vive en el fichero de prompts; el código solo serializa datos.**
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.errors import ConfigurationError

__all__ = ["PromptTemplate", "fmt_dict", "fmt_list", "fmt_records"]


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Un prompt con su versión.

    ``version`` no es decoración: cuando una respuesta sale mal en producción, la
    primera pregunta es con qué prompt se generó, y un literal en el código no
    sabe contestarla.
    """

    content: str
    version: str = "1.0"
    description: str = ""
    variables: Mapping[str, Any] = field(default_factory=dict)

    def render(self, **values: Any) -> str:
        """Interpola las variables.  Los argumentos ganan sobre los del template.

        Raises:
            ConfigurationError: si falta alguna variable.  Dejar la llave sin
                sustituir no falla — llega al modelo como texto literal y
                produce una respuesta peor sin que nadie se entere.
        """
        context = {**self.variables, **values}
        try:
            return self.content.format_map(_Strict(context))
        except KeyError as missing:
            esperadas = sorted(self.placeholders)
            raise ConfigurationError(
                f"Falta la variable {missing} al renderizar el prompt "
                f"(versión {self.version}). Espera: {esperadas}."
            ) from missing

    @property
    def placeholders(self) -> set[str]:
        """Nombres de las variables que el template usa."""
        return {
            name
            for _, name, _, _ in string.Formatter().parse(self.content)
            if name
        }

    def __str__(self) -> str:
        return self.content


class _Strict(dict):
    """Diccionario que se niega a inventar un valor ausente."""

    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


# ── Serialización de datos para meter en un prompt ────────────────────────────
#
# Producen **datos y nada más**: sin encabezados, sin viñetas, sin texto de
# encuadre.  Todo eso pertenece al fichero de prompts, porque es donde se puede
# cambiar sin tocar código — que es el único motivo por el que los prompts viven
# fuera.

def fmt_dict(data: Mapping[str, Any], *, max_value: int = 500) -> str:
    """Serializa un diccionario como líneas ``clave: valor``."""
    from ..core.types import dumps

    lines = []
    for key, value in data.items():
        rendered = dumps(value)[:max_value] if isinstance(value, (dict, list)) else str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def fmt_list(items: list[Any], *, prefix: str = "· ") -> str:
    """Serializa una lista como líneas con prefijo."""
    return "\n".join(f"{prefix}{item}" for item in items)


def fmt_records(items: list[Mapping[str, Any]], template: str) -> str:
    """Serializa una lista de diccionarios con una plantilla por línea."""
    return "\n".join(template.format_map(dict(item)) for item in items)
