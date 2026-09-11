"""
SYN-16 · El protocolo ``Schema`` — salida estructurada sin atarse a una librería.

Un agente que debe producir un objeto concreto necesita dos cosas: **el JSON
Schema** que se le manda al proveedor, y **la validación** de lo que vuelve.
Nada de eso obliga a elegir una librería de modelos.

``Schema`` es un ``typing.Protocol``, así que quien lo implemente no hereda ni
importa nada de aquí. Y ``schema_for`` acepta directamente lo que la gente ya
tiene a mano — un modelo de Pydantic, un dataclass, o un JSON Schema escrito a
mano — sin que el bucle sepa cuál de los tres recibió.

Por qué no basta con «usa Pydantic»
------------------------------------
Pydantic es excelente y es la elección por defecto de casi todo el mundo, pero
convertirlo en dependencia dura significa que **el núcleo deja de tener cero
dependencias** por una funcionalidad que no todos usan. Con este protocolo,
Pydantic es un extra: quien lo tiene lo usa, y quien no, usa dataclasses de la
librería estándar y obtiene lo mismo.

El coste honesto: un JSON Schema pasado a mano **no se valida**. No hay
validador en la librería estándar y no vamos a traer uno para eso. Se dice en
voz alta en vez de dejar creer que sí.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Protocol, runtime_checkable

from ..core.errors import ConfigurationError, NoObjectGeneratedError
from .derive import schema_of_dataclass

__all__ = ["Schema", "schema_for", "DataclassSchema", "PydanticSchema", "RawSchema"]


@runtime_checkable
class Schema(Protocol):
    """Un tipo de salida: su esquema y cómo validar lo que vuelve."""

    def json_schema(self) -> Mapping[str, Any]:
        """El JSON Schema que viaja al proveedor."""
        ...

    def validate(self, data: Any) -> Any:
        """Convierte datos ya parseados en el objeto tipado.

        Raises:
            NoObjectGeneratedError: si los datos no encajan.  Es reintentable
                a propósito: el muestreo es estocástico y una segunda pasada
                suele acertar.
        """
        ...

    def dump(self, obj: Any) -> Any:
        """Devuelve el objeto como datos JSON, para el journal.

        Lo implementa el adaptador y no el núcleo porque **quien construyó el
        objeto es quien sabe desmontarlo**.  Meter aquí un caso especial por
        cada librería de modelos convertiría el núcleo en el sitio que tiene que
        conocerlas todas, que es justo lo que este protocolo evita.
        """
        ...


# ── Adaptadores ───────────────────────────────────────────────────────────────

class DataclassSchema:
    """Salida estructurada con la librería estándar y nada más.

    La validación reutiliza el mismo decodificador que reconstruye eventos del
    journal: uno solo que sepa convertir datos JSON en objetos tipados, no dos
    que acaben discrepando en los bordes.
    """

    def __init__(self, target: type) -> None:
        self.target = target
        self._schema = schema_of_dataclass(target, where=target.__name__)

    @property
    def name(self) -> str:
        return self.target.__name__

    def json_schema(self) -> Mapping[str, Any]:
        return self._schema

    def validate(self, data: Any) -> Any:
        from ..core.codec import decode

        try:
            return decode(self.target, data)
        except (TypeError, ValueError, KeyError) as mismatch:
            raise NoObjectGeneratedError(
                f"La salida no encaja en {self.target.__name__}: {mismatch}",
                raw=repr(data),
            ) from mismatch

    def dump(self, obj: Any) -> Any:
        from ..core.types import to_jsonable

        return to_jsonable(obj)


class PydanticSchema:
    """Salida estructurada con Pydantic, cuando está instalado."""

    def __init__(self, target: type) -> None:
        self.target = target

    @property
    def name(self) -> str:
        return self.target.__name__

    def json_schema(self) -> Mapping[str, Any]:
        return self.target.model_json_schema()  # type: ignore[attr-defined]

    def validate(self, data: Any) -> Any:
        from pydantic import ValidationError

        try:
            return self.target.model_validate(data)  # type: ignore[attr-defined]
        except ValidationError as mismatch:
            raise NoObjectGeneratedError(
                f"La salida no encaja en {self.target.__name__}: {mismatch}",
                raw=repr(data),
            ) from mismatch

    def dump(self, obj: Any) -> Any:
        return obj.model_dump(mode="json")


class RawSchema:
    """Un JSON Schema escrito a mano.

    **No valida.** La librería estándar no trae validador y no vamos a arrastrar
    uno para este caso; lo que vuelve se entrega tal cual. Es la opción para
    esquemas que no se pueden expresar como tipo, y el precio de esa libertad es
    que la comprobación queda de tu lado.
    """

    def __init__(self, schema: Mapping[str, Any], *, name: str = "output") -> None:
        self._schema = dict(schema)
        self.name = name

    def json_schema(self) -> Mapping[str, Any]:
        return self._schema

    def validate(self, data: Any) -> Any:
        return data

    def dump(self, obj: Any) -> Any:
        return obj


# ── Resolución ────────────────────────────────────────────────────────────────

def schema_for(spec: Any) -> Schema:
    """Convierte lo que el usuario tenga a mano en un ``Schema``.

    Acepta, por este orden: algo que ya cumple el protocolo, un modelo de
    Pydantic, un dataclass, o un JSON Schema como diccionario.
    """
    if isinstance(spec, Schema):
        return spec

    if isinstance(spec, Mapping):
        return RawSchema(spec)

    if isinstance(spec, type):
        if _is_pydantic_model(spec):
            return PydanticSchema(spec)
        if dataclasses.is_dataclass(spec):
            return DataclassSchema(spec)

    raise ConfigurationError(
        f"No sé usar {spec!r} como tipo de salida. Pasa un dataclass, un modelo "
        "de Pydantic, un JSON Schema como dict, o algo que implemente json_schema(), "
        "validate() y dump()."
    )


def _is_pydantic_model(target: type) -> bool:
    """Detecta un modelo de Pydantic **sin importar Pydantic**.

    Importarlo aquí lo convertiría en dependencia dura por el mero hecho de
    preguntar, que es justo lo que este módulo existe para evitar.
    """
    return hasattr(target, "model_validate") and hasattr(target, "model_json_schema")
