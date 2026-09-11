"""
SYN-16 · Derivación de JSON Schema desde tipos de Python.

Una sola implementación, usada por dos sitios que la necesitan igual: el
decorador ``@tool``, que deriva el esquema de los parámetros de una función, y
la salida estructurada, que lo deriva del tipo que el agente debe producir.

Tenerla duplicada habría garantizado que las dos versiones divergieran, y que la
divergencia apareciera como un modelo mandando argumentos que la función no
acepta — lejos del cambio que la causó.

Un tipo que no sabemos traducir **falla al derivar**, no al usar: romper el
arranque es mejor que romper una llamada ya pagada.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import types
import typing
from typing import Any, Callable, Mapping, get_args, get_origin

from ..core.errors import ConfigurationError

__all__ = ["json_schema_for", "schema_of_type", "hints_of"]


_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def schema_of_type(
    annotation: Any, *, where: str, localns: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], str | None]:
    """Traduce una anotación a JSON Schema.  Devuelve ``(esquema, descripción)``."""
    origin = get_origin(annotation)

    # Annotated[T, "descripción", ...] — la vía estándar para documentar un campo.
    if origin is typing.Annotated:
        base, *meta = get_args(annotation)
        schema, inherited = schema_of_type(base, where=where, localns=localns)
        description = next((m for m in meta if isinstance(m, str)), inherited)
        return schema, description

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}, None

    if annotation is type(None):
        return {"type": "null"}, None

    if annotation is Any:
        return {}, None

    if origin is typing.Literal:
        return {"enum": list(get_args(annotation))}, None

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"enum": [member.value for member in annotation]}, None

    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        items, _ = schema_of_type(args[0], where=where, localns=localns) if args else ({}, None)
        return {"type": "array", "items": items}, None

    if origin is dict:
        args = get_args(annotation)
        values, _ = schema_of_type(args[1], where=where, localns=localns) if len(args) == 2 else ({}, None)
        return {"type": "object", "additionalProperties": values}, None

    if origin in (typing.Union, types.UnionType):
        options = [schema_of_type(a, where=where, localns=localns)[0] for a in get_args(annotation)]
        return {"anyOf": options}, None

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return schema_of_dataclass(annotation, where=where, localns=localns), None

    raise ConfigurationError(
        f"No sé traducir el tipo {annotation!r} de '{where}' a JSON Schema. "
        "Usa un primitivo, Literal, Enum, list, dict, unión, o un dataclass — o "
        "declara el esquema a mano con parameters=."
    )


def schema_of_dataclass(
    klass: type, *, where: str, localns: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    hints = hints_of(klass, localns)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for f in dataclasses.fields(klass):
        schema, description = schema_of_type(hints.get(f.name, Any), where=f"{where}.{f.name}", localns=localns)
        if description:
            schema["description"] = description
        properties[f.name] = schema
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if not has_default:
            required.append(f.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def hints_of(target: Any, localns: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resuelve anotaciones, tolerando tipos declarados dentro de una función.

    Con ``from __future__ import annotations`` las anotaciones son cadenas que
    se evalúan en el espacio del módulo — donde una clase definida dentro de una
    función no existe.  Reintentamos con el ámbito local del sitio donde se
    aplicó el decorador antes de rendirnos, porque declarar un dataclass junto a
    la tool que lo usa es lo natural y no debería castigarse.
    """
    try:
        return typing.get_type_hints(target, include_extras=True)
    except NameError:
        if not localns:
            raise
        return typing.get_type_hints(target, localns=dict(localns), include_extras=True)


def json_schema_for(
    fn: Callable[..., Any], *, localns: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Deriva el JSON Schema de los parámetros de una función tipada."""
    signature = inspect.signature(fn)
    hints = hints_of(fn, localns)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            raise ConfigurationError(
                f"'{fn.__name__}' usa *args o **kwargs. Una tool necesita parámetros "
                "nombrados: el modelo solo sabe rellenar un esquema."
            )
        if name not in hints:
            raise ConfigurationError(
                f"El parámetro '{name}' de '{fn.__name__}' no tiene anotación de tipo. "
                "Sin ella no hay esquema, y sin esquema el modelo adivina."
            )

        schema, description = schema_of_type(hints[name], where=f"{fn.__name__}.{name}", localns=localns)
        if description:
            schema["description"] = description
        if parameter.default is not inspect.Parameter.empty:
            if isinstance(parameter.default, (str, int, float, bool, type(None))):
                schema["default"] = parameter.default
        else:
            required.append(name)
        properties[name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
