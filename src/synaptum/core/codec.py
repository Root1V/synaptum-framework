"""
Decodificación de eventos — la vuelta de ``dumps``.

Un journal que solo sabe escribir no sirve para reanudar: si el proceso muere,
lo que hay en disco tiene que poder volver a ser un evento tipado.  Sin esto,
``Checkpointer.load`` no puede existir fuera de memoria.

La reconstrucción se apoya en el discriminante ``kind`` que ya llevan las
uniones etiquetadas.  El registro de tipos **se deriva de la propia unión**, no
se mantiene a mano: añadir una parte de contenido o un tipo de evento no obliga
a acordarse de registrarlo en un sitio aparte, que es exactamente el paso que
todo el mundo olvida.

Los campos ausentes se dejan en su valor por defecto.  Es lo que hace que el
formato sea compatible hacia atrás: un journal escrito por una versión anterior
—que no conocía un campo nuevo— sigue siendo legible.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any, Mapping, get_args, get_origin

from .events import ApprovalStep, DelegateStep, Event, FinalStep, ModelStep, ToolStep

__all__ = ["decode", "decode_event", "registry_of"]


_UNIONS: dict[Any, dict[str, type]] = {}


def registry_of(union: Any) -> dict[str, type]:
    """Construye ``{valor de kind: clase}`` a partir de una unión etiquetada.

    Memoizado: la tabla se deriva una vez por unión, no en cada evento leído.
    """
    cached = _UNIONS.get(union)
    if cached is not None:
        return cached

    table: dict[str, type] = {}
    for member in get_args(union):
        tag = _tag_of(member)
        if tag is not None:
            table[tag] = member
    _UNIONS[union] = table
    return table


def _tag_of(klass: type) -> str | None:
    """Extrae el valor de ``kind``, sea un ``Literal`` o un default de campo."""
    if not dataclasses.is_dataclass(klass):
        return None
    for field in dataclasses.fields(klass):
        if field.name != "kind":
            continue
        if isinstance(field.default, str):
            return field.default
        hint = typing.get_type_hints(klass).get("kind")
        if get_origin(hint) is typing.Literal:
            return str(get_args(hint)[0])
    return None


_EVENT: dict[str, type] = {
    "model": ModelStep,
    "tool": ToolStep,
    "delegate": DelegateStep,
    "approval": ApprovalStep,
    "final": FinalStep,
}


def decode(annotation: Any, value: Any) -> Any:
    """Reconstruye un valor tipado a partir de su forma JSON."""
    if value is None:
        return None

    origin = get_origin(annotation)

    if annotation is Any or annotation is None:
        return value

    if origin is typing.Annotated:
        return decode(get_args(annotation)[0], value)

    if origin in (typing.Union, types.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        # Unión etiquetada: el discriminante decide.
        if isinstance(value, Mapping) and "kind" in value:
            table = registry_of(annotation)
            target = table.get(str(value["kind"]))
            if target is not None:
                return _decode_dataclass(target, value)
        # Unión simple (por ejemplo ``int | None``): el primer tipo que encaje.
        for option in options:
            try:
                return decode(option, value)
            except (TypeError, ValueError, KeyError):
                continue
        return value

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation(value)

    if origin in (tuple, list, set, frozenset):
        args = get_args(annotation)
        inner = args[0] if args else Any
        decoded = [decode(inner, item) for item in value]
        return tuple(decoded) if origin is tuple else origin(decoded)

    if origin in (dict, Mapping) or annotation is Mapping:
        return dict(value)

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _decode_dataclass(annotation, value)

    return value


def _decode_dataclass(klass: type, payload: Mapping[str, Any]) -> Any:
    hints = typing.get_type_hints(klass)
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(klass):
        if field.name not in payload:
            # Ausente: se queda con su default.  Es lo que hace legible un
            # journal escrito por una versión que no conocía este campo.
            continue
        kwargs[field.name] = decode(hints.get(field.name, Any), payload[field.name])
    return klass(**kwargs)


def decode_event(payload: Mapping[str, Any]) -> Event:
    """Reconstruye un evento del bucle desde su forma JSON.

    Raises:
        ValueError: si el discriminante no corresponde a ningún tipo conocido.
            Fallar aquí es mejor que devolver un evento a medias que el replay
            interpretará como un paso sin hacer.
    """
    kind = str(payload.get("kind", ""))
    target = _EVENT.get(kind)
    if target is None:
        raise ValueError(
            f"Evento desconocido en el journal: kind={kind!r}. "
            f"Conocidos: {sorted(_EVENT)}."
        )
    return _decode_dataclass(target, payload)
