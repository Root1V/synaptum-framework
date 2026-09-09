"""
SYN-20 · SYN-21 · ``@tool`` — el esquema sale de la firma.

Una herramienta se declara una sola vez.  El nombre, la descripción y el JSON
Schema salen de la función tipada; el riesgo y la idempotencia se declaran en el
decorador porque no se pueden deducir de una firma::

    @tool(risk=Risk.DESTRUCTIVE)
    async def borrar_fichero(path: Annotated[str, "Ruta absoluta"]) -> str:
        "Borra un fichero del disco."
        ...

Por qué derivar y no escribir el esquema a mano
------------------------------------------------
Un esquema escrito aparte **se desincroniza**, y cuando lo hace el modelo manda
argumentos que la función no acepta.  El fallo aparece en tiempo de ejecución,
lejos del cambio que lo causó, y con una llamada al modelo ya pagada.
Derivándolo, cambiar la firma cambia el esquema.

Un tipo que no sabemos traducir **falla al decorar**, no al invocar: es la
diferencia entre romper el arranque y romper una llamada en producción.

Qué se declara y qué se deduce
-------------------------------
``risk`` e ``idempotent`` no están en la firma porque no son propiedades del
tipo, sino del **efecto**.  Ninguna anotación puede saber que una función que
devuelve ``str`` mueve dinero.  Y ambos tienen defaults deliberadamente
asimétricos: permisivo en riesgo, conservador en durabilidad.  Quien calla
obtiene la clase de riesgo más inocua y la garantía más cara.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import sys
import types
import typing
from typing import Any, Callable, Mapping, get_args, get_origin

from ..core.errors import ConfigurationError, InvalidToolCallError, ToolExecutionError
from ..core.types import Risk, Text, ToolDefinition, ToolResult

__all__ = ["Tool", "tool", "json_schema_for"]


# ── Derivación de esquema ─────────────────────────────────────────────────────

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _schema_of(
    annotation: Any, *, where: str, localns: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], str | None]:
    """Traduce una anotación a JSON Schema.  Devuelve ``(esquema, descripción)``."""
    origin = get_origin(annotation)

    # Annotated[T, "descripción", ...] — la vía estándar para documentar un campo.
    if origin is typing.Annotated:
        base, *meta = get_args(annotation)
        schema, inherited = _schema_of(base, where=where, localns=localns)
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
        items, _ = _schema_of(args[0], where=where, localns=localns) if args else ({}, None)
        return {"type": "array", "items": items}, None

    if origin is dict:
        args = get_args(annotation)
        values, _ = _schema_of(args[1], where=where, localns=localns) if len(args) == 2 else ({}, None)
        return {"type": "object", "additionalProperties": values}, None

    if origin in (typing.Union, types.UnionType):
        options = [_schema_of(a, where=where, localns=localns)[0] for a in get_args(annotation)]
        return {"anyOf": options}, None

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _schema_of_dataclass(annotation, where=where, localns=localns), None

    raise ConfigurationError(
        f"No sé traducir el tipo {annotation!r} de '{where}' a JSON Schema. "
        "Usa un primitivo, Literal, Enum, list, dict, unión, o un dataclass — o "
        "declara el esquema a mano con parameters=."
    )


def _schema_of_dataclass(
    klass: type, *, where: str, localns: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    hints = _hints_of(klass, localns)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for f in dataclasses.fields(klass):
        schema, description = _schema_of(hints.get(f.name, Any), where=f"{where}.{f.name}", localns=localns)
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


def _hints_of(target: Any, localns: Mapping[str, Any] | None) -> dict[str, Any]:
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
    hints = _hints_of(fn, localns)

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

        schema, description = _schema_of(hints[name], where=f"{fn.__name__}.{name}", localns=localns)
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


# ── La herramienta ────────────────────────────────────────────────────────────

class Tool:
    """Una función con su contrato.

    Sigue siendo invocable con normalidad — el decorador no la esconde — y
    además expone ``definition``, que es lo que viaja en el ``Hello`` y lo que
    el modelo ve.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        risk: Risk = Risk.READ,
        idempotent: bool = False,
        parameters: Mapping[str, Any] | None = None,
        localns: Mapping[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(fn)
        doc = inspect.getdoc(fn) or ""
        self.definition = ToolDefinition(
            name=name or fn.__name__,
            description=description or doc.split("\n\n")[0].strip(),
            parameters=parameters if parameters is not None else json_schema_for(fn, localns=localns),
            risk=risk,
            idempotent=idempotent,
        )

    @property
    def name(self) -> str:
        return self.definition.name

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Tool({self.name!r}, risk={self.definition.risk.value})"

    async def invoke(self, call_id: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Ejecuta y envuelve el resultado.

        Un fallo de la función **no se propaga**: vuelve como ``ToolResult`` con
        ``is_error``, porque un modelo que no ve el error no puede corregirlo.
        Lo que sí se propaga es una llamada mal formada — argumentos que no
        encajan en la firma no son algo que el modelo pueda arreglar leyendo un
        mensaje, son un desajuste entre el esquema y la función.
        """
        try:
            bound = inspect.signature(self.fn).bind(**arguments)
        except TypeError as mismatch:
            raise InvalidToolCallError(
                f"Argumentos inválidos para '{self.name}': {mismatch}",
                tool=self.name,
                call_id=call_id,
            ) from mismatch

        try:
            outcome = self.fn(*bound.args, **bound.kwargs)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except ToolExecutionError as failure:
            return ToolResult.of(call_id, str(failure), is_error=True)
        except Exception as failure:  # noqa: BLE001 — la evidencia vuelve al modelo
            return ToolResult.of(
                call_id, f"{type(failure).__name__}: {failure}", is_error=True
            )

        if isinstance(outcome, ToolResult):
            return outcome
        if isinstance(outcome, str):
            return ToolResult.of(call_id, outcome)
        return ToolResult(call_id=call_id, content=(Text(_render(outcome)),))


def _render(value: Any) -> str:
    """Serializa un resultado para que lo lea el modelo.

    Compacto y determinista: un resultado que cambia de forma entre llamadas
    reescribe el prefijo y tira la caché del proveedor.
    """
    from ..core.types import dumps

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dumps(value)
    if isinstance(value, (Mapping, list, tuple, int, float, bool)) or value is None:
        return dumps(value)
    return str(value)


# ── Decorador ─────────────────────────────────────────────────────────────────

def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: Risk = Risk.READ,
    idempotent: bool = False,
    parameters: Mapping[str, Any] | None = None,
) -> Any:
    """Convierte una función tipada en una herramienta.

    Se puede usar con o sin paréntesis::

        @tool
        def buscar(q: str) -> str: ...

        @tool(risk=Risk.HARD_WRITE, idempotent=False)
        async def transferir(cuenta: str, importe: float) -> str: ...

    Args:
        name: por defecto, el nombre de la función.
        description: por defecto, el primer párrafo del docstring.
        risk: clase de riesgo del efecto.  Synaptum declara; el harness decide.
        idempotent: si el efecto puede repetirse sin consecuencias.  Determina
            si el replay puede reintentar el paso tras una caída, y si su
            registro en el journal puede diferirse.
        parameters: esquema a mano, para el caso raro en el que la firma no
            baste.  Salta la derivación por completo.
    """

    # El ámbito donde se aplicó el decorador, para resolver tipos declarados
    # dentro de una función.  Se captura aquí y no dentro de `wrap` porque la
    # profundidad de pila difiere entre `@tool` y `@tool(...)`: en cambio, a
    # `tool` siempre se le llama desde el sitio de la decoración.
    scope = sys._getframe(1).f_locals

    def wrap(target: Callable[..., Any]) -> Tool:
        return Tool(
            target,
            name=name,
            description=description,
            risk=risk,
            idempotent=idempotent,
            parameters=parameters,
            localns=scope,
        )

    return wrap if fn is None else wrap(fn)
