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
import inspect
import sys
from typing import Any, Callable, Mapping

from ..core.errors import InvalidToolCallError, ToolExecutionError
from ..core.types import Risk, Text, ToolDefinition, ToolResult
from ..schema.derive import json_schema_for

__all__ = ["Tool", "tool", "json_schema_for"]


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
