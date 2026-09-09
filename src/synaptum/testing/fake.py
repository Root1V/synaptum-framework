"""
SYN-65 · ``FakeGateway`` — la vía principal de desarrollo, no una utilidad de test.

Un desarrollador no tiene acceso a una instancia de Prometheus, y la única
puerta a inferencia local es el SDK de Axonium.  De ahí sale una consecuencia
que hay que asumir de frente: **en modo autónomo Synaptum no tiene inferencia
local.**  O proveedores cloud de pago, o modelos simulados.

Eso convierte esto en infraestructura, no en un stub de veinte líneas.  Tiene
que poder ejercitar todo lo que el bucle sabe hacer — tool calls, streaming con
cancelación, las tres disposiciones de denegación, la taxonomía de errores y el
``Usage`` de tres estados — o habrá caminos del framework que nadie puede
recorrer sin gastar dinero.

Uso::

    from synaptum.testing import FakeGateway, calls, says

    gateway = FakeGateway(
        calls("leer", path="/x"),
        says("el fichero dice hola"),
        tools=[leer],
    )

    async for step in agent.run("lee /x", session=Session("run-1", gateway)):
        ...
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from ..core.errors import Denied
from ..core.events import Decision
from ..core.protocols import SEAM_VERSION, CallContext, Hello, Welcome, negotiate
from ..core.types import (
    Finish,
    FinishReason,
    Message,
    Request,
    Response,
    Risk,
    Role,
    StreamEvent,
    StreamStart,
    Text,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    Usage,
)
from ..tools.decorator import Tool

__all__ = ["FakeGateway", "says", "calls", "DEFAULT_USAGE"]


DEFAULT_USAGE = Usage(input=100, output=20, reasoning=0, cache_read=0, cache_write=0)
"""Consumo por defecto: **medido y en cero** donde corresponde, no ``None``.

Un fake que devolviera ``None`` en todo haría creer que la economía de contexto
no funciona.  Para simular una fuente que no reporta —Prometheus con llama.cpp,
que no expone ``reasoning`` ni ``cache_write``— pásalo explícitamente.
"""


def says(text: str, *, usage: Usage | None = None) -> Response:
    """Respuesta de texto que cierra el turno."""
    return Response(
        message=Message.assistant(text),
        finish_reason=FinishReason.STOP,
        usage=usage or DEFAULT_USAGE,
        model="fake",
    )


def calls(name: str, *, id: str = "call-1", usage: Usage | None = None, **arguments: Any) -> Response:
    """Respuesta que pide una herramienta."""
    return Response(
        message=Message(Role.ASSISTANT, (ToolCall(id=id, name=name, arguments=arguments),)),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=usage or DEFAULT_USAGE,
        model="fake",
    )


ScriptItem = Response | BaseException | Decision | str | Callable[[Request], Any]


class FakeGateway:
    """``Gateway`` guionizado.  Ejecuta de verdad las tools que se le registran.

    El guion se consume en orden.  Cada elemento puede ser:

    * ``Response`` — se devuelve tal cual.
    * ``str`` — atajo de ``says(...)``.
    * ``BaseException`` — se lanza.  Con un ``ProviderError`` se ejercita el
      camino de reintentos.
    * ``Decision`` — se lanza como ``Denied``.  Es como se prueban las tres
      disposiciones sin montar un motor de políticas.
    * ``callable(request)`` — se llama y se trata su retorno como lo anterior.
      Sirve para responder según lo que el bucle acabe de enviar.
    """

    def __init__(
        self,
        *script: ScriptItem,
        tools: Sequence[Tool] = (),
        chunk_size: int = 8,
        deny_tools: Mapping[str, Decision] | None = None,
    ) -> None:
        self.script: list[ScriptItem] = list(script)
        self.tools = {t.name: t for t in tools}
        self.chunk_size = chunk_size
        self.deny_tools = dict(deny_tools or {})

        self.model_calls = 0
        self.tool_calls = 0
        self.requests: list[Request] = []
        self.chunks_emitted = 0
        """Cuántos fragmentos llegó a producir el último stream.

        Es el testigo de la cancelación: si el consumidor cierra el iterador a
        mitad, este número se queda muy por debajo del total — igual que el
        `upstream stopped after N of M chunks` que mide el harness.
        """
        self.cancelled = False

    # ── Costura ───────────────────────────────────────────────────────────────

    async def handshake(self, hello: Hello) -> Welcome:
        return Welcome(
            version=negotiate(hello.versions, current=SEAM_VERSION),
            tool_refs={t.name: f"{t.name}@fake" for t in hello.tools},
            session_id="fake-session",
        )

    async def invoke_model(self, request: Request, ctx: CallContext) -> Response:
        self.model_calls += 1
        self.requests.append(request)
        return self._next(request)

    async def stream_model(self, request: Request, ctx: CallContext) -> AsyncIterator[StreamEvent]:
        """Emite la respuesta troceada, honrando la cancelación.

        Cerrar el iterador —``aclose()``, salir de un ``async with``, o cancelar
        la tarea que lo consume— detiene la emisión.  Es el mismo contrato que
        el gateway real debe cumplir contra el proveedor: si la generación
        siguiera viva tras cerrar el canal, se seguiría pagando.
        """
        self.model_calls += 1
        self.requests.append(request)
        self.chunks_emitted = 0
        self.cancelled = False

        response = self._next(request)
        yield StreamStart(model=response.model)

        try:
            for index, part in enumerate(response.message.content):
                if isinstance(part, Text):
                    yield TextStart(index=index)
                    for piece in _slice(part.text, self.chunk_size):
                        yield TextDelta(text=piece, index=index)
                        self.chunks_emitted += 1
                        await asyncio.sleep(0)
                    yield TextEnd(index=index)
                elif isinstance(part, ToolCall):
                    yield ToolCallStart(id=part.id, name=part.name, index=index)
                    from ..core.types import dumps

                    for piece in _slice(dumps(part.arguments), self.chunk_size):
                        yield ToolCallDelta(arguments_delta=piece, index=index)
                        self.chunks_emitted += 1
                        await asyncio.sleep(0)
                    yield ToolCallEnd(index=index)

            yield Finish(response=response)
        except GeneratorExit:
            # El consumidor cerró el iterador: dejamos de generar.
            self.cancelled = True
            raise

    async def invoke_tool(
        self,
        call: ToolCall,
        ctx: CallContext,
        *,
        risk: Risk = Risk.READ,
        tool_ref: str | None = None,
    ) -> ToolResult:
        self.tool_calls += 1

        denial = self.deny_tools.get(call.name)
        if denial is not None:
            raise Denied(denial)

        target = self.tools.get(call.name)
        if target is None:
            return ToolResult.of(
                call.id,
                f"No existe la herramienta '{call.name}'. Disponibles: "
                f"{sorted(self.tools) or 'ninguna'}.",
                is_error=True,
            )
        return await target.invoke(call.id, call.arguments)

    # ── Guion ─────────────────────────────────────────────────────────────────

    def _next(self, request: Request) -> Response:
        if not self.script:
            raise AssertionError(
                "El guion del FakeGateway se agotó: el bucle pidió una respuesta "
                "más de las previstas. Añade otro elemento o revisa por qué no "
                "terminó."
            )

        item = self.script.pop(0)
        if callable(item) and not isinstance(item, BaseException):
            item = item(request)
        if isinstance(item, str):
            return says(item)
        if isinstance(item, Decision):
            raise Denied(item)
        if isinstance(item, BaseException):
            raise item
        return item


def _slice(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]
