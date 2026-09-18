"""
SYN-26 · ``LocalGateway`` — la costura de aplicación, en modo permisivo.

Es lo que hace que Synaptum funcione **sin harness**: un cuaderno, un test, un
script. Ejecuta contra las credenciales del entorno y registra cada decisión que
un gateway real habría tomado.

Esto no aplica nada, y merece decirse sin rodeos
-------------------------------------------------
Una comprobación que corre **dentro del proceso gobernado** es advisoria: la
cumple un bucle correcto y se la salta uno con un fallo — o uno comprometido.
Es la misma razón por la que un sandbox no se implementa como una función que
el código encerrado decide llamar.

Por eso avisa al construirse, y por eso cada comprobación queda marcada con
``enforced=False``.  Que un run pase por aquí sin denegaciones **no dice nada**
sobre si pasaría por el gateway real.  Sirve para ver la forma de las
decisiones, no para confiar en ellas.

Uso::

    gateway = LocalGateway(model=mi_adaptador, tools=[leer, escribir])

    async for step in agent.run(tarea, session=Session("run-1", gateway)):
        ...

    for check in gateway.checks:
        print(check)      # lo que un gateway real habría decidido
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence

from ..core.errors import Denied
from ..core.events import Decision, Disposition
from ..core.protocols import SEAM_VERSION, CallContext, Hello, Welcome, negotiate
from ..core.types import (
    Finish,
    Request,
    Response,
    Risk,
    StreamEvent,
    Text,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    dumps,
)
from ..tools.decorator import Tool

__all__ = ["LocalGateway", "Check", "Policy"]


ModelCall = Callable[..., Awaitable[Response]]
ModelStream = Callable[..., AsyncIterator[StreamEvent]]
"""``(Request)`` o ``(Request, CallContext)``.

Lo segundo es opcional y se detecta **una vez, al construir**. Existe porque hay
proveedores que pueden aprovechar la identidad del paso —una clave de
idempotencia, una traza— y obligar a todos a aceptar un parámetro que la mayoría
ignora sería peor que mirar la firma una vez."""
Policy = Callable[["Check"], Decision]
"""Política simulada.  Permite ver los tres caminos de denegación en local, con
la advertencia de que aquí no aplican nada."""


@dataclass(frozen=True, slots=True)
class Check:
    """Una decisión que un gateway real habría tomado."""

    kind: str
    """``"model"`` o ``"tool"``."""
    name: str
    step_id: str
    risk: Risk = Risk.READ
    arguments: Mapping[str, Any] = field(default_factory=dict)
    """Con qué se llamó.  Vacío para un paso de modelo.

    Es campo propio y no una clave de ``detail`` porque **es lo que una política
    decide**: negar «capturar» sin ver el importe no es una política, es un
    interruptor.  Estaba dentro de ``detail`` y obligaba a un acceso por cadena
    para llegar a lo único que casi siempre hace falta.
    """
    decision: Decision | None = None
    enforced: bool = False
    """Siempre ``False``.  Está en el dato, no solo en la documentación,
    para que nadie construya un informe que sugiera lo contrario."""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        verdict = self.decision.disposition.value if self.decision else "allow"
        return f"[no aplicado] {self.kind}:{self.name} risk={self.risk.value} → {verdict}"


class LocalGateway:
    """``Gateway`` de referencia para modo autónomo.

    Args:
        model: callable ``(Request) -> Awaitable[Response]``.  Cualquier
            adaptador de proveedor encaja sin que este módulo lo conozca.
        stream: opcional.  Si falta, el streaming se sirve igual entregando la
            respuesta completa como un único fragmento — y el ``Finish`` lo
            declara en ``provider_metadata["streamed"]``.  Ocultar la
            diferencia haría creer a quien corta por presupuesto que está
            ahorrando cuando ya ha pagado la respuesta entera.
        tools: herramientas decoradas con ``@tool``, que se ejecutan de verdad.
        policy: política simulada, opcional.
        warn: pon ``False`` solo si ya sabes que esto no aplica nada.
    """

    def __init__(
        self,
        *,
        model: ModelCall,
        stream: ModelStream | None = None,
        tools: Sequence[Tool] = (),
        policy: Policy | None = None,
        warn: bool = True,
    ) -> None:
        if warn:
            warnings.warn(
                "LocalGateway no aplica política: corre dentro del proceso que "
                "gobernaría, así que sus comprobaciones son advisorias. Un run que "
                "pasa por aquí sin denegaciones no dice nada sobre si pasaría por "
                "el gateway real.",
                UserWarning,
                stacklevel=2,
            )
        self._model = model
        self._stream = stream
        self._model_wants_ctx = _acepta_ctx(model)
        self._stream_wants_ctx = _acepta_ctx(stream) if stream is not None else False
        self.tools = {t.name: t for t in tools}
        self.policy = policy
        self.checks: list[Check] = []

    # ── Costura ───────────────────────────────────────────────────────────────

    async def handshake(self, hello: Hello) -> Welcome:
        return Welcome(
            version=negotiate(hello.versions, current=SEAM_VERSION),
            tool_refs={t.name: f"{t.name}@local" for t in hello.tools},
            session_id="local",
        )

    async def invoke_model(self, request: Request, ctx: CallContext) -> Response:
        self._check("model", request.model, ctx.step_id, Risk.READ)
        if self._model_wants_ctx:
            return await self._model(request, ctx)
        return await self._model(request)

    async def stream_model(self, request: Request, ctx: CallContext) -> AsyncIterator[StreamEvent]:
        self._check("model", request.model, ctx.step_id, Risk.READ)

        if self._stream is not None:
            fuente = (
                self._stream(request, ctx) if self._stream_wants_ctx else self._stream(request)
            )
            try:
                async for event in fuente:
                    yield event
            finally:
                cerrar = getattr(fuente, "aclose", None)
                if cerrar is not None:
                    await cerrar()
            return

        # Sin streaming del proveedor: se sirve igual, y el resultado dice qué
        # garantía tocó.  Es el mismo criterio que aplica el harness.
        response = await self._model(request)
        for index, part in enumerate(response.message.content):
            if isinstance(part, Text):
                yield TextStart(index=index)
                yield TextDelta(text=part.text, index=index)
                yield TextEnd(index=index)
            elif isinstance(part, ToolCall):
                yield ToolCallStart(id=part.id, name=part.name, index=index)
                yield ToolCallDelta(arguments_delta=dumps(part.arguments), index=index)
                yield ToolCallEnd(index=index)

        yield Finish(
            response=Response(
                message=response.message,
                finish_reason=response.finish_reason,
                usage=response.usage,
                model=response.model,
                provider_metadata={**dict(response.provider_metadata), "streamed": False},
            )
        )

    async def invoke_tool(
        self,
        call: ToolCall,
        ctx: CallContext,
        *,
        risk: Risk = Risk.READ,
        tool_ref: str | None = None,
    ) -> ToolResult:
        self._check("tool", call.name, ctx.step_id, risk, arguments=dict(call.arguments))

        target = self.tools.get(call.name)
        if target is None:
            return ToolResult.of(
                call.id,
                f"No existe la herramienta '{call.name}'. Disponibles: "
                f"{sorted(self.tools) or 'ninguna'}.",
                is_error=True,
            )
        return await target.invoke(call.id, call.arguments)

    # ── Registro de decisiones ────────────────────────────────────────────────

    def _check(
        self,
        kind: str,
        name: str,
        step_id: str,
        risk: Risk,
        *,
        arguments: Mapping[str, Any] | None = None,
        **detail: Any,
    ) -> None:
        comun = {
            "kind": kind, "name": name, "step_id": step_id, "risk": risk,
            "arguments": dict(arguments or {}), "detail": detail,
        }
        check = Check(**comun)
        decision = self.policy(check) if self.policy else None
        if decision is not None:
            check = Check(**comun, decision=decision)
        self.checks.append(check)

        if decision is not None and decision.disposition is not Disposition.ALLOW:
            raise Denied(decision)

    # ── Informe ───────────────────────────────────────────────────────────────

    def report(self) -> str:
        """Qué habría decidido un gateway real, en texto.

        Encabezado incluido: un informe que no dice que no se aplicó nada acaba
        pegado en un ticket como si fuera una auditoría.
        """
        lines = [
            "Comprobaciones simuladas (NO aplicadas — proceso local):",
            *(f"  {check}" for check in self.checks),
        ]
        return "\n".join(lines)


def _acepta_ctx(llamable: Any) -> bool:
    """¿La función quiere el ``CallContext`` además de la petición?

    Se mira una vez, al construir el gateway, y no en cada llamada: inspeccionar
    una firma en el camino caliente sería pagar por algo que no cambia.
    """
    import inspect

    try:
        firma = inspect.signature(llamable)
    except (TypeError, ValueError):   # builtins y objetos sin firma legible
        return False
    parametros = [
        p for p in firma.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if any(p.kind is p.VAR_POSITIONAL for p in firma.parameters.values()):
        return False
    return len(parametros) >= 2
