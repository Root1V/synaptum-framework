"""
SYN-66 · ``ReplayGateway`` — respuestas reales grabadas, sin acceso y sin coste.

Los cuerpos del corpus dorado sirven **dos veces**: para probar que dos
implementaciones normalizan igual, y para respaldar un modelo que reproduce
respuestas reales. La segunda es la mejor respuesta disponible a no tener
inferencia local — un guion escrito a mano dice lo que uno espera; una respuesta
grabada dice lo que el proveedor hizo.

La diferencia se nota en los sitios que uno no piensa: un `content` nulo junto a
tool calls, argumentos como cadena JSON, `usage` ausente donde se creía
presente, un stream que se corta a mitad. Contra un guion a mano esos caminos no
se recorren nunca, porque nadie escribe a mano el caso que no se le ocurre.

Uso::

    gateway = ReplayGateway(
        "contratos/gateway-prometheus/fixtures/chat_completion_tool_calls.json",
        "contratos/gateway-prometheus/fixtures/chat_stream_ok.sse",
        tools=[leer],
    )

Los cuerpos pasan por el **adaptador real**, no por un atajo: lo que llega al
bucle es exactamente lo que llegaría en producción con esa misma respuesta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Mapping, Sequence

from ..core.errors import ConfigurationError, Denied
from ..core.events import Decision
from ..core.protocols import SEAM_VERSION, CallContext, Hello, Welcome, negotiate
from ..core.types import Request, Response, Risk, StreamEvent, ToolCall, ToolResult
from ..providers import get as get_provider
from ..tools.decorator import Tool

__all__ = ["ReplayGateway", "split_sse"]


def split_sse(text: str) -> list[dict[str, Any]]:
    """Separa los fragmentos de un cuerpo SSE.

    Vive aquí y no en el adaptador porque **cómo llegan los fragmentos por el
    cable no es normalización**: es transporte, y esa separación es lo que
    permite ejercitar la normalización con un fichero.
    """
    chunks: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ").strip()
        if payload and payload != "[DONE]":
            chunks.append(json.loads(payload))
    return chunks


class ReplayGateway:
    """``Gateway`` que sirve respuestas grabadas, normalizadas de verdad.

    Args:
        bodies: rutas a cuerpos grabados.  ``.sse`` se sirve como stream;
            cualquier otra extensión, como respuesta completa.
        provider: adaptador con el que normalizar.  Por defecto, el dialecto
            que habla Prometheus.
        tools: herramientas decoradas, que se ejecutan de verdad.
        deny_tools: denegaciones simuladas por nombre de herramienta.
    """

    def __init__(
        self,
        *bodies: str | Path,
        provider: str = "openai-compatible",
        tools: Sequence[Tool] = (),
        deny_tools: Mapping[str, Decision] | None = None,
    ) -> None:
        self.provider = get_provider(provider)
        self.bodies = [Path(b) for b in bodies]
        self.tools = {t.name: t for t in tools}
        self.deny_tools = dict(deny_tools or {})

        missing = [str(b) for b in self.bodies if not b.exists()]
        if missing:
            raise ConfigurationError(f"No existen estos cuerpos grabados: {missing}")

        self._pending = list(self.bodies)
        self.model_calls = 0
        self.tool_calls = 0
        self.requests: list[Request] = []

    # ── Costura ───────────────────────────────────────────────────────────────

    async def handshake(self, hello: Hello) -> Welcome:
        return Welcome(
            version=negotiate(hello.versions, current=SEAM_VERSION),
            tool_refs={t.name: f"{t.name}@replay" for t in hello.tools},
            session_id="replay",
        )

    async def invoke_model(self, request: Request, ctx: CallContext) -> Response:
        self.model_calls += 1
        self.requests.append(request)
        body = self._next()

        if body.suffix == ".sse":
            # Un cuerpo de stream servido como respuesta completa: se consume el
            # ciclo y se entrega lo acumulado del `Finish`.
            events = list(self.provider.stream_from_wire(split_sse(body.read_text())))
            return events[-1].response  # type: ignore[union-attr]

        return self.provider.from_wire(json.loads(body.read_text()))

    async def stream_model(
        self, request: Request, ctx: CallContext
    ) -> AsyncIterator[StreamEvent]:
        self.model_calls += 1
        self.requests.append(request)
        body = self._next()

        chunks: Iterable[Mapping[str, Any]]
        if body.suffix == ".sse":
            chunks = split_sse(body.read_text())
        else:
            # Una respuesta completa servida como stream: el mismo criterio que
            # aplica el arnés con un proveedor sin streaming — se entrega igual,
            # y el resultado dice qué garantía tocó.
            chunks = [json.loads(body.read_text())]

        for event in self.provider.stream_from_wire(chunks):
            yield event

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

    def _next(self) -> Path:
        if not self._pending:
            raise AssertionError(
                "Se agotaron los cuerpos grabados: el bucle pidió una respuesta "
                "más de las previstas. Añade otro cuerpo o revisa por qué no "
                "terminó."
            )
        return self._pending.pop(0)

    def rewind(self) -> None:
        """Vuelve al primer cuerpo.  Útil para reanudar el mismo run dos veces."""
        self._pending = list(self.bodies)
