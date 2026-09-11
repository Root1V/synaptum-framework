"""
SYN-52 · Puente hacia Axonium — la única puerta a la inferencia local.

**Esto no es un ``Provider``.** Un `Provider` normaliza y no transporta; Axonium
hace las dos cosas: abre la conexión y **ya devuelve respuestas normalizadas**.
Lo que falta es traducir entre dos vocabularios que ya son normalizados, no
interpretar un cuerpo crudo.

La distinción tiene consecuencias, y son la razón de que este módulo exista en
vez de reutilizar el lector del dialecto OpenAI-compatible: el ``Usage`` de
Axonium lleva ``cache_read_tokens`` y ``estimated``, campos que **su**
normalización produjo. Pasarlo por un lector de cable los buscaría donde no
están y los perdería **en silencio** — que es exactamente el fallo que hemos
estado persiguiendo entre los tres proyectos toda la semana.

Encaja como el modelo de ``LocalGateway``::

    from synaptum.providers.axonium import AxoniumModel
    from synaptum import LocalGateway, Session

    modelo  = AxoniumModel()                       # lee AXONIUM_* del entorno
    gateway = LocalGateway(model=modelo.complete, stream=modelo.stream, tools=[...])

Solo tiene sentido en **modo autónomo**. En el camino gobernado, la llamada la
hace el gateway del arnés con Axonium-Go dentro, porque ahí vive la credencial.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping

from ..core.errors import ConfigurationError, ProviderError
from ..core.types import (
    Finish,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamStart,
    Text,
    TextDelta,
    TextEnd,
    TextStart,
    Thinking,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)
from .openai_compatible import OpenAICompatible

__all__ = ["AxoniumModel", "message_from_axonium", "usage_from_axonium"]


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Lee un campo de un objeto tipado **o** de un diccionario.

    Axonium deja hoy las tool calls como diccionarios crudos mientras el resto
    del mensaje sí es un modelo. Acceder solo por atributo las perdería en
    silencio — y si mañana las tipan, esto sigue funcionando.
    """
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class AxoniumModel:
    """Llama a Prometheus a través del SDK de Axonium.

    Args:
        client: cliente ya construido.  Si falta, se crea uno leyendo las
            variables ``AXONIUM_*`` del entorno — el modo autónomo, donde el
            SDK acuña y refresca sus propios tokens.
        **settings: se pasan al cliente si hay que construirlo.
    """

    def __init__(self, client: Any = None, **settings: Any) -> None:
        self._client = client if client is not None else _build_client(**settings)
        self._wire = OpenAICompatible()

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()

    # ── Llamada ───────────────────────────────────────────────────────────────

    async def complete(self, request: Request) -> Response:
        body = self._wire.to_wire(request)
        body.pop("model", None)
        try:
            completion = await self._client.chat.completions.create(
                model=_model_name(request.model), **body
            )
        except Exception as failure:  # noqa: BLE001
            raise _translate(failure) from failure
        return _response_from(completion)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        body = self._wire.to_wire(request)
        body.pop("model", None)
        yield StreamStart(model=request.model)

        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = FinishReason.STOP
        usage = Usage()
        open_text = False

        try:
            stream = self._client.chat.completions.stream(
                model=_model_name(request.model), **body
            )
            async with stream as chunks:
                async for chunk in chunks:
                    if getattr(chunk, "usage", None):
                        usage = usage_from_axonium(chunk.usage)

                    for choice in getattr(chunk, "choices", ()) or ():
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue

                        if getattr(delta, "reasoning_content", None):
                            reasoning.append(str(delta.reasoning_content))

                        if getattr(delta, "content", None):
                            if not open_text:
                                yield TextStart()
                                open_text = True
                            piece = str(delta.content)
                            text.append(piece)
                            yield TextDelta(text=piece)

                        for raw in _field(delta, "tool_calls") or ():
                            index = int(_field(raw, "index", 0) or 0)
                            function = _field(raw, "function")
                            if index not in calls:
                                calls[index] = {
                                    "id": _field(raw, "id", "") or "",
                                    "name": _field(function, "name", "") or "",
                                    "args": [],
                                }
                                yield ToolCallStart(
                                    id=calls[index]["id"],
                                    name=calls[index]["name"],
                                    index=index,
                                )
                            fragment = _field(function, "arguments")
                            if fragment:
                                calls[index]["args"].append(str(fragment))
                                yield ToolCallDelta(
                                    arguments_delta=str(fragment), index=index
                                )

                        if getattr(choice, "finish_reason", None):
                            finish = _FINISH.get(choice.finish_reason, FinishReason.STOP)
        except Exception as failure:  # noqa: BLE001
            # Los deltas ya emitidos no se retiran: se generaron y se pagaron.
            raise _translate(failure) from failure

        if open_text:
            yield TextEnd()
        for index in sorted(calls):
            yield ToolCallEnd(index=index)

        parts: list[Any] = []
        if reasoning:
            parts.append(Thinking(text="".join(reasoning)))
        if text:
            parts.append(Text("".join(text)))
        for index in sorted(calls):
            entry = calls[index]
            parts.append(
                ToolCall(
                    id=entry["id"],
                    name=entry["name"],
                    arguments=_arguments("".join(entry["args"]), entry["name"]),
                )
            )

        yield Finish(
            response=Response(
                message=Message(Role.ASSISTANT, tuple(parts)),
                finish_reason=finish,
                usage=usage,
                model=request.model,
            )
        )


# ── Traducción entre dos vocabularios ya normalizados ─────────────────────────

def usage_from_axonium(source: Any) -> Usage:
    """Mapea campo a campo, **sin pasar por un lector de cable**.

    Su ``Usage`` ya distingue lo cacheado y lo derivado; leerlo como si fuera
    un cuerpo crudo buscaría esos campos donde no están y los perdería sin
    error.  Los dos contadores que Prometheus no puede alimentar quedan en
    ``None`` — sin medir, no cero.
    """
    if source is None:
        return Usage()
    return Usage(
        input=getattr(source, "prompt_tokens", None),
        output=getattr(source, "completion_tokens", None),
        reasoning=None,
        cache_read=getattr(source, "cache_read_tokens", None),
        cache_write=None,
        estimated=bool(getattr(source, "estimated", False)),
    )


def message_from_axonium(source: Any) -> Message:
    """Traduce un mensaje de Axonium al vocabulario unificado."""
    parts: list[Any] = []

    if _field(source, "reasoning_content"):
        parts.append(Thinking(text=str(_field(source, "reasoning_content"))))

    # `content` nulo junto a tool calls no es una respuesta vacía: no se añade
    # una parte de texto vacía.
    if _field(source, "content"):
        parts.append(Text(str(_field(source, "content"))))

    for raw in _field(source, "tool_calls") or ():
        function = _field(raw, "function")
        name = _field(function, "name", "") or ""
        parts.append(
            ToolCall(
                id=_field(raw, "id", "") or "",
                name=name,
                arguments=_arguments(_field(function, "arguments"), name),
            )
        )

    return Message(Role.ASSISTANT, tuple(parts))


def _response_from(completion: Any) -> Response:
    choices = getattr(completion, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None) if choice is not None else None
    reason = getattr(choice, "finish_reason", None) if choice is not None else None

    return Response(
        message=message_from_axonium(message) if message is not None else Message(Role.ASSISTANT),
        finish_reason=_FINISH.get(reason or "", FinishReason.STOP),
        usage=usage_from_axonium(getattr(completion, "usage", None)),
        model=str(getattr(completion, "model", "") or ""),
    )


def _arguments(raw: Any, tool: str) -> Mapping[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as broken:
        raise ProviderError(
            f"Argumentos ilegibles para '{tool}': {broken}",
            provider="axonium",
            retryable=False,
        ) from broken
    return parsed if isinstance(parsed, Mapping) else {}


# ── Errores ───────────────────────────────────────────────────────────────────

def _translate(failure: Exception) -> Exception:
    """Traduce un error de Axonium a nuestra taxonomía.

    La reintentabilidad viaja en el tipo, así que se toma del estado HTTP que
    el SDK ya trae en vez de volver a deducirla del mensaje.
    """
    from ..core.errors import SynaptumError

    if isinstance(failure, SynaptumError):
        return failure

    status = getattr(failure, "status_code", None) or getattr(failure, "status", None)
    return ProviderError(str(failure), status=status, provider="axonium")


def _build_client(**settings: Any) -> Any:
    try:
        from axonium import AsyncAxonium
    except ImportError as missing:
        raise ConfigurationError(
            "Axonium no está instalado. Es la única puerta a la inferencia local: "
            "instálalo con `pip install synaptum[axonium]`."
        ) from missing
    return AsyncAxonium(**settings)


def _model_name(spec: str) -> str:
    """Quita el prefijo de proveedor si lo lleva: Axonium recibe el nombre a secas."""
    _, _, model = spec.partition(":")
    return model or spec
