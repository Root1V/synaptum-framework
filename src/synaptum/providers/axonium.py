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

import hashlib
import json
from typing import Any, AsyncIterator, Mapping

from ..core.errors import ConfigurationError, ProviderError
from ..core.types import (
    Finish,
    FinishReason,
    Message,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
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
    dumps,
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

    async def complete(self, request: Request, ctx: Any = None) -> Response:
        body = self._wire.to_wire(request)
        body.pop("model", None)
        try:
            completion = await self._client.chat.completions.create(
                model=_model_name(request.model),
                idempotency_key=_clave(ctx, body),
                **body,
            )
        except Exception as failure:  # noqa: BLE001
            raise _translate(failure) from failure
        return _response_from(completion)

    async def stream(self, request: Request, ctx: Any = None) -> AsyncIterator[StreamEvent]:
        body = self._wire.to_wire(request)
        body.pop("model", None)
        yield StreamStart(model=request.model)

        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = FinishReason.STOP
        usage = Usage()
        meta: dict[str, Any] = {}
        open_text = False
        open_reasoning = False

        try:
            stream = self._client.chat.completions.stream(
                model=_model_name(request.model),
                idempotency_key=_clave(ctx, body),
                **body,
            )
            async with stream as chunks:
                async for chunk in chunks:
                    if getattr(chunk, "usage", None):
                        usage = usage_from_axonium(chunk.usage)
                    meta.update(metadata_from_axonium(chunk))

                    for choice in getattr(chunk, "choices", ()) or ():
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue

                        if getattr(delta, "reasoning_content", None):
                            # El razonamiento tiene su propio ciclo.  Acumularlo
                            # sin emitirlo deja a quien consume sin nada que ver
                            # durante toda la fase — y con un modelo de
                            # razonamiento esa fase puede ser la respuesta
                            # entera.
                            if not open_reasoning:
                                yield ReasoningStart()
                                open_reasoning = True
                            piece = str(delta.reasoning_content)
                            reasoning.append(piece)
                            yield ReasoningDelta(text=piece)

                        if getattr(delta, "content", None):
                            if open_reasoning:
                                yield ReasoningEnd()
                                open_reasoning = False
                            if not open_text:
                                yield TextStart()
                                open_text = True
                            piece = str(delta.content)
                            text.append(piece)
                            yield TextDelta(text=piece)

                        for raw in _field(delta, "tool_calls") or ():
                            # La fase de razonamiento termina cuando empieza
                            # cualquier otra cosa, no solo el texto: un modelo
                            # que razona y llama a una herramienta no emite ni
                            # un token de respuesta.
                            if open_reasoning:
                                yield ReasoningEnd()
                                open_reasoning = False
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

        if open_reasoning:
            yield ReasoningEnd()
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
                provider_metadata=meta,
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
        provider_metadata=metadata_from_axonium(completion),
    )


def metadata_from_axonium(source: Any) -> dict[str, Any]:
    """Lo que la plataforma dice sobre **esta** llamada, más allá del contenido.

    Se descartaba entero, y dos de estos campos no son decorativos:

    * ``idempotent_replay`` — una respuesta servida desde una clave previa no se
      generó ahora.  Su ``usage`` describe la generación **original**, así que
      sumarlo a un total acumulado lo falsea: cuenta dos veces algo que se pagó
      una.  El bucle lo mira antes de acumular.
    * ``instance_id`` — qué réplica atendió.  Es el valor que hay que citar al
      reportar una respuesta lenta o rara, y sin él la pregunta «¿cuál
      respondió?» no tiene respuesta cuando más falta hace.

    Se copian con los nombres del origen: reetiquetarlos obligaría a traducir de
    vuelta al hablar con quien los emitió.
    """
    meta = getattr(source, "meta", None)
    if meta is None:
        return {}
    recogido = {
        campo: getattr(meta, campo, None)
        for campo in (
            "request_id",
            "trace_id",
            "instance",
            "instance_id",
            "idempotent_replay",
            # Un replay trae su propio `request_id`, y ese id **no lleva a
            # ninguna fila de facturación**: nombra la respuesta que se sirvió,
            # no la generación que se cobró.  Este campo nombra la que sí, y sin
            # él una auditoría que parta del request_id de un replay no
            # encuentra nada y no sabe por qué.
            "idempotent_replay_of",
            # Cuánto esperó el SDK a propósito, y en cuántos intentos (`rc5`).
            #
            # Nos importa más que a nadie porque **hay dos reintentos apilados**:
            # el suyo, dentro de una llamada, y el nuestro encima. Sin esto, una
            # llamada que tardó veinte segundos porque el SDK respetó un
            # `Retry-After` es indistinguible de una plataforma lenta — tres
            # equipos ya la reportaron como un cuelgue, y ninguno como espera.
            #
            # Restado del reloj de fuera queda lo que tardó la plataforma. No se
            # usa para decidir nada todavía; entra ahora porque el sitio donde
            # hay que mirarlo es el journal, y al journal solo llega lo que se
            # recoge en el momento.
            "waited_s",
            "attempts",
        )
    }
    return {campo: valor for campo, valor in recogido.items() if valor is not None}


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
    # `Retry-After` viene del otro extremo y sabe algo que nosotros no: cuándo
    # estará libre.  Perderlo convierte una espera informada en una adivinada.
    espera = getattr(failure, "retry_after", None)
    return ProviderError(
        str(failure),
        status=status,
        provider="axonium",
        retry_after=float(espera) if espera is not None else None,
    )


def _build_client(**settings: Any) -> Any:
    try:
        from axonium import AsyncAxonium
    except ImportError as missing:
        raise ConfigurationError(
            "Axonium no está instalado. Es la única puerta a la inferencia local: "
            "instálalo con `pip install synaptum[axonium]`."
        ) from missing
    return AsyncAxonium(**settings)


def _clave(ctx: Any, body: Mapping[str, Any]) -> str | None:
    """La identidad del paso **más la huella de la petición**.

    ``(run_id, step_id)`` es determinista por construcción — el ordinal del paso
    dentro del run, nunca un UUID ni un reloj — así que reanudar produce la misma
    clave sin coordinar nada.  Es exactamente lo que la plataforma necesita para
    devolver la generación anterior en vez de cobrar otra.

    Cierra un agujero que el journal solo no puede tapar: si el proceso muere
    **después** de mandar la petición y **antes** de registrar el resultado, al
    reanudar el replay reintenta —una llamada al modelo es repetible— y sin clave
    esa repetición es una segunda generación facturable.

    **La huella del cuerpo no es prudencia: corrige la clave.** Una clave
    identifica *una* petición, y reutilizarla para otra distinta es un rechazo,
    no un replay.  Sin la huella, dos runs con el mismo ``run_id`` y distinta
    tarea chocaban — y también un mismo run tras cambiar las instrucciones del
    agente, que es lo que pasa mientras se desarrolla.  El síntoma era un error
    del proveedor durante 24 h y ninguna pista de por qué.

    Con ella, la propiedad que importa se mantiene intacta: al reanudar, el
    contexto se **vuelve a derivar** de los mismos resultados en el mismo orden,
    así que el cuerpo es idéntico y la huella también.  Y si el cuerpo *no* es
    idéntico, es otra petición y generar otra vez es lo correcto.

    ``dumps`` basta y no hace falta una canonicalización entre lenguajes: esta
    clave la produce y la consume el mismo proceso, no la compara nadie más.

    Sin ``ctx`` no se manda clave.  Inventar una sería peor que no mandarla: una
    clave no determinista convierte cada reintento en una generación nueva con la
    etiqueta de que no lo es.
    """
    if ctx is None:
        return None
    run_id = getattr(ctx, "run_id", None)
    step_id = getattr(ctx, "step_id", None)
    if not run_id or not step_id:
        return None

    huella = hashlib.sha256(dumps(body).encode()).hexdigest()[:16]
    clave = f"{run_id}/{step_id}/{huella}"
    # La plataforma corta en 255 caracteres.  Se recorta por delante: lo que
    # distingue dos peticiones está al final.
    return clave[-255:]


def _model_name(spec: str) -> str:
    """Quita el prefijo de proveedor si lo lleva: Axonium recibe el nombre a secas."""
    _, _, model = spec.partition(":")
    return model or spec
