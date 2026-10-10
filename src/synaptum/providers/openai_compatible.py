"""
SYN-18 · Adaptador del dialecto OpenAI-compatible.

Implementa `contratos/normalizacion/spec.md` para el dialecto OpenAI-compatible,
que es el que habla casi todo: OpenAI, los servidores locales
(llama.cpp, vLLM, Ollama, LM Studio) y la mayoría de los gateways.

Cuando existe una segunda implementación de la misma especificación en otro
lenguaje, lo que impide que diverjan no es la confianza: es un corpus dorado que
ambas ejecutan contra los mismos cuerpos.

Puro por construcción: no abre conexiones ni lee credenciales. El transporte va
aparte, y en el camino gobernado ni siquiera ocurre en este proceso.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator, Mapping

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

__all__ = ["OpenAICompatible", "tool_call_from_wire"]


_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAICompatible:
    name = "openai-compatible"

    # ── Request ───────────────────────────────────────────────────────────────

    def to_wire(self, request: Request) -> dict[str, Any]:
        body: dict[str, Any] = {"model": request.model, "messages": []}

        # El prompt de sistema no es un turno: se extrae, como en todos los
        # proveedores, y por eso el tipo lo guarda en su propio campo.
        if request.system:
            body["messages"].append({"role": "system", "content": request.system})

        for message in request.messages:
            body["messages"].extend(_message_to_wire(message))

        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in request.tools
            ]
        if request.tool_choice.mode != "auto":
            body["tool_choice"] = (
                {"type": "function", "function": {"name": request.tool_choice.name}}
                if request.tool_choice.mode == "named"
                else request.tool_choice.mode
            )
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        for field, value in (("temperature", request.temperature), ("top_p", request.top_p)):
            if value is not None:
                body[field] = value
        if request.stop:
            body["stop"] = list(request.stop)
        if request.response_format is not None:
            body["response_format"] = _format_to_wire(request.response_format)

        body.update(request.provider_options)
        return body

    # ── Response ──────────────────────────────────────────────────────────────

    def from_wire(self, body: Mapping[str, Any]) -> Response:
        choices = body.get("choices") or [{}]
        choice = choices[0]
        wire = choice.get("message") or {}

        return Response(
            message=Message(Role.ASSISTANT, _content_from_wire(wire)),
            finish_reason=_FINISH.get(choice.get("finish_reason") or "", FinishReason.STOP),
            usage=_usage_from_wire(body),
            model=str(body.get("model", "")),
        )

    # ── Stream ────────────────────────────────────────────────────────────────

    def stream_from_wire(
        self, chunks: Iterable[Mapping[str, Any]]
    ) -> Iterator[StreamEvent]:
        yield StreamStart()

        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = FinishReason.STOP
        model = ""
        usage_source: Mapping[str, Any] = {}
        open_text = False
        open_reasoning = False

        for chunk in chunks:
            if "error" in chunk:
                # Los deltas ya emitidos no se retiran: se generaron y se
                # pagaron.  Quien consume conserva lo parcial y sabe que el
                # turno no terminó.
                raise ProviderError(str(chunk["error"]), provider=self.name)

            model = model or str(chunk.get("model", ""))
            if chunk.get("usage") or chunk.get("timings"):
                usage_source = chunk

            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}

                if delta.get("reasoning_content"):
                    # El razonamiento tiene su propio ciclo, igual que el texto.
                    # Acumularlo sin emitirlo dejaría a quien consume sin nada
                    # que ver durante toda la fase — y hay respuestas que son
                    # solo razonamiento.
                    if not open_reasoning:
                        yield ReasoningStart()
                        open_reasoning = True
                    piece = str(delta["reasoning_content"])
                    reasoning.append(piece)
                    yield ReasoningDelta(text=piece)

                if delta.get("content"):
                    if open_reasoning:
                        yield ReasoningEnd()
                        open_reasoning = False
                    if not open_text:
                        yield TextStart()
                        open_text = True
                    piece = str(delta["content"])
                    text.append(piece)
                    yield TextDelta(text=piece)

                for raw in delta.get("tool_calls") or ():
                    if open_reasoning:
                        # La fase de razonamiento termina cuando empieza
                        # *cualquier* otra cosa, no solo el texto.  Una
                        # grabación real de un modelo de razonamiento que llama
                        # a una herramienta son 75 deltas de razonamiento y
                        # ningún token de respuesta: cerrar solo con `content`
                        # dejaba el `reasoning_end` cayendo en mitad de la tool
                        # call, con el bloque de pensamiento abierto mientras
                        # los argumentos ya estaban llegando.
                        yield ReasoningEnd()
                        open_reasoning = False
                    index = int(raw.get("index", 0))
                    if index not in calls:
                        calls[index] = {"id": raw.get("id", ""), "name": "", "args": []}
                        function = raw.get("function") or {}
                        calls[index]["name"] = str(function.get("name", ""))
                        yield ToolCallStart(
                            id=calls[index]["id"], name=calls[index]["name"], index=index
                        )
                    fragment = (raw.get("function") or {}).get("arguments")
                    if fragment:
                        calls[index]["args"].append(str(fragment))
                        yield ToolCallDelta(arguments_delta=str(fragment), index=index)

                if choice.get("finish_reason"):
                    finish = _FINISH.get(choice["finish_reason"], FinishReason.STOP)

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
            # Los fragmentos no son JSON válido hasta el final: se acumulan y
            # se parsean una sola vez, aquí.
            parts.append(
                tool_call_from_wire(entry["id"], entry["name"], "".join(entry["args"]))
            )

        yield Finish(
            response=Response(
                message=Message(Role.ASSISTANT, tuple(parts)),
                finish_reason=finish,
                usage=_usage_from_wire(usage_source),
                model=model,
            )
        )


# ── Piezas ────────────────────────────────────────────────────────────────────

def _content_from_wire(wire: Mapping[str, Any]) -> tuple[Any, ...]:
    parts: list[Any] = []

    if wire.get("reasoning_content"):
        parts.append(Thinking(text=str(wire["reasoning_content"])))

    # `content` nulo con tool calls no es una respuesta vacía: no se añade una
    # parte de texto vacía, porque no es lo mismo que la ausencia de texto
    # cuando alguien las concatena.
    if wire.get("content"):
        parts.append(Text(str(wire["content"])))

    for raw in wire.get("tool_calls") or ():
        function = raw.get("function") or {}
        parts.append(tool_call_from_wire(
            str(raw.get("id", "")),
            str(function.get("name", "")),
            function.get("arguments"),
        ))

    return tuple(parts)


def tool_call_from_wire(id: str, name: str, raw: Any) -> ToolCall:
    """Una tool call del cable, con sus argumentos leídos — o sin leer.

    El cable entrega los argumentos como cadena JSON. Una cadena que no parsea
    **no se convierte en objeto vacío**: un ``{}`` silencioso ejecutaría la
    herramienta sin argumentos, y el modelo vería un resultado en vez de su
    error. Tampoco mata el run, que es lo que hacía antes —un `ProviderError`
    no reintentable por un argumento mal escrito— y lo que pidió Veritium
    cambiar en `VRT-SYN-005`: el texto crudo viaja en la llamada, el bucle lo
    devuelve al modelo como resultado de error y le cuesta **un turno**.

    Lo mismo con algo que parsea y no es un objeto: `"[1,2]"` son argumentos
    tan inservibles como `"{roto"`, y antes se volvía `{}` **en silencio** —el
    caso que el aviso de arriba decía estar cubriendo y no cubría—.

    Vive aquí y la usan los dos lectores del dialecto —este y el del puente de
    SDK— porque es una regla, no dos: dos copias escritas a mano del mismo
    criterio es exactamente lo que acabamos de ver fallar en otro equipo, donde
    las dos perdieron el campo nuevo.
    """
    if raw is None or raw == "":
        return ToolCall(id=id, name=name)
    if isinstance(raw, Mapping):
        return ToolCall(id=id, name=name, arguments=dict(raw))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ToolCall(id=id, name=name, unreadable_arguments=str(raw))
    if not isinstance(parsed, Mapping):
        return ToolCall(id=id, name=name, unreadable_arguments=str(raw))
    return ToolCall(id=id, name=name, arguments=parsed)


def _usage_from_wire(body: Mapping[str, Any]) -> Usage:
    """Consumo reportado si lo hay; derivado de ``timings`` si no.

    ``input`` es **inclusivo**: contiene los tokens servidos desde caché, y
    ``cache_read`` dice cuántos de ellos lo fueron.  Se copia lo que reporta el
    proveedor en vez de restar, porque olvidar una resta contaría dos veces lo
    cacheado sin producir ningún error.
    """
    reported = body.get("usage")
    if reported:
        details = reported.get("prompt_tokens_details") or {}
        return Usage(
            input=reported.get("prompt_tokens"),
            output=reported.get("completion_tokens"),
            reasoning=(reported.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cache_read=details.get("cached_tokens"),
            cache_write=None,
        )

    timings = body.get("timings")
    if timings:
        prompt = timings.get("prompt_n")
        cached = timings.get("cache_n")
        total = None if prompt is None else prompt + (cached or 0)
        return Usage(
            input=total,
            output=timings.get("predicted_n"),
            reasoning=None,
            cache_read=cached,
            cache_write=None,
            estimated=True,
        )

    # Ni medido ni derivable: todo sin medir.  Cero diría «no hubo».
    return Usage()


def _message_to_wire(message: Message) -> list[dict[str, Any]]:
    if message.role is Role.TOOL:
        return [
            {
                "role": "tool",
                "tool_call_id": part.call_id,
                "content": "".join(p.text for p in part.content if isinstance(p, Text)),
            }
            for part in message.content
            if hasattr(part, "call_id")
        ]

    _rechaza_lo_que_no_sabe_mandar(message)

    text = message.text
    calls = message.tool_calls
    imagenes = [p for p in message.content if p.kind == "image"]

    # Que **haya** una parte de texto, aunque esté vacía, no es lo mismo que que
    # no haya ninguna: la primera es un turno que dijo la cadena vacía y la
    # segunda es un turno que no tiene nada que decir. El dialecto también las
    # distingue —`content: ""` es enviable y lo acepta el servidor; medido por
    # el equipo del arnés contra el suyo al arreglar su mitad de VRT-SYN-004— y
    # colapsarlas aquí cambiaría el cuerpo de peticiones que hoy funcionan.
    dijo_algo = any(parte.kind == "text" for parte in message.content)

    if message.role is Role.ASSISTANT and not (dijo_algo or calls or imagenes):
        # **Un turno del asistente que no lleva nada enviable no se envuelve.**
        # El caso real: el modelo contesta solo con `reasoning_content`, la
        # respuesta se normaliza a un mensaje cuya única parte es `Thinking`, y
        # el razonamiento no se devuelve al proveedor —eso está decidido arriba,
        # en `_TRANSPORTABLE`—.  Lo que quedaba era el sobre vacío,
        # `{"role": "assistant", "content": null}`, que **el propio dialecto
        # declara inválido**: un mensaje del asistente lleva `content` o
        # `tool_calls`.  llama-server lo rechaza con un 400 no reintentable y se
        # pierde el run entero, en el turno *siguiente* y solo cuando el modelo
        # razona sin hablar — intermitente, y más probable cuanto más largo es el
        # run (VRT-SYN-004).
        #
        # Omitir el sobre no es descartar en silencio, que es lo que este
        # adaptador se niega a hacer: la parte que no viaja ya no viajaba, y el
        # turno sigue completo donde importa que lo esté —en el diario, que es
        # de donde lee el replay—.  Lo que desaparece es un envoltorio sin nada
        # dentro.
        #
        # No se manda `content: ""` en su lugar, aunque este servidor lo
        # aceptaría: hay dialectos que rechazan un bloque de texto vacío, así
        # que la forma «válida» dependería de quién esté al otro lado.  No
        # mandar nada vale en todos.
        return []

    wire: dict[str, Any] = {"role": message.role.value}

    if imagenes:
        # Con partes no textuales, `content` deja de ser una cadena y pasa a ser
        # una lista de partes.  Es la forma que espera el dialecto, y la cadena
        # simple se conserva cuando no hay imágenes: cambiarla siempre haría
        # distinto el cuerpo de todos los runs que hoy funcionan.
        partes: list[dict[str, Any]] = []
        if text:
            partes.append({"type": "text", "text": text})
        partes.extend({"type": "image_url", "image_url": {"url": _imagen_a_url(p)}}
                      for p in imagenes)
        wire["content"] = partes
    else:
        # `None` y `""` no son lo mismo: el primero es «este turno no trae
        # texto» —lo normal junto a unas tool calls— y el segundo es «dijo la
        # cadena vacía». Con `text or None` los dos salían como `null`.
        wire["content"] = text if dijo_algo else None
    if calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": dumps(call.arguments)},
            }
            for call in calls
        ]
    return [wire]


#: Lo que este adaptador sabe poner en el cable.  El razonamiento se omite a
#: propósito —no se devuelve al proveedor— y por eso no cuenta como pérdida.
_TRANSPORTABLE = {
    "text", "image", "tool_call", "tool_result", "thinking", "redacted_thinking",
}


def _rechaza_lo_que_no_sabe_mandar(message: Message) -> None:
    """Falla si el mensaje lleva algo que este adaptador no transporta.

    Antes se descartaba en silencio: una imagen puesta en el mensaje
    desaparecía, el modelo contestaba sobre un texto sin ella, y **nada
    fallaba**. La respuesta parecía mala y lo que estaba mal era el envío.

    Negarse es peor para quien ya tenía un atajo y mejor para todos los demás:
    un fallo ruidoso se arregla una vez, y uno silencioso se paga en cada
    respuesta sin que nadie sepa por qué.
    """
    perdidas = sorted({
        parte.kind for parte in message.content if parte.kind not in _TRANSPORTABLE
    })
    if perdidas:
        raise ConfigurationError(
            f"El adaptador `openai-compatible` no sabe transportar {perdidas} y "
            f"no lo descarta en silencio: el modelo respondería sin eso y nadie "
            f"se enteraría. Las imágenes sí viajan; un documento **se rechaza a "
            f"propósito** — convertirlo aquí sería decidir por quien lo manda "
            f"cómo se ve una página, y eso lo decide quien la recortó."
        )


def _imagen_a_url(imagen: Any) -> str:
    """Una imagen, como la pide el dialecto: una URL o un *data URI*.

    El tipo ya garantiza que hay exactamente una de las dos —lo comprueba al
    construirse— así que aquí no hay caso ambiguo que resolver.
    """
    if imagen.url:
        return imagen.url
    return f"data:{imagen.media_type};base64,{imagen.data}"


def _format_to_wire(response_format: Any) -> dict[str, Any]:
    if response_format.kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.name or "output",
                "schema": dict(response_format.schema),
                "strict": response_format.strict,
            },
        }
    return {"type": response_format.kind}
