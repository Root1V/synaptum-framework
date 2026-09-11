"""
SYN-18 · Adaptador del dialecto OpenAI-compatible.

Implementa `contratos/normalizacion/spec.md` para el dialecto que habla
Prometheus y casi todo lo demás. Es la **implementación en Python** de una
especificación que el gateway implementa en Go; lo que impide que diverjan es el
corpus dorado compartido, que ambas ejecutan contra los mismos cuerpos.

Puro por construcción: no abre conexiones ni lee credenciales. El transporte va
aparte, y en el camino gobernado ni siquiera ocurre en este proceso.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator, Mapping

from ..core.errors import ProviderError
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
    dumps,
)

__all__ = ["OpenAICompatible"]


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
                    reasoning.append(str(delta["reasoning_content"]))

                if delta.get("content"):
                    if not open_text:
                        yield TextStart()
                        open_text = True
                    piece = str(delta["content"])
                    text.append(piece)
                    yield TextDelta(text=piece)

                for raw in delta.get("tool_calls") or ():
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
                    # Los fragmentos no son JSON válido hasta el final: se
                    # acumulan y se parsean una sola vez, aquí.
                    arguments=_parse_arguments("".join(entry["args"]), entry["name"]),
                )
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
        name = str(function.get("name", ""))
        parts.append(
            ToolCall(
                id=str(raw.get("id", "")),
                name=name,
                arguments=_parse_arguments(function.get("arguments"), name),
            )
        )

    return tuple(parts)


def _parse_arguments(raw: Any, tool: str) -> Mapping[str, Any]:
    """El cable entrega los argumentos como cadena JSON; aquí llegan decodificados.

    Una cadena que no parsea **no se convierte en objeto vacío**: un ``{}``
    silencioso ejecutaría la herramienta sin argumentos.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as broken:
        raise ProviderError(
            f"Argumentos ilegibles para '{tool}': {broken}", retryable=False
        ) from broken
    return parsed if isinstance(parsed, Mapping) else {}


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

    wire: dict[str, Any] = {"role": message.role.value}
    text = message.text
    calls = message.tool_calls
    wire["content"] = text or None
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
