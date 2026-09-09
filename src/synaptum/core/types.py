"""
RM-01 · Vocabulario unificado de modelo.

Contrato compartido y versionado entre Synaptum, Aeon y Axonium — no un detalle
interno de este paquete.  Es la forma que viaja por la costura de aplicación y
la que cada implementación de la especificación de normalización (RM-02) debe
producir a partir de la respuesta nativa de su proveedor.

Reglas de normalización que el tipo hace cumplir por construcción
-----------------------------------------------------------------
* **El mensaje de sistema no es un mensaje.**  Viaja en ``Request.system``,
  porque ningún proveedor lo trata como turno: OpenAI lo extrae a
  ``instructions``, Anthropic a ``system`` y Gemini a ``systemInstruction``.
  Meterlo en la lista de mensajes obliga a cada adaptador a volver a sacarlo.

* **Los resultados de tool son partes de contenido, no un formato por
  proveedor.**  OpenAI usa un rol ``tool``, Anthropic los embebe en mensajes de
  usuario y Gemini usa ``functionResponse``.  Aquí son ``ToolResult`` y el
  adaptador traduce.

* **``Usage`` distingue cinco contadores.**  Los tokens de razonamiento y los de
  caché no son opcionales ni derivables: la economía de contexto del bucle
  depende de ellos, y por H3 vuelven por la costura aunque el span lo emita
  quien ejecuta la llamada.

* **Serialización determinista.**  ``dumps`` ordena claves y fija separadores.
  Sin esto no hay prefijo estable de caché ni comparación de replay.

* **Todo es inmutable.**  Los eventos del journal (RM-05) referencian estos
  tipos; si mutasen, el registro dejaría de describir lo que ocurrió.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

__all__ = [
    "Risk",
    "Role",
    "Text",
    "Image",
    "Audio",
    "Document",
    "ToolCall",
    "ToolResult",
    "Thinking",
    "RedactedThinking",
    "ContentPart",
    "Message",
    "Usage",
    "FinishReason",
    "ToolDefinition",
    "ToolChoice",
    "ResponseFormat",
    "Request",
    "Response",
    "StreamStart",
    "TextStart",
    "TextDelta",
    "TextEnd",
    "ReasoningStart",
    "ReasoningDelta",
    "ReasoningEnd",
    "ToolCallStart",
    "ToolCallDelta",
    "ToolCallEnd",
    "Finish",
    "StreamEvent",
    "to_jsonable",
    "dumps",
    "b64",
]


# ── Roles ─────────────────────────────────────────────────────────────────────

class Risk(str, Enum):
    """Nivel de riesgo del efecto de una herramienta — RM-21.

    Synaptum **declara**; el harness **decide**.  El bucle garantiza que un paso
    destructivo no se ejecuta antes de tener una decisión; cuál sea esa decisión
    no es asunto suyo.

    Vive aquí, junto a ``ToolDefinition``, porque es parte del contrato de la
    herramienta: viaja con ella en el handshake y el gateway lo necesita para
    decidir antes de ejecutar.
    """

    READ = "read"
    SOFT_WRITE = "soft_write"
    HARD_WRITE = "hard_write"
    DESTRUCTIVE = "destructive"


class Role(str, Enum):
    """Los cinco roles que cualquier proveedor sabe expresar."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


# ── Partes de contenido — ocho tipos ──────────────────────────────────────────
#
# Unión etiquetada por el campo ``kind``.  El discriminante permite serializar y
# reconstruir sin registro de clases y sin isinstance en el camino caliente.

@dataclass(frozen=True, slots=True)
class Text:
    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class Image:
    media_type: str
    data: str | None = None
    url: str | None = None
    kind: Literal["image"] = "image"

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise ValueError("Image requiere exactamente uno de 'data' o 'url'.")


@dataclass(frozen=True, slots=True)
class Audio:
    media_type: str
    data: str | None = None
    url: str | None = None
    kind: Literal["audio"] = "audio"

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise ValueError("Audio requiere exactamente uno de 'data' o 'url'.")


@dataclass(frozen=True, slots=True)
class Document:
    media_type: str
    data: str | None = None
    url: str | None = None
    name: str | None = None
    kind: Literal["document"] = "document"

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise ValueError("Document requiere exactamente uno de 'data' o 'url'.")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Invocación pedida por el modelo.

    ``arguments`` ya viene decodificado.  Los proveedores que lo entregan como
    cadena JSON lo parsean en su adaptador: el bucle no debería tener que
    adivinar si recibió un objeto o su serialización.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    kind: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Resultado devuelto al modelo.

    ``is_error`` conserva la evidencia del fallo en vez de ocultarla: un modelo
    que no ve el error no puede corregirlo.
    """

    call_id: str
    content: tuple["ContentPart", ...] = ()
    is_error: bool = False
    kind: Literal["tool_result"] = "tool_result"

    @staticmethod
    def of(call_id: str, text: str, *, is_error: bool = False) -> "ToolResult":
        return ToolResult(call_id=call_id, content=(Text(text),), is_error=is_error)


@dataclass(frozen=True, slots=True)
class Thinking:
    """Razonamiento visible.

    ``signature`` transporta el token de verificación que algunos proveedores
    exigen devolver intacto en el siguiente turno.
    """

    text: str
    signature: str | None = None
    kind: Literal["thinking"] = "thinking"


@dataclass(frozen=True, slots=True)
class RedactedThinking:
    """Razonamiento cifrado por el proveedor.

    Opaco para nosotros, pero debe reenviarse tal cual o el proveedor pierde el
    hilo de su propio razonamiento.
    """

    data: str
    kind: Literal["redacted_thinking"] = "redacted_thinking"


ContentPart = (
    Text | Image | Audio | Document | ToolCall | ToolResult | Thinking | RedactedThinking
)


# ── Mensaje ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: tuple[ContentPart, ...] = ()
    name: str | None = None

    # Constructores de conveniencia.  El rol SYSTEM no tiene uno a propósito:
    # el prompt de sistema va en Request.system, no en la lista de mensajes.
    @staticmethod
    def user(text: str, *, name: str | None = None) -> "Message":
        return Message(Role.USER, (Text(text),), name)

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(Role.ASSISTANT, (Text(text),))

    @staticmethod
    def developer(text: str) -> "Message":
        return Message(Role.DEVELOPER, (Text(text),))

    @staticmethod
    def tool_results(*results: ToolResult) -> "Message":
        return Message(Role.TOOL, tuple(results))

    @property
    def text(self) -> str:
        """Concatena las partes de texto.  Ignora razonamiento y binarios."""
        return "".join(p.text for p in self.content if isinstance(p, Text))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(p for p in self.content if isinstance(p, ToolCall))


# ── Consumo ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Usage:
    """Contadores de tokens — H3.

    Cinco campos, no dos.  ``reasoning`` lo expone OpenAI en
    ``output_tokens_details``, Anthropic por estimación de bloques y Gemini en
    ``thoughtsTokenCount``.  ``cache_read`` y ``cache_write`` son la entrada de
    la economía de contexto: sin ellos el bucle no puede decidir si compactar
    sale más barato que conservar.
    """

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.reasoning

    @property
    def cache_hit_ratio(self) -> float:
        """Fracción de la entrada servida desde caché.  0.0 si no hubo entrada."""
        billed = self.input + self.cache_read
        return self.cache_read / billed if billed else 0.0

    def __add__(self, other: object) -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


# ── Herramientas ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Definición de una tool.

    ``ref`` es la referencia versionada de H5.  Cuando está presente, el esquema
    no viaja en cada llamada: el handshake lo resolvió una vez y el prefijo
    cacheado se mantiene estable.  ``parameters`` sigue disponible para el modo
    autónomo, donde no hay registro contra el que resolver.
    """

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    ref: str | None = None
    risk: Risk = Risk.READ
    """Conservador por defecto en durabilidad, permisivo por defecto en riesgo:
    quien no declara nada obtiene la clase más inocua, y declararse destructivo
    es un acto explícito."""
    idempotent: bool = False
    """Si el efecto puede repetirse sin consecuencias.  Conservador por defecto:
    determina si el replay puede reintentar el paso tras una caída."""


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: Literal["auto", "none", "required", "named"] = "auto"
    name: str | None = None

    def __post_init__(self) -> None:
        if (self.mode == "named") != (self.name is not None):
            raise ValueError("ToolChoice 'named' requiere 'name', y solo ese modo lo admite.")


AUTO = ToolChoice()


# ── Salida estructurada ───────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ResponseFormat:
    kind: Literal["text", "json_object", "json_schema"] = "text"
    schema: Mapping[str, Any] | None = None
    name: str | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        if self.kind == "json_schema" and self.schema is None:
            raise ValueError("ResponseFormat 'json_schema' requiere 'schema'.")


# ── Petición y respuesta ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Request:
    """Lo que cruza la costura hacia quien ejecuta la llamada.

    ``provider_options`` es la válvula de escape para lo que un proveedor
    concreto expone y el vocabulario común no cubre.  Existe para que nadie
    tenga que bifurcar el tipo; usarla para algo que sí es común es una señal de
    que falta un campo en la especificación.
    """

    model: str
    messages: tuple[Message, ...] = ()
    system: str | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice = AUTO
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    response_format: ResponseFormat | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(m.role is Role.SYSTEM for m in self.messages):
            raise ValueError(
                "El prompt de sistema va en Request.system, no en la lista de mensajes. "
                "Ningún proveedor lo trata como turno."
            )


@dataclass(frozen=True, slots=True)
class Response:
    message: Message
    finish_reason: FinishReason
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.message.tool_calls


# ── Eventos de stream — H2 ────────────────────────────────────────────────────
#
# Ciclo start / delta / end uniforme para texto, razonamiento y tool calls.
# ``index`` distingue bloques concurrentes dentro del mismo turno.
#
# No hay evento de error: los fallos se propagan como excepciones de la
# taxonomía (RM-04).  Un generador asíncrono las propaga limpiamente, y un
# evento de error obligaría a cada consumidor a comprobarlo en cada iteración.

@dataclass(frozen=True, slots=True)
class StreamStart:
    model: str = ""
    kind: Literal["stream_start"] = "stream_start"


@dataclass(frozen=True, slots=True)
class TextStart:
    index: int = 0
    kind: Literal["text_start"] = "text_start"


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str = ""
    index: int = 0
    kind: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, slots=True)
class TextEnd:
    index: int = 0
    kind: Literal["text_end"] = "text_end"


@dataclass(frozen=True, slots=True)
class ReasoningStart:
    index: int = 0
    kind: Literal["reasoning_start"] = "reasoning_start"


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str = ""
    index: int = 0
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(frozen=True, slots=True)
class ReasoningEnd:
    index: int = 0
    signature: str | None = None
    kind: Literal["reasoning_end"] = "reasoning_end"


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    id: str = ""
    name: str = ""
    index: int = 0
    kind: Literal["tool_call_start"] = "tool_call_start"


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """Fragmento de los argumentos, tal como llegan del proveedor.

    Los argumentos se transmiten troceados y sin garantía de ser JSON válido
    hasta el final; quien consume acumula y solo parsea al recibir el
    ``ToolCallEnd`` correspondiente.
    """

    arguments_delta: str = ""
    index: int = 0
    kind: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    index: int = 0
    kind: Literal["tool_call_end"] = "tool_call_end"


@dataclass(frozen=True, slots=True)
class Finish:
    """Cierre del stream con la respuesta acumulada.

    Lleva el ``Response`` completo para que quien consumió los deltas no tenga
    que reconstruirlo, y para que ``Usage`` vuelva por la costura (H3) aunque el
    span de ``chat`` lo emita quien ejecutó la llamada (A5).
    """

    response: Response
    kind: Literal["finish"] = "finish"


StreamEvent = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ReasoningStart
    | ReasoningDelta
    | ReasoningEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | Finish
)


# ── Serialización determinista ────────────────────────────────────────────────

def b64(data: bytes) -> str:
    """Codifica binario para los campos ``data`` de las partes de contenido."""
    return base64.b64encode(data).decode("ascii")


def to_jsonable(obj: Any) -> Any:
    """Convierte a estructuras JSON puras, recursivamente.

    Los ``None`` se omiten: reducen el tamaño del journal y evitan que añadir un
    campo opcional cambie la serialización de valores que no lo usan, lo que
    rompería la comparación de replay entre versiones.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if value is None:
                continue
            out[f.name] = to_jsonable(value)
        return out
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (str, bytes)):
        return b64(obj) if isinstance(obj, bytes) else obj
    if isinstance(obj, Sequence):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"No serializable: {type(obj).__name__}")


def dumps(obj: Any) -> str:
    """Serializa de forma determinista.

    Claves ordenadas y separadores fijos: dos objetos iguales producen la misma
    cadena byte a byte, en cualquier proceso y en cualquier ejecución.  Es lo
    que hace comparables los eventos entre ejecución y replay, y lo que permite
    que un prefijo de prompt sea estable frente a la caché del proveedor.
    """
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
