"""SYN-52 · El puente hacia Axonium.

Se prueba contra los **tipos reales del SDK**, construidos a mano. Lo que no se
puede probar aquí es la llamada de red: exige una instancia de Prometheus, y por
P10 no hay ninguna al alcance. Eso queda dicho en vez de simulado.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("axonium", reason="extra 'axonium' no instalado")

from axonium import ChatCompletion, Usage as AxUsage  # noqa: E402

from synaptum import FinishReason, Request, Response, Thinking, ToolCall  # noqa: E402
from synaptum.providers.axonium import (  # noqa: E402
    AxoniumModel,
    message_from_axonium,
    usage_from_axonium,
)


def _completion(**message) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "llama3-8b-q4",
            "choices": [{"index": 0, "message": {"role": "assistant", **message},
                         "finish_reason": message.pop("_finish", "stop")}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 20,
                      "total_tokens": 35, "cache_read_tokens": 3, "estimated": True},
        }
    )


# ── Lo que motivó que este módulo exista ──────────────────────────────────────

def test_the_cache_and_estimated_fields_survive_the_translation():
    """Pasarlo por un lector de cable los buscaría donde no están y los perdería."""
    unificado = usage_from_axonium(
        AxUsage(prompt_tokens=15, completion_tokens=20, total_tokens=35,
                cache_read_tokens=3, estimated=True)
    )
    assert unificado.input == 15, "inclusivo: contiene lo cacheado"
    assert unificado.cache_read == 3
    assert unificado.estimated is True


def test_what_prometheus_cannot_measure_stays_unmeasured():
    """Sin medir, no cero — cero diría «no hubo»."""
    unificado = usage_from_axonium(
        AxUsage(prompt_tokens=15, completion_tokens=20, total_tokens=35)
    )
    assert unificado.reasoning is None
    assert unificado.cache_write is None
    assert unificado.cache_read is None, "este backend no lo reportó"


def test_no_usage_at_all_is_everything_unmeasured():
    assert usage_from_axonium(None).input is None


# ── Mensajes ──────────────────────────────────────────────────────────────────

def test_reasoning_and_answer_stay_separate():
    completion = _completion(content="la respuesta", reasoning_content="pensando")
    message = message_from_axonium(completion.choices[0].message)

    assert isinstance(message.content[0], Thinking)
    assert message.text == "la respuesta", "el razonamiento no se concatena"


def test_a_null_content_with_tool_calls_produces_no_empty_text_part():
    completion = _completion(
        content=None,
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Lima"}'}}],
    )
    message = message_from_axonium(completion.choices[0].message)

    assert [p.kind for p in message.content] == ["tool_call"]
    assert message.text == ""


def test_tool_call_arguments_arrive_decoded():
    completion = _completion(
        content=None,
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Lima"}'}}],
    )
    call = message_from_axonium(completion.choices[0].message).tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.arguments == {"city": "Lima"}


def test_unreadable_arguments_are_not_turned_into_an_empty_object():
    """Un `{}` silencioso ejecutaría la herramienta sin argumentos."""
    from synaptum import ProviderError

    completion = _completion(
        content=None,
        tool_calls=[{"id": "c", "type": "function",
                     "function": {"name": "t", "arguments": "{roto"}}],
    )
    with pytest.raises(ProviderError, match="ilegibles"):
        message_from_axonium(completion.choices[0].message)


# ── El modelo, contra un cliente falso ────────────────────────────────────────

class _ClienteFalso:
    def __init__(self, completion):
        self.completion = completion
        self.llamadas: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.llamadas.append(kwargs)
        return self.completion


def test_the_request_reaches_axonium_with_the_bare_model_name():
    """Axonium recibe el nombre a secas: el prefijo dice quién normaliza, no qué modelo."""
    cliente = _ClienteFalso(_completion(content="hola"))
    modelo = AxoniumModel(client=cliente)

    respuesta = asyncio.run(
        modelo.complete(Request(model="axonium:llama3-8b-q4", system="sé breve"))
    )

    enviado = cliente.llamadas[0]
    assert enviado["model"] == "llama3-8b-q4"
    assert enviado["messages"][0] == {"role": "system", "content": "sé breve"}
    assert isinstance(respuesta, Response)
    assert respuesta.text == "hola"
    assert respuesta.finish_reason is FinishReason.STOP
    assert respuesta.usage.cache_read == 3, "el desglose llegó hasta el bucle"


def test_a_missing_sdk_says_where_to_get_it():
    from synaptum.core.errors import ConfigurationError
    from synaptum.providers import axonium as puente

    original = puente._build_client
    try:
        def explota(**_):
            raise ConfigurationError("Axonium no está instalado.")

        puente._build_client = explota
        with pytest.raises(ConfigurationError, match="no está instalado"):
            AxoniumModel()
    finally:
        puente._build_client = original


# ── Streaming del puente ──────────────────────────────────────────────────────
#
# El puente tiene su PROPIA implementación de streaming: no reutiliza la del
# adaptador de cable, porque Axonium ya entrega objetos normalizados.  Eso
# significa que un arreglo en el adaptador no lo alcanza — y no lo alcanzó.

class _StreamFalso:
    """Cliente falso que entrega los fragmentos como los entrega el SDK."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.chat = self
        self.completions = self

    def stream(self, **_):
        chunks = self.chunks

        class _Ctx:
            async def __aenter__(self):
                async def gen():
                    for c in chunks:
                        yield c

                return gen()

            async def __aexit__(self, *_):
                return False

        return _Ctx()


def _delta(**campos):
    from types import SimpleNamespace

    completos = {"content": None, "reasoning_content": None, "tool_calls": None}
    completos.update(campos)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(**completos), finish_reason=None)],
        usage=None,
    )


def _cierre(motivo="stop"):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(delta=None, finish_reason=motivo)], usage=None
    )


def _eventos(chunks):
    modelo = AxoniumModel(client=_StreamFalso(chunks))

    async def go():
        return [e async for e in modelo.stream(Request(model="axonium:m"))]

    return asyncio.run(go())


def test_the_bridge_emits_the_reasoning_cycle():
    """Acumularlo sin emitirlo deja a quien consume sin nada que ver.

    Encontrado ejecutando el puente contra la plataforma por primera vez: el
    adaptador de cable ya lo hacía bien —lo fijó el corpus dorado— y el puente
    no, porque el corpus prueba normalización de cuerpos y esto no pasa por
    ningún cuerpo. Un arreglo en un sitio no alcanza al otro.
    """
    eventos = _eventos([
        _delta(reasoning_content="pien"),
        _delta(reasoning_content="so"),
        _delta(content="hola"),
        _cierre(),
    ])
    tipos = [e.kind for e in eventos]

    assert tipos == [
        "stream_start",
        "reasoning_start", "reasoning_delta", "reasoning_delta", "reasoning_end",
        "text_start", "text_delta", "text_end",
        "finish",
    ]
    assert "".join(e.text for e in eventos if e.kind == "reasoning_delta") == "pienso"
    assert eventos[-1].response.text == "hola", "el razonamiento no se cuela en la respuesta"


def test_the_bridge_closes_the_reasoning_cycle_when_a_tool_call_starts():
    """La fase termina al empezar cualquier otra cosa, no solo el texto.

    Con un modelo de razonamiento que llama a una herramienta no llega ni un
    token de respuesta, así que cerrar solo con `content` deja el bloque de
    pensamiento abierto mientras los argumentos ya están llegando.
    """
    eventos = _eventos([
        _delta(reasoning_content="pienso"),
        _delta(tool_calls=[{"index": 0, "id": "c1",
                            "function": {"name": "f", "arguments": '{"a":1}'}}]),
        _cierre("tool_calls"),
    ])
    tipos = [e.kind for e in eventos]

    assert tipos.index("reasoning_end") < tipos.index("tool_call_start")
    assert eventos[-1].response.tool_calls[0].arguments == {"a": 1}
