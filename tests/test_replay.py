"""SYN-66 · Respuestas reales grabadas, sin acceso y sin coste."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from synaptum import Agent, CallContext, ConfigurationError, Request, Session, tool
from synaptum.testing import ReplayGateway, split_sse

_BODIES = Path(
    "/Users/emericespiritusantiago/Documents/Victor/coordinacion_project"
    "/contratos/gateway-prometheus/fixtures"
)

pytestmark = pytest.mark.skipif(
    not _BODIES.exists(), reason="cuerpos grabados compartidos no disponibles"
)


@tool(idempotent=True)
def get_weather(city: str) -> str:
    """Consulta el tiempo."""
    return f"soleado en {city}"


def drain(agent: Agent, task: str, session: Session) -> list:
    async def go():
        return [e async for e in agent.run(task, session=session)]

    return asyncio.run(go())


def test_a_recorded_body_reaches_the_loop_exactly_as_production_would():
    gateway = ReplayGateway(_BODIES / "chat_completion_finished.json")
    events = drain(Agent("a", model="openai-compatible:qwen3"), "di hola",
                   Session("run-1", gateway))

    assert events[-1].output == "hello", "el razonamiento no se cuela en la respuesta"
    assert events[-1].usage.input == 13, "el consumo es el que reportó el proveedor"
    assert events[-1].usage.cache_read == 3


def test_a_real_recording_exposes_the_answer_that_is_only_reasoning():
    """Grabación real: el modelo nunca sale de la fase de razonamiento.

    `content` vacío, `finish_reason` `length`, y consumo que pagar. Un guion
    escrito a mano no produce este caso porque nadie lo escribe — y un cliente
    que solo mire el texto ve una respuesta vacía sin explicación.
    """
    from synaptum import Thinking

    gateway = ReplayGateway(_BODIES / "chat_completion.json")
    events = drain(Agent("a", model="openai-compatible:qwen3"), "di hola",
                   Session("run-1", gateway))

    assert events[-1].output == "", "no hubo texto"
    assert events[-1].usage.output == 24, "pero sí hubo 24 tokens que pagar"

    from synaptum import ModelStep, Phase

    respuesta = next(
        e for e in events if isinstance(e, ModelStep) and e.phase is Phase.COMPLETED
    ).response
    assert isinstance(respuesta.message.content[0], Thinking), "el razonamiento se conserva"
    assert respuesta.finish_reason.value == "length"


def test_a_recorded_tool_call_exercises_a_path_nobody_writes_by_hand():
    """`content` nulo junto a tool calls y argumentos como cadena JSON.

    Contra un guion escrito a mano ese camino no se recorre: nadie escribe el
    caso que no se le ocurre.
    """
    gateway = ReplayGateway(
        _BODIES / "chat_completion_tool_calls.json",
        _BODIES / "chat_completion_finished.json",
        tools=[get_weather],
    )
    agent = Agent("a", model="openai-compatible:llama3-8b-q4", tools=[get_weather])
    events = drain(agent, "qué tiempo hace", Session("run-1", gateway))

    from synaptum import ToolStep

    resultado = next(e for e in events if isinstance(e, ToolStep) and e.result is not None)
    assert resultado.result.content[0].text == "soleado en Lima", (
        "los argumentos llegaron decodificados desde la cadena JSON del cable"
    )
    assert gateway.tool_calls == 1


def test_a_derived_usage_keeps_its_estimated_flag_through_the_loop():
    """El bit que Axonium pidió tiene que sobrevivir hasta el cierre del run."""
    gateway = ReplayGateway(_BODIES / "chat_stream_ok.sse")
    events = drain(Agent("a", model="openai-compatible:llama3-8b-q4"), "hola",
                   Session("run-1", gateway))

    consumo = events[-1].usage
    assert consumo.input == 15, "prompt_n=1 + cache_n=14, convención inclusiva"
    assert consumo.cache_read == 14
    assert consumo.cache_write is None, "llama.cpp no lo reporta: sin medir, no cero"
    assert consumo.estimated is True


def test_a_stream_body_can_be_served_as_a_full_response():
    gateway = ReplayGateway(_BODIES / "chat_stream_usage.sse")
    events = drain(Agent("a", model="openai-compatible:m"), "hola",
                   Session("run-1", gateway))
    assert events[-1].output == "Hi"


def test_streaming_yields_the_unified_cycle():
    gateway = ReplayGateway(_BODIES / "chat_stream_ok.sse")

    async def go():
        ctx = CallContext(run_id="run-1", step_id="000000-model")
        return [
            e async for e in gateway.stream_model(Request(model="openai-compatible:m"), ctx)
        ]

    events = asyncio.run(go())
    assert events[0].kind == "stream_start" and events[-1].kind == "finish"
    # La grabación real es de un modelo de razonamiento que no llega a responder:
    # todo lo que emite va por el ciclo de razonamiento, no por el de texto.
    assert [e.kind for e in events[1:3]] == ["reasoning_start", "reasoning_delta"]
    assert events[-2].kind == "reasoning_end"
    assert "".join(e.text for e in events if e.kind == "reasoning_delta").startswith("Okay")


def test_a_missing_body_fails_at_construction():
    with pytest.raises(ConfigurationError, match="No existen"):
        ReplayGateway(_BODIES / "no-existe.json")


def test_running_out_of_bodies_says_so_clearly():
    gateway = ReplayGateway(_BODIES / "chat_completion_tool_calls.json", tools=[get_weather])
    agent = Agent("a", model="openai-compatible:llama3-8b-q4", tools=[get_weather])
    with pytest.raises(AssertionError, match="agotaron"):
        drain(agent, "x", Session("run-1", gateway))


def test_rewind_allows_replaying_the_same_run_twice():
    gateway = ReplayGateway(_BODIES / "chat_completion_finished.json")
    agent = Agent("a", model="openai-compatible:qwen3")
    drain(agent, "hola", Session("run-1", gateway))
    gateway.rewind()
    events = drain(agent, "hola", Session("run-2", gateway))
    assert events[-1].output == "hello"


def test_the_sse_splitter_drops_the_terminator():
    chunks = split_sse('data: {"a": 1}\n\ndata: [DONE]\n')
    assert chunks == [{"a": 1}]
