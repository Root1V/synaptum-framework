"""SYN-37 · La estructura del bucle, como trazas.

Dos preguntas distintas, y la segunda es la que importa:

* **¿hay spans?** — la fácil, y la que deja pasar el defecto;
* **¿forman una traza?** — dos spans pueden existir, compartir `trace_id` y ser
  los dos raíz. En una lista se leen como una traza; en un waterfall son dos
  historias desconectadas.

Aeon dio por cerrada su instrumentación con tres aserciones ciertas —«existe el
span `chat`», «existe `execute_tool»`, «existe `invoke_agent`»— y una conclusión
falsa: el propagador global de Go es no-op por defecto, así que nunca formaron
una traza. Estos tests preguntan lo segundo.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest

from synaptum import (
    Agent,
    Decision,
    Disposition,
    Risk,
    Session,
    tool,
)
from synaptum.testing import FakeGateway, calls, says

otel = pytest.importorskip(
    "opentelemetry.sdk.trace", reason="las trazas necesitan el extra `otel`"
)

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from synaptum.telemetry import ATRIBUTO_GUARDRAIL, describe_tracing, traced  # noqa: E402


@tool(idempotent=True)
async def mirar(cosa: Annotated[str, "Qué mirar"]) -> str:
    """Mira algo."""
    return f"{cosa}: bien"


@tool(risk=Risk.DESTRUCTIVE)
async def borrar(ruta: Annotated[str, "Ruta"]) -> str:
    """Borra."""
    return "borrado"


@pytest.fixture
def spans():
    """Un proveedor de verdad, con su exportador en memoria."""
    exportador = InMemorySpanExporter()
    proveedor = TracerProvider()
    proveedor.add_span_processor(SimpleSpanProcessor(exportador))
    anterior = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = proveedor          # el global se fija una sola vez
    yield exportador
    trace._TRACER_PROVIDER = anterior if not isinstance(anterior, TracerProvider) else None


def _correr(agente, puerta, run_id="r1", tarea="mira el disco"):
    async def ir():
        sesion = Session(run_id, puerta)
        return [
            p
            async for p in traced(
                agente.run(tarea, session=sesion),
                run_id=run_id, agent=agente.name, model=agente.model,
            )
        ]

    return asyncio.run(ir())


def test_the_spans_of_a_run_are_one_trace_and_not_several_roots(spans):
    """La pregunta no es «¿hay spans?», es «¿son una traza?»."""
    agente = Agent("vigía", model="fake:m", instructions="Mira.", tools=[mirar])
    puerta = FakeGateway(calls("mirar", cosa="disco"), says("Bien."), tools=[mirar])

    _correr(agente, puerta)
    emitidos = spans.get_finished_spans()

    assert emitidos, "no se emitió ningún span"

    trazas = {s.context.trace_id for s in emitidos}
    assert len(trazas) == 1, f"{len(emitidos)} spans repartidos en {len(trazas)} trazas"

    raices = [s for s in emitidos if s.parent is None]
    assert len(raices) == 1, (
        f"{len(raices)} spans sin padre: comparten traza y aun así son historias "
        "desconectadas en cualquier waterfall"
    )
    assert raices[0].name == "agent.run"


def test_a_turn_boundary_becomes_a_span(spans):
    agente = Agent("vigía", model="fake:m", instructions="Mira.", tools=[mirar])
    puerta = FakeGateway(calls("mirar", cosa="disco"), says("Bien."), tools=[mirar])

    _correr(agente, puerta)
    turnos = [s for s in spans.get_finished_spans() if s.name == "agent.turn"]

    assert len(turnos) == 2, "dos llamadas al modelo son dos turnos"
    assert [s.attributes["synaptum.turn"] for s in turnos] == [1, 2]


def test_the_model_call_is_not_traced_here(spans):
    """El `chat` lo emite quien lo ejecuta, y duplicarlo sería peor que no tenerlo.

    Dos spans para un mismo hecho traen dos duraciones que nunca coinciden y un
    lector eligiendo la que le parezca.
    """
    agente = Agent("vigía", model="fake:m", instructions="Mira.", tools=[mirar])
    puerta = FakeGateway(calls("mirar", cosa="disco"), says("Bien."), tools=[mirar])

    _correr(agente, puerta)
    nombres = {s.name for s in spans.get_finished_spans()}

    assert "chat" not in nombres and "execute_tool" not in nombres
    assert nombres == {"agent.turn", "agent.run"}


def test_a_governance_denial_is_marked_hot_and_is_not_an_error(spans):
    """Una denegación de política es el sistema funcionando.

    Marcarla como error enterraría los fallos de verdad bajo un flujo de
    rechazos correctos — y el atributo la enruta igual, porque es su propio
    disparador.
    """
    from opentelemetry.trace import StatusCode

    agente = Agent("vigía", model="fake:m", instructions="Borra.", tools=[borrar])
    puerta = FakeGateway(
        calls("borrar", ruta="/"),
        says("No pude."),
        tools=[borrar],
        deny_tools={"borrar": Decision(Disposition.DENY_STEP, reason_code="politica-dura")},
    )

    _correr(agente, puerta, tarea="borra /")
    marcados = [
        s for s in spans.get_finished_spans() if ATRIBUTO_GUARDRAIL in (s.attributes or {})
    ]

    assert marcados, "una denegación de gobierno no llegó al camino caliente"
    span = marcados[0]
    assert span.attributes[ATRIBUTO_GUARDRAIL] == "deny_step", "lleva el tipo, no un booleano"
    assert span.status.status_code is not StatusCode.ERROR
    assert any(e.name == "synaptum.denied" for e in span.events)


def test_the_span_carries_the_time_of_the_event_not_of_the_observation(spans):
    """Un paso reproducido del diario cuenta su duración original.

    Si el span se midiera al observarlo, una reanudación diría que el turno duró
    microsegundos — y la traza de un run reanudado sería la del replay, no la
    del run.
    """
    agente = Agent("vigía", model="fake:m", instructions="Mira.", tools=[mirar])
    puerta = FakeGateway(calls("mirar", cosa="disco"), says("Bien."), tools=[mirar])

    pasos = _correr(agente, puerta)
    turno = next(s for s in spans.get_finished_spans() if s.name == "agent.turn")
    primero = next(p for p in pasos if getattr(p, "at", None))

    assert abs(turno.start_time - int(primero.at * 1_000_000_000)) < 1_000_000_000


def test_unconfigured_tracing_says_it_goes_nowhere():
    """El modo de fallo que no produce ningún error.

    Sin proveedor, los spans se emiten, no llegan a ningún sitio, y la
    aplicación funciona. No hay excepción que mirar: solo la ausencia de
    trazas, que es lo que nadie mira al montar esto.
    """
    from opentelemetry.trace import NoOpTracerProvider

    anterior = trace._TRACER_PROVIDER
    trace._TRACER_PROVIDER = NoOpTracerProvider()
    try:
        assert "EN NINGUNA PARTE" in describe_tracing()
    finally:
        trace._TRACER_PROVIDER = anterior
