"""SYN-38 · Economía de contexto de un run.

`Usage` dice cuánto se gastó; esto dice **por qué**, que es lo único accionable.
Lo que se comprueba aquí es que el informe no mienta en los dos sitios donde es
fácil: cuando nadie midió, y cuando el prefijo se rompe solo.
"""

from __future__ import annotations

import asyncio

from synaptum import (
    Agent,
    MemoryCheckpointer,
    ModelStep,
    Phase,
    Request,
    RunState,
    Session,
    ToolDefinition,
    Usage,
    tool,
)
from synaptum.context import economy
from synaptum.testing import FakeGateway, calls, says


@tool
async def leer(path: str) -> str:
    """Lee un fichero."""
    return "contenido"


def _paso(step_id: str, *, request=None, usage=None, fase=Phase.ATTEMPTED) -> ModelStep:
    return ModelStep(
        run_id="r", step_id=step_id, step_seq=0, phase=fase,
        request=request, usage=usage or Usage(),
    )


def _diario(*pares) -> RunState:
    """Un journal armado a mano: `(request, usage)` por turno."""
    eventos = []
    for i, (peticion, consumo) in enumerate(pares):
        eventos.append(_paso(f"{i:06d}-model", request=peticion))
        eventos.append(_paso(f"{i:06d}-model", usage=consumo, fase=Phase.COMPLETED))
    return RunState(run_id="r", events=tuple(eventos))


_PREFIJO = Request(model="m", system="Sé breve.", tools=(ToolDefinition(name="leer"),))


# ── Lo que nadie midió no se inventa ──────────────────────────────────────────

def test_an_unmeasured_cache_is_none_not_zero():
    """«Nadie lo midió» y «no hubo acierto» son cosas distintas.

    Confundirlas hace creer que un run sin instrumentar tiene la caché rota, y
    manda a alguien a arreglar lo que no está mal.
    """
    informe = economy(_diario((_PREFIJO, Usage(input=100, output=10))))

    assert informe.cache_hit_ratio is None
    assert "sin medir" in informe.report()


def test_a_measured_zero_is_reported_as_zero():
    informe = economy(_diario((_PREFIJO, Usage(input=100, output=10, cache_read=0))))
    assert informe.cache_hit_ratio == 0.0


def test_the_ratio_weighs_by_size_not_by_turn():
    """Una media de turnos pesaría igual uno de 50 tokens y uno de 50.000."""
    informe = economy(
        _diario(
            (_PREFIJO, Usage(input=50, cache_read=50)),        # 100 % y diminuto
            (_PREFIJO, Usage(input=50_000, cache_read=0)),     # 0 % y enorme
        )
    )
    assert informe.cache_hit_ratio < 0.01, "la media por turno habría dado ~50 %"


# ── El prefijo que se rompe solo ──────────────────────────────────────────────

def test_a_stable_prefix_reports_no_rewrites():
    informe = economy(_diario((_PREFIJO, Usage(input=100)), (_PREFIJO, Usage(input=150))))

    assert informe.prefix_rewrites == 0
    assert "estable durante todo el run" in informe.report()


def test_a_prefix_that_changes_mid_run_is_caught_and_explained():
    """El caso que este informe existe para cazar.

    Reanudar con otra configuración ya está prohibido, pero **dentro** de un run
    el prefijo puede cambiar solo: unas instrucciones con la fecha dentro lo
    reescriben en cada turno, y el síntoma es una caché que nunca arranca.
    """
    from dataclasses import replace

    informe = economy(
        _diario(
            (_PREFIJO, Usage(input=100, cache_read=0)),
            (replace(_PREFIJO, system="Son las 03:14."), Usage(input=150, cache_read=0)),
        )
    )

    assert informe.prefix_rewrites == 1
    texto = informe.report()
    assert "⚠" in texto and "instrucciones de sistema" in texto, texto


def test_the_first_turn_is_never_a_rewrite():
    """No hay nada anterior: contar el estreno como regresión sería absurdo."""
    informe = economy(_diario((_PREFIJO, Usage(input=100))))
    assert informe.prefix_rewrites == 0


# ── Crecimiento ───────────────────────────────────────────────────────────────

def test_growth_is_none_with_a_single_turn():
    """Con un punto no hay pendiente, y devolver 0 fingiría que no crece."""
    assert economy(_diario((_PREFIJO, Usage(input=100)))).input_growth is None


def test_growth_is_the_slope_between_first_and_last():
    informe = economy(
        _diario(
            (_PREFIJO, Usage(input=100)),
            (_PREFIJO, Usage(input=200)),
            (_PREFIJO, Usage(input=300)),
        )
    )
    assert informe.input_growth == 100


# ── Sobre un run de verdad ────────────────────────────────────────────────────

def test_it_works_on_a_real_journal():
    """Se calcula del diario, así que funciona sobre un run ya terminado."""
    store = MemoryCheckpointer()
    agente = Agent("a", model="openai-compatible:m", instructions="Sé breve.", tools=[leer])
    gateway = FakeGateway(calls("leer", path="/x"), says("listo"), tools=[leer])

    async def ir():
        async for _ in agente.run("t", session=Session("r1", gateway, store)):
            pass

    asyncio.run(ir())
    informe = economy(asyncio.run(store.load("r1")))

    assert len(informe.turns) == 2, "un turno por llamada al modelo"
    assert informe.prefix_rewrites == 0
    assert informe.total.input == sum(t.usage.input for t in informe.turns)


def test_a_turn_without_a_result_is_not_counted():
    """Se intentó y no se sabe qué costó: contarlo como cero sería inventar."""
    eventos = (_paso("000000-model", request=_PREFIJO),)   # solo la intención
    assert economy(RunState(run_id="r", events=eventos)).turns == ()
