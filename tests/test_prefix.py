"""SYN-32 · El prefijo estable de un run.

Lo que se protege aquí no es la caché: es que el journal no describa un run que
ninguna configuración produjo. La caché es la consecuencia visible; la historia
mezclada es la que hace daño y no se ve.
"""

from __future__ import annotations

import asyncio

import pytest

from synaptum import (
    Agent,
    ConfigurationError,
    MemoryCheckpointer,
    Request,
    Session,
    ToolDefinition,
    tool,
)
from synaptum.context import describe_prefix_change, prefix_fingerprint
from synaptum.testing import FakeGateway, calls, says


@tool
async def leer(path: str) -> str:
    """Lee un fichero."""
    return "contenido"


@tool
async def borrar(path: str) -> str:
    """Borra un fichero."""
    return "borrado"


# ── La huella, sola ───────────────────────────────────────────────────────────

def test_the_messages_are_not_part_of_the_prefix():
    """Los mensajes crecen, y crecer al final no invalida nada."""
    from synaptum import Message

    vacio = Request(model="m", system="s")
    con_turnos = Request(model="m", system="s", messages=(Message.user("hola"),))

    assert prefix_fingerprint(vacio) == prefix_fingerprint(con_turnos)


@pytest.mark.parametrize(
    "cambio",
    [
        {"model": "otro"},
        {"system": "otras instrucciones"},
        {"tools": (ToolDefinition(name="leer"), ToolDefinition(name="borrar"))},
    ],
    ids=["modelo", "instrucciones", "herramientas"],
)
def test_anything_in_the_stable_prefix_changes_the_fingerprint(cambio):
    base = Request(model="m", system="s", tools=(ToolDefinition(name="leer"),))
    from dataclasses import replace

    assert prefix_fingerprint(base) != prefix_fingerprint(replace(base, **cambio))


def test_reordering_tools_counts_as_a_change():
    """Reordenar cambia el prefijo por el cable, así que tira la caché igual.

    Fingir que da lo mismo sería mentir sobre lo que cuesta.
    """
    a = Request(model="m", tools=(ToolDefinition(name="leer"), ToolDefinition(name="borrar")))
    b = Request(model="m", tools=(ToolDefinition(name="borrar"), ToolDefinition(name="leer")))

    assert prefix_fingerprint(a) != prefix_fingerprint(b)


def test_a_changed_tool_schema_is_detected_even_with_the_same_names():
    """El caso que más cuesta ver a ojo: mismos nombres, otra firma."""
    antes = Request(model="m", tools=(ToolDefinition(name="leer", parameters={"a": 1}),))
    ahora = Request(model="m", tools=(ToolDefinition(name="leer", parameters={"a": 2}),))

    assert prefix_fingerprint(antes) != prefix_fingerprint(ahora)
    assert "esquema o descripción" in describe_prefix_change(antes, ahora)


def test_the_message_says_what_to_fix_not_that_a_hash_differs():
    """«La huella no coincide» no lo arregla nadie."""
    antes = Request(model="m", tools=(ToolDefinition(name="leer"),))
    ahora = Request(model="otro", tools=(ToolDefinition(name="leer"), ToolDefinition(name="borrar")))

    texto = describe_prefix_change(antes, ahora)
    assert "'m' → 'otro'" in texto
    assert "['leer'] → ['leer', 'borrar']" in texto


# ── En el bucle ───────────────────────────────────────────────────────────────

class _Corte(Exception):
    pass


def _run_a_medias(store, agente, gateway, run_id="r1"):
    async def ir():
        try:
            async for paso in agente.run("t", session=Session(run_id, gateway, store)):
                if paso.kind == "tool" and paso.phase.value == "completed":
                    raise _Corte
        except _Corte:
            pass

    asyncio.run(ir())


def test_resuming_with_the_same_configuration_works():
    """Lo normal tiene que seguir siendo normal."""
    store = MemoryCheckpointer()
    agente = Agent("a", model="openai-compatible:m", instructions="Sé breve.", tools=[leer])

    _run_a_medias(store, agente, FakeGateway(calls("leer", path="/x"), says("ok"), tools=[leer]))

    segundo = FakeGateway(says("ok"), tools=[leer])

    async def reanudar():
        return [p async for p in agente.run("t", session=Session("r1", segundo, store))]

    assert asyncio.run(reanudar())[-1].output == "ok"


def test_resuming_with_another_configuration_is_refused():
    """Reanudar es continuar *ese* run. Otra configuración es otro run.

    Sin esto, la primera mitad del run la ejecuta un agente y la segunda otro, y
    el journal lo registra como uno solo: una auditoría de «qué hizo el agente»
    devuelve una historia que ninguna configuración produjo nunca.
    """
    store = MemoryCheckpointer()
    original = Agent("a", model="openai-compatible:m", instructions="Sé breve.", tools=[leer])
    _run_a_medias(store, original, FakeGateway(calls("leer", path="/x"), says("ok"), tools=[leer]))

    otro = Agent(
        "a", model="openai-compatible:m", instructions="Sé breve.", tools=[leer, borrar]
    )

    async def reanudar():
        return [
            p async for p in otro.run("t", session=Session("r1", FakeGateway(says("ok")), store))
        ]

    with pytest.raises(ConfigurationError, match="herramientas"):
        asyncio.run(reanudar())


def test_the_refusal_names_a_new_run_id_as_the_way_out():
    """Un error que no dice cómo salir deja a alguien atascado."""
    store = MemoryCheckpointer()
    original = Agent("a", model="openai-compatible:m", tools=[leer])
    _run_a_medias(store, original, FakeGateway(calls("leer", path="/x"), says("ok"), tools=[leer]))

    otro = Agent("a", model="openai-compatible:otro", tools=[leer])

    async def reanudar():
        async for _ in otro.run("t", session=Session("r1", FakeGateway(says("ok")), store)):
            pass

    with pytest.raises(ConfigurationError) as fallo:
        asyncio.run(reanudar())

    assert "run_id nuevo" in str(fallo.value)


def test_a_fresh_run_is_never_refused():
    """No hay nada con lo que comparar, y comparar contra nada no es comparar."""
    agente = Agent("a", model="openai-compatible:m", tools=[leer])

    async def ir():
        return [
            p async for p in agente.run("t", session=Session("nuevo", FakeGateway(says("ok"))))
        ]

    assert asyncio.run(ir())[-1].output == "ok"
