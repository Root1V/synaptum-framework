"""Probar un agente sin gastar en inferencia.

El doble **no es un mock**: las herramientas se ejecutan de verdad, el journal se
escribe y el consumo se calcula. Lo único que no hace es inferir. Probar un
agente es iterar una lista — no hay que arrancar nada.
"""

from __future__ import annotations

from synaptum import FinalStep, MemoryCheckpointer, Phase, Session, ToolStep
from synaptum.testing import FakeGateway, calls, says

from mi_agente.agente import HERRAMIENTAS, Respuesta, construir

BIEN = (
    '{"respuesta": "El README tiene 34 líneas.", "ficheros_consultados": ["README.md"]}'
)


def _gateway(*guion):
    return FakeGateway(*guion, tools=HERRAMIENTAS)


async def test_consulta_antes_de_responder():
    """Lo que se prueba de un agente es **qué decide**, no qué dice."""
    gateway = _gateway(calls("contar_lineas", fichero="README.md"), says(BIEN))

    pasos = [
        paso
        async for paso in construir().run("¿cuántas líneas?", session=Session("t1", gateway))
    ]

    # Filtrar por fase, no solo por tipo: un `ToolStep` llega **dos veces** —una
    # al intentarse y otra al completarse— y las dos llevan `call`. Sin el filtro
    # cada herramienta se cuenta doble.
    usadas = [
        p.call.name
        for p in pasos
        if isinstance(p, ToolStep) and p.phase is Phase.ATTEMPTED and p.call
    ]
    assert usadas == ["contar_lineas"], "usó otras herramientas, o ninguna"


async def test_la_salida_llega_tipada():
    gateway = _gateway(calls("contar_lineas", fichero="README.md"), says(BIEN))

    final = [
        paso
        async for paso in construir().run("¿cuántas líneas?", session=Session("t2", gateway))
    ][-1]

    assert isinstance(final, FinalStep)
    assert isinstance(final.output, Respuesta)
    assert final.output.ficheros_consultados == ["README.md"]


async def test_un_error_de_herramienta_vuelve_al_modelo():
    """Un modelo que no ve el error no puede corregirlo.

    Aquí pide un fichero que no existe, la herramienta lo dice, y el modelo
    rectifica — sin que el run se rompa.
    """
    gateway = _gateway(
        calls("contar_lineas", fichero="no-existe.txt"),
        calls("contar_lineas", fichero="README.md"),
        says(BIEN),
    )

    pasos = [
        paso
        async for paso in construir().run("¿cuántas líneas?", session=Session("t3", gateway))
    ]

    resultados = [
        p.result for p in pasos
        if isinstance(p, ToolStep) and p.phase is Phase.COMPLETED and p.result
    ]
    assert "No existe" in resultados[0].content[0].text
    assert isinstance(pasos[-1], FinalStep), "el run se rompió por un error de herramienta"


async def test_reanudar_no_vuelve_a_pagar_la_inferencia():
    """La propiedad que justifica el runtime, comprobada en tu propio agente."""
    store = MemoryCheckpointer()
    agente = construir()

    primero = _gateway(calls("contar_lineas", fichero="README.md"), says(BIEN))
    async for _ in agente.run("¿cuántas líneas?", session=Session("t4", primero, store)):
        pass

    # Mismo run_id, gateway nuevo: no debería hacer falta ni una llamada.
    segundo = _gateway()
    async for _ in agente.run("¿cuántas líneas?", session=Session("t4", segundo, store)):
        pass

    assert segundo.model_calls == 0, "un run cerrado se reabrió y volvió a pagar"
