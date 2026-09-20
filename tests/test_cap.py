"""SYN-33 · Recorte de la salida de herramientas.

Lo que se comprueba aquí no es que recorte —eso es aritmética— sino las cuatro
propiedades que hacen que recortar sea seguro: que el journal conserve lo
entero, que reanudar dé el mismo contexto, que se diga lo que se quitó, y que el
error conserve su final.
"""

from __future__ import annotations

import asyncio

import pytest

from synaptum import Agent, Limits, MemoryCheckpointer, Session, Text, ToolResult, tool
from synaptum.context import cap_tool_output
from synaptum.context.cap import DEFAULT_MAX_CHARS
from synaptum.testing import FakeGateway, calls, says

LARGO = "línea de log con mucho ruido\n" * 3000


@tool
async def leer_log(path: str) -> str:
    """Lee un fichero de log."""
    return LARGO


def _texto(resultado: ToolResult) -> str:
    return "".join(p.text for p in resultado.content if isinstance(p, Text))


def _contexto(gateway) -> str:
    """Todo el texto que llegó al modelo en la última petición."""
    return "".join(
        t.text
        for m in gateway.requests[-1].messages
        for parte in m.content
        for t in (getattr(parte, "content", None) or [parte])
        if hasattr(t, "text")
    )


# ── La función, sola ──────────────────────────────────────────────────────────

def test_a_result_that_fits_is_returned_untouched():
    """Mismo objeto, no una copia: quien compare por identidad sabe si se tocó."""
    cabe = ToolResult.of("c1", "corto")
    assert cap_tool_output(cabe, max_chars=100) is cabe


def test_an_oversized_result_is_cut_to_the_cap():
    recortado = cap_tool_output(ToolResult.of("c1", LARGO), max_chars=1000)
    assert len(_texto(recortado)) <= 1000


def test_the_cut_says_how_much_it_removed():
    """Un texto truncado en silencio hace concluir sobre datos incompletos."""
    texto = _texto(cap_tool_output(ToolResult.of("c1", LARGO), max_chars=1000))
    assert "recortado" in texto
    assert str(len(LARGO))[:2] in texto.replace(".", ""), "no dice el tamaño original"


def test_both_ends_survive():
    """El encabezado dice qué es; el final trae el total o la excepción."""
    texto = "PRINCIPIO" + ("x" * 5000) + "FINAL"
    recortado = _texto(cap_tool_output(ToolResult.of("c1", texto), max_chars=800))
    assert recortado.startswith("PRINCIPIO")
    assert recortado.endswith("FINAL")


def test_an_error_keeps_more_of_its_tail():
    """El mensaje de una traza está abajo: cortar por el final lo tira."""
    traza = "Traceback:\n" + ("  File x\n" * 500) + "ValueError: el dato no cuadra"
    error = cap_tool_output(ToolResult.of("c1", traza, is_error=True), max_chars=600)
    normal = cap_tool_output(ToolResult.of("c1", traza), max_chars=600)

    assert "ValueError: el dato no cuadra" in _texto(error)
    cola_error = len(_texto(error).split("…]")[-1])
    cola_normal = len(_texto(normal).split("…]")[-1])
    assert cola_error > cola_normal, "un error no conservó más cola que un resultado normal"


def test_non_text_parts_are_left_alone():
    """Una imagen no se recorta por la mitad."""
    from synaptum import Image

    resultado = ToolResult(
        call_id="c1",
        content=(Text(LARGO), Image(media_type="image/png", data="AAAA")),
    )
    recortado = cap_tool_output(resultado, max_chars=500)
    assert any(p.kind == "image" for p in recortado.content)


def test_none_disables_it():
    entero = ToolResult.of("c1", LARGO)
    assert cap_tool_output(entero, max_chars=None) is entero


def test_a_cap_too_small_for_the_notice_still_respects_the_cap():
    """Quien puso el número manda, aunque el número sea absurdo."""
    assert len(_texto(cap_tool_output(ToolResult.of("c1", LARGO), max_chars=20))) <= 20


# ── En el bucle ───────────────────────────────────────────────────────────────

def _correr(limites: Limits, store=None, run_id: str = "r1"):
    gateway = FakeGateway(calls("leer_log", path="/x"), says("ok"), tools=[leer_log])
    agente = Agent("a", model="openai-compatible:m", tools=[leer_log], limits=limites)

    async def ir():
        sesion = Session(run_id, gateway, store) if store else Session(run_id, gateway)
        return [p async for p in agente.run("lee el log", session=sesion)]

    return gateway, asyncio.run(ir())


def test_the_cap_is_on_by_default():
    """No recortar falla en silencio: revienta con ventana pequeña y cuesta con la grande."""
    gateway, _ = _correr(Limits())
    sin_tope, _ = _correr(Limits(max_tool_chars=None))

    assert len(_contexto(gateway)) < len(_contexto(sin_tope)) / 5
    assert len(_contexto(gateway)) <= DEFAULT_MAX_CHARS + 500


def test_the_journal_keeps_the_whole_thing():
    """El diario cuenta lo que ocurrió; el contexto lleva lo que el modelo ve."""
    _, pasos = _correr(Limits(max_tool_chars=500))
    paso = next(p for p in pasos if p.kind == "tool" and p.phase.value == "completed")

    assert len(_texto(paso.result)) == len(LARGO), "el journal guardó el resultado recortado"


def test_resuming_rebuilds_the_very_same_context():
    """La propiedad que hace seguro recortar.

    Al reanudar, el contexto se vuelve a derivar del journal. Si el recorte se
    aplicara solo al ejecutar —y no también sobre lo que viene del journal— el
    prompt reconstruido sería distinto del original: caché fallada, y
    potencialmente otra respuesta a la misma pregunta.

    Se comparan las dos formas de construir **la misma petición**: la que lleva
    el resultado de la herramienta recién ejecutada, y la que lo lleva traído del
    diario.
    """
    limites = Limits(max_tool_chars=800)

    # ── Camino A: de una sentada ──────────────────────────────────────────────
    fresco = FakeGateway(calls("leer_log", path="/x"), says("ok"), tools=[leer_log])
    agente = Agent("a", model="openai-compatible:m", tools=[leer_log], limits=limites)

    async def entero():
        async for _ in agente.run("lee el log", session=Session("a1", fresco)):
            pass

    asyncio.run(entero())
    de_una_sentada = _contexto(fresco)

    # ── Camino B: cortado tras la herramienta, y reanudado ────────────────────
    class _Corte(Exception):
        pass

    store = MemoryCheckpointer()
    primero = FakeGateway(calls("leer_log", path="/x"), says("ok"), tools=[leer_log])

    async def hasta_la_herramienta():
        async for paso in agente.run("lee el log", session=Session("b1", primero, store)):
            if paso.kind == "tool" and paso.phase.value == "completed":
                raise _Corte

    with pytest.raises(_Corte):
        asyncio.run(hasta_la_herramienta())

    segundo = FakeGateway(says("ok"), tools=[leer_log])

    async def reanudar():
        async for _ in agente.run("lee el log", session=Session("b1", segundo, store)):
            pass

    asyncio.run(reanudar())

    assert segundo.requests, "la reanudación no llegó a llamar al modelo"
    assert _contexto(segundo) == de_una_sentada, (
        "el contexto reconstruido no es idéntico al original: la caché fallaría "
        "y el modelo podría responder otra cosa a la misma pregunta"
    )
