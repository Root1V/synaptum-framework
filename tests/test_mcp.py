"""SYN-39 · Cliente MCP, contra un servidor MCP de verdad.

El servidor de `servidores/mcp_demo.py` se arranca como proceso hijo por stdio,
igual que uno real. Un doble de nuestro propio cliente comprobaría que llamamos
a nuestras funciones, que es lo único que no hace falta comprobar.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from synaptum import Agent, LocalGateway, Risk, Session
from synaptum.testing import FakeGateway, calls, says

mcp = pytest.importorskip("mcp", reason="el cliente MCP es un extra opcional")

from synaptum.mcp import MCPTools, mcp_risk  # noqa: E402

SERVIDOR = Path(__file__).parent / "servidores" / "mcp_demo.py"


def con_servidor(cuerpo):
    """Arranca el servidor, ejecuta el cuerpo y lo cierra."""

    async def ir():
        async with MCPTools.stdio(sys.executable, str(SERVIDOR)) as herramientas:
            return await cuerpo(herramientas)

    return asyncio.run(ir())


# ── Descubrimiento y traducción del riesgo ────────────────────────────────────

def test_the_tools_a_server_publishes_arrive_with_their_schema():
    nombres = con_servidor(lambda h: _nombres(h))
    assert {"leer", "borrar", "anotar", "sin_anotaciones"} <= set(nombres)


async def _nombres(herramientas):
    return {t.name for t in herramientas}


def test_hints_become_risk_and_absence_means_destructive():
    """Sin anotaciones, destructiva — y no es una elección nuestra.

    MCP define `destructiveHint` con defecto verdadero y `readOnlyHint` con
    falso: quien no dice nada está diciendo «puede destruir». Nuestro `@tool`
    usa READ por defecto porque el autor está delante y puede declarar; aquí el
    autor no está, y suponer en su lugar es suponer a favor.
    """

    async def cuerpo(herramientas):
        return {t.name: (t.definition.risk, t.definition.idempotent) for t in herramientas}

    por_nombre = con_servidor(cuerpo)

    assert por_nombre["leer"] == (Risk.READ, True)
    assert por_nombre["borrar"][0] is Risk.DESTRUCTIVE
    assert por_nombre["anotar"][0] is Risk.SOFT_WRITE, "no destruye, pero escribe"
    assert por_nombre["sin_anotaciones"] == (Risk.DESTRUCTIVE, False)


def test_an_absent_annotation_block_is_not_a_promise():
    """La unidad de la regla, sin servidor de por medio."""
    assert mcp_risk(None) is Risk.DESTRUCTIVE


def test_the_destructive_ones_can_be_inspected_before_handing_them_over():
    """Con un servidor que no anota nada son **todas**, y conviene verlo."""

    async def cuerpo(herramientas):
        return {t.name for t in herramientas.destructivas}

    assert {"borrar", "sin_anotaciones"} <= con_servidor(cuerpo)


# ── Ejecución ─────────────────────────────────────────────────────────────────

def test_a_tool_runs_on_the_server_and_the_result_comes_back():
    async def cuerpo(herramientas):
        leer = next(t for t in herramientas if t.name == "leer")
        return await leer.invoke("c1", {"ruta": "/x"})

    resultado = con_servidor(cuerpo)
    assert "contenido de /x" in resultado.content[0].text
    assert resultado.is_error is False


def test_a_failure_on_the_server_comes_back_as_evidence_not_as_a_crash():
    """Un modelo que no ve el error no puede corregirlo.

    Y una propiedad de MCP que conviene tener escrita: **el servidor no pasa el
    mensaje original.** Nuestra herramienta lanza «el servidor no pudo» y al
    cliente llega «Error executing tool revienta». Es una decisión razonable del
    servidor —no filtrar sus internos— con una consecuencia para quien construye
    agentes: con herramientas MCP, el modelo recibe *que* falló y casi nunca
    *por qué*, así que no puede corregir como corrige con una tool local.
    """

    async def cuerpo(herramientas):
        revienta = next(t for t in herramientas if t.name == "revienta")
        return await revienta.invoke("c1", {})

    resultado = con_servidor(cuerpo)
    assert resultado.is_error is True, "el fallo llegó como éxito"
    texto = resultado.content[0].text
    assert "revienta" in texto, texto
    assert "el servidor no pudo" not in texto, (
        "si el mensaje original llegara, este test documentaría lo contrario "
        "de lo que ocurre — y habría que cambiar el aviso del ejemplo"
    )


def test_names_can_be_prefixed_so_two_servers_do_not_collide():
    """Dos servidores que publiquen `search` no son distinguibles para el modelo.

    Y el que gana es el que se registró último, en silencio.
    """

    async def ir():
        async with MCPTools.stdio(sys.executable, str(SERVIDOR), prefix="fs.") as h:
            return {t.name for t in h}

    assert "fs.leer" in asyncio.run(ir())


# ── El bucle entero ───────────────────────────────────────────────────────────

def test_an_mcp_tool_works_in_the_loop_like_any_other():
    """Ni el agente ni el gateway notan que la herramienta es remota."""

    async def ir():
        async with MCPTools.stdio(sys.executable, str(SERVIDOR)) as herramientas:
            leer = next(t for t in herramientas if t.name == "leer")
            gateway = FakeGateway(
                calls("leer", ruta="/x"), says("listo"), tools=[leer]
            )
            agente = Agent("a", model="openai-compatible:m", tools=[leer])
            return [
                paso
                async for paso in agente.run("lee /x", session=Session("r1", gateway))
            ]

    pasos = asyncio.run(ir())
    resultado = next(
        p for p in pasos if p.kind == "tool" and p.phase.value == "completed"
    )
    assert "contenido de /x" in resultado.result.content[0].text
    assert pasos[-1].output == "listo"


def test_the_declared_risk_reaches_the_seam():
    """Lo que el servidor insinuó llega al gateway como declaración.

    Es lo que permite que una política deniegue una herramienta ajena sin
    saber nada de ella salvo lo que declaró.
    """
    from synaptum import ALLOW, Decision, Denied, Disposition

    vistos: list = []

    def politica(check):
        vistos.append((check.name, check.risk))
        if check.risk is Risk.DESTRUCTIVE:
            return Decision(disposition=Disposition.DENY_STEP, reason_code="sin_declarar")
        return ALLOW

    async def ir():
        async with MCPTools.stdio(sys.executable, str(SERVIDOR)) as herramientas:
            desconocida = next(t for t in herramientas if t.name == "sin_anotaciones")
            modelo = FakeGateway(calls("sin_anotaciones", x="hola"), says("ya"))
            gateway = LocalGateway(
                model=modelo.invoke_model, tools=[desconocida], policy=politica, warn=False
            )
            agente = Agent("a", model="openai-compatible:m", tools=[desconocida])
            return [
                paso
                async for paso in agente.run("usa la herramienta", session=Session("r1", gateway))
            ]

    asyncio.run(ir())
    assert ("sin_anotaciones", Risk.DESTRUCTIVE) in vistos, (
        "una herramienta ajena sin declarar entró como inofensiva"
    )
