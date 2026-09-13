"""El bucle entero, en un fichero.

Un agente con dos herramientas que responde una pregunta sobre un repositorio.
Lo interesante no es la tarea: es que **todo lo que el agente hace pasa por el
stream de eventos**, así que verlo es hacer `async for`.

    uv run python examples/01_agente.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from synaptum import (
    Agent,
    FinalStep,
    ModelStep,
    Phase,
    Risk,
    Session,
    ToolStep,
    tool,
)
from synaptum.testing import calls, says

from _comun import encabezado, gateway, nombre_del_modelo

RAIZ = Path(__file__).resolve().parent.parent


@tool
async def listar(
    carpeta: Annotated[str, "Ruta relativa a la raíz del proyecto"] = ".",
) -> str:
    """Lista los ficheros de una carpeta del proyecto."""
    destino = (RAIZ / carpeta).resolve()
    if RAIZ not in destino.parents and destino != RAIZ:
        return f"Fuera del proyecto: {carpeta}"
    return "\n".join(sorted(p.name for p in destino.iterdir() if not p.name.startswith(".")))


@tool(risk=Risk.READ, idempotent=True)
async def contar_lineas(
    fichero: Annotated[str, "Ruta relativa a la raíz del proyecto"],
) -> str:
    """Cuenta las líneas de un fichero del proyecto."""
    destino = (RAIZ / fichero).resolve()
    if not destino.is_file():
        return f"No existe: {fichero}"
    return f"{fichero}: {len(destino.read_text().splitlines())} líneas"


# `risk` e `idempotent` se declaran y no se deducen: ninguna anotación puede
# saber que una función que devuelve `str` mueve dinero.  Synaptum **declara**;
# quien gobierne **decide**.


async def main() -> None:
    encabezado("01 · El bucle entero")

    agente = Agent(
        "explorador",
        model=nombre_del_modelo(),
        instructions="Respondes sobre el repositorio usando las herramientas. Sé breve.",
        tools=[listar, contar_lineas],
    )

    # El guion solo se usa cuando no hay modelo real.  Con SYNAPTUM_BASE_URL
    # puesto, el agente es exactamente el mismo y las respuestas las da el
    # modelo.
    guion = [
        calls("contar_lineas", fichero="README.md"),
        # El doble puede responder *según lo que el bucle acaba de mandar*, así
        # que la respuesta final no contradice a la herramienta.  Es la
        # diferencia entre un guion y un mock que devuelve una constante.
        lambda peticion: says(f"El {_ultimo_resultado(peticion)}."),
    ]

    puerta = gateway(guion, tools=[listar, contar_lineas])
    sesion = Session("ejemplo-01", puerta)

    async for paso in agente.run("¿Cuántas líneas tiene el README?", session=sesion):
        match paso:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                argumentos = ", ".join(f"{k}={v!r}" for k, v in llamada.arguments.items())
                print(f"  herramienta  {llamada.name}({argumentos})")
            case ToolStep(phase=Phase.COMPLETED, result=resultado) if resultado:
                print(f"  ↳            {_texto(resultado)}")
            case ModelStep(phase=Phase.COMPLETED, usage=consumo):
                print(f"  modelo       {_consumo(consumo)}")
            case FinalStep(output=salida, usage=total):
                print(f"\n  {salida}")
                print(f"\n  total        {_consumo(total)}")


def _ultimo_resultado(peticion) -> str:
    """Lo que la última herramienta devolvió, leído del propio contexto."""
    from synaptum import Role, Text

    for mensaje in reversed(peticion.messages):
        if mensaje.role is Role.TOOL:
            return " ".join(
                parte.text
                for resultado in mensaje.content
                for parte in getattr(resultado, "content", ())
                if isinstance(parte, Text)
            ).strip()
    return "no hay resultado"


def _texto(resultado) -> str:
    from synaptum import Text

    return " ".join(p.text for p in resultado.content if isinstance(p, Text)).strip()[:70]


def _consumo(u) -> str:
    """Los tres estados de `Usage`, dichos como son.

    `None` no es cero. Un cero diría «no hubo»; la verdad es «nadie lo midió»,
    y confundirlas hace creer que escribir en caché es gratis.
    """
    partes = []
    for nombre in ("input", "output", "cache_read"):
        valor = getattr(u, nombre)
        partes.append(f"{nombre}={'sin medir' if valor is None else valor}")
    if u.estimated:
        partes.append("(estimado)")
    return " · ".join(partes)


if __name__ == "__main__":
    asyncio.run(main())
