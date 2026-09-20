"""Cómo se ejecuta, y cómo se elige por dónde salen los efectos.

    uv run python -m mi_agente
"""

from __future__ import annotations

import asyncio
import os

from synaptum import FinalStep, LocalGateway, Phase, Session, ToolStep

from .agente import HERRAMIENTAS, construir


def puerta():
    """El gateway: la única puerta por la que salen los efectos.

    Con `SYNAPTUM_BASE_URL` apunta a un modelo real; sin él, a un doble
    guionizado. **El agente no nota la diferencia**, y eso no es comodidad: es lo
    que permite construir y probar sin gastar.
    """
    if os.environ.get("SYNAPTUM_BASE_URL"):
        from synaptum import HttpModel

        modelo = HttpModel(
            os.environ["SYNAPTUM_BASE_URL"], api_key=os.environ.get("SYNAPTUM_API_KEY")
        )
        # `LocalGateway` **no aplica política**: corre dentro del proceso que
        # gobernaría, así que sus comprobaciones son advisorias. En producción
        # el gateway es un proceso aparte, con las credenciales y capaz de negar.
        return LocalGateway(
            model=modelo, stream=modelo.stream, tools=HERRAMIENTAS, warn=False
        )

    from synaptum.testing import FakeGateway, calls, says

    return FakeGateway(
        calls("contar_lineas", fichero="README.md"),
        says('{"respuesta": "El README tiene 34 líneas.", '
             '"ficheros_consultados": ["README.md"]}'),
        tools=HERRAMIENTAS,
    )


async def main() -> None:
    agente = construir()
    sesion = Session("run-1", puerta())

    # Cada `yield` es una frontera: ver lo que hace el agente es iterar, y
    # pararlo es dejar de iterar.
    async for paso in agente.run("¿Cuántas líneas tiene README.md?", session=sesion):
        match paso:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                print(f"  → {llamada.name}({llamada.arguments})")
            case FinalStep(output=salida, usage=consumo):
                print(f"\n  {salida.respuesta}")
                print(f"  consultó: {salida.ficheros_consultados}")
                print(f"  consumo: entrada={consumo.input} salida={consumo.output}")


if __name__ == "__main__":
    asyncio.run(main())
