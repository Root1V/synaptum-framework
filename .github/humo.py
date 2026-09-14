"""Lo mínimo que un consumidor hace, contra la copia instalada desde el índice.

Vive en un fichero y no incrustado en el workflow porque un heredoc dentro de un
bloque YAML dentro de un bucle de reintentos es tres capas de citado que nadie
quiere depurar en un incidente. Además así se puede ejecutar en local, que es
donde conviene descubrir que está roto.

Importa **desde el paquete**, nunca desde un submódulo interno: es la diferencia
que dejó a Axonium con un rc publicado al que le faltaban seis exportaciones y
el CI en verde.
"""

import asyncio

import synaptum
from synaptum import Agent, Session, tool
from synaptum.testing import FakeGateway, says


@tool
async def eco(texto: str) -> str:
    """Devuelve lo que recibe."""
    return texto


async def main() -> None:
    agente = Agent("humo", model="openai-compatible:m", tools=[eco])
    pasos = [
        paso
        async for paso in agente.run(
            "hola", session=Session("r1", FakeGateway(says("hola")))
        )
    ]
    assert pasos[-1].output == "hola", pasos[-1]
    print("instalado desde el índice y funcionando:", synaptum.__version__)


if __name__ == "__main__":
    asyncio.run(main())
