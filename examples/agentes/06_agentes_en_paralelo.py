"""06 · Varios agentes a la vez, y uno que decide.

**Dominio: Argus bajo tormenta** — 10.446 señales se deduplican en 45
notificaciones. Cuando cae un servicio central, media plataforma se queja a la
vez y las hipótesis compiten: ¿es la red, es el despliegue, es el disco?

Investigarlas en serie multiplica el tiempo hasta el aviso, y ese tiempo está
medido en ~2 s para la detección: tirarlo en la investigación sería desperdiciar
lo caro. En paralelo cuesta lo mismo en tokens y una fracción en reloj.

El patrón es **orchestrator-worker**, y hoy se escribe con `asyncio.gather`
porque el bucle de cada agente es un generador asíncrono de verdad — el motor es
`await`, no una cola de mensajes. Nada que aprender más allá de asyncio.

    uv run python examples/agentes/06_agentes_en_paralelo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import Agent, FinalStep, Session, Usage, tool
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo


@dataclass
class Hipotesis:
    descartada: bool
    confianza: float
    hallazgo: str


# ── Herramientas: una por línea de investigación ──────────────────────────────

@tool(idempotent=True)
async def red(servicio: Annotated[str, "Servicio a inspeccionar"]) -> str:
    """Errores de conexión, reintentos y latencia de red."""
    await asyncio.sleep(0.3)          # una consulta a ClickHouse tarda
    return "conn_refused: 0 · retries: 12 (normal) · rtt p99: 1.2ms"


@tool(idempotent=True)
async def despliegues(ventana: Annotated[str, "Ventana temporal"]) -> str:
    """Qué cambió y cuándo."""
    await asyncio.sleep(0.3)
    return "02:09 prometheus-gateway 2.7.0 · 18:30 (ayer) argus-collector 1.4.2"


@tool(idempotent=True)
async def recursos(host: Annotated[str, "Host a inspeccionar"]) -> str:
    """CPU, RAM y disco del host."""
    await asyncio.sleep(0.3)
    return "cpu 34% · RAM 61.2/64 GB ← al límite · disco 41%"


# ── Los trabajadores ──────────────────────────────────────────────────────────

LINEAS = [
    ("red", "¿Es la red? Mira errores de conexión y latencia en prometheus-gateway.", red,
     [calls("red", servicio="prometheus-gateway"),
      says('{"descartada": true, "confianza": 0.9, '
           '"hallazgo": "cero conexiones rechazadas y rtt normal: la red está sana"}')]),
    ("despliegue", "¿Fue un cambio? Mira qué se desplegó en las últimas 6 horas.", despliegues,
     [calls("despliegues", ventana="6h"),
      says('{"descartada": false, "confianza": 0.75, '
           '"hallazgo": "2.7.0 entró a las 02:09, cinco minutos antes del síntoma"}')]),
    ("recursos", "¿Es el host? Mira CPU, memoria y disco de inference-01.", recursos,
     [calls("recursos", host="inference-01"),
      says('{"descartada": false, "confianza": 0.8, '
           '"hallazgo": "61.2 de 64 GB de RAM: el host está al límite"}')]),
]


async def investigar(nombre: str, brief: str, herramienta, guion) -> tuple[str, Hipotesis, Usage]:
    """Una línea de investigación, aislada de las demás.

    Cada una tiene su `run_id`, su gateway y su contexto. No comparten nada, que
    es lo que permite lanzarlas a la vez sin que se pisen — y lo que hace que una
    que falle no se lleve a las otras por delante.
    """
    agente = Agent(
        f"investigador-{nombre}",
        model=nombre_del_modelo(),
        instructions=(
            "Investigas UNA hipótesis sobre un incidente. Usa tu herramienta y "
            "responde si la hipótesis queda descartada o sigue viva, con el dato "
            "concreto que lo sostiene."
        ),
        tools=[herramienta],
        output=Hipotesis,
    )
    sesion = Session(f"tormenta-{nombre}", gateway(guion, tools=[herramienta]))

    async for paso in agente.run(brief, session=sesion):
        if isinstance(paso, FinalStep):
            return nombre, paso.output, paso.usage
    raise RuntimeError(f"{nombre} terminó sin resultado")


# ── El orquestador ────────────────────────────────────────────────────────────

async def main() -> None:
    encabezado("06 · Agentes en paralelo")

    print("  tres líneas a la vez…\n")
    reloj = time.perf_counter()

    resultados = await asyncio.gather(
        *(investigar(n, b, h, g) for n, b, h, g in LINEAS)
    )

    paralelo = time.perf_counter() - reloj

    total = Usage.zero()
    vivas = []
    for nombre, h, consumo in resultados:
        total += consumo
        marca = "descartada" if h.descartada else f"VIVA ({h.confianza:.0%})"
        print(f"  {nombre:12} {marca:16} {h.hallazgo}")
        if not h.descartada:
            vivas.append((nombre, h))

    # ── El sintetizador ──────────────────────────────────────────────────────
    #
    # Recibe **los hallazgos**, no las tres conversaciones. Tres contextos
    # completos serían tres veces el coste para decir lo mismo.

    brief = "Hipótesis que siguen vivas:\n" + "\n".join(
        f"- {n} ({h.confianza:.0%}): {h.hallazgo}" for n, h in vivas
    )

    sintetizador = Agent(
        "sintetizador",
        model=nombre_del_modelo(),
        instructions=(
            "Recibes hipótesis de varios investigadores y decides cuál explica el "
            "incidente. Si dos encajan, di cómo se relacionan. Dos frases."
        ),
    )
    puerta = gateway([says(
        "El despliegue 2.7.0 y la RAM al límite son la misma causa, no dos: subió "
        "el número de modelos residentes por encima de lo que cabe en 64 GB. "
        "Revertir 2.7.0 resuelve las dos."
    )])

    async for paso in sintetizador.run(brief, session=Session("tormenta-sint", puerta)):
        if isinstance(paso, FinalStep):
            total += paso.usage
            print(f"\n  síntesis   {paso.output}")

    serie = sum(0.3 for _ in LINEAS)
    print(f"\n  reloj: {paralelo:.1f}s en paralelo frente a ~{serie:.1f}s en serie")
    print(f"  coste: entrada={total.input} salida={total.output} "
          "— el mismo que en serie, porque los tokens no saben de concurrencia")


# Lo que hay que llevarse:
#
# * **Aislar contexto es lo que hace barato paralelizar.** Si los tres
#   investigadores compartieran historial, habría que serializarlos.
# * **El orquestador decide con hallazgos, no con transcripciones.** Es la misma
#   regla del ejemplo 05, y es lo que impide que el coste crezca con el número
#   de trabajadores.
# * **Una línea que falle no se lleva a las otras.** Con `asyncio.gather` y
#   `return_exceptions=True` se decide qué hacer con la que reventó; aquí se
#   deja explotar a propósito, porque en un incidente un investigador mudo es
#   peor que uno que grita.

if __name__ == "__main__":
    asyncio.run(main())
