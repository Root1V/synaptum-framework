"""01 · Lo mínimo: un agente, una herramienta.

**Dominio: Argus** — la plataforma de observabilidad. Su camino caliente detecta
un incidente en ~2 s sin tocar la base de datos, y deja una señal. Alguien —o
algo— tiene que mirarla y decir si merece despertar a una persona.

Ese "algo" es el agente más simple que se puede escribir con este framework.

    uv run python examples/agentes/01_triaje.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import Agent, FinalStep, Phase, Session, ToolStep, tool
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

# ── Datos que en Argus vendrían de ClickHouse ──────────────────────────────────

SEÑALES = {
    "sig-4471": {
        "servicio": "prometheus-gateway",
        "sintoma": "latencia p95 de /v1/chat/completions pasó de 820ms a 6.4s",
        "desde": "2026-09-18T02:14:00Z",
        "afectados": 3,
        "spans": 412,
    }
}


# ── La herramienta ────────────────────────────────────────────────────────────
#
# El esquema sale de la firma.  Un esquema escrito aparte se desincroniza, y
# cuando lo hace el modelo manda argumentos que la función no acepta — lejos del
# cambio que lo causó y con una inferencia ya pagada.

@tool
async def leer_señal(
    señal_id: Annotated[str, "Identificador de la señal, por ejemplo 'sig-4471'"],
) -> str:
    """Devuelve los datos de una señal detectada por el camino caliente."""
    señal = SEÑALES.get(señal_id)
    if señal is None:
        return f"No existe la señal {señal_id}."
    return (
        f"servicio: {señal['servicio']}\n"
        f"síntoma: {señal['sintoma']}\n"
        f"desde: {señal['desde']}\n"
        f"servicios afectados: {señal['afectados']}\n"
        f"spans implicados: {señal['spans']}"
    )


# ── El agente ─────────────────────────────────────────────────────────────────

async def main() -> None:
    encabezado("01 · Triaje de una señal")

    agente = Agent(
        "triaje",
        model=nombre_del_modelo(),
        instructions=(
            "Eres el triaje de una plataforma de observabilidad. Lee la señal y "
            "decide si despertar a una persona ahora o dejarlo para mañana. "
            "Responde en dos frases: el veredicto y por qué."
        ),
        tools=[leer_señal],
    )

    # El guion solo se usa cuando no hay modelo configurado.  Con uno, el agente
    # es exactamente el mismo: no sabe quién hay al otro lado.
    guion = [
        calls("leer_señal", señal_id="sig-4471"),
        says(
            "Despertar ahora. La latencia del gateway de inferencia se multiplicó "
            "por ocho y afecta a tres servicios: todo lo que dependa de un modelo "
            "está degradado, no solo lento."
        ),
    ]

    sesion = Session("triaje-4471", gateway(guion, tools=[leer_señal]))

    async for paso in agente.run("Tría la señal sig-4471.", session=sesion):
        match paso:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                print(f"  → {llamada.name}({llamada.arguments})")
            case FinalStep(output=veredicto, usage=consumo):
                print(f"\n  {veredicto}\n")
                print(f"  consumo: entrada={consumo.input} salida={consumo.output}")


# Eso es todo. Tres piezas y ninguna más:
#
#   @tool        una función con tipos, que se ejecuta de verdad
#   Agent        el modelo, las instrucciones y las herramientas
#   Session      dónde se ejecuta (gateway) y dónde se recuerda (checkpointer)
#
# El `async for` no es decoración: el bucle **cede el control en cada frontera**,
# así que ver lo que hace el agente es iterar, y pararlo es dejar de iterar.

if __name__ == "__main__":
    asyncio.run(main())
