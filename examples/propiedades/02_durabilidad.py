"""Matar el proceso a mitad y reanudar sin volver a pagar la inferencia.

Es la propiedad que justifica toda la arquitectura, y este ejemplo la **mide**
en vez de afirmarla: cuenta las llamadas al modelo y a la herramienta de cada
vuelta. La segunda tiene que dar cero.

    uv run python examples/02_durabilidad.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

from synaptum import Agent, FinalStep, Session, SqliteCheckpointer, tool
from synaptum.testing import calls, says

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comun import encabezado, gateway, nombre_del_modelo

LLAMADAS = {"herramienta": 0}


@tool
async def consultar_saldo(
    cuenta: Annotated[str, "IBAN de la cuenta"],
) -> str:
    """Consulta el saldo de una cuenta."""
    LLAMADAS["herramienta"] += 1
    return f"{cuenta}: 4.200,00 €"


class _Cortado(Exception):
    """El proceso muriéndose a mitad del run."""


def responder(peticion):
    """El doble contesta **según el contexto**, como haría un modelo.

    Importa para este ejemplo: al reanudar, el gateway es nuevo y no recuerda
    nada, pero el contexto sí trae el resultado de la herramienta —se vuelve a
    derivar del journal—. Un guion posicional respondería lo mismo que la
    primera vez y volvería a pedir la herramienta, que es justo lo que la
    reanudación evita.
    """
    from synaptum import Role

    ya_consultado = any(m.role is Role.TOOL for m in peticion.messages)
    if ya_consultado:
        return says("El saldo es de 4.200,00 €.")
    return calls("consultar_saldo", cuenta="ES91 2100 0418 45")


def guion():
    return [responder] * 6


async def main() -> None:
    encabezado("02 · Reanudar sin volver a pagar")

    almacen = Path(tempfile.mkdtemp()) / "runs.db"
    agente = Agent(
        "cajero",
        model=nombre_del_modelo(),
        instructions="Consultas saldos con la herramienta. Responde en una frase.",
        tools=[consultar_saldo],
    )
    tarea = "¿Cuánto hay en ES91 2100 0418 45?"

    # ── Primera vuelta: se corta después de ejecutar la herramienta ────────────
    #
    # El corte es lo peor que puede pasar: el efecto ya ocurrió y el proceso
    # muere antes de contárselo a nadie.

    puerta = gateway(guion(), tools=[consultar_saldo])
    checkpointer = SqliteCheckpointer(almacen)

    try:
        async for paso in agente.run(tarea, session=Session("run-1", puerta, checkpointer)):
            if paso.kind == "tool" and paso.phase.value == "completed":
                print("  1ª vuelta    herramienta ejecutada · el proceso muere aquí")
                raise _Cortado
    except _Cortado:
        pass

    primera = (puerta.model_calls, LLAMADAS["herramienta"])
    print(f"               modelo ×{primera[0]} · herramienta ×{primera[1]}")

    estado = await checkpointer.load("run-1")
    print(f"               journal: {estado.next_seq} eventos en {almacen.name}")

    # ── Segunda vuelta: otro proceso, otra conexión, mismo run_id ──────────────
    #
    # Gateway nuevo y contadores a la vista.  Lo que no se vuelve a hacer es lo
    # que ya está en el journal.

    LLAMADAS["herramienta"] = 0
    otra_puerta = gateway(guion(), tools=[consultar_saldo])
    otro_checkpointer = SqliteCheckpointer(almacen)   # otra conexión al mismo fichero

    salida = None
    async for paso in agente.run(tarea, session=Session("run-1", otra_puerta, otro_checkpointer)):
        if isinstance(paso, FinalStep):
            salida = paso.output

    segunda = (otra_puerta.model_calls, LLAMADAS["herramienta"])
    print(f"  2ª vuelta    modelo ×{segunda[0]} · herramienta ×{segunda[1]}")
    print(f"\n  {salida}")

    # El primer paso de modelo y la herramienta ya estaban pagados: no se
    # repiten.  La llamada que sí ocurre es la que nunca llegó a hacerse.
    assert segunda[1] == 0, "la herramienta se volvió a ejecutar"
    assert segunda[0] < primera[0] + 2, "se repitió inferencia ya pagada"

    # ── Tercera vuelta: un run cerrado no se reabre ────────────────────────────

    tercera_puerta = gateway(guion(), tools=[consultar_saldo])
    pasos = [
        paso
        async for paso in agente.run(
            tarea, session=Session("run-1", tercera_puerta, SqliteCheckpointer(almacen))
        )
    ]
    print(f"  3ª vuelta    modelo ×{tercera_puerta.model_calls} · "
          f"{len(pasos)} evento(s): devuelve lo que pasó, no lo recorre otra vez")
    assert tercera_puerta.model_calls == 0

    print("\n  La ventana de contexto no se guarda: se vuelve a derivar de los")
    print("  mismos resultados en el mismo orden. Guardarla sería guardar dos")
    print("  veces lo mismo y arriesgarse a que discrepen.")


if __name__ == "__main__":
    asyncio.run(main())
