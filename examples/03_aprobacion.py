"""Un paso destructivo se detiene, alguien decide, y el run sigue donde estaba.

Aquí es donde la durabilidad deja de ser una optimización de coste: **no se le
pregunta dos veces a una persona porque el proceso se cayó.**

    uv run python examples/03_aprobacion.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

from synaptum import (
    ApprovalStep,
    Decision,
    Disposition,
    FinalStep,
    Agent,
    Risk,
    Session,
    SqliteCheckpointer,
    ToolStep,
    tool,
)
from synaptum.testing import FakeGateway, calls, says

from _comun import encabezado

HECHAS: list[str] = []


@tool(risk=Risk.DESTRUCTIVE)
async def transferir(
    destino: Annotated[str, "IBAN de destino"],
    importe: Annotated[float, "Importe en euros"],
) -> str:
    """Ordena una transferencia."""
    HECHAS.append(f"{importe:.2f} € → {destino}")
    return f"Transferencia de {importe:.2f} € enviada a {destino}."


# `risk=DESTRUCTIVE` se declara y no se deduce: ninguna anotación puede saber
# que una función que devuelve `str` mueve dinero. Synaptum **declara**, el
# gateway **decide**.


def responder(peticion):
    from synaptum import Role

    if any(m.role is Role.TOOL for m in peticion.messages):
        return says("Transferencia completada.")
    return calls("transferir", destino="ES76 0049 1500 05", importe=250.0)


async def main() -> None:
    encabezado("03 · Aprobación humana a mitad de un run")

    almacen = Path(tempfile.mkdtemp()) / "runs.db"
    agente = Agent("tesorero", model="openai-compatible:doble", tools=[transferir])
    tarea = "Transfiere 250 € a ES76 0049 1500 05"

    # ── 1ª vuelta: el gateway exige aprobación ────────────────────────────────

    pendiente = Decision(
        disposition=Disposition.REQUIRE_APPROVAL,
        reason_code="destructive_requires_human",
        message="Una transferencia necesita aprobación de una persona.",
    )
    puerta = FakeGateway(
        *[responder] * 4,
        tools=[transferir],
        deny_tools={"transferir": pendiente},
    )
    checkpointer = SqliteCheckpointer(almacen)

    async for paso in agente.run(tarea, session=Session("run-2", puerta, checkpointer)):
        match paso:
            case ToolStep(decision=veredicto) if veredicto is not None:
                # La denegación se registra como **desenlace del paso**, no como
                # silencio. Es lo que permite distinguir «no se ejecutó porque
                # se denegó» de «no se sabe si ejecutó».
                print(f"  1ª vuelta    tool → {veredicto.disposition.value} "
                      f"· {veredicto.reason_code}")
            case ApprovalStep(subject=asunto):
                print(f"  1ª vuelta    detenido · {asunto}")

    print(f"               transferencias hechas: {len(HECHAS)}")
    assert not HECHAS, "el efecto ocurrió pese a la denegación"

    estado = await checkpointer.load("run-2")
    print(f"               journal: {estado.next_seq} eventos · el run queda suspendido")

    # ── Una persona decide.  El bucle no estaba corriendo cuando pasó ─────────

    print("\n  … alguien mira la petición y la aprueba …\n")

    # ── 2ª vuelta: el gateway ya no deniega ───────────────────────────────────

    otra_puerta = FakeGateway(*[responder] * 4, tools=[transferir])

    salida = None
    async for paso in agente.run(
        tarea, session=Session("run-2", otra_puerta, SqliteCheckpointer(almacen))
    ):
        if isinstance(paso, FinalStep):
            salida = paso.output

    print(f"  2ª vuelta    modelo ×{otra_puerta.model_calls} · "
          f"transferencias hechas: {len(HECHAS)}")
    print(f"\n  {salida}")

    assert len(HECHAS) == 1, "la transferencia se ordenó más de una vez"
    print(f"  efecto       {HECHAS[0]} · exactamente una vez")

    print("\n  Sin esto, reanudar un run suspendido levantaba UncertainEffect: el")
    print("  replay veía intención sin resultado y lo leía como «no se sabe si")
    print("  ejecutó». Pero aquí no hay incertidumbre — se denegó ANTES de")
    print("  ejecutar, así que la denegación se registra como desenlace del paso.")


if __name__ == "__main__":
    asyncio.run(main())
