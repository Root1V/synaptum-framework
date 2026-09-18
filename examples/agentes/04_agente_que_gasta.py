"""04 · Un agente que mueve dinero, y el runtime que lo frena.

**Dominio: Aerarium + Mercatus** — el tesoro y el mercado. Un agente descubre un
servicio, paga una décima de céntimo por una llamada, y **hay que poder decirle
que no en el momento en que se pasa de lo que su dueño autorizó**.

Aquí es donde un framework de agentes deja de ser un bucle bonito. Tres piezas:

* **`risk` se declara, no se deduce.** Ninguna anotación puede saber que una
  función que devuelve `str` mueve dinero. El framework **declara**; quien
  gobierna **decide**.
* **La decisión se toma fuera del proceso.** Una comprobación dentro del código
  gobernado es advisoria: la cumple un bucle correcto y se la salta uno con un
  fallo. Aquí se simula con una política local, y el gateway **avisa de que no
  aplica nada**.
* **El run sobrevive a la espera.** Cuando hace falta una persona, el run se
  suspende en disco y se reanuda cuando alguien decide — sin repetir lo pagado
  y sin volver a preguntar.

    uv run python examples/agentes/04_agente_que_gasta.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import (
    ALLOW,
    Agent,
    ApprovalStep,
    Decision,
    Disposition,
    FinalStep,
    Risk,
    Session,
    SqliteCheckpointer,
    ToolStep,
    tool,
)
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

# ── El libro mayor, en micro-unidades ─────────────────────────────────────────
#
# Aerarium lleva el ledger en 10⁻⁶ para que una llamada de $0.001 liquide de
# verdad. Aquí se imita con enteros por el mismo motivo: un float no es un saldo.

RETENCIONES: list[str] = []
PAGOS: list[str] = []
SALDO_MICROS = 5_000_000          # 5,00 USD

# Dos listas y no una, y la distinción es todo el dominio: **una retención no es
# un pago.** Caduca sola a los 15 minutos y no mueve nada; una captura sí. La
# primera versión de este ejemplo las metía en la misma lista y afirmaba «no
# hubo efecto» mirando su longitud — contra un modelo real que decidió reservar
# antes de capturar, eso daba un fallo que no era tal.


@tool(idempotent=True)
async def catalogo(
    capacidad: Annotated[str, "Qué se busca, por ejemplo 'transcripción'"],
) -> str:
    """Servicios del mercado que ofrecen una capacidad, con su precio por llamada."""
    return (
        "servicio                precio/llamada\n"
        "whisper-api-x402        $0.0008\n"
        "deepgram-mandate        $0.0021\n"
        "transcribe-mega         $1.9500"
    )


@tool(risk=Risk.SOFT_WRITE)
async def reservar(
    servicio: Annotated[str, "Servicio del catálogo"],
    micros: Annotated[int, "Importe a retener, en micro-unidades"],
) -> str:
    """Retiene fondos sin moverlos — el `hold` de una autorización."""
    RETENCIONES.append(f"hold {micros}µ → {servicio}")
    return f"Retenidos {micros}µ para {servicio}. Caduca en 15 minutos."


@tool(risk=Risk.DESTRUCTIVE)
async def capturar(
    servicio: Annotated[str, "Servicio del catálogo"],
    micros: Annotated[int, "Importe a capturar, en micro-unidades"],
) -> str:
    """Convierte una retención en un pago. **Esto sí mueve dinero.**"""
    PAGOS.append(f"capture {micros}µ → {servicio}")
    return f"Pagados {micros}µ a {servicio}."


# `reservar` es SOFT_WRITE y `capturar` es DESTRUCTIVE, y esa diferencia es todo
# el ejemplo: retener se deshace solo al caducar, pagar no. Quien calla obtiene
# `risk=READ` e `idempotent=False` — permisivo en riesgo, conservador en
# durabilidad: la clase más inocua y la garantía más cara.


# ── La política ───────────────────────────────────────────────────────────────
#
# En producción esto vive en el arnés, fuera del proceso, y por eso puede
# aplicar. Aquí es una función local que **solo enseña la forma** de la decisión.

# Medio céntimo. Deliberadamente por debajo de **cualquier** servicio del
# catálogo: así la aprobación no depende de qué elija el modelo.
#
# La primera versión de este ejemplo ponía el tope en 0,10 USD y afirmaba que el
# pago quedaría detenido. Con el doble se cumplía siempre; contra un modelo real
# falló, porque eligió el servicio barato y pasó por debajo del tope. El ejemplo
# afirmaba una conducta del modelo creyendo que afirmaba una de la política.
TOPE_SIN_APROBACION = 500         # 0,0005 USD


def politica_del_dueño(check) -> Decision:
    """Lo que el dueño del agente autorizó, expresado como código."""
    if check.kind != "tool":
        return ALLOW
    if check.name != "capturar":
        return ALLOW

    importe = int(check.arguments.get("micros", 0))
    if importe > SALDO_MICROS:
        return Decision(
            disposition=Disposition.TERMINATE_RUN,
            reason_code="insufficient_funds",
            message=f"{importe}µ supera el saldo de {SALDO_MICROS}µ.",
        )
    if importe > TOPE_SIN_APROBACION:
        return Decision(
            disposition=Disposition.REQUIRE_APPROVAL,
            reason_code="above_delegated_cap",
            message=f"{importe}µ pasa del tope delegado de {TOPE_SIN_APROBACION}µ.",
        )
    return ALLOW


# Tres disposiciones distintas para tres situaciones distintas, y la diferencia
# importa: sin fondos **no hay nada que aprobar** y el run termina; pasarse del
# tope delegado es exactamente lo que una persona puede resolver.


def responder(peticion):
    """Doble que imita a un modelo que decide gastar de más."""
    from synaptum import Role

    turnos = sum(1 for m in peticion.messages if m.role is Role.TOOL)
    if turnos == 0:
        return calls("catalogo", capacidad="transcripción")
    if turnos == 1:
        return calls("capturar", servicio="whisper-api-x402", micros=800)
    return says("Transcripción contratada con whisper-api-x402 por 800µ.")


async def main() -> None:
    encabezado("04 · Un agente que gasta")

    almacen = Path(tempfile.mkdtemp()) / "tesoro.db"
    herramientas = [catalogo, reservar, capturar]
    agente = Agent(
        "comprador",
        model=nombre_del_modelo(),
        instructions=(
            "Contratas servicios del mercado con el presupuesto del dueño. "
            "Consulta el catálogo, elige el más barato que sirva, y **paga con "
            "`capturar`**: sin pagar no hay servicio contratado. Los importes van "
            "en micro-unidades: $0.0008 son 800."
        ),
        tools=herramientas,
    )
    tarea = "Necesito transcribir un audio. Contrata y paga el servicio más barato."

    # ── 1ª vuelta: el agente se pasa del tope y el run se detiene ─────────────

    detenido = Decision(
        disposition=Disposition.REQUIRE_APPROVAL,
        reason_code="above_delegated_cap",
        message=f"el importe pasa del tope delegado de {TOPE_SIN_APROBACION}µ.",
    )
    puerta = gateway(
        [responder] * 6,
        tools=herramientas,
        policy=politica_del_dueño,
        deny_tools={"capturar": detenido},
    )
    checkpointer = SqliteCheckpointer(almacen)

    detenciones = []
    async for paso in agente.run(tarea, session=Session("compra-1", puerta, checkpointer)):
        match paso:
            case ToolStep(decision=veredicto) if veredicto is not None:
                detenciones.append(veredicto)
                print(f"  ✋ {veredicto.reason_code}: {veredicto.message}")
            case ApprovalStep(subject=asunto):
                print(f"  ⏸  run suspendido · {asunto}")

    print(f"     retenciones: {len(RETENCIONES)} · pagos: {len(PAGOS)}")

    # Lo que el framework garantiza: si se intentó el paso peligroso, no ocurrió.
    assert not PAGOS, "se movió dinero pese a la denegación"

    if not detenciones:
        # Que un modelo pequeño no llegue a intentar el pago **no es un fallo**:
        # es el modelo siendo cauto o despistado. El ejemplo lo dice en vez de
        # reventar, porque afirmar aquí sería afirmar una conducta del modelo.
        print("\n  ⚠ el modelo no llegó a intentar el pago, así que la política no")
        print("    tuvo nada que denegar. Con el doble sí ocurre; vuelve a probar")
        print("    o usa un modelo con mejor seguimiento de instrucciones.")
        return

    estado = await checkpointer.load("compra-1")
    print(f"     journal: {estado.next_seq} eventos en disco")

    print("\n  … el dueño mira la petición y la aprueba …\n")

    # ── 2ª vuelta: otro proceso, mismo run_id ────────────────────────────────
    #
    # El agente no vuelve a consultar el catálogo: eso ya se pagó. Solo ocurre
    # lo que no llegó a ocurrir.

    otra_puerta = gateway([responder] * 6, tools=herramientas)
    salida = None
    async for paso in agente.run(
        tarea, session=Session("compra-1", otra_puerta, SqliteCheckpointer(almacen))
    ):
        if isinstance(paso, FinalStep):
            salida = paso.output

    print(f"  ▶  reanudado · llamadas al modelo: {otra_puerta.model_calls}")
    print(f"     {salida}")
    print(f"     pagos: {PAGOS}")
    assert len(PAGOS) == 1, "el pago se ordenó más de una vez"
    print("     ✓ el pago ocurrió exactamente una vez")


# El fallo que esto evita, y que solo aparece cuando la durabilidad es de verdad:
# un `require_approval` deja la herramienta con intención y sin resultado. Al
# reanudar, eso se parece a «no se sabe si se ejecutó» — pero **aquí sí se
# sabe**: se denegó *antes* de ejecutar. Por eso la denegación se registra como
# desenlace del paso y no como silencio.

if __name__ == "__main__":
    asyncio.run(main())
