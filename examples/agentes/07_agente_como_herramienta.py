"""07 · Un agente como herramienta de otro, y un supervisor que enruta.

**Dominio: una mesa de entrada que reparte entre varios sistemas.** Llega una
petición en lenguaje natural y puede ser de tres sitios distintos: un incidente
de observabilidad, un documento para extraer, o un pago. Cada uno tiene su
especialista, sus herramientas y su riesgo — y esa es la situación de cualquier
organización con más de un sistema en producción.

El patrón: **envolver un agente en un `@tool`**. No hace falta nada del
framework para esto — un `@tool` es una función asíncrona, y un agente se
ejecuta con `async for`. Pero tiene consecuencias que conviene entender antes de
usarlo, y están al final del fichero.

    uv run python examples/agentes/07_agente_como_herramienta.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import Agent, FinalStep, Phase, Risk, Session, ToolStep, Usage, tool
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

# Consumo de los especialistas, que hay que sumar a mano: desde fuera, una
# llamada a un subagente parece una herramienta barata y no lo es.
COSTE = Usage.zero()


# ── Herramientas de los especialistas ─────────────────────────────────────────

@tool(idempotent=True)
async def buscar_spans(consulta: Annotated[str, "Consulta sobre trazas"]) -> str:
    """Busca en las trazas de la plataforma de observabilidad."""
    return "chat.completions p95 6.4s · model.load 5.9s de esos · 412 spans"


@tool(idempotent=True)
async def clasificar_documento(ruta: Annotated[str, "Ruta del documento"]) -> str:
    """Clasifica un documento en una de las categorías soportadas."""
    return "tipo: boleta_de_pago · confianza 0.94 · 1 página · texto nativo (sin OCR)"


@tool(risk=Risk.SOFT_WRITE)
async def consultar_saldo(cuenta: Annotated[str, "Identificador de cuenta"]) -> str:
    """Saldo disponible de una cuenta del libro mayor."""
    return f"{cuenta}: 4.812.400µ disponibles · 150.000µ retenidos"


# ── Los especialistas, envueltos como herramientas ────────────────────────────

async def _ejecutar(agente: Agent, brief: str, run_id: str, puerta) -> str:
    """Corre un agente hasta el final y devuelve su salida.

    El historial del subagente **no sale de aquí**: lo que vuelve es su
    respuesta. Es lo que hace que delegar aísle contexto en vez de duplicarlo.
    """
    global COSTE
    async for paso in agente.run(brief, session=Session(run_id, puerta)):
        if isinstance(paso, FinalStep):
            COSTE += paso.usage
            return str(paso.output)
    return "el especialista no devolvió nada"


@tool
async def especialista_observabilidad(
    pregunta: Annotated[str, "La pregunta, tal cual, para el especialista"],
) -> str:
    """Responde preguntas sobre incidentes, latencias y trazas de la plataforma."""
    agente = Agent(
        "obs", model=nombre_del_modelo(),
        instructions="Respondes sobre observabilidad usando las trazas.",
        tools=[buscar_spans],
    )
    puerta = gateway(
        [calls("buscar_spans", consulta="p95 gateway"),
         says("El p95 son 6.4s y 5.9s de ellos son carga de modelo: se está "
              "recargando el modelo en cada petición.")],
        tools=[buscar_spans],
    )
    return await _ejecutar(agente, pregunta, "sup-obs", puerta)


@tool
async def especialista_documentos(
    pregunta: Annotated[str, "La pregunta, tal cual, para el especialista"],
) -> str:
    """Clasifica y extrae datos de documentos empresariales."""
    agente = Agent(
        "docs", model=nombre_del_modelo(),
        instructions="Clasificas documentos y dices qué se puede extraer.",
        tools=[clasificar_documento],
    )
    puerta = gateway(
        [calls("clasificar_documento", ruta="/entrada/88231.pdf"),
         says("Es una boleta de pago con 94% de confianza y texto nativo, así que "
              "no hace falta OCR.")],
        tools=[clasificar_documento],
    )
    return await _ejecutar(agente, pregunta, "sup-docs", puerta)


@tool
async def especialista_tesoreria(
    pregunta: Annotated[str, "La pregunta, tal cual, para el especialista"],
) -> str:
    """Consulta saldos, retenciones y movimientos del libro mayor."""
    agente = Agent(
        "tesoro", model=nombre_del_modelo(),
        instructions="Respondes sobre el estado de las cuentas.",
        tools=[consultar_saldo],
    )
    puerta = gateway(
        [calls("consultar_saldo", cuenta="agt-7741"),
         says("Hay 4.812.400µ disponibles y 150.000µ retenidos.")],
        tools=[consultar_saldo],
    )
    return await _ejecutar(agente, pregunta, "sup-tes", puerta)


# ── El supervisor ─────────────────────────────────────────────────────────────

async def main() -> None:
    encabezado("07 · Un agente como herramienta")

    especialistas = [
        especialista_observabilidad,
        especialista_documentos,
        especialista_tesoreria,
    ]

    supervisor = Agent(
        "mesa-de-entrada",
        model=nombre_del_modelo(),
        instructions=(
            "Enrutas cada petición al especialista adecuado y devuelves su "
            "respuesta. No contestes tú: no tienes acceso a los datos. Si una "
            "petición toca dos dominios, consulta a los dos."
        ),
        tools=especialistas,
    )

    peticion = "¿Por qué va lento el gateway de inferencia y cuánto saldo le queda al agente agt-7741?"

    guion = [
        calls("especialista_observabilidad",
              pregunta="¿Por qué va lento el gateway de inferencia?"),
        calls("especialista_tesoreria", pregunta="¿Cuánto saldo le queda a agt-7741?"),
        says("El gateway recarga el modelo en cada petición —5.9s de los 6.4s del "
             "p95—, y agt-7741 tiene 4.812.400µ disponibles con 150.000µ retenidos."),
    ]

    print(f"  petición   {peticion}\n")

    sesion = Session("mesa-1", gateway(guion, tools=especialistas))
    async for paso in supervisor.run(peticion, session=sesion):
        match paso:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                print(f"  ↳ delega a {llamada.name.removeprefix('especialista_')}")
            case FinalStep(output=respuesta, usage=consumo_supervisor):
                print(f"\n  {respuesta}\n")
                print(f"  coste supervisor   entrada={consumo_supervisor.input}")
                print(f"  coste especialistas entrada={COSTE.input}")
                print(f"  coste real         entrada={consumo_supervisor.input + COSTE.input}")


# Cuándo usar esto, y cuándo no
# ─────────────────────────────
#
# **A favor:** el supervisor no necesita conocer las herramientas de nadie. Sus
# tres herramientas tienen una firma de una línea, mientras cada especialista
# puede tener quince. Añadir un dominio es añadir una función, no reescribir un
# prompt.
#
# **En contra, y es lo que nadie cuenta:**
#
# 1. **El coste desaparece de la vista.** `FinalStep.usage` del supervisor mide
#    *sus* llamadas, no las de dentro. Aquí se suma a mano en `COSTE`, y por eso
#    el ejemplo imprime las tres cifras: si solo miras la del supervisor, un
#    sistema que gasta cinco veces más parece igual de barato.
#
#    **Esto ya está resuelto** si delegas como primitiva en vez de a mano:
#    `Agent(delegates=[…])` transporta el consumo del subagente en el
#    `DelegateStep` y lo suma al total del padre. Ver el ejemplo 09.
#
# 2. **Un subagente envuelto a mano no es un paso durable.** Si el proceso muere
#    a mitad de un especialista, al reanudar la herramienta se reejecuta entera
#    — el journal la ve como una llamada, no como un run con sus propios pasos.
#    Para lectura da igual; para algo que mueva dinero, no.
#
#    **También resuelto delegando como primitiva**: el subagente tiene su propio
#    diario y no se repite.
#
# 3. **El riesgo no se propaga.** `especialista_tesoreria` es `Risk.READ` por
#    defecto aunque por dentro llame a algo que escribe, y aquí hay que
#    declararlo a mano en el envoltorio.
#
#    **Delegando como primitiva sí se deriva**: el riesgo de delegar es el mayor
#    de lo que el subagente puede hacer. Envolviendo a mano el framework no
#    puede deducirlo —ve una función que devuelve `str`—; como delegado, sí ve
#    sus herramientas.
#
# Los tres desaparecen con `Agent(delegates=[…])`, que es el ejemplo
# [`09`](09_delegar.py). Este patrón sigue siendo el correcto cuando lo de
# dentro **no es un `Agent`**: una API ajena, un servicio heredado, cualquier
# cosa que no tenga un bucle que ceder.

if __name__ == "__main__":
    asyncio.run(main())
