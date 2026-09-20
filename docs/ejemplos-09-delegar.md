# 09 · Delegar como primitiva, no como patrón

> **Generada de [`examples/agentes/09_delegar.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/09_delegar.py).** El fichero corre; esta página lo
> transcribe. Si los dos no coinciden, falla un test.

**Dominio: Aerarium** — el cierre de mes. Hay dos trabajos distintos y no
conviene mezclarlos en un solo agente: conciliar es leer mucho y decidir poco,
pagar es leer poco y decidir algo irreversible.

El [`07`](ejemplos-07-agente-como-herramienta.md) ya enrutaba a especialistas envolviendo
cada uno en un `@tool`. Funciona, y tiene tres agujeros que el propio ejemplo
documenta al final. Este fichero hace lo mismo con una línea distinta —
`Agent(delegates=[…])`— y los tres desaparecen:

1. el consumo del subagente **sube** al total de quien delega,
2. la delegación es un **paso durable** con su propio diario,
3. el riesgo **se deriva**: delegar en alguien que borra es destructivo.

El tercero es el que no se puede conseguir a mano, y es el que importa.

```bash
uv run python examples/agentes/09_delegar.py
```

```python
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import (
    Agent,
    DelegateStep,
    FinalStep,
    MemoryCheckpointer,
    ModelStep,
    Phase,
    Risk,
    Role,
    Session,
    Usage,
    tool,
)
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo
```

## Las herramientas de cada especialista

Fíjate en el riesgo declarado: el conciliador solo lee, la tesorería mueve
dinero.  Esa diferencia es la que va a viajar sola hasta el supervisor.

```python
@tool(idempotent=True)
async def listar_pendientes(
    periodo: Annotated[str, "Periodo contable, por ejemplo '2026-08'"],
) -> str:
    """Asientos del periodo que no cuadran con el extracto bancario."""
    return (
        "3 pendientes en 2026-08:\n"
        "  as-8812  proveedor Cloudflare   1.204.000µ  sin contrapartida\n"
        "  as-8840  nómina septiembre     8.900.000µ  duplicado aparente\n"
        "  as-8851  reembolso viaje          84.500µ  falta justificante"
    )

@tool(risk=Risk.HARD_WRITE)
async def ejecutar_pago(
    asiento: Annotated[str, "Identificador del asiento a pagar"],
    importe_micros: Annotated[int, "Importe en micros"],
) -> str:
    """Ordena un pago contra el banco. **Irreversible.**"""
    return f"pago ordenado · {asiento} · {importe_micros}µ · ref BK-33901"
```

## Los especialistas

Son agentes normales.  Ni saben ni tienen por qué saber que alguien los va a
usar como subagentes: eso lo decide quien los compone, no ellos.

```python
def especialistas() -> tuple[Agent, Agent]:
    conciliador = Agent(
        "conciliador",
        model=nombre_del_modelo(),
        instructions=(
            "Concilias asientos contra el extracto bancario. Enumera lo que no "
            "cuadra y di qué falta en cada caso. No ordenas pagos."
        ),
        tools=[listar_pendientes],
    )
    tesoreria = Agent(
        "tesoreria",
        model=nombre_del_modelo(),
        instructions=(
            "Ejecutas pagos ya aprobados. Un pago por asiento, con su importe "
            "exacto. Si el brief no trae importe, no pagas."
        ),
        tools=[ejecutar_pago],
    )
    return conciliador, tesoreria
```

## El supervisor

```python
async def main() -> None:
    encabezado("09 · Delegar como primitiva")

    conciliador, tesoreria = especialistas()

    cierre = Agent(
        "cierre",
        model=nombre_del_modelo(),
        instructions=(
            "Coordinas el cierre de mes. Primero pides la conciliación, y solo "
            "después ordenas los pagos que hayan quedado claros. Termina con un "
            "resumen de dos frases."
        ),
        # Esta es la línea.  Un `Agent` suelto se envuelve solo; también se
        # acepta cualquier cosa que cumpla el contrato —un delegado remoto, por
        # ejemplo (ver el `12`)— y el bucle no distingue.
        delegates=[conciliador, tesoreria],
    )

    # Lo que el supervisor ve de cada especialista: un nombre, para qué sirve, y
    # **un solo parámetro**.  No hereda su catálogo, así que delegar no infla el
    # prefijo de quien delega — un especialista con quince herramientas se
    # presenta igual que uno con una.
    print("  el catálogo del supervisor:")
    for definicion in cierre.tools:
        print(
            f"    {definicion.name:<14} riesgo={definicion.risk.value:<10} "
            f"argumentos={sorted(definicion.parameters['properties'])}"
        )
    print(
        "\n  «tesoreria» sale HARD_WRITE sin que nadie lo declare aquí: el riesgo\n"
        "  de delegar es el mayor de lo que el otro puede hacer.\n"
    )

    sesion = Session(
        "cierre-2026-08",
        # Siete respuestas: tres del supervisor y dos de cada especialista.
        gateway([guion] * 7, tools=[listar_pendientes, ejecutar_pago]),
        MemoryCheckpointer(),
    )

    propio = Usage.zero()   # lo que gastó el supervisor por su cuenta

    async for paso in cierre.run("Cierra agosto de 2026.", session=sesion):
        match paso:
            case ModelStep(phase=Phase.COMPLETED, response=respuesta) if respuesta:
                propio += respuesta.usage
            case DelegateStep(phase=Phase.ATTEMPTED, agent=quien, brief=encargo):
                print(f"  → delega en {quien}: {_corto(encargo)}")
            case DelegateStep(phase=Phase.COMPLETED, agent=quien, usage=consumo, step_id=paso_id):
                print(f"    ← {quien} · entrada={consumo.input} salida={consumo.output}")
                print(f"      su diario vive en 'cierre-2026-08/{paso_id}'")
            case FinalStep(output=resumen, usage=total):
                print(f"\n  {resumen}\n")
                print(f"  gasto propio del supervisor: entrada={propio.input} salida={propio.output}")
                print(f"  total del run:               entrada={total.input} salida={total.output}")
                print("  la diferencia es lo que costaron los especialistas, y sin")
                print("  delegar como primitiva no aparecería en ninguna parte.")

def _corto(texto: str, tope: int = 60) -> str:
    return texto if len(texto) <= tope else f"{texto[:tope]}…"
```

## El guion, solo para cuando no hay modelo

Es una función y no una lista porque aquí hay **tres bucles** compartiendo
gateway —el supervisor y sus dos especialistas— y un guion posicional no
sabría a cuál le toca. Un modelo de verdad responde por lo que ve; este
también.

```python
def guion(peticion):
    nombres = {t.name for t in peticion.tools}
    resultados = sum(1 for m in peticion.messages if m.role is Role.TOOL)

    if "conciliador" in nombres:                       # el supervisor
        if resultados == 0:
            return calls("conciliador", id="d1", brief="Concilia el periodo 2026-08.")
        if resultados == 1:
            return calls(
                "tesoreria", id="d2",
                brief="Paga el asiento as-8812 por 1.204.000 micros.",
            )
        return says(
            "Agosto queda cerrado con una salvedad: pagado as-8812 (Cloudflare, "
            "1.204.000µ). as-8840 y as-8851 quedan retenidos — un duplicado "
            "aparente de nómina y un reembolso sin justificante."
        )

    if "listar_pendientes" in nombres:                 # el conciliador
        if resultados == 0:
            return calls("listar_pendientes", id="c1", periodo="2026-08")
        return says(
            "Tres asientos sin cuadrar: as-8812 es un cargo real de Cloudflare "
            "sin contrapartida contable y se puede pagar; as-8840 parece un "
            "duplicado de nómina y as-8851 no tiene justificante."
        )

    if resultados == 0:                                # la tesorería
        return calls("ejecutar_pago", id="t1", asiento="as-8812", importe_micros=1_204_000)
    return says("Pagado as-8812 por 1.204.000µ, referencia bancaria BK-33901.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Lo que esto enseña

Lo que cambia respecto al `07`, dicho entero:

**Sigue habiendo un sitio para envolver a mano.** Si lo que hay detrás no es
un `Agent` —una API ajena, un servicio heredado— un `@tool` es la respuesta
correcta y esto no aplica.

**Pero entre agentes, delegar es una primitiva por un motivo concreto:** un
subagente necesita **su propio diario**. Envuelto en una función, una caída a
mitad lo reejecuta entero, porque el journal del padre vio una llamada y no un
run. Como primitiva, su identidad se deriva de la del padre
—`{run_id}/{step_id}`, determinista como todo lo demás— así que al reanudar se
reencuentra con lo suyo y no vuelve a pagarlo.

Y lo que **no** cambia, porque es la misma regla de siempre:

```text
Entre agentes viaja el resultado, nunca el historial.
```

Al especialista le llega el brief y nada más. De vuelta suben el resultado y
el consumo. Su conversación se queda en su diario, donde se puede auditar, y
no en el prompt del supervisor, donde solo costaría dinero.

Un aviso honesto sobre el riesgo derivado: se **declara** —el modelo lo ve en
el catálogo, el arnés en el handshake— pero la delegación en sí no cruza la
costura, así que una política no puede denegarla *antes* de que empiece. Las
herramientas del especialista sí la cruzan cuando las llama, así que un efecto
destructivo se detiene igual; lo que se pierde es detenerlo antes de pagar la
inferencia del hijo. Cerrarlo exige un método nuevo en la costura, y eso va
por el canal de coordinación.
