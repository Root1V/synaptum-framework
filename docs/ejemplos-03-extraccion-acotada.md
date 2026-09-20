# 03 · Un bucle acotado, y qué pasa cuando el modelo se equivoca

> **Esto es un fichero que se ejecuta:** [`examples/agentes/03_extraccion_acotada.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/03_extraccion_acotada.py) ↗
> Esta página lo transcribe y enseña lo que imprime. Si dejan de coincidir, falla un test.

**Dominio: la plataforma de inteligencia documental** — extracción agéntica
*acotada* sobre boletas de pago, con validación determinista detrás. Su README lo
dice en esas palabras, y las dos importan:

* **acotada** — un bucle sin techo es un incidente esperando: el modelo se
  atasca, reintenta y factura. `Limits` pone el techo.
* **determinista detrás** — el modelo extrae, el código valida. Si la validación
  falla, el error vuelve al modelo para que rectifique en vez de propagarse.

Lo que este ejemplo añade sobre el 02: **el camino de error**. Un agente que solo
se prueba cuando acierta no está probado.

## Cómo correrlo

```bash
uv run python examples/agentes/03_extraccion_acotada.py
```

No hace falta configurar nada: sin modelo, las respuestas van guionizadas y **todo lo
demás es real** — las herramientas se ejecutan, el journal se escribe, el consumo se mide.
Con `AXONIUM_CLIENT_ID` o `SYNAPTUM_BASE_URL` en el entorno, **el mismo fichero sin tocar**
habla con un modelo de verdad; lo que cambia entonces es lo que diga el modelo, no el
código. Ver [Modelos](04-modelos.md).

## Lo que imprime

```text
03 · Extracción acotada, con el camino de error
───────────────────────────────────────────────
sin inferencia · respuestas guionizadas (exporta SYNAPTUM_BASE_URL para usar un modelo real)

  → leer_pagina
  → validar_ruc

  JOHN DOE · 2026-09
  bruto 4,302.50 − descuentos 548.55 = neto 3,753.95
  ✓ el neto cuadra con bruto − descuentos

  llamadas al modelo: 3 (una se perdió en un JSON truncado y se reintentó sola)
```

```python
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import Agent, FinalStep, Limits, ModelStep, Phase, Session, ToolStep, tool
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

@dataclass
class Boleta:
    ruc: str
    trabajador: str
    periodo: str
    bruto: float
    descuentos: float
    neto: float
```

## Herramientas: OCR y una validación que el modelo NO hace

Los datos de una boleta tienen que ser **inequívocamente inventados**, no
solo inventados.

Aquí había un nombre peruano verosímil, con dos de los apellidos más comunes
del país, al lado de un sueldo y una AFP. No salió de ningún sitio —está
compuesto— y aun así es un nombre que casi con seguridad lleva alguien, en un
repositorio público, en un sitio web y dentro de los artefactos publicados en
PyPI. Que el dato sea falso no ayuda a quien se llame así.

La regla, para el próximo ejemplo: si un nombre, un RUC o una cuenta pueden
ser de alguien, hay que cambiarlos aunque te los hayas inventado.

Y se usan los marcadores de siempre —`John Doe`, `ACME`— en vez de inventar
uno nuevo. No es capricho: un nombre que el lector ya reconoce como marcador
se lee como «aquí va un nombre» sin tener que pensarlo, y nadie se pregunta
si detrás hay alguien. Un «PERSONA DE EJEMPLO UNO» cumple pero hay que
leerlo dos veces.

El RUC de abajo **falla el dígito verificador** a propósito, así que no es de
ninguna empresa; se comprueba con los pesos 5,4,3,2,7,6,5,4,3,2.

```python
PAGINA = """
ACME CONSTRUCTORA S.A.C.
RUC 20481234567
BOLETA DE PAGO - SETIEMBRE 2026
Trabajador: JOHN DOE
Remuneración básica        4,200.00
Asignación familiar          102.50
--------------------------------
Total bruto                4,302.50
AFP Integra (10%)            430.25
EsSalud                        0.00
Impuesto 5ta categoría       118.30
--------------------------------
Total descuentos             548.55
NETO A PAGAR               3,753.95
"""

@tool(idempotent=True)
async def leer_pagina(
    documento_id: Annotated[str, "Identificador del documento en el almacén"],
    pagina: Annotated[int, "Número de página, empezando en 1"] = 1,
) -> str:
    """Texto de una página, tal como lo devuelve el OCR."""
    return PAGINA.strip()

@tool(idempotent=True)
async def validar_ruc(
    ruc: Annotated[str, "RUC de 11 dígitos"],
) -> str:
    """Comprueba el formato de un RUC y busca su razón social en el padrón.

    Esto **no lo hace el modelo**, y son dos razones distintas: un padrón es un
    dato que hay fuera, y un modelo no lo tiene; y el formato es aritmética,
    que pedírsela a un modelo de lenguaje convierte algo exacto en algo
    probable.
    """
    if len(ruc) != 11 or not ruc.isdigit():
        return f"RUC {ruc} inválido: deben ser 11 dígitos."
    return f"RUC {ruc}: hallado en el padrón · ACME CONSTRUCTORA S.A.C."

async def main() -> None:
    encabezado("03 · Extracción acotada, con el camino de error")

    agente = Agent(
        "extractor",
        model=nombre_del_modelo(),
        instructions=(
            "Extraes datos de boletas de pago peruanas. Lee la página, valida el "
            "RUC con la herramienta y devuelve el objeto. Los importes van como "
            "número, sin separador de miles."
        ),
        tools=[leer_pagina, validar_ruc],
        output=Boleta,
        # El techo. Sin él, un modelo que no converge reintenta hasta que alguien
        # mira la factura. `max_steps` es corrección, no política: la política de
        # cuánto puede gastar un run vive en el arnés, no aquí.
        limits=Limits(max_steps=8, max_retries=2),
    )

    guion = [
        calls("leer_pagina", documento_id="doc-88231", pagina=1),
        calls("validar_ruc", ruc="20481234567"),
        # Primer intento del modelo: JSON mal formado. Pasa, y pasa más de lo que
        # nadie admite. El bucle lo trata como reintentable —el muestreo es
        # estocástico— y vuelve a pedirlo sin que el llamante se entere.
        says('{"ruc": "20481234567", "trabajador": "JOHN DOE",'),
        says(
            '{"ruc": "20481234567", "trabajador": "JOHN DOE", '
            '"periodo": "2026-09", "bruto": 4302.50, "descuentos": 548.55, '
            '"neto": 3753.95}'
        ),
    ]

    sesion = Session("doc-88231", gateway(guion, tools=[leer_pagina, validar_ruc]))

    llamadas_al_modelo = 0
    async for paso in agente.run("Extrae la boleta doc-88231.", session=sesion):
        match paso:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                print(f"  → {llamada.name}")
            case ModelStep(phase=Phase.COMPLETED):
                llamadas_al_modelo += 1
            case FinalStep(output=b):
                print(f"\n  {b.trabajador} · {b.periodo}")
                print(f"  bruto {b.bruto:,.2f} − descuentos {b.descuentos:,.2f} "
                      f"= neto {b.neto:,.2f}")

                # La validación que el modelo no puede garantizar, hecha aquí.
                assert abs(b.bruto - b.descuentos - b.neto) < 0.01, "el neto no cuadra"
                print("  ✓ el neto cuadra con bruto − descuentos")

    print(f"\n  llamadas al modelo: {llamadas_al_modelo} "
          f"(una se perdió en un JSON truncado y se reintentó sola)")

if __name__ == "__main__":
    asyncio.run(main())
```

## Lo que esto enseña

Lo que hay que llevarse: **la validación estructurada vive dentro del
reintento**. Un objeto mal formado no es un fallo del run, es una muestra
mala — y el bucle lo sabe porque la taxonomía de errores lo dice, no porque
alguien escribiera un `try` aquí.

---

**El fichero entero, para clonarlo y tocarlo:** [`examples/agentes/03_extraccion_acotada.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/03_extraccion_acotada.py) ↗

Está en [`examples/`](https://github.com/Root1V/synaptum-framework/tree/main/examples) con los otros quince, y todos corren igual.
