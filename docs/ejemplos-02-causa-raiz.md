# 02 · Varias herramientas y una respuesta tipada

> **Esto es un fichero que se ejecuta:** [`examples/agentes/02_causa_raiz.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/02_causa_raiz.py) ↗
> Esta página lo transcribe y enseña lo que imprime. Si dejan de coincidir, falla un test.

**Dominio: Argus** — AIOps agéntico. El triaje decidió que hay que mirarlo; ahora
hay que averiguar **por qué**. El agente encadena consultas hasta tener una
hipótesis, y la devuelve como un objeto y no como prosa.

Lo que este ejemplo añade sobre el 01:

* **Varias herramientas.** El agente decide cuáles usa y en qué orden; nadie
  escribe ese orden. Con una sola, un agente es una llamada con pasos extra.
* **Salida estructurada.** `output=` valida la respuesta contra un tipo. Lo que
  sale de aquí va a un incidente, a una notificación y a un panel — tres
  consumidores que no pueden parsear prosa.

## Cómo correrlo

```bash
uv run python examples/agentes/02_causa_raiz.py
```

No hace falta configurar nada: sin modelo, las respuestas van guionizadas y **todo lo
demás es real** — las herramientas se ejecutan, el journal se escribe, el consumo se mide.
Con `AXONIUM_CLIENT_ID` o `SYNAPTUM_BASE_URL` en el entorno, **el mismo fichero sin tocar**
habla con un modelo de verdad; lo que cambia entonces es lo que diga el modelo, no el
código. Ver [Modelos](04-modelos.md).

## Lo que imprime

```text
02 · Causa raíz con salida tipada
─────────────────────────────────
sin inferencia · respuestas guionizadas (exporta SYNAPTUM_BASE_URL para usar un modelo real)

  → spans_lentos
  → despliegues_recientes
  → metrica

  causa      El despliegue 2.7.0 subió el límite de modelos residentes de 3 a 5, y la RAM solo da para 3: cada petición desaloja un modelo y lo recarga
  culpable   prometheus-gateway
  confianza  85%
  acción     revertir a 2.6.x o bajar el límite a 3
  evidencia:
    · model.load es 5.9s de los 6.4s del p95
    · despliegue 2.7.0 a las 02:09, cinco minutos antes del síntoma
    · modelos_cargados pasó de 2 a 5 con un límite de RAM de 3
```

```python
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import Agent, FinalStep, Phase, Session, ToolStep, tool
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo
```

## Lo que el agente tiene que devolver

Una `dataclass` de stdlib basta: `output=` deriva el JSON Schema, se lo pide al
modelo, valida la respuesta y entrega el objeto.  Pydantic es un extra para
quien ya lo use, no un requisito.

```python
@dataclass
class Diagnostico:
    causa: str
    servicio_culpable: str
    confianza: float
    evidencia: list[str]
    accion: str
```

## Las herramientas

```python
DESPLIEGUES = [
    {"servicio": "prometheus-gateway", "version": "2.7.0", "cuando": "2026-09-18T02:09:00Z"},
    {"servicio": "argus-collector", "version": "1.4.2", "cuando": "2026-09-17T18:30:00Z"},
]

@tool(idempotent=True)
async def spans_lentos(
    servicio: Annotated[str, "Nombre del servicio"],
    desde: Annotated[str, "Instante ISO-8601 desde el que buscar"],
) -> str:
    """Los spans más lentos de un servicio desde un instante dado."""
    return (
        "operación                        p95      n\n"
        "chat.completions (gen_ai)       6.4s    412\n"
        "  └ model.load                  5.9s     11\n"
        "  └ token.generate              0.4s    412\n"
        "auth.verify_token              0.01s    412"
    )

@tool(idempotent=True)
async def despliegues_recientes(
    desde: Annotated[str, "Instante ISO-8601"],
) -> str:
    """Qué se desplegó y cuándo, en todos los servicios."""
    return "\n".join(
        f"{d['cuando']}  {d['servicio']} {d['version']}" for d in DESPLIEGUES
    )

@tool(idempotent=True)
async def metrica(
    nombre: Annotated[str, "Nombre de la métrica, por ejemplo 'modelos_cargados'"],
    servicio: Annotated[str, "Servicio del que leerla"],
) -> str:
    """Serie temporal reciente de una métrica."""
    if nombre == "modelos_cargados":
        return "02:00 → 2 · 02:10 → 2 · 02:14 → 5 · 02:20 → 5   (límite de RAM: 3)"
    return f"sin datos para {nombre}"

# `idempotent=True` no es una optimización: dice que **repetir la llamada no
# tiene consecuencias**. El runtime lo usa para decidir si puede reintentar tras
# una caída sin arriesgarse a duplicar un efecto. Leer es repetible; cobrar no.

async def main() -> None:
    encabezado("02 · Causa raíz con salida tipada")

    agente = Agent(
        "causa-raiz",
        model=nombre_del_modelo(),
        instructions=(
            "Investigas incidentes. Usa las herramientas hasta tener una hipótesis "
            "sostenida por evidencia concreta. No adivines: si la evidencia no "
            "alcanza, dilo bajando la confianza."
        ),
        tools=[spans_lentos, despliegues_recientes, metrica],
        output=Diagnostico,
    )

    guion = [
        calls("spans_lentos", servicio="prometheus-gateway", desde="2026-09-18T02:00:00Z"),
        calls("despliegues_recientes", desde="2026-09-18T00:00:00Z"),
        calls("metrica", nombre="modelos_cargados", servicio="prometheus-gateway"),
        says(
            '{"causa": "El despliegue 2.7.0 subió el límite de modelos residentes '
            'de 3 a 5, y la RAM solo da para 3: cada petición desaloja un modelo y '
            'lo recarga", "servicio_culpable": "prometheus-gateway", '
            '"confianza": 0.85, "evidencia": ['
            '"model.load es 5.9s de los 6.4s del p95", '
            '"despliegue 2.7.0 a las 02:09, cinco minutos antes del síntoma", '
            '"modelos_cargados pasó de 2 a 5 con un límite de RAM de 3"], '
            '"accion": "revertir a 2.6.x o bajar el límite a 3"}'
        ),
    ]

    sesion = Session(
        "causa-4471",
        gateway(guion, tools=[spans_lentos, despliegues_recientes, metrica]),
    )

    async for paso in agente.run(
        "El gateway tiene el p95 disparado desde las 02:14. ¿Por qué?", session=sesion
    ):
        match paso:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                print(f"  → {llamada.name}")
            case FinalStep(output=d):
                print(f"\n  causa      {d.causa}")
                print(f"  culpable   {d.servicio_culpable}")
                print(f"  confianza  {d.confianza:.0%}")
                print(f"  acción     {d.accion}")
                print("  evidencia:")
                for linea in d.evidencia:
                    print(f"    · {linea}")

                # Es un objeto, no texto. Se puede pasar a un panel, a un
                # webhook o a una regla — sin volver a parsear.
                assert isinstance(d, Diagnostico) and 0 <= d.confianza <= 1

if __name__ == "__main__":
    asyncio.run(main())
```

---

**El fichero entero, para clonarlo y tocarlo:** [`examples/agentes/02_causa_raiz.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/02_causa_raiz.py) ↗

Está en [`examples/`](https://github.com/Root1V/synaptum-framework/tree/main/examples) con los otros quince, y todos corren igual.
