# 16 · Ver los tokens según llegan, y cortar a mitad

> **Generada de [`examples/propiedades/04_streaming.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/propiedades/04_streaming.py).** El fichero corre; esta página lo
> transcribe. Si los dos no coinciden, falla un test.

`agent.stream(...)` es el mismo bucle y el mismo journal que `agent.run(...)`:
lo único que cambia es que los fragmentos del modelo se ceden intercalados entre
la intención del paso y su resultado.

```bash
uv run python examples/propiedades/04_streaming.py
```

```python
from __future__ import annotations

import asyncio
import sys
from typing import Annotated

from synaptum import FinalStep, Agent, Phase, Session, StepEvent, ToolStep, tool
from synaptum.testing import calls, says

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comun import encabezado, gateway, nombre_del_modelo

@tool
async def cotizacion(
    valor: Annotated[str, "Ticker del valor"],
) -> str:
    """Devuelve la última cotización de un valor."""
    return f"{valor}: 187,34 USD (+1,2 %)"

def responder(peticion):
    from synaptum import Role

    if any(m.role is Role.TOOL for m in peticion.messages):
        return says(
            "La acción cotiza a 187,34 USD, con una subida del 1,2 % en la sesión. "
            "El movimiento no es significativo por sí solo."
        )
    return calls("cotizacion", valor="ACME")

async def ver_llegar() -> None:
    print("  Los fragmentos se ceden dentro del paso que los produce:\n")

    agente = Agent(
        "analista",
        model=nombre_del_modelo(),
        instructions="Respondes sobre valores usando las herramientas.",
        tools=[cotizacion],
    )
    puerta = gateway([responder] * 4, tools=[cotizacion])

    async for evento in agente.stream("¿Cómo va ACME?", session=Session("ej-04", puerta)):
        match evento:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                print(f"\n  · herramienta {llamada.name}", flush=True)
            case FinalStep():
                print("\n")
            case StepEvent():
                pass                      # los pasos ya se ven por lo que emiten
            case _ if evento.kind == "text_start":
                print("  ", end="", flush=True)
            case _ if evento.kind == "text_delta":
                print(evento.text, end="", flush=True)
                await asyncio.sleep(0.02)   # solo para que se aprecie
            case _ if evento.kind == "reasoning_delta":
                sys.stdout.write(".")       # el razonamiento no es la respuesta
                sys.stdout.flush()

async def cortar_a_mitad() -> None:
    print("  Cortar es dejar de iterar — no hay evento de cancelación:\n")

    largo = (
        "Una respuesta larga que nadie va a leer entera, porque el punto del "
        "ejemplo es irse a mitad y comprobar que arriba se para."
    )
    agente = Agent("analista", model=nombre_del_modelo())

    # Cuántos fragmentos habría si se drenara entero.  Es la referencia contra
    # la que medir: «recibí pocos» no dice nada sin saber cuántos había.
    completo = gateway([lambda _: says(largo)])
    total = 0
    async for evento in agente.stream("largo", session=Session("ej-04-ref", completo)):
        total += evento.kind == "text_delta"

    puerta = gateway([lambda _: says(largo)])
    flujo = agente.stream("cuéntame algo largo", session=Session("ej-04b", puerta))
    recibidos = 0
    async for evento in flujo:
        if evento.kind == "text_delta":
            recibidos += 1
            if recibidos == 3:
                break
    await flujo.aclose()

    print(f"  recibidos {recibidos} de {total} fragmentos")
    if getattr(puerta, "cancelled", False):
        print(f"  el proveedor produjo {puerta.chunks_emitted} y registró el cierre")
    print("\n  Un canal que se está cerrando no es sitio para mandar el aviso de")
    print("  que se cierra. Cerrar el iterador cierra el cuerpo de la respuesta,")
    print("  y eso es lo que de verdad para la generación arriba.")

async def main() -> None:
    encabezado("04 · Streaming")
    await ver_llegar()
    await cortar_a_mitad()

if __name__ == "__main__":
    asyncio.run(main())
```
