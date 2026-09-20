# 12 · Un agente que vive en otro contenedor

> **Generada de [`examples/agentes/12_agente_remoto.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/12_agente_remoto.py).** El fichero corre; esta página lo
> transcribe. Si los dos no coinciden, falla un test.

**Dominio: el pipeline de doblaje.** El revisor de sincronía no es tuyo: lo
mantiene otro equipo, se despliega por su cuenta y se actualiza cuando ellos
quieren. No puedes importarlo — solo hablarle.

Esa es la pregunta que este ejemplo contesta: **¿se pueden desplegar los agentes
en contenedores distintos y seguir orquestándolos?** Sí, y la respuesta no es
nuestra: es **A2A**, el protocolo que estandarizó la Linux Foundation, la misma
pieza que MCP ocupa para las herramientas. MCP conecta un agente con sus
herramientas; A2A conecta un agente con **otro agente**.

Y lo que hace este framework con eso cabe en una frase: un agente remoto entra
por la misma ranura que uno local.

    coordinador = Agent("…", delegates=[local, remoto])

```bash
uv run python examples/agentes/12_agente_remoto.py
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
    Phase,
    Risk,
    Role,
    Session,
    tool,
)
from synaptum.a2a import A2AClient, RemoteDelegate
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

sys.path.insert(0, str(Path(__file__).parent))
from servidor_a2a.agente_remoto import AgenteRemoto
```

## Un especialista local, para comparar

```python
@tool(idempotent=True)
async def leer_guion(
    episodio: Annotated[str, "Identificador del episodio"],
) -> str:
    """Guion original y sus marcas de tiempo."""
    return (
        f"{episodio} · 47 intervenciones · duración 22:41\n"
        "marcas densas entre 04:00-05:00 y 17:30-18:10"
    )

def transcriptor() -> Agent:
    return Agent(
        "transcriptor",
        model=nombre_del_modelo(),
        instructions="Lees el guion original y resumes su estructura.",
        tools=[leer_guion],
    )

async def main() -> None:
    encabezado("12 · Un agente en otro contenedor")

    remoto_proc = AgenteRemoto()          # en producción: otro despliegue
    try:
        await parte_uno(remoto_proc)
        await parte_dos(remoto_proc)
    finally:
        remoto_proc.cerrar()
```

## 1 · Quién es, antes de hablarle

```python
async def parte_uno(remoto_proc: AgenteRemoto) -> None:
    tarjeta = await A2AClient(remoto_proc.url).agent_card()
    print(f"  su tarjeta dice: {tarjeta.name} — {tarjeta.description}")
    print(f"  habilidades:     {[h.get('name') for h in tarjeta.skills]}")
    print(
        "\n  Lo que **no** dice es qué riesgo tiene, porque A2A no tiene ese\n"
        "  campo: una tarjeta declara habilidades. Por eso `risk` es obligatorio\n"
        "  al conectarlo y no tiene valor por defecto — un defecto conservador\n"
        "  sería correcto y silencioso, y la decisión la tomaría un valor en vez\n"
        "  de una persona.\n"
    )
```

## 2 · Delegar sin que el bucle note la diferencia

```python
async def parte_dos(remoto_proc: AgenteRemoto) -> None:
    revisor = RemoteDelegate(
        name="revisor",
        url=remoto_proc.url,
        # Lo declara quien lo conecta, con su nombre. Nadie más puede saberlo:
        # al otro lado hay un modelo decidiendo, y puede cambiar sin avisarnos.
        risk=Risk.READ,
        description="Revisa la sincronía de una pista doblada contra el original.",
        poll_every=0.05,          # en producción, segundos
    )

    coordinador = Agent(
        "doblaje",
        model=nombre_del_modelo(),
        instructions=(
            "Coordinas la revisión de un episodio doblado. Pide la estructura "
            "del guion y la revisión de sincronía, y resume en dos frases qué "
            "hay que retocar."
        ),
        # Uno local y uno al otro lado de una red. El bucle no los distingue:
        # acepta cualquier cosa con `execute`, no una clase concreta.
        delegates=[transcriptor(), revisor],
    )

    print("  el catálogo del coordinador:")
    for definicion in coordinador.tools:
        print(f"    {definicion.name:<14} riesgo={definicion.risk.value}")
    print()

    sesion = Session(
        "doblaje-ep12",
        gateway([_guion] * 5, tools=[leer_guion]),
        MemoryCheckpointer(),
    )

    async for paso in coordinador.run("Revisa el episodio 12.", session=sesion):
        match paso:
            case DelegateStep(phase=Phase.ATTEMPTED, agent=quien):
                print(f"  → delega en {quien}")
            case DelegateStep(phase=Phase.COMPLETED, agent=quien, usage=consumo):
                medido = (
                    f"entrada={consumo.input} salida={consumo.output}"
                    if consumo.input is not None
                    else "sin medir — el remoto no reportó consumo"
                )
                print(f"    ← {quien} · {medido}")
            case FinalStep(output=salida, usage=total):
                print(f"\n  {salida}")
                print(f"\n  total del run: entrada={total.input} salida={total.output}")

    print(
        f"\n  mensajes que recibió el agente remoto: {len(remoto_proc.enviados)}\n"
        f"  contextId que usó el cliente: "
        f"{remoto_proc.enviados[0].get('contextId')!r}\n"
        "  — es el `run_id` del sub-run, `{run_id del padre}/{step_id}`. Un\n"
        "  identificador, dos sistemas: para nosotros es la identidad del paso;\n"
        "  para A2A, la conversación.\n"
    )

    # Y lo que hace ese identificador, medido:
    await reanudar(remoto_proc, revisor)

async def reanudar(remoto_proc: AgenteRemoto, revisor: RemoteDelegate) -> None:
    """Volver a entrar en la misma delegación no vuelve a encargarla."""
    antes = len(remoto_proc.enviados)
    contexto = remoto_proc.enviados[0].get("contextId")

    sesion = Session("doblaje-ep12", gateway([], tools=[]), MemoryCheckpointer())
    resultado, _ = await revisor.execute("Revisa el episodio 12.", sesion, contexto)

    print("  reanudando con el mismo contextId:")
    print(f"    mensajes enviados: {antes} → {len(remoto_proc.enviados)}")
    print(f"    y aun así devuelve el resultado: {resultado[:48]}…")

def _guion(peticion):
    nombres = {t.name for t in peticion.tools}
    resultados = sum(1 for m in peticion.messages if m.role is Role.TOOL)

    if "revisor" in nombres:                       # el coordinador
        if resultados == 0:
            return calls("transcriptor", id="d1", brief="Resume la estructura del episodio 12.")
        if resultados == 1:
            return calls("revisor", id="d2", brief="Revisa la sincronía del episodio 12 en es-419.")
        return says(
            "El episodio 12 tiene tres intervenciones fuera de tolerancia "
            "(04:12, 09:38 y 17:55), dos de ellas en el tramo denso del guion. "
            "Con retocar esas tres queda listo."
        )

    if resultados == 0:                            # el transcriptor
        return calls("leer_guion", id="t1", episodio="ep12")
    return says("47 intervenciones en 22:41, con marcas densas en 04:00-05:00 y 17:30-18:10.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Lo que esto enseña

Las tres cosas que pasan aquí y no se ven:

**Reanudar es una consulta, no una apuesta.** El `taskId` lo asigna el
servidor y se pierde en una caída. El `contextId` lo ponemos nosotros, así que
al volver se pregunta `tasks/list(contextId)` antes de enviar nada — y si ya
había una tarea, se espera a esa en vez de encargar el trabajo otra vez. Es la
diferencia entre reanudar y pagar dos veces un trabajo irreversible. Contra un
servidor sin `tasks/list` se baja un peldaño —un `messageId` determinista, que
garantiza menos— y se dice.

**Esperar es parte del protocolo.** Un agente remoto puede tardar minutos, y
puede quedarse esperando a una persona (`input-required`, `auth-required`):
eso no es un fallo ni un resultado, es lo mismo que nuestro `ApprovalStep` del
otro lado de la red, y vuelve al modelo dicho con esas palabras. Cancelar es
dejar de esperar: al cerrar el iterador se cancela la tarea remota.

**El consumo puede no venir.** A2A no define un campo para él. Si el servidor
lo pone en `metadata` se lee; si no, queda `None` — «nadie lo midió», que no
es lo mismo que cero. Un agente remoto siempre cuesta algo.

Y una advertencia sobre dónde poner la `url`: **en un despliegue gobernado
apunta al proxy del arnés**, no al agente. Un control en el framework es una
petición; una frontera en el camino es una frontera. El código no cambia — solo
a dónde apunta.
