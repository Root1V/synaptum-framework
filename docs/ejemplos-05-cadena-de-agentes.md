# 05 · Varios agentes en cadena, cada uno con su contexto

> **Esto es un fichero que se ejecuta:** [`examples/agentes/05_cadena_de_agentes.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/05_cadena_de_agentes.py) ↗
> Esta página lo transcribe y enseña lo que imprime. Si dejan de coincidir, falla un test.

**Dominio: el pipeline de doblaje (Prosodia)** — vídeo en inglés a español, todo
local:
Whisper transcribe, pyannote separa hablantes, un LLM traduce con contexto, y
IndexTTS clona cada voz. Las etapas del medio son las que necesitan criterio, y
ahí es donde entran los agentes.

**Esto es composición a mano, y sigue siendo la forma correcta aquí.** Delegar
como primitiva —`Agent(delegates=[…])`, ver el ejemplo 09— sirve cuando **el
modelo decide** a quién llamar. Cuando el orden lo decides tú, como en una
cadena fija, componer con asyncio es más simple y más explícito.

La regla vale para las dos formas:

    **Entre agentes viaja el resultado, nunca el historial.**

Duplicar el contexto de un agente en otro es la forma más cara de equivocarse:
se paga dos veces por los mismos tokens y el segundo hereda los errores del
primero sin poder distinguirlos de sus datos.

## Cómo correrlo

```bash
uv run python examples/agentes/05_cadena_de_agentes.py
```

No hace falta configurar nada: sin modelo, las respuestas van guionizadas y **todo lo
demás es real** — las herramientas se ejecutan, el journal se escribe, el consumo se mide.
Con `AXONIUM_CLIENT_ID` o `SYNAPTUM_BASE_URL` en el entorno, **el mismo fichero sin tocar**
habla con un modelo de verdad; lo que cambia entonces es lo que diga el modelo, no el
código. Ver [Modelos](04-modelos.md).

## Lo que imprime

```text
05 · Cadena de agentes
──────────────────────
sin inferencia · respuestas guionizadas (exporta SYNAPTUM_BASE_URL para usar un modelo real)

  traductor  «Mira, no estoy diciendo que sea imposible. Estoy diciendo que nadie lo ha hecho todavía.»
             registro coloquial · se mantiene «Mira,» por coherencia con seg-0007
  revisor    corregida · cabe en el hueco: False
             «Mira, no digo que sea imposible. Digo que nadie lo ha hecho aún.»

  a síntesis «Mira, no digo que sea imposible. Digo que nadie lo ha hecho aún.»
  coste total del segmento: entrada=500 salida=100
```

```python
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import Agent, FinalStep, Session, Usage, tool
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

@dataclass
class Traduccion:
    texto_es: str
    registro: str
    notas: list[str]

@dataclass
class Revision:
    aprobada: bool
    duracion_ok: bool
    correccion: str
```

## Herramientas de cada etapa

```python
SEGMENTO = {
    "id": "seg-0042",
    "hablante": "SPEAKER_01",
    "inicio": 128.4,
    "fin": 133.9,
    "texto_en": "Look, I'm not saying it's impossible — I'm saying nobody's done it yet.",
}

@tool(idempotent=True)
async def leer_segmento(
    segmento_id: Annotated[str, "Identificador del segmento diarizado"],
) -> str:
    """Transcripción y metadatos de un segmento, tal como salen de Whisper + pyannote."""
    s = SEGMENTO
    return (
        f"hablante: {s['hablante']}\n"
        f"duración: {s['fin'] - s['inicio']:.1f}s\n"
        f"texto: {s['texto_en']}"
    )

@tool(idempotent=True)
async def glosario(
    termino: Annotated[str, "Término o expresión en inglés"],
) -> str:
    """Cómo se ha traducido antes un término en este proyecto."""
    return "«Look,» al inicio de frase → «Mira,» (decidido en seg-0007)"

@tool(idempotent=True)
async def estimar_duracion(
    texto: Annotated[str, "Texto en español"],
    voz: Annotated[str, "Identificador de la voz clonada"],
) -> str:
    """Cuánto duraría este texto con esa voz. El doblaje tiene que caber en el hueco."""
    palabras = len(texto.split())
    segundos = palabras / 2.6
    return f"{palabras} palabras ≈ {segundos:.1f}s con {voz}"
```

## Los agentes

Dos agentes, dos trabajos, dos contextos. El traductor no sabe nada de
duraciones y el revisor no ve el razonamiento del traductor: solo su resultado.

```python
def traductor() -> Agent:
    return Agent(
        "traductor",
        model=nombre_del_modelo(),
        instructions=(
            "Traduces diálogo de vídeo al español neutro para doblaje. Consulta el "
            "glosario para mantener la coherencia con lo ya traducido. Prioriza que "
            "suene natural dicho en voz alta sobre la literalidad."
        ),
        tools=[leer_segmento, glosario],
        output=Traduccion,
    )

def revisor() -> Agent:
    return Agent(
        "revisor",
        model=nombre_del_modelo(),
        instructions=(
            "Revisas una traducción para doblaje. Comprueba que quepa en el hueco "
            "temporal del original con un margen del 10%. Si no cabe, acórtala sin "
            "perder el sentido."
        ),
        tools=[estimar_duracion],
        output=Revision,
    )

async def main() -> None:
    encabezado("05 · Cadena de agentes")

    total = Usage.zero()

    # ── Etapa 1 · traducir ───────────────────────────────────────────────────

    guion_traductor = [
        calls("leer_segmento", segmento_id="seg-0042"),
        calls("glosario", termino="Look,"),
        says(
            '{"texto_es": "Mira, no estoy diciendo que sea imposible. Estoy diciendo '
            'que nadie lo ha hecho todavía.", "registro": "coloquial", '
            '"notas": ["se mantiene «Mira,» por coherencia con seg-0007"]}'
        ),
    ]
    puerta = gateway(guion_traductor, tools=[leer_segmento, glosario])

    traduccion = None
    async for paso in traductor().run(
        "Traduce el segmento seg-0042.", session=Session("dub-0042-tr", puerta)
    ):
        if isinstance(paso, FinalStep):
            traduccion, total = paso.output, total + paso.usage

    print(f"  traductor  «{traduccion.texto_es}»")
    print(f"             registro {traduccion.registro} · {traduccion.notas[0]}")

    # ── Etapa 2 · revisar ────────────────────────────────────────────────────
    #
    # Al revisor le llega **el resultado**, no la conversación del traductor. Su
    # brief es estrecho a propósito: es lo que hace que delegar aísle contexto en
    # vez de duplicarlo.

    brief = (
        f"Traducción propuesta: «{traduccion.texto_es}»\n"
        f"El original dura 5.5s y lo dice la voz SPEAKER_01."
    )

    guion_revisor = [
        calls("estimar_duracion", texto=traduccion.texto_es, voz="SPEAKER_01"),
        says(
            '{"aprobada": false, "duracion_ok": false, '
            '"correccion": "Mira, no digo que sea imposible. Digo que nadie lo ha '
            'hecho aún."}'
        ),
    ]
    puerta_revisor = gateway(guion_revisor, tools=[estimar_duracion])

    revision = None
    async for paso in revisor().run(brief, session=Session("dub-0042-rev", puerta_revisor)):
        if isinstance(paso, FinalStep):
            revision, total = paso.output, total + paso.usage

    veredicto = "aprobada" if revision.aprobada else "corregida"
    print(f"  revisor    {veredicto} · cabe en el hueco: {revision.duracion_ok}")
    print(f"             «{revision.correccion}»")

    final = revision.correccion if not revision.aprobada else traduccion.texto_es
    print(f"\n  a síntesis «{final}»")
    print(f"  coste total del segmento: entrada={total.input} salida={total.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Lo que esto enseña

Tres cosas que este ejemplo enseña y que no se ven en uno de un solo agente:

1. **Cada agente tiene su propio `run_id`.** Son runs distintos, se reanudan
```text
por separado, y el journal de uno no contamina el del otro.
```

2. **El brief es texto, no historial.** El revisor recibe 2 frases donde el
```text
traductor manejó una transcripción, un glosario y su razonamiento.
```

3. **El coste se suma explícitamente**, porque aquí el orquestador eres tú.
```text
Delegando como primitiva, el `DelegateStep` lo atribuye solo — es la
diferencia principal entre las dos formas, y está en el ejemplo 09.
```

---

**El fichero entero, para clonarlo y tocarlo:** [`examples/agentes/05_cadena_de_agentes.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/05_cadena_de_agentes.py) ↗

Está en [`examples/`](https://github.com/Root1V/synaptum-framework/tree/main/examples) con los otros quince, y todos corren igual.
