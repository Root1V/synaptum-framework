"""05 · Varios agentes en cadena, cada uno con su contexto.

**Dominio: el pipeline de doblaje** — vídeo en inglés a español, todo local:
Whisper transcribe, pyannote separa hablantes, un LLM traduce con contexto, y
IndexTTS clona cada voz. Las etapas del medio son las que necesitan criterio, y
ahí es donde entran los agentes.

**Lo primero, sin rodeos: `delegate()` no existe todavía** — es `SYN-41`, y el
tipo `DelegateStep` está escrito pero nada lo emite. Esto no es un problema para
componer agentes, porque el bucle **es** un generador asíncrono: orquestar varios
es código async normal, y eso ya funciona. Lo que `SYN-41` añadirá es que la
delegación quede **en el journal** como un paso propio, con su consumo atribuido.

La regla que sí se puede seguir hoy, y es la que importa:

    **Entre agentes viaja el resultado, nunca el historial.**

Duplicar el contexto de un agente en otro es la forma más cara de equivocarse:
se paga dos veces por los mismos tokens y el segundo hereda los errores del
primero sin poder distinguirlos de sus datos.

    uv run python examples/agentes/05_cadena_de_agentes.py
"""

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


# ── Herramientas de cada etapa ────────────────────────────────────────────────

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


# ── Los agentes ───────────────────────────────────────────────────────────────
#
# Dos agentes, dos trabajos, dos contextos. El traductor no sabe nada de
# duraciones y el revisor no ve el razonamiento del traductor: solo su resultado.

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


# Tres cosas que este ejemplo enseña y que no se ven en uno de un solo agente:
#
# 1. **Cada agente tiene su propio `run_id`.** Son runs distintos, se reanudan
#    por separado, y el journal de uno no contamina el del otro.
# 2. **El brief es texto, no historial.** El revisor recibe 2 frases donde el
#    traductor manejó una transcripción, un glosario y su razonamiento.
# 3. **El coste se suma explícitamente.** Hoy lo hace el orquestador; con
#    `SYN-41` el `DelegateStep` lo atribuirá solo.

if __name__ == "__main__":
    asyncio.run(main())
