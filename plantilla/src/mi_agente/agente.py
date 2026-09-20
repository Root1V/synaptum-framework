"""El agente: su configuración, y nada de comportamiento.

Un `Agent` no se hereda ni tiene métodos que sobrescribir. Es su configuración
más el bucle, así que puedes reutilizar el mismo objeto para mil runs a la vez.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from synaptum import Agent, Limits

from .herramientas import borrar, contar_lineas, listar_ficheros

HERRAMIENTAS = [listar_ficheros, contar_lineas, borrar]


@dataclass
class Respuesta:
    """La salida tipada del agente.

    Una `dataclass` de stdlib basta: el esquema se deriva de ella, se le pide al
    modelo y se valida la respuesta. Pydantic es un extra para quien ya lo use.

    Lo que sale de aquí puede ir a un panel, a un webhook o a una regla — tres
    consumidores que no pueden parsear prosa.
    """

    respuesta: str
    ficheros_consultados: list[str]


def nombre_del_modelo() -> str:
    """`proveedor:modelo`.

    El prefijo dice **quién normaliza la respuesta**, no solo qué modelo quieres.
    Es nuestro y no viaja por el cable.
    """
    return f"openai-compatible:{os.environ.get('SYNAPTUM_MODEL', 'qwen3-8b')}"


def construir() -> Agent:
    return Agent(
        "mi-agente",
        model=nombre_del_modelo(),
        instructions=(
            "Respondes preguntas sobre ficheros usando las herramientas. "
            "No inventes rutas: si no sabes qué hay, lista primero."
        ),
        tools=HERRAMIENTAS,
        output=Respuesta,
        # Topes del bucle. Son **corrección, no política**: evitan que un bucle
        # mal formado no termine nunca. Cuánto puede gastar un run es política, y
        # eso pertenece a quien gobierna, no aquí.
        limits=Limits(max_steps=10, max_retries=2),
    )
