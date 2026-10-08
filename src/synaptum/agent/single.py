"""
Una llamada al modelo **gobernada y durable**, sin escribir un agente.

Pedida por el segundo consumidor: una plataforma tiene llamadas sueltas
—clasificar, segmentar, decidir un enrutado— que no son un bucle agéntico y aun
así deben pasar por la misma puerta. Si no, parte del tráfico de modelo queda
fuera del gobierno: sin presupuesto, sin atribución y sin diario.

**Es un run de un solo turno, no un atajo por fuera.** Registra un `ModelStep`
con su identidad de paso, así que al reanudar **no se vuelve a inferir** — que
era su criterio de aceptación— y lo que gobierne la costura lo gobierna igual.
Lo que no tiene es herramientas: si hacen falta, eso es un agente.
"""

from __future__ import annotations

from typing import Any

from ..core.events import FinalStep
from .agent import Agent, Entrada, Limits, Sampling, Session

__all__ = ["generate"]


async def generate(
    task: Entrada,
    *,
    model: str,
    session: Session,
    instructions: Any = None,
    output: Any = None,
    sampling: Sampling | None = None,
    max_steps: int = 4,
) -> Any:
    """Pide una respuesta y devuelve el objeto tipado, o el texto.

    Args:
        task: la petición — texto, un mensaje, o las partes de uno.
        model: qué modelo.
        session: por dónde sale y dónde se recuerda.  **Hace falta**: sin ella
            no habría diario, y sin diario esto sería la llamada suelta que
            viene a sustituir.
        instructions: prompt de sistema, o una plantilla versionada.
        output: el esquema de la respuesta.  Sin él, devuelve texto.
        sampling: temperatura y demás.  Sin él, lo que decida quien ejecuta.
        max_steps: techo de turnos.  Más de uno porque una salida que no valida
            se vuelve a pedir **enseñando el error**, y eso cuesta un turno; con
            uno solo, un modelo que falla la primera vez no tendría ocasión de
            corregir.

    Returns:
        El objeto validado contra ``output``, o el texto si no se pidió esquema.
    """
    agente = Agent(
        "generate",
        model=model,
        instructions=instructions,
        output=output,
        sampling=sampling,
        limits=Limits(max_steps=max_steps),
    )
    async for paso in agente.run(task, session=session):
        if isinstance(paso, FinalStep):
            return paso.output
    raise AssertionError("el bucle terminó sin paso final")   # pragma: no cover
