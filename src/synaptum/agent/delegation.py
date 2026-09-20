"""
SYN-41 · Delegar a un subagente con contexto aislado.

Componer agentes a mano ya funcionaba —el bucle es un generador asíncrono, así
que orquestar varios es asyncio normal— y tenía tres agujeros que estaban
documentados como limitaciones:

1. **El coste desaparecía de la vista.** El `usage` de quien delega medía *sus*
   llamadas, no las de dentro. Un sistema que gasta cinco veces más parecía
   igual de barato.
2. **Un subagente no era un paso durable.** Si el proceso moría a mitad, al
   reanudar se reejecutaba entero: el journal lo veía como una llamada, no como
   un run con sus propios pasos.
3. **El riesgo no se propagaba.** El envoltorio entraba como ``READ`` aunque por
   dentro llamara a algo que borra.

Los tres se cierran aquí, y el segundo es el que obliga a que esto sea una
primitiva y no un patrón: **un subagente necesita su propio journal**, y su
identidad tiene que derivarse de la del padre para que reanudar lo encuentre.

La identidad del sub-run
-------------------------
``{run_id del padre}/{step_id de la delegación}``. Determinista, como todo lo
demás: el paso 3 de un run es siempre el paso 3, así que al reanudar el
subagente se reencuentra con su propio diario y no vuelve a pagar lo suyo.

Lo que viaja, y lo que no
--------------------------
Al subagente le llega **el brief y nada más**. No su historial, no el del padre.
Duplicar el contexto de un agente en otro se paga dos veces y hace que el
segundo herede los errores del primero sin poder distinguirlos de sus datos.

De vuelta sube el **resultado** y el **consumo agregado**. El historial del
subagente se queda en su diario, donde se puede auditar, y no en el prompt del
padre, donde solo costaría dinero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.types import Risk, ToolDefinition

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent

__all__ = ["Delegate", "delegate_risk"]


def delegate_risk(agente: "Agent") -> Risk:
    """El riesgo de delegar es el mayor de lo que el subagente puede hacer.

    Se **deriva** en vez de declararse, y aquí sí se puede: quien delega no sabe
    qué herramientas tiene el otro, pero el framework sí. Es lo contrario que en
    ``@tool``, donde ninguna anotación puede saber que una función que devuelve
    ``str`` mueve dinero.

    Si un subagente tiene una herramienta ``DESTRUCTIVE``, **delegar en él es
    destructivo**. Sin esto, envolver un agente en una función lo blanqueaba a
    ``READ``.

    Hasta dónde llega esto hoy, dicho con precisión
    ------------------------------------------------
    El riesgo **se declara** —el modelo lo ve en el catálogo, y el arnés en el
    handshake— pero **la delegación no cruza la costura**: el bucle arranca al
    subagente sin preguntar. Así que una política de gateway no puede denegar
    una delegación *antes* de que empiece.

    Lo que sí sigue funcionando es lo que importa para la seguridad: las
    herramientas del subagente **sí** cruzan la costura cuando las llama, así
    que un efecto destructivo se detiene igual. Lo que se pierde es detenerlo
    antes de pagar la inferencia del hijo.

    Denegar la delegación en sí exigiría un método de la costura que autorice
    sin ejecutar, y eso es un cambio de contrato: va por el canal de
    coordinación, no por aquí.

    Se mira también a sus propios subagentes: un riesgo que se pierde a dos
    saltos se pierde igual.
    """
    orden = (Risk.READ, Risk.SOFT_WRITE, Risk.HARD_WRITE, Risk.DESTRUCTIVE)
    mayor = Risk.READ

    for herramienta in agente.tools:
        if orden.index(herramienta.risk) > orden.index(mayor):
            mayor = herramienta.risk

    for sub in getattr(agente, "delegates", ()):
        heredado = delegate_risk(sub.agent)
        if orden.index(heredado) > orden.index(mayor):
            mayor = heredado

    return mayor


@dataclass(frozen=True, slots=True)
class Delegate:
    """Un subagente, tal como lo ve quien delega.

    Se presenta al modelo como una herramienta de **un solo parámetro**: el
    brief. No se le ofrecen las herramientas del subagente, y eso es lo que hace
    barato delegar — el catálogo del padre no crece con el del hijo.
    """

    agent: "Agent"
    description: str = ""
    """Cuándo usarlo.  Si falta, se toma de las instrucciones del subagente."""

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def risk(self) -> Risk:
        return delegate_risk(self.agent)

    async def execute(self, brief: str, session: Any, run_id: str) -> tuple[Any, Any]:
        """Corre el subagente y devuelve ``(resultado, consumo)``.

        **Es el único punto que sabe dónde vive el subagente.** Aquí, en este
        proceso; en un delegado remoto, al otro lado de una red. Todo lo demás
        —el paso durable, la reanudación, el consumo agregado, el riesgo
        declarado— es idéntico en los dos casos, y por eso está fuera de aquí.
        """
        from ..core.types import Usage
        from .agent import Session

        salida: Any = None
        consumo = Usage.zero()
        hija = Session(run_id, session.gateway, session.checkpointer)

        async for paso in self.agent._loop(brief, hija, stream=False, depth=self._depth):
            if paso.kind == "final":
                salida, consumo = paso.output, paso.usage
        return salida, consumo

    _depth: int = 0
    """Profundidad que se le pasa al subagente.  Lo rellena el bucle."""

    @property
    def definition(self) -> ToolDefinition:
        """Lo que el modelo ve: un nombre, cuándo usarlo, y un hueco para el brief."""
        return ToolDefinition(
            name=self.name,
            description=self.description or _cuando_usarlo(self.agent),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "brief": {
                        "type": "string",
                        "description": (
                            "La tarea para el especialista, completa y autónoma. "
                            "No ve esta conversación."
                        ),
                    }
                },
                "required": ["brief"],
            },
            risk=self.risk,
            # Delegar **nunca** es idempotente: dentro puede haber cualquier
            # cosa. Suponer que repetir no cuesta sería suponer por el otro.
            idempotent=False,
        )


def _cuando_usarlo(agente: "Agent") -> str:
    """La primera frase de sus instrucciones, que es lo que responde «para qué es»."""
    instrucciones = (agente.instructions or "").strip()
    if not instrucciones:
        return f"Delega una tarea al especialista '{agente.name}'."
    primera = instrucciones.split("\n")[0].split(". ")[0].strip().rstrip(".")
    return f"{primera}. Delega una tarea a este especialista."


def sub_run_id(parent_run_id: str, step_id: str) -> str:
    """Identidad determinista del sub-run.

    Que se derive de la del padre es lo que permite reanudar: al volver a entrar
    en la misma delegación, el subagente encuentra **su** diario y salta lo que
    ya pagó.
    """
    return f"{parent_run_id}/{step_id}"
