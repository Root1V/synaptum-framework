"""
RM-05 · Taxonomía de eventos del bucle.

El bucle del agente no es un ``while`` oculto: cede el control en cada frontera
significativa emitiendo un evento tipado.  Quien itera puede mirar, medir,
aprobar, denegar o cortar — sin que el bucle sepa quién está al otro lado.

Cada evento se emite **dos veces**: una con ``Phase.INTENT``, antes del efecto,
y otra con ``Phase.RESULT``, después.  Ambas comparten ``step_id`` y difieren en
``phase``.  Eso es lo que permite tres cosas a la vez:

* **Escritura anticipada.**  Si el proceso cae entre la intención y el
  resultado, el journal dice que el efecto pudo haber ocurrido, y el replay
  puede decidir en consecuencia en vez de adivinar.
* **Aplicación.**  La costura de aplicación ve la intención antes de que el
  efecto exista, que es el único momento en el que denegarlo sirve de algo.
* **Idempotencia.**  ``(run_id, step_id, phase)`` es la clave con la que el
  ``Checkpointer`` deduplica, de modo que el reintento *at-least-once* de una
  Activity de Temporal no corrompe el journal con duplicados.

``step_id`` es **determinista**: se deriva de la posición del paso en el run, no
de un UUID ni de un reloj.  Un run reproducido acuña exactamente los mismos
identificadores, y por eso el replay puede emparejar un paso con su resultado ya
registrado y **saltarse una inferencia ya pagada**.  Esa es la propiedad que
justifica toda la arquitectura, y se apoya entera en esta línea.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .types import Request, Response, Risk, ToolCall, ToolResult, Usage

__all__ = [
    "Phase",
    "Durability",
    "Risk",
    "Disposition",
    "Decision",
    "ALLOW",
    "StepEvent",
    "ModelStep",
    "ToolStep",
    "DelegateStep",
    "ApprovalStep",
    "FinalStep",
    "Event",
    "make_step_id",
    "idempotency_key",
]


# ── Fases y durabilidad ───────────────────────────────────────────────────────

class Phase(str, Enum):
    INTENT = "intent"
    RESULT = "result"


class Durability(str, Enum):
    """Con qué urgencia debe persistirse un evento — RM-14.

    Separar las dos costuras no sirve para que el journal sea asíncrono: sirve
    para que cada evento tenga la política que le toca en vez de heredar la peor
    de las dos.
    """

    DURABLE = "durable"
    """Persistir de forma síncrona antes de continuar.  Si se pierde, tras una
    caída es imposible saber si el efecto ocurrió."""

    DEFERRABLE = "deferrable"
    """Puede diferirse o agruparse por lotes.  Perderlo cuesta como mucho
    repetir trabajo que no tiene consecuencias externas."""


# ── Decisión de la costura de aplicación — H4 ─────────────────────────────────

class Disposition(str, Enum):
    """«No» no es una sola cosa.

    Sin distinguir estos casos el bucle no sabe si reintentar con otra cosa,
    esperar o parar, y cada implementación acabaría inventando su convención.
    """

    ALLOW = "allow"
    DENY_STEP = "deny_step"
    """Este efecto no.  El bucle puede intentar otra cosa."""

    TERMINATE_RUN = "terminate_run"
    """Se acabó.  No hay alternativa: presupuesto agotado, política dura."""

    REQUIRE_APPROVAL = "require_approval"
    """Pendiente de una persona.  El run se suspende, no falla."""


@dataclass(frozen=True, slots=True)
class Decision:
    disposition: Disposition
    reason_code: str = ""
    """Código estable, apto para métricas y alertas.  No es el mensaje."""
    message: str = ""
    """Explicación legible.  Puede acabar delante de una persona."""

    @property
    def allowed(self) -> bool:
        return self.disposition is Disposition.ALLOW


ALLOW = Decision(Disposition.ALLOW)


# ── Identidad de paso — RM-06 ─────────────────────────────────────────────────

def make_step_id(seq: int, kind: str) -> str:
    """Acuña un identificador de paso determinista.

    Deriva de la posición del paso en el run, nunca de un UUID ni de un reloj:
    un run reproducido debe acuñar exactamente los mismos identificadores, o el
    replay no puede reconocer lo ya hecho.

    El ancho fijo mantiene el orden lexicográfico alineado con el numérico, de
    modo que un almacén que ordene por clave devuelve el journal en orden de
    ejecución sin índice adicional.
    """
    if seq < 0:
        raise ValueError("seq no puede ser negativo.")
    return f"{seq:06d}-{kind}"


def idempotency_key(event: "StepEvent") -> tuple[str, str, str]:
    """La clave con la que el ``Checkpointer`` deduplica — A3.

    Un ``append`` repetido con esta misma clave es un no-op, nunca un error.
    """
    return (event.run_id, event.step_id, event.phase.value)


# ── Eventos ───────────────────────────────────────────────────────────────────
#
# ``kw_only`` en toda la jerarquía: los eventos se construyen por nombre, así
# que añadir un campo nunca reordena una llamada existente.

@dataclass(frozen=True, kw_only=True)
class StepEvent:
    """Base común.  No se instancia directamente."""

    run_id: str
    step_id: str
    seq: int
    phase: Phase
    at: float | None = None
    """Marca temporal en épocas, solo para observabilidad.

    **No forma parte de la identidad.**  La deduplicación usa
    ``idempotency_key``, de modo que un reintento con distinto reloj sigue
    reconociéndose como el mismo paso.
    """
    meta: Mapping[str, Any] = field(default_factory=dict)
    """Contexto de traza W3C y cualquier acompañante opaco.  Se propaga sin
    interpretarse."""

    kind: str = "step"

    @property
    def durability(self) -> Durability:
        return Durability.DURABLE

    @property
    def key(self) -> tuple[str, str, str]:
        return idempotency_key(self)


@dataclass(frozen=True, kw_only=True)
class ModelStep(StepEvent):
    """Una llamada al modelo.

    Siempre ``DURABLE``: una inferencia cuesta dinero y no es reproducible, así
    que perder el registro de que ocurrió es exactamente el fallo que la
    durabilidad existe para evitar.
    """

    kind: str = "model"
    request: Request | None = None
    """Presente en ``INTENT``."""
    response: Response | None = None
    """Presente en ``RESULT``."""
    usage: Usage = field(default_factory=Usage)
    """Vuelve por la costura aunque el span lo emita quien ejecutó — H3 con A5."""

    @property
    def durability(self) -> Durability:
        """La intención es diferible; el resultado, no.

        Una llamada al modelo no tiene efecto externo más allá de su coste, así
        que saber que *se intentó* no cambia ninguna decisión: si el registro se
        pierde, se vuelve a inferir, que es caro pero correcto.  El resultado sí
        es durable, y esa es la garantía que importa — **una vez escrito, no se
        repite**.

        Lo que se ahorra no es una escritura, es una **espera antes del efecto**.
        """
        return Durability.DEFERRABLE if self.phase is Phase.INTENT else Durability.DURABLE


@dataclass(frozen=True, kw_only=True)
class ToolStep(StepEvent):
    """La ejecución de una herramienta.

    Es el único evento cuya durabilidad depende del efecto: releer un fichero se
    puede repetir sin consecuencias, ordenar una transferencia no.  Esa
    diferencia la declara el contrato de la tool, dentro del framework — es
    justamente lo que un harness no puede deducir desde fuera.
    """

    kind: str = "tool"
    call: ToolCall | None = None
    """Presente en ``INTENT``."""
    result: ToolResult | None = None
    """Presente en ``RESULT``."""
    risk: Risk = Risk.READ
    idempotent: bool = False
    """Conservador por defecto: quien no declara nada paga durabilidad."""

    @property
    def durability(self) -> Durability:
        """Depende del efecto, y ambas fases por igual.

        Si el efecto no puede repetirse, la intención tiene que estar en disco
        **antes** de que ocurra: es la única forma de que, tras una caída, se
        sepa que pudo haber ocurrido.  Es la escritura anticipada clásica, y es
        la única espera bloqueante que el bucle impone en todo un turno.
        """
        return Durability.DEFERRABLE if self.idempotent else Durability.DURABLE


@dataclass(frozen=True, kw_only=True)
class DelegateStep(StepEvent):
    """Delegación a un subagente con contexto aislado.

    El worker recibe un brief estrecho y devuelve resultado y referencias, nunca
    su historial: es lo que hace que delegar aísle contexto en vez de duplicarlo.
    """

    kind: str = "delegate"
    agent: str = ""
    brief: str = ""
    """Presente en ``INTENT``."""
    result: Any = None
    """Presente en ``RESULT``."""
    usage: Usage = field(default_factory=Usage)
    """Consumo agregado del subagente, para atribuir el coste de orquestar."""

    @property
    def durability(self) -> Durability:
        return Durability.DURABLE


@dataclass(frozen=True, kw_only=True)
class ApprovalStep(StepEvent):
    """Una pausa a la espera de decisión.

    Siempre ``DURABLE``, y por una razón distinta a las demás: no se le pregunta
    dos veces a una persona porque el proceso se cayó.
    """

    kind: str = "approval"
    subject: str = ""
    """Qué se somete a decisión, en términos legibles."""
    decision: Decision | None = None
    """Presente en ``RESULT``."""

    @property
    def durability(self) -> Durability:
        return Durability.DURABLE


@dataclass(frozen=True, kw_only=True)
class FinalStep(StepEvent):
    """Cierre del run."""

    kind: str = "final"
    output: Any = None
    usage: Usage = field(default_factory=Usage)
    """Consumo acumulado de todo el run."""

    @property
    def durability(self) -> Durability:
        return Durability.DURABLE


Event = ModelStep | ToolStep | DelegateStep | ApprovalStep | FinalStep
