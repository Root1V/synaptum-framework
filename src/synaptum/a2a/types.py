"""
SYN-44 · El vocabulario de A2A que usamos.

Solo lo que necesitamos para delegar: no es una implementación de la
especificación entera, y decirlo evita que alguien la tome por una.

A2A v1.0 (abril de 2026, Linux Foundation). Binding **HTTP+JSON**; los otros dos
—gRPC y JSON-RPC sobre WebSocket— quedan fuera hasta que alguien los pida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = ["TaskState", "AgentCard", "Task", "Artifact", "TERMINALES", "ESPERANDO"]


class TaskState(str, Enum):
    """Los ocho estados de una tarea A2A.

    Se distinguen tres clases, y la distinción es la que decide qué hace el
    bucle: los **terminales** no aceptan más mensajes, los que **esperan a
    alguien** sí, y el resto siguen su curso.
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"


#: No aceptan más mensajes.  Preguntar otra vez no cambia nada.
TERMINALES = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED}
)

#: Interrumpidos a la espera de una persona o de credenciales.
#:
#: Es exactamente nuestro ``ApprovalStep``: un run que se detiene porque alguien
#: de fuera tiene que decidir.  Que A2A tenga los mismos dos estados no es
#: casualidad — es el mismo problema.
ESPERANDO = frozenset({TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED})


@dataclass(frozen=True, slots=True)
class AgentCard:
    """Lo que un agente publica en ``/.well-known/agent-card.json``.

    Declara **qué sabe hacer**, no **qué puede romper**: no hay campo de riesgo
    en la especificación. De ahí que un agente remoto entre como destructivo
    mientras alguien no diga lo contrario.
    """

    name: str
    description: str = ""
    url: str = ""
    version: str = ""
    skills: tuple[Mapping[str, Any], ...] = ()
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    security_schemes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def streaming(self) -> bool:
        return bool(self.capabilities.get("streaming"))

    @property
    def push_notifications(self) -> bool:
        return bool(self.capabilities.get("pushNotifications"))


@dataclass(frozen=True, slots=True)
class Artifact:
    """Un entregable producido por una tarea.

    Distinto de un mensaje: un mensaje es conversación, un artefacto es
    resultado.
    """

    artifact_id: str = ""
    name: str = ""
    parts: tuple[Mapping[str, Any], ...] = ()

    @property
    def text(self) -> str:
        return "".join(str(p.get("text", "")) for p in self.parts if "text" in p)


@dataclass(frozen=True, slots=True)
class Task:
    """Una unidad de trabajo al otro lado.

    ``task_id`` lo asigna **el servidor** y no podemos aportarlo — es el hecho
    que decide todo nuestro diseño de reanudación. ``context_id`` sí lo ponemos
    nosotros, y es por donde se reencuentra.
    """

    task_id: str
    context_id: str = ""
    state: TaskState = TaskState.SUBMITTED
    message: str = ""
    """El texto del último mensaje del agente, si lo hay."""
    artifacts: tuple[Artifact, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)
    """El cuerpo tal cual.  Lo que no traducimos no se pierde."""

    @property
    def terminal(self) -> bool:
        return self.state in TERMINALES

    @property
    def esperando(self) -> bool:
        return self.state in ESPERANDO

    @property
    def result(self) -> str:
        """Lo que vuelve al agente que delegó.

        Prioriza los artefactos —son el entregable— y cae al último mensaje si
        no hay ninguno. Un agente que responde en prosa sin producir artefactos
        es legítimo y frecuente.
        """
        de_artefactos = "\n".join(a.text for a in self.artifacts if a.text)
        return de_artefactos or self.message
