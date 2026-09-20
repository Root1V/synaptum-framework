"""
SYN-44 · Normalización de A2A — del cable a nuestro vocabulario.

**Pura por construcción**: no abre conexiones ni lee credenciales. Es la misma
separación que en los adaptadores de proveedor, y por el mismo motivo — permite
ejercitar la traducción **contra un fichero**, sin servidor y sin red.

Esa separación ya encontró un fallo real en el adaptador OpenAI-compatible: un
stream que era solo razonamiento no emitía ni un evento, y solo se vio porque el
cuerpo grabado se podía pasar por la función sin montar nada.
"""

from __future__ import annotations

from typing import Any, Mapping

from .types import AgentCard, Artifact, Task, TaskState

__all__ = [
    "task_from_wire",
    "agent_card_from_wire",
    "message_to_wire",
    "deterministic_message_id",
]


def _campo(origen: Any, *nombres: str, por_defecto: Any = None) -> Any:
    """Lee el primer nombre que exista, por atributo o por clave.

    A2A define los campos en `camelCase` sobre el cable, y hay servidores que
    responden en `snake_case` —los generados desde los `.proto` de gRPC—. Aceptar
    los dos cuesta una línea; no aceptarlos cuesta un fallo que solo aparece
    contra ciertos servidores.
    """
    for nombre in nombres:
        if isinstance(origen, Mapping):
            if nombre in origen:
                return origen[nombre]
        elif hasattr(origen, nombre):
            return getattr(origen, nombre)
    return por_defecto


def _estado(valor: Any) -> TaskState:
    """Traduce el estado, y **no inventa uno** si no lo conoce.

    La especificación tiene un mecanismo de extensión desde la v1.0.1, así que
    van a aparecer estados nuevos. Uno desconocido se trata como `WORKING` —que
    es «sigue su curso»— y nunca como terminal: dar por terminada una tarea que
    no lo está pierde su resultado en silencio.
    """
    if isinstance(valor, Mapping):
        valor = _campo(valor, "state", "status")
    texto = str(valor or "").lower().removeprefix("task_state_")
    try:
        return TaskState(texto)
    except ValueError:
        return TaskState.WORKING


def _texto_de_partes(partes: Any) -> str:
    return "".join(
        str(_campo(p, "text", por_defecto="")) for p in (partes or ()) if _campo(p, "text")
    )


def task_from_wire(cuerpo: Mapping[str, Any]) -> Task:
    """Un `Task` desde la respuesta del servidor."""
    tarea = _campo(cuerpo, "task", "result", por_defecto=cuerpo) or cuerpo
    estado_bruto = _campo(tarea, "status", "state")

    mensaje = ""
    if isinstance(estado_bruto, Mapping):
        mensaje = _texto_de_partes(_campo(_campo(estado_bruto, "message") or {}, "parts"))
    if not mensaje:
        historial = _campo(tarea, "history") or ()
        if historial:
            mensaje = _texto_de_partes(_campo(historial[-1], "parts"))

    return Task(
        task_id=str(_campo(tarea, "id", "taskId", "task_id", por_defecto="") or ""),
        context_id=str(_campo(tarea, "contextId", "context_id", por_defecto="") or ""),
        state=_estado(estado_bruto),
        message=mensaje,
        artifacts=tuple(
            Artifact(
                artifact_id=str(_campo(a, "artifactId", "artifact_id", por_defecto="") or ""),
                name=str(_campo(a, "name", por_defecto="") or ""),
                parts=tuple(_campo(a, "parts") or ()),
            )
            for a in (_campo(tarea, "artifacts") or ())
        ),
        raw=dict(tarea) if isinstance(tarea, Mapping) else {},
    )


def agent_card_from_wire(cuerpo: Mapping[str, Any]) -> AgentCard:
    """Un `AgentCard` desde `/.well-known/agent-card.json`."""
    return AgentCard(
        name=str(_campo(cuerpo, "name", por_defecto="") or ""),
        description=str(_campo(cuerpo, "description", por_defecto="") or ""),
        url=str(_campo(cuerpo, "url", por_defecto="") or ""),
        version=str(_campo(cuerpo, "version", "protocolVersion", por_defecto="") or ""),
        skills=tuple(_campo(cuerpo, "skills") or ()),
        capabilities=dict(_campo(cuerpo, "capabilities") or {}),
        security_schemes=dict(
            _campo(cuerpo, "securitySchemes", "security_schemes") or {}
        ),
    )


def message_to_wire(
    brief: str, *, context_id: str, message_id: str, task_id: str | None = None
) -> dict[str, Any]:
    """El cuerpo de un `SendMessage`.

    `contextId` lo ponemos nosotros —la especificación lo permite— y es lo que
    después hace localizable la tarea. `taskId` solo va cuando continuamos una
    que ya existe.
    """
    mensaje: dict[str, Any] = {
        "role": "user",
        "messageId": message_id,
        "contextId": context_id,
        "parts": [{"text": brief}],
    }
    if task_id:
        mensaje["taskId"] = task_id
    return {"message": mensaje}


def deterministic_message_id(context_id: str, intento: str = "") -> str:
    """Un `messageId` que se repite al reanudar.

    Determinista a propósito: la especificación dice que `SendMessage` **puede**
    detectar duplicados por este campo. No es una garantía y no construimos
    sobre ella —para eso está `ListTasks`— pero donde el servidor sí deduplica,
    cierra el hueco sin costar nada.

    Y no lleva contador de intento: eso lo haría no determinista, y entonces
    reanudar generaría trabajo nuevo **siempre**, que es justo lo que todo esto
    existe para evitar.
    """
    import hashlib

    semilla = f"{context_id}{intento}"
    return f"syn-{hashlib.sha256(semilla.encode()).hexdigest()[:24]}"
