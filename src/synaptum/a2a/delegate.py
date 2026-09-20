"""
SYN-44 · Un subagente que vive en otro sitio.

Cumple el mismo contrato que un ``Delegate`` local — ``name``, ``risk``,
``definition``, ``execute``— así que **el bucle no nota la diferencia**: el paso
sigue siendo durable, reanudar sigue sin repetirlo, el consumo sigue subiendo al
padre y el riesgo sigue viajando por la costura.

Lo único distinto es dónde ocurre el trabajo.

Reconciliar al reanudar
------------------------
El problema que decide el diseño: **A2A asigna el ``taskId`` en el servidor** y
el cliente no puede aportarlo. Tras una caída no sabemos a qué preguntar.

La salida no es pedirle a A2A que acepte nuestro id, es usar el que **sí**
ponemos nosotros. ``contextId`` es del cliente, y lo hacemos igual al
``run_id`` del sub-run — que ya era determinista. Entonces la reanudación deja
de ser una apuesta y pasa a ser una consulta:

    ListTasks(contextId) → vacío / una tarea / una terminal

Y una escalera explícita, porque no todo servidor implementa ``ListTasks``:

===========================  ===================================================
``tasks/list``               **Exacta.** Sabemos si hay tarea y cuál
``messageId`` determinista   **Depende del servidor.** «Puede» deduplicar
Ninguno                      **Ninguna.** ``UncertainEffect``, y decide el arnés
===========================  ===================================================

El ``messageId`` determinista se manda siempre: no cuesta nada, y donde el
servidor deduplica cierra el hueco antes de llegar al último peldaño.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ProviderError, ToolExecutionError, UncertainEffect
from ..core.types import Risk, ToolDefinition, Usage
from .client import A2AClient
from .types import ESPERANDO, Task, TaskState
from .wire import deterministic_message_id

__all__ = ["RemoteDelegate"]


@dataclass(frozen=True, slots=True)
class RemoteDelegate:
    """Un agente remoto, tal como lo ve quien delega.

    Args:
        name: con qué nombre lo ve el modelo.
        url: el agente, o **el proxy que lo gobierna**. En el camino gobernado
            es lo segundo: el framework apunta al endpoint del arnés y no cambia
            nada más.
        description: cuándo usarlo. Si falta se toma de su tarjeta.
        risk: **obligatorio, sin valor por defecto.** Ver abajo por qué no lo tiene.
        poll_every: segundos entre consultas mientras la tarea trabaja.
        timeout: tope total de espera.
    """

    name: str
    url: str
    risk: Risk
    description: str = ""
    poll_every: float = 1.0
    timeout: float = 600.0
    headers: Any = None
    _depth: int = 0

    # `risk` no tiene valor por defecto, y eso es deliberado.
    #
    # Un `AgentCard` declara `skills`, no riesgo: la especificación no tiene ese
    # campo. No hay forma de saber qué puede hacer un agente ajeno, y a
    # diferencia de una herramienta MCP **puede cambiar sin avisarnos**, porque
    # al otro lado hay un modelo decidiendo.
    #
    # La primera versión ponía `DESTRUCTIVE` por defecto. Lo corrigió quien
    # opera un arnés real, y tenía razón: un defecto conservador es **correcto y
    # silencioso** — nadie se entera nunca de que el riesgo no se declaró, y la
    # decisión queda tomada por un valor por defecto en vez de por una persona.
    #
    # Aquí sí hay alguien delante: quien construye el `RemoteDelegate`. Con el
    # `@tool` ese alguien es el autor de la función y por eso `READ` por defecto
    # es razonable. Con un agente ajeno, el autor no está — así que la decisión
    # es de quien lo conecta, y se le pide que la tome con su nombre.

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description or f"Delega una tarea al agente remoto '{self.name}'.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "brief": {
                        "type": "string",
                        "description": (
                            "La tarea para el agente remoto, completa y autónoma. "
                            "No ve esta conversación."
                        ),
                    }
                },
                "required": ["brief"],
            },
            risk=self.risk,
            idempotent=False,
        )

    # ── Ejecución ─────────────────────────────────────────────────────────────

    async def execute(self, brief: str, session: Any, run_id: str) -> tuple[Any, Usage]:
        """Manda el brief, espera, y devuelve el resultado.

        ``run_id`` es el del sub-run y **es también el ``contextId`` de A2A**.
        No es un apaño: A2A define ``contextId`` como la agrupación de una
        conversación, y una delegación es una conversación —de varios turnos si
        el remoto pide entrada o credenciales—. Un identificador, dos sistemas.
        """
        cliente = A2AClient(self.url, headers=self.headers)

        tarea = await self._reconciliar(cliente, run_id)
        if tarea is None:
            tarea = await cliente.send_message(
                brief,
                context_id=run_id,
                message_id=deterministic_message_id(run_id),
            )

        tarea = await self._esperar(cliente, tarea)
        return self._resultado(tarea), _consumo(tarea)

    async def _reconciliar(self, cliente: A2AClient, context_id: str) -> Task | None:
        """¿Hay ya una tarea para este contexto?

        Devuelve ``None`` cuando toca enviar. Si el servidor no implementa
        ``tasks/list``, se baja un peldaño en silencio **aquí** y se dice en el
        peldaño siguiente: el ``messageId`` determinista ya va en el envío.
        """
        try:
            tareas = await cliente.list_tasks(context_id=context_id)
        except ProviderError:
            return None          # sin `tasks/list`: se envía y se confía en el messageId

        if not tareas:
            return None
        if len(tareas) == 1:
            return tareas[0]

        # Varias tareas para un contexto que solo debería tener una. Puede ser
        # legítimo —un servidor que agrupa turnos— así que se toma la que sigue
        # viva; si hay dos vivas, no se elige por nosotros.
        vivas = [t for t in tareas if not t.terminal]
        if len(vivas) > 1:
            raise UncertainEffect(
                context_id,
                detail=(
                    f"el agente remoto tiene {len(vivas)} tareas sin terminar para este "
                    "contexto y no se puede saber cuál es la nuestra"
                ),
            )
        return vivas[0] if vivas else tareas[-1]

    async def _esperar(self, cliente: A2AClient, tarea: Task) -> Task:
        """Consulta hasta que termine, se detenga a esperar, o se agote el plazo.

        Cancelar es dejar de esperar: al cerrarse este iterador se cancela la
        tarea al otro lado. Igual que con un stream, la señal es irse — no un
        mensaje por el mismo canal que se está cerrando.
        """
        limite = asyncio.get_running_loop().time() + self.timeout
        try:
            while not tarea.terminal and not tarea.esperando:
                if asyncio.get_running_loop().time() > limite:
                    raise ToolExecutionError(
                        f"el agente '{self.name}' no terminó en {self.timeout:.0f}s "
                        f"(tarea {tarea.task_id}, estado {tarea.state.value})",
                        tool=self.name,
                        retryable=False,
                    )
                await asyncio.sleep(self.poll_every)
                tarea = await cliente.get_task(tarea.task_id)
            return tarea
        except asyncio.CancelledError:
            if tarea.task_id:
                await cliente.cancel_task(tarea.task_id)
            raise

    def _resultado(self, tarea: Task) -> str:
        """Traduce el desenlace a algo que el modelo que delegó pueda usar."""
        if tarea.state is TaskState.COMPLETED:
            return tarea.result

        if tarea.state in ESPERANDO:
            # El remoto se detuvo a esperar a una persona o a credenciales. No
            # es un fallo, y tampoco un resultado: es lo mismo que nuestro
            # `ApprovalStep`, del otro lado de la red.
            return (
                f"El agente '{self.name}' quedó esperando ({tarea.state.value}): "
                f"{tarea.message or 'sin detalle'}. Tarea {tarea.task_id}."
            )

        # `failed`, `rejected` o `canceled`: vuelve como evidencia al modelo,
        # que suele rectificar. Uno que no ve el error no puede corregirlo.
        return (
            f"El agente '{self.name}' no completó la tarea ({tarea.state.value}): "
            f"{tarea.message or 'sin detalle'}."
        )


def _consumo(tarea: Task) -> Usage:
    """Lo que el remoto dice haber gastado, si lo dice.

    A2A **no define un campo de consumo**. Hay servidores que lo ponen en
    `metadata`, así que se mira ahí — y si no está, queda **sin medir**.

    `Usage()` con todo a ``None`` es exactamente eso: «nadie lo midió». Poner
    cero diría «no costó nada», y un agente remoto siempre cuesta algo.
    """
    meta = tarea.raw.get("metadata") or {}
    uso = meta.get("usage") or {}
    if not isinstance(uso, dict) or not uso:
        return Usage()
    return Usage(
        input=uso.get("input") or uso.get("promptTokens"),
        output=uso.get("output") or uso.get("completionTokens"),
    )
