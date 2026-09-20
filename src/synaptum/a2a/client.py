"""
SYN-44 · Cliente A2A sobre el binding HTTP+JSON.

Solo biblioteca estándar. El SDK oficial arrastra once dependencias —protobuf,
`google-auth`, `requests`— y la superficie que necesitamos son cuatro métodos
sobre JSON-RPC. En un framework cuya identidad es no tener dependencias, eso no
sale a cuenta.

La normalización vive aparte (`wire`), así que lo de aquí es transporte y nada
más — y lo de allá se puede probar contra un fichero.

**En el camino gobernado esto no se usa.** La llamada sale por el proxy del
arnés, que es quien tiene la credencial y quien puede denegar. Este cliente es
para modo autónomo, igual que ``HttpModel``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from ..core.errors import NetworkError, ProviderError, RequestTimeoutError
from .types import AgentCard, Task
from .wire import agent_card_from_wire, message_to_wire, task_from_wire

__all__ = ["A2AClient"]


class A2AClient:
    """Los cuatro métodos que hacen falta para delegar.

    Args:
        base_url: raíz del agente remoto, o del proxy que lo gobierna.
        headers: cabeceras extra — lo que pida el `securitySchemes` de su tarjeta.
        timeout: segundos por lectura. Un agente puede tardar mucho en el primer
            byte y eso no es un fallo.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = dict(headers or {})

    # ── Descubrimiento ────────────────────────────────────────────────────────

    async def agent_card(self) -> AgentCard:
        """Lee `/.well-known/agent-card.json`."""
        cuerpo = await asyncio.to_thread(
            self._get, f"{self.base_url}/.well-known/agent-card.json"
        )
        return agent_card_from_wire(cuerpo)

    # ── Los cuatro métodos ────────────────────────────────────────────────────

    async def send_message(
        self, brief: str, *, context_id: str, message_id: str, task_id: str | None = None
    ) -> Task:
        cuerpo = message_to_wire(
            brief, context_id=context_id, message_id=message_id, task_id=task_id
        )
        return task_from_wire(await self._rpc("message/send", cuerpo))

    async def get_task(self, task_id: str) -> Task:
        return task_from_wire(await self._rpc("tasks/get", {"id": task_id}))

    async def list_tasks(self, *, context_id: str) -> list[Task]:
        """Las tareas de un contexto.

        **Es la pieza que convierte la reanudación en una consulta.** El
        `taskId` lo pone el servidor y no lo tenemos tras una caída; el
        `contextId` lo ponemos nosotros, así que por ahí se reencuentra.

        Un servidor que no implemente esto devuelve un error de método, y
        entonces se cae al peldaño siguiente de la escalera — que garantiza
        menos, y lo dice.
        """
        respuesta = await self._rpc("tasks/list", {"contextId": context_id})
        tareas = respuesta.get("tasks") or respuesta.get("result") or []
        if isinstance(tareas, Mapping):
            tareas = tareas.get("tasks") or []
        return [task_from_wire({"result": t}) for t in tareas]

    async def cancel_task(self, task_id: str) -> Task:
        """Cancelar es lo que hacemos al cerrar el iterador, también por red."""
        return task_from_wire(await self._rpc("tasks/cancel", {"id": task_id}))

    # ── Transporte ────────────────────────────────────────────────────────────

    async def _rpc(self, metodo: str, parametros: Mapping[str, Any]) -> dict[str, Any]:
        cuerpo = {
            "jsonrpc": "2.0",
            "id": parametros.get("messageId") or metodo,
            "method": metodo,
            "params": dict(parametros),
        }
        respuesta = await asyncio.to_thread(self._post, self.base_url, cuerpo)

        if "error" in respuesta:
            error = respuesta["error"] or {}
            # Un método que el servidor no implementa no es un fallo de red: es
            # una capacidad que no tiene, y quien llama decide qué hacer.
            raise ProviderError(
                f"{metodo}: {error.get('message', error)}",
                status=None if error.get("code") != -32601 else 501,
                provider="a2a",
                retryable=False,
            )
        return respuesta.get("result") or respuesta

    def _peticion(self, url: str, cuerpo: Any | None) -> urllib.request.Request:
        cabeceras = {"content-type": "application/json", **self._headers}
        return urllib.request.Request(
            url,
            data=json.dumps(cuerpo).encode() if cuerpo is not None else None,
            headers=cabeceras,
            method="POST" if cuerpo is not None else "GET",
        )

    def _abrir(self, url: str, cuerpo: Any | None) -> str:
        try:
            with urllib.request.urlopen(
                self._peticion(url, cuerpo), timeout=self.timeout
            ) as respuesta:
                return respuesta.read().decode()
        except urllib.error.HTTPError as fallo:
            detalle = fallo.read().decode(errors="replace")[:400]
            raise ProviderError(
                f"{fallo.code} {fallo.reason}: {detalle}", status=fallo.code, provider="a2a"
            ) from fallo
        except TimeoutError as fallo:
            raise RequestTimeoutError(f"el agente no respondió en {self.timeout}s") from fallo
        except urllib.error.URLError as fallo:
            if isinstance(fallo.reason, TimeoutError):
                raise RequestTimeoutError(f"sin respuesta en {self.timeout}s") from fallo
            raise NetworkError(f"no se pudo llegar a {url}: {fallo.reason}") from fallo

    def _post(self, url: str, cuerpo: Any) -> dict[str, Any]:
        return json.loads(self._abrir(url, cuerpo))

    def _get(self, url: str) -> dict[str, Any]:
        return json.loads(self._abrir(url, None))
