"""El otro lado del cable: un agente A2A mínimo.

En producción esto es **otro despliegue** —otro contenedor, otro equipo, quizá
otro lenguaje— y por eso no importa qué framework use por dentro. Lo único que
compartimos con él es el protocolo.

Aquí está en el mismo proceso por comodidad, y solo implementa lo que el cliente
necesita del binding HTTP+JSON de A2A:

    GET  /.well-known/agent-card.json   quién es y qué sabe hacer
    POST message/send                   encárgate de esto
    POST tasks/get                      ¿cómo va?
    POST tasks/list                     ¿tenías ya algo mío?   ← la importante
    POST tasks/cancel                   déjalo

`tasks/list` es la que convierte reanudar en una **consulta**. El `taskId` lo
pone el servidor y se pierde en una caída; el `contextId` lo ponemos nosotros,
así que por ahí se reencuentra lo que ya se encargó.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

INFORME = (
    "Doblaje es-419 del episodio 12: 47 intervenciones, 3 con desincronía "
    "> 120ms (min 04:12, 09:38, 17:55). El resto dentro de tolerancia."
)


class AgenteRemoto:
    """Un agente A2A que tarda un par de consultas en terminar.

    `turnos_trabajando` existe para que el ejemplo no mienta: un agente remoto
    **no** contesta en el mismo instante, y el cliente tiene que esperar.
    """

    def __init__(self, *, turnos_trabajando: int = 2) -> None:
        self.tareas: dict[str, dict] = {}
        self.enviados: list[dict] = []
        self.turnos_trabajando = turnos_trabajando
        self._n = 0
        servidor = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):     # sin ruido en la salida del ejemplo
                pass

            def do_GET(self):
                if self.path.endswith("agent-card.json"):
                    return self._json({
                        "name": "revisor-de-doblaje",
                        "description": "Revisa sincronía y calidad de una pista doblada.",
                        "version": "1.0",
                        "capabilities": {"streaming": False},
                        # Una tarjeta A2A declara **habilidades**, no riesgo: la
                        # especificación no tiene ese campo. Por eso quien lo
                        # conecta tiene que decirlo con su nombre.
                        "skills": [{"id": "revisar", "name": "Revisar doblaje"}],
                    })
                self.send_error(404)

            def do_POST(self):
                largo = int(self.headers.get("content-length", 0))
                peticion = json.loads(self.rfile.read(largo) or b"{}")
                metodo = (peticion.get("method") or "").replace("/", "_")
                try:
                    resultado = getattr(servidor, f"_{metodo}")(peticion.get("params") or {})
                except AttributeError:
                    return self._json({
                        "jsonrpc": "2.0", "id": peticion.get("id"),
                        "error": {"code": -32601, "message": "method not found"},
                    })
                self._json({"jsonrpc": "2.0", "id": peticion.get("id"), "result": resultado})

            def _json(self, cuerpo):
                crudo = json.dumps(cuerpo).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(crudo)))
                self.end_headers()
                self.wfile.write(crudo)

        self._http = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._http.server_port}"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    # ── Los métodos del protocolo ────────────────────────────────────────────

    def _message_send(self, params):
        mensaje = params.get("message") or {}
        self.enviados.append(mensaje)
        self._n += 1
        tid = f"task-{self._n}"
        self.tareas[tid] = {
            "id": tid,
            "contextId": mensaje.get("contextId", ""),
            "status": {"state": "working"},
            "artifacts": [],
            "_restantes": self.turnos_trabajando,
        }
        return dict(self.tareas[tid])

    def _tasks_get(self, params):
        tarea = self.tareas[params["id"]]
        if tarea["_restantes"] > 0:
            tarea["_restantes"] -= 1
            return dict(tarea)
        tarea["status"] = {"state": "completed", "message": {"parts": [{"text": INFORME}]}}
        tarea["artifacts"] = [
            {"artifactId": "a1", "name": "informe", "parts": [{"text": INFORME}]}
        ]
        # A2A **no define un campo de consumo**. Hay servidores que lo ponen en
        # `metadata`, así que se mira ahí; si no está, queda sin medir.
        tarea["metadata"] = {"usage": {"input": 2140, "output": 96}}
        return dict(tarea)

    def _tasks_list(self, params):
        ctx = params.get("contextId")
        return {"tasks": [t for t in self.tareas.values() if t["contextId"] == ctx]}

    def _tasks_cancel(self, params):
        self.tareas[params["id"]]["status"] = {"state": "canceled"}
        return dict(self.tareas[params["id"]])

    def cerrar(self) -> None:
        self._http.shutdown()
        self._http.server_close()
