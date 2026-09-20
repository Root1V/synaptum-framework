"""Servidor A2A mínimo para los tests, con la forma del binding HTTP+JSON.

No es una implementación de la especificación: es lo justo para que el cliente
hable con algo que no sea un doble de sí mismo. Se le puede pedir que se
comporte mal —que no implemente `tasks/list`, que tarde, que falle— porque eso
es lo que hay que probar.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class Servidor:
    def __init__(self, *, con_list=True, estado_final="completed", turnos_trabajando=0):
        self.tareas: dict[str, dict] = {}
        self.enviados: list[dict] = []
        self.con_list = con_list
        self.estado_final = estado_final
        self.turnos_trabajando = turnos_trabajando
        self._n = 0
        sv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path.endswith("agent-card.json"):
                    return self._json({
                        "name": "analista", "description": "Analiza valores.",
                        "version": "1.0", "capabilities": {"streaming": False},
                        "skills": [{"id": "cotizar", "name": "Cotizar"}],
                    })
                self.send_error(404)

            def do_POST(self):
                largo = int(self.headers.get("content-length", 0))
                peticion = json.loads(self.rfile.read(largo) or b"{}")
                metodo, params = peticion.get("method"), peticion.get("params") or {}
                try:
                    resultado = getattr(sv, f"_{metodo.replace('/', '_')}")(params)
                except AttributeError:
                    return self._json({"jsonrpc": "2.0", "id": peticion.get("id"),
                                       "error": {"code": -32601, "message": "method not found"}})
                self._json({"jsonrpc": "2.0", "id": peticion.get("id"), "result": resultado})

            def _json(self, cuerpo):
                crudo = json.dumps(cuerpo).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(crudo)))
                self.end_headers()
                self.wfile.write(crudo)

        self._http = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self._http.server_port}"
        self._hilo = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._hilo.start()

    # ── Los métodos A2A ──────────────────────────────────────────────────────

    def _message_send(self, params):
        mensaje = params.get("message") or {}
        self.enviados.append(mensaje)
        self._n += 1
        tid = f"task-{self._n}"
        self.tareas[tid] = {
            "id": tid,
            "contextId": mensaje.get("contextId", ""),
            "status": {"state": "working"},
            "_restantes": self.turnos_trabajando,
            "artifacts": [],
        }
        return dict(self.tareas[tid])

    def _tasks_get(self, params):
        tarea = self.tareas[params["id"]]
        if tarea["_restantes"] > 0:
            tarea["_restantes"] -= 1
            return dict(tarea)
        tarea["status"] = {"state": self.estado_final,
                           "message": {"parts": [{"text": "ACME cotiza a 187,34 USD."}]}}
        if self.estado_final == "completed":
            tarea["artifacts"] = [{"artifactId": "a1", "name": "informe",
                                   "parts": [{"text": "187,34 USD"}]}]
        tarea["metadata"] = {"usage": {"input": 120, "output": 40}}
        return dict(tarea)

    def _tasks_cancel(self, params):
        tarea = self.tareas[params["id"]]
        tarea["status"] = {"state": "canceled"}
        return dict(tarea)

    def _tasks_list(self, params):
        if not self.con_list:
            raise AttributeError("sin tasks/list")
        ctx = params.get("contextId")
        return {"tasks": [t for t in self.tareas.values() if t["contextId"] == ctx]}

    def cerrar(self):
        self._http.shutdown()
        self._http.server_close()
