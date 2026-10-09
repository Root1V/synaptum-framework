"""Un diario HTTP mínimo con la forma del contrato `costura-durabilidad`.

No es el arnés: es lo justo para que el cliente hable con algo que no sea un
doble de sí mismo. Lo que importa de él es que la clave primaria incluya el
**sub-run**, porque es ahí donde se decide si dos sub-runs de un mismo padre se
pisan — y ese es el caso que el cliente existe para no romper.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


class Diario:
    def __init__(self, *, exige_token: str | None = None) -> None:
        #: (run_id, sub_run_id, step_id, phase) -> fila.  La clave es el contrato.
        self.filas: dict[tuple[str, str, str, str], dict] = {}
        self.orden: list[dict] = []
        self.peticiones: list[dict] = []
        sv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def _auth(self) -> bool:
                if exige_token is None:
                    return True
                return self.headers.get("authorization") == f"Bearer {exige_token}"

            def _run_id(self) -> str | None:
                partes = urlparse(self.path).path.strip("/").split("/")
                # /runs/{run_id}/checkpoints — un solo segmento, como el real.
                if len(partes) != 3 or partes[0] != "runs" or partes[2] != "checkpoints":
                    return None
                return partes[1]

            def do_POST(self):
                if not self._auth():
                    return self.send_error(401)
                run = self._run_id()
                if run is None:
                    return self.send_error(404)
                largo = int(self.headers.get("content-length", 0))
                cuerpo = json.loads(self.rfile.read(largo) or b"{}")
                sv.peticiones.append({"run_id": run, **cuerpo})

                clave = (run, cuerpo.get("sub_run_id", ""), cuerpo["step_id"], cuerpo["phase"])
                if clave in sv.filas:
                    ya = sv.filas[clave]
                    return self._json({
                        "seq": ya["seq"],
                        "duplicate": True,
                        "payload_diverged": ya["payload"] != cuerpo.get("payload"),
                    })
                fila = {
                    "run_id": run,
                    "sub_run_id": cuerpo.get("sub_run_id", ""),
                    "step_id": cuerpo["step_id"],
                    "phase": cuerpo["phase"],
                    "payload": cuerpo.get("payload"),
                    "seq": len([f for f in sv.orden if f["run_id"] == run]),
                    "recorded_at": "2026-10-08T00:00:00Z",
                }
                sv.filas[clave] = fila
                sv.orden.append(fila)
                self._json({"seq": fila["seq"], "duplicate": False, "payload_diverged": False})

            def do_GET(self):
                if not self._auth():
                    return self.send_error(401)
                run = self._run_id()
                if run is None:
                    return self.send_error(404)
                sub = parse_qs(urlparse(self.path).query).get("sub_run_id", [""])[0]
                filas = [
                    f for f in sv.orden
                    if f["run_id"] == run and f["sub_run_id"] == sub
                ]
                self._json({"run_id": run, "next_seq": len(filas), "records": filas})

            def _json(self, cuerpo):
                crudo = json.dumps(cuerpo).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(crudo)))
                self.end_headers()
                self.wfile.write(crudo)

        self._http = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self._http.server_port}"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def cerrar(self) -> None:
        self._http.shutdown()
        self._http.server_close()
