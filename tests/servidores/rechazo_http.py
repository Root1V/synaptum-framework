"""Un servidor que contesta **una grabación**, tal cual.

No imita a un arnés: reproduce el cuerpo y el estado que uno real sirvió, byte a
byte, para que el cliente se escriba contra la forma que existe y no contra la
que supondríamos. La grabación la produce y la verifica un test del otro lado
contra Postgres real, así que no puede quedarse rancia sin que allí se note.

Lo único que este servidor decide es **cuándo** contestar qué: las grabaciones
de gobierno traen varios intercambios por superficie, y un cliente que lea uno e
implemente contra el otro es el fallo que la grabación existe para evitar.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class Grabado:
    """Sirve el intercambio cuyo `path` coincide con el pedido."""

    def __init__(self, intercambios: list[dict]) -> None:
        self.intercambios = intercambios
        self.peticiones: list[dict] = []
        self._servidos: set[int] = set()
        sv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_POST(self):
                largo = int(self.headers.get("content-length", 0))
                cuerpo = json.loads(self.rfile.read(largo) or b"{}")
                sv.peticiones.append({
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": cuerpo,
                })

                self._servir("POST")

            def do_GET(self):
                sv.peticiones.append({"path": self.path, "headers": {}, "body": None})
                self._servir("GET")

            def _servir(self, metodo: str) -> None:
                # Se consume **en orden**: varios intercambios comparten método
                # y ruta —tres escrituras al mismo `/checkpoints`— y lo que los
                # distingue es la secuencia, que es justamente lo que la
                # grabación afirma. Servir siempre el primero haría pasar un
                # cliente que no conserva el orden.
                for i, intercambio in enumerate(sv.intercambios):
                    if i in sv._servidos:
                        continue
                    if intercambio.get("method", "POST") != metodo:
                        continue
                    if intercambio["path"] == self.path:
                        sv._servidos.add(i)
                        return self._contestar(
                            intercambio["status"], intercambio["response"]
                        )
                self._contestar(404, {"error": f"sin grabación para {metodo} {self.path}"})

            def _contestar(self, status: int, cuerpo: dict) -> None:
                crudo = json.dumps(cuerpo).encode()
                self.send_response(status)
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
