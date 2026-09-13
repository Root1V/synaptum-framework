"""SYN-67 · Transporte HTTP de desarrollo.

Se prueba contra un servidor de verdad en `localhost` —de la biblioteca
estándar, levantado por el test— y no contra un `urlopen` parcheado. Un mock de
`urlopen` comprobaría que llamo a `urlopen`, que es lo único que no hace falta
comprobar; lo que puede fallar es el cable: cabeceras, el prefijo del proveedor
colándose en el cuerpo, un SSE leído entero antes de emitir nada, o un stream
que sigue generando después de que el consumidor se va.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from synaptum import HttpModel, NetworkError, ProviderError, Request


class _Servidor:
    """Servidor mínimo que guarda lo que recibió y sirve lo que se le diga."""

    def __init__(self, responder) -> None:
        self.peticiones: list[dict] = []
        self.enviados = 0          # fragmentos que llegaron a salir por el cable
        servidor = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):  # sin ruido en la salida del test
                pass

            def do_POST(self):
                largo = int(self.headers.get("content-length", 0))
                cuerpo = json.loads(self.rfile.read(largo) or b"{}")
                servidor.peticiones.append({
                    "body": cuerpo,
                    # Las cabeceras HTTP no distinguen mayúsculas y urllib las
                    # capitaliza: comparar por la forma exacta probaría urllib.
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "path": self.path,
                })
                responder(self, servidor)

        self._http = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._http.server_port}/v1"
        self._hilo = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._hilo.start()

    def cerrar(self) -> None:
        self._http.shutdown()
        self._http.server_close()


def _json(handler, payload: dict, status: int = 200) -> None:
    crudo = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(crudo)))
    handler.end_headers()
    handler.wfile.write(crudo)


_RESPUESTA = {
    "model": "qwen3-0.6b",
    "choices": [{"message": {"role": "assistant", "content": "hola"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2,
              "prompt_tokens_details": {"cached_tokens": 9}},
}


@pytest.fixture
def servidor():
    creados: list[_Servidor] = []

    def montar(responder):
        s = _Servidor(responder)
        creados.append(s)
        return s

    yield montar
    for s in creados:
        s.cerrar()


def test_a_full_response_comes_back_normalized(servidor):
    s = servidor(lambda h, _: _json(h, _RESPUESTA))
    modelo = HttpModel(s.url, api_key="k")

    respuesta = asyncio.run(modelo(Request(model="openai-compatible:qwen3-0.6b")))

    assert respuesta.text == "hola"
    assert respuesta.usage.input == 10 and respuesta.usage.cache_read == 9
    assert respuesta.usage.estimated is False


def test_the_provider_prefix_never_reaches_the_wire(servidor):
    """``openai-compatible:`` es nuestro, no suyo.

    Mandarlo entero hace que el proveedor responda «no conozco ese modelo» con
    un 404 que parece un problema de despliegue y es de formato.
    """
    s = servidor(lambda h, _: _json(h, _RESPUESTA))
    modelo = HttpModel(s.url, api_key="k")

    asyncio.run(modelo(Request(model="openai-compatible:qwen3-0.6b")))

    assert s.peticiones[0]["body"]["model"] == "qwen3-0.6b"
    assert s.peticiones[0]["headers"]["authorization"] == "Bearer k"
    assert s.peticiones[0]["path"] == "/v1/chat/completions"


def test_without_a_key_no_authorization_header_is_sent(servidor):
    """Hay despliegues locales que no la piden, y ``Bearer None`` es peor que nada."""
    s = servidor(lambda h, _: _json(h, _RESPUESTA))

    asyncio.run(HttpModel(s.url)(Request(model="openai-compatible:m")))

    assert "authorization" not in s.peticiones[0]["headers"]


def test_a_4xx_is_not_retryable_and_a_5xx_is(servidor):
    """La clasificación la decide el tipo del error, no quien lo recibe."""
    for status, retryable in ((422, False), (503, True)):
        s = servidor(lambda h, _, c=status: _json(h, {"error": "no"}, status=c))
        with pytest.raises(ProviderError) as fallo:
            asyncio.run(HttpModel(s.url)(Request(model="openai-compatible:m")))
        assert fallo.value.status == status
        assert fallo.value.retryable is retryable


def test_an_unreachable_host_is_a_network_error():
    modelo = HttpModel("http://127.0.0.1:1/v1", timeout=2.0)
    with pytest.raises(NetworkError):
        asyncio.run(modelo(Request(model="openai-compatible:m")))


# ── Stream ────────────────────────────────────────────────────────────────────

def _sse(handler, servidor, fragmentos: list[str], *, lento: bool = False) -> None:
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.end_headers()
    for fragmento in fragmentos:
        try:
            handler.wfile.write(f"data: {fragmento}\n\n".encode())
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return          # el cliente se fue: es lo que el test quiere ver
        servidor.enviados += 1
        if lento:
            threading.Event().wait(0.05)


_DELTAS = [
    json.dumps({"choices": [{"delta": {"content": t}}]}) for t in ("ho", "la")
] + [
    json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
    "[DONE]",
]


def test_a_stream_arrives_as_the_unified_cycle(servidor):
    s = servidor(lambda h, sv: _sse(h, sv, _DELTAS))
    modelo = HttpModel(s.url)

    async def ir():
        return [e async for e in modelo.stream(Request(model="openai-compatible:m"))]

    eventos = asyncio.run(ir())

    assert [e.kind for e in eventos] == [
        "stream_start", "text_start", "text_delta", "text_delta", "text_end", "finish"
    ]
    assert eventos[-1].response.text == "hola"
    assert eventos[-1].response.usage.input == 3
    assert s.peticiones[0]["body"]["stream"] is True


def test_events_are_emitted_while_the_body_is_still_open(servidor):
    """Un stream que solo emite al terminar no es un stream.

    Es el fallo que no se ve en un fixture: leer el cuerpo entero y luego
    normalizarlo da los mismos eventos y ninguna de las propiedades.
    """
    s = servidor(lambda h, sv: _sse(h, sv, _DELTAS, lento=True))
    modelo = HttpModel(s.url)

    async def ir():
        async for evento in modelo.stream(Request(model="openai-compatible:m")):
            if evento.kind == "text_delta":
                return s.enviados          # cuántos había mandado el servidor
        return -1

    enviados_al_primer_delta = asyncio.run(ir())
    assert 0 < enviados_al_primer_delta < len(_DELTAS), (
        "el primer delta llegó cuando el cuerpo ya estaba entero: no hay streaming"
    )


def test_leaving_the_iterator_stops_the_upstream(servidor):
    """Dejar de iterar **es** la señal de cancelación.

    Se mide lo único que importa: que el servidor **deje de mandar**. Y se mide
    así porque la primera versión de este test comparaba «mandó menos de los
    que había», que pasaba igual con la cancelación arrancada — el hilo se
    bloqueaba con la cola llena y el servidor se paraba por contrapresión, no
    por el cierre. Un test que pasa por el motivo equivocado es peor que no
    tenerlo.

    Aquí la cola es mayor que el número de fragmentos, así que la
    contrapresión no puede explicar nada: si tras irse el consumidor el
    contador sigue subiendo, el cuerpo no se cerró.
    """
    muchos = [json.dumps({"choices": [{"delta": {"content": str(i)}}]}) for i in range(30)]
    s = servidor(lambda h, sv: _sse(h, sv, muchos, lento=True))
    modelo = HttpModel(s.url)

    async def ir():
        flujo = modelo.stream(Request(model="openai-compatible:m"))
        async for evento in flujo:
            if evento.kind == "text_delta":
                break
        await flujo.aclose()

    asyncio.run(ir())
    al_irse = s.enviados
    threading.Event().wait(0.6)     # ~12 fragmentos más, si siguiera vivo

    assert s.enviados <= al_irse + 1, (
        f"el servidor mandó {s.enviados - al_irse} fragmentos más después de que "
        "el cliente se fuera: irse del iterador no cerró el cuerpo"
    )
    assert al_irse < len(muchos), "el servidor terminó antes de que se pudiera cancelar"
