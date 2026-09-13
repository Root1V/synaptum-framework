"""
SYN-67 · Transporte HTTP de desarrollo.

Los adaptadores de proveedor son **puros**: traducen entre el vocabulario
unificado y el cable, y no abren conexiones. Es lo que permite ejercitar la
normalización contra un fichero, y es la razón de que el corpus dorado exista.

Pero deja un hueco: sin transporte, no había forma de ir de ``pip install
synaptum`` a un agente hablando con un modelo de verdad. El adaptador sabía
*qué* mandar y nadie sabía *por dónde*. Esto lo cierra con la biblioteca
estándar y nada más — ``urllib`` en un hilo, porque el núcleo no tiene
dependencias y no las va a tener por esto.

**Esto es transporte de desarrollo, y conviene decirlo entero.** En el camino
gobernado la llamada al modelo **no ocurre en este proceso**: sale por la
costura hacia el gateway, que es quien tiene las credenciales y quien puede
denegar. Un transporte dentro del proceso tiene la clave en su memoria y llama
a quien le digan. Sirve para un cuaderno, un script o un ejemplo; no para
producción.

Uso::

    modelo = HttpModel("http://localhost:8080/v1", api_key=os.environ["API_KEY"])

    gateway = LocalGateway(model=modelo, stream=modelo.stream, tools=[leer])
    async for step in agent.run("...", session=Session("run-1", gateway)):
        ...

Cancelar es dejar de iterar, como en todo lo demás: al cerrarse el generador se
cierra el cuerpo de la respuesta, que es lo que de verdad para la generación
arriba. No hay evento de cancelación, y no lo habrá — un canal que se está
cerrando no es sitio para mandar el aviso de que se cierra.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, AsyncIterator, Iterator, Mapping

from ..core.errors import NetworkError, ProviderError, RequestTimeoutError
from ..core.types import Request, Response, StreamEvent
from ..providers import base as _providers

__all__ = ["HttpModel"]

_FIN = object()


class HttpModel:
    """Llama a un endpoint HTTP y devuelve respuestas ya normalizadas.

    Args:
        base_url: raíz del servicio, por ejemplo ``http://localhost:8080/v1``.
        api_key: si falta, no se manda cabecera de autorización — hay
            despliegues locales que no la piden, y mandar ``Bearer None`` es
            peor que no mandar nada.
        provider: nombre del adaptador. Por defecto se toma del prefijo de
            ``Request.model`` (``"openai-compatible:qwen3-0.6b"``), que es lo
            que el agente ya escribe.
        path: ruta del endpoint de chat, relativa a ``base_url``.
        headers: cabeceras extra. Se mandan tal cual.
        timeout: segundos de espera **por lectura**, no por respuesta completa.
            Un modelo de razonamiento puede tardar mucho en emitir el primer
            token y eso no es un fallo.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        provider: str | None = None,
        path: str = "/chat/completions",
        headers: Mapping[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.path = path if path.startswith("/") else f"/{path}"
        self.api_key = api_key
        self.provider = provider
        self.timeout = timeout
        self._extra = dict(headers or {})

    # ── Respuesta completa ────────────────────────────────────────────────────

    async def __call__(self, request: Request) -> Response:
        adapter, body = self._prepare(request, stream=False)
        raw = await asyncio.to_thread(self._post, body)
        return adapter.from_wire(json.loads(raw))

    # ── Stream ────────────────────────────────────────────────────────────────
    #
    # `stream_from_wire` es síncrono a propósito: la normalización no sabe de
    # concurrencia.  Así que el hilo hace las dos cosas —leer del socket y
    # normalizar— y por la cola solo pasan eventos ya tipados.  El bucle de
    # eventos nunca se bloquea leyendo.

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        adapter, body = self._prepare(request, stream=True)

        buzon: queue.Queue[Any] = queue.Queue(maxsize=64)
        parar = threading.Event()
        abierto: list[Any] = []      # la respuesta viva, para poder cerrarla desde fuera

        def trabajar() -> None:
            try:
                with self._open(body) as respuesta:
                    abierto.append(respuesta)
                    for event in adapter.stream_from_wire(_chunks(respuesta, parar)):
                        if parar.is_set():
                            break
                        buzon.put(event)
            except BaseException as fallo:  # se re-lanza en el consumidor
                if not parar.is_set():      # tras cancelar, el fallo es el cierre
                    buzon.put(fallo)
            finally:
                abierto.clear()
                buzon.put(_FIN)

        hilo = threading.Thread(target=trabajar, name="synaptum-stream", daemon=True)
        hilo.start()
        try:
            while True:
                item = await asyncio.to_thread(buzon.get)
                if item is _FIN:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Dejar de iterar cierra el cuerpo de la respuesta, y eso es lo que
            # para la generación arriba.  Sin esto el modelo seguiría generando
            # —y facturando— para nadie.
            #
            # Lo que de verdad para arriba es **cerrar el cuerpo**; la bandera
            # sola no basta, y creerlo costó un test que pasaba por el motivo
            # equivocado.  El hilo puede estar bloqueado en `put` con la cola
            # llena, sin llegar nunca a mirar la bandera: entonces ni lee ni
            # cierra, y el socket queda colgando con la generación viva al otro
            # lado.  Así que se cierra la respuesta —lo que revienta la lectura
            # en el hilo— y se vacía la cola para desbloquearlo.
            #
            # La bandera se queda como camino rápido: evita reportar como fallo
            # la excepción que provoca el propio cierre.  Quitarla no cambia
            # ninguna conducta observable, y eso está comprobado — no se
            # pretende que sea lo que cancela.
            parar.set()
            for respuesta in list(abierto):
                try:
                    respuesta.close()
                except Exception:       # ya cerrado, o cerrándose: da igual
                    pass
            while True:
                try:
                    if buzon.get_nowait() is _FIN:
                        break
                except queue.Empty:
                    break

    # ── Piezas ────────────────────────────────────────────────────────────────

    def _prepare(self, request: Request, *, stream: bool) -> tuple[Any, dict[str, Any]]:
        """Resuelve el adaptador y arma el cuerpo.

        El nombre del modelo viaja como ``"proveedor:modelo"`` y por el cable
        tiene que ir solo el modelo: el prefijo es nuestro, no suyo.
        """
        if self.provider is not None:
            adapter = _providers.get(self.provider)
            nombre = request.model.partition(":")[2] or request.model
        else:
            adapter, nombre = _providers.resolve(request.model)

        from dataclasses import replace

        body = dict(adapter.to_wire(replace(request, model=nombre)))
        if stream:
            body["stream"] = True
            # Se pide aunque haya despliegues que lo ignoren: donde se atiende,
            # convierte un consumo derivado en uno medido.
            body.setdefault("stream_options", {"include_usage": True})
        return adapter, body

    def _request(self, body: Mapping[str, Any]) -> urllib.request.Request:
        cabeceras = {"content-type": "application/json", **self._extra}
        if self.api_key:
            cabeceras.setdefault("authorization", f"Bearer {self.api_key}")
        return urllib.request.Request(
            f"{self.base_url}{self.path}",
            data=json.dumps(body).encode(),
            headers=cabeceras,
            method="POST",
        )

    def _open(self, body: Mapping[str, Any]):
        try:
            return urllib.request.urlopen(self._request(body), timeout=self.timeout)
        except urllib.error.HTTPError as fallo:
            detalle = fallo.read().decode(errors="replace")[:500]
            # El estado decide si se reintenta, y lo decide el tipo del error:
            # un 4xx repetido cuesta lo mismo y da lo mismo.
            raise ProviderError(
                f"{fallo.code} {fallo.reason}: {detalle}",
                status=fallo.code,
                provider=self.provider or "",
            ) from fallo
        except TimeoutError as fallo:
            raise RequestTimeoutError(f"sin respuesta en {self.timeout}s") from fallo
        except urllib.error.URLError as fallo:
            if isinstance(fallo.reason, TimeoutError):
                raise RequestTimeoutError(f"sin respuesta en {self.timeout}s") from fallo
            raise NetworkError(f"no se pudo llegar a {self.base_url}: {fallo.reason}") from fallo

    def _post(self, body: Mapping[str, Any]) -> str:
        with self._open(body) as respuesta:
            return respuesta.read().decode()


def _chunks(respuesta, parar: threading.Event) -> Iterator[dict[str, Any]]:
    """Lee un cuerpo SSE línea a línea y para en el primer centinela.

    Parar en el primero no es un detalle: hasta el manifest v8 el gateway de
    Prometheus mandaba dos, y todo lo que registraba la petición vivía pasado
    ese punto — así que el cliente correcto era justo el que no se facturaba.
    """
    for linea in respuesta:
        if parar.is_set():
            break
        texto = linea.decode(errors="replace").strip()
        if not texto.startswith("data:"):
            continue
        carga = texto[5:].strip()
        if carga == "[DONE]":
            break
        if carga:
            yield json.loads(carga)
