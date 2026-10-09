"""
`Checkpointer` contra el diario de un arnés, por HTTP.

El journal deja de vivir en este proceso y pasa a vivir donde el arnés lo
gobierna: presupuestos, atribución y durabilidad compartidas. El bucle no se
entera — la costura es un `typing.Protocol` y esto es otro relleno.

Dos cosas que el endpoint decidió y que esta clase tiene que respetar:

**El `run_id` de la ruta es el run padre, un solo segmento.** Un sub-run tiene
identidad ``{run}/{step}``, y esa barra **no viaja en la ruta**: ni cruda —el
enrutador casa un segmento y daría 404— ni escapada como ``%2F``, que funciona
hoy y es lo que normalizan o rechazan los proxies. O sea, la peor forma de
fallo disponible: pasa en tu test y falla detrás de un gateway.

Así que la identidad se parte aquí: el padre va en la ruta y el resto viaja en
el cuerpo como ``sub_run_id``. Es **una ruta, no un paso**: la delegación anida,
y un nieto es ``000001-delegate/000002-delegate``. Para el otro lado es opaca.

**Y el sub-run está en la clave primaria del almacén**, que es lo que impide que
dos sub-runs del mismo padre se pisen: el primer paso de cualquier run se llama
``000000-model``, así que sin esa columna el segundo sub-run entraría como
duplicado — y un duplicado es un no-op silencioso, no un error.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

from ..core.codec import decode_event
from ..core.errors import NetworkError, ProviderError, RequestTimeoutError
from ..core.events import StepEvent
from ..core.protocols import AppendResult, RunState
from ..core.types import dumps

__all__ = ["HttpCheckpointer"]


class HttpCheckpointer:
    """Habla con `POST/GET /runs/{run_id}/checkpoints`.

    Args:
        base_url: raíz del servicio que **posee el diario**.  Ojo: no es
            necesariamente el mismo que arranca los runs — en el despliegue
            donde se midió, `POST /runs` y `POST /runs/{id}/checkpoints` son el
            mismo prefijo en dos servicios distintos, así que un cliente con un
            solo `base_url` recibe 404 en uno de los dos.
        token: bearer.  El diario está detrás de autenticación como el resto.
        timeout: segundos por petición.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._token = token

    # ── Costura ───────────────────────────────────────────────────────────────

    async def append(self, run_id: str, event: StepEvent) -> AppendResult:
        padre, sub = partir(run_id)
        cuerpo: dict[str, Any] = {
            "step_id": event.step_id,
            "phase": event.phase.value,
            "payload": json.loads(dumps(event)),
        }
        if sub:
            # Se **omite** cuando no hay sub-run en vez de mandar cadena vacía:
            # el otro lado distingue «este cliente no sabe de sub-runs» de «este
            # cliente dice que no hay», y la columna ya tiene su valor por
            # defecto para el primero.
            cuerpo["sub_run_id"] = sub

        respuesta = await asyncio.to_thread(
            self._pedir, f"/runs/{padre}/checkpoints", cuerpo
        )
        return AppendResult(
            seq=int(respuesta["seq"]),
            duplicate=bool(respuesta.get("duplicate", False)),
            payload_diverged=bool(respuesta.get("payload_diverged", False)),
        )

    async def load(self, run_id: str) -> RunState:
        padre, sub = partir(run_id)
        ruta = f"/runs/{padre}/checkpoints"
        if sub:
            ruta += f"?sub_run_id={urllib.parse.quote(sub, safe='')}"
        respuesta = await asyncio.to_thread(self._pedir, ruta, None)

        eventos = tuple(
            decode_event(fila["payload"])
            for fila in respuesta.get("records", ())
            if isinstance(fila.get("payload"), Mapping)
        )
        return RunState(run_id=run_id, events=eventos)

    # ── Transporte ────────────────────────────────────────────────────────────

    def _pedir(self, ruta: str, cuerpo: Any | None) -> dict[str, Any]:
        cabeceras = {"content-type": "application/json"}
        if self._token:
            cabeceras["authorization"] = f"Bearer {self._token}"

        peticion = urllib.request.Request(
            f"{self.base_url}{ruta}",
            data=dumps(cuerpo).encode() if cuerpo is not None else None,
            headers=cabeceras,
            method="POST" if cuerpo is not None else "GET",
        )
        try:
            with urllib.request.urlopen(peticion, timeout=self.timeout) as respuesta:
                return json.loads(respuesta.read().decode() or "{}")
        except urllib.error.HTTPError as fallo:
            detalle = fallo.read().decode(errors="replace")[:400]
            if fallo.code == 404 and ruta.startswith("/runs/"):
                # El 404 más probable aquí no es «ese run no existe»: es apuntar
                # al servicio que arranca runs en vez de al que posee el diario.
                # Decirlo ahorra buscar en la ruta un fallo que está en el
                # puerto.
                detalle += (
                    " · Comprueba que `base_url` apunte al servicio dueño del "
                    "diario y no al que arranca los runs: comparten prefijo."
                )
            raise ProviderError(
                f"{fallo.code} {fallo.reason}: {detalle}",
                status=fallo.code,
                provider="checkpointer",
            ) from fallo
        except TimeoutError as fallo:
            raise RequestTimeoutError(f"el diario no respondió en {self.timeout}s") from fallo
        except urllib.error.URLError as fallo:
            if isinstance(fallo.reason, TimeoutError):
                raise RequestTimeoutError(f"sin respuesta en {self.timeout}s") from fallo
            raise NetworkError(
                f"no se pudo llegar al diario en {self.base_url}: {fallo.reason}"
            ) from fallo


#: Tope del identificador de sub-run que acepta el otro lado.
LIMITE_SUB_RUN = 512


def partir(run_id: str) -> tuple[str, str]:
    """Separa el run padre del sub-run.

    ``"r1"`` → ``("r1", "")`` · ``"r1/000001-delegate"`` → ``("r1", "000001-delegate")``

    Y un nieto conserva su ruta entera: ``("r1", "000001-delegate/000002-delegate")``.
    """
    padre, _, sub = run_id.partition("/")
    if len(sub) > LIMITE_SUB_RUN:
        from ..core.errors import ConfigurationError

        raise ConfigurationError(
            f"El sub-run mide {len(sub)} caracteres y el diario acepta "
            f"{LIMITE_SUB_RUN}. Una delegación no llega a eso por profundidad, "
            "así que probablemente el `run_id` del padre ya traía una barra."
        )
    return padre, sub
