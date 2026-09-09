"""
RM-04 · Taxonomía de errores.

Un bucle que reintenta lo que no debe quema dinero; uno que no reintenta lo que
debe falla por un pico de red.  La diferencia no la puede decidir el bucle: la
sabe el adaptador que habló con el proveedor, y por eso viaja **en el tipo del
error**, no en una tabla de códigos repartida por el código.

Regla de clasificación
----------------------
* **No reintentable** — 400, 401, 403, 404, 413, 422.  La petición está mal, o
  no hay permiso, o no existe: repetirla da el mismo resultado y cuesta lo
  mismo.
* **Reintentable** — 429, 5xx, timeouts y fallos de red.  El fallo es del otro
  lado o del camino, y la misma petición puede funcionar dentro de un momento.
* **Desconocido** — reintentable por defecto.  Un error que no sabemos
  clasificar se parece más a un problema transitorio que a una petición
  inválida, y el coste de equivocarse es menor.

``Denied`` no es un fallo
-------------------------
Una denegación de la costura de aplicación es una **decisión**, no un error del
sistema.  Nunca es reintentable: repetir la llamada no cambia la política.  Es
una excepción porque tiene que interrumpir el efecto de forma ineludible —
incluso a mitad de un stream — y porque un valor de retorno que se puede ignorar
no sirve para hacer cumplir nada.
"""

from __future__ import annotations

from .events import Decision, Disposition

__all__ = [
    "SynaptumError",
    "ConfigurationError",
    "ProviderError",
    "RequestTimeoutError",
    "NetworkError",
    "AbortError",
    "Denied",
    "InvalidToolCallError",
    "ToolExecutionError",
    "NoObjectGeneratedError",
    "LimitExceeded",
    "SeamVersionError",
    "retryable_for_status",
]


# ── Base ──────────────────────────────────────────────────────────────────────

class SynaptumError(Exception):
    """Raíz de todo lo que este framework lanza a propósito.

    ``retryable`` es la única pregunta que el bucle le hace a un error.
    """

    retryable: bool = False

    def __init__(self, message: str = "", *, retryable: bool | None = None) -> None:
        super().__init__(message)
        if retryable is not None:
            self.retryable = retryable


# ── Configuración ─────────────────────────────────────────────────────────────

class ConfigurationError(SynaptumError):
    """Falta algo, sobra algo, o dos cosas se contradicen.

    Nunca reintentable: el entorno no se arregla solo entre dos intentos.
    """

    retryable = False


class SeamVersionError(ConfigurationError):
    """No hay versión de costura común entre los dos extremos — RM-12."""


# ── Proveedor y transporte ────────────────────────────────────────────────────

class ProviderError(SynaptumError):
    """Error devuelto por el proveedor, con su estado HTTP si lo hubo.

    Cuando no se pasa ``retryable`` explícito, se deduce del estado según la
    regla de clasificación.  Un error sin estado se considera transitorio.
    """

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        provider: str = "",
        retryable: bool | None = None,
    ) -> None:
        self.status = status
        self.provider = provider
        resolved = retryable if retryable is not None else retryable_for_status(status)
        super().__init__(message, retryable=resolved)


class RequestTimeoutError(SynaptumError):
    """La petición no respondió dentro del plazo."""

    retryable = True


class NetworkError(SynaptumError):
    """Falló el camino, no el destino."""

    retryable = True


class AbortError(SynaptumError):
    """Alguien canceló deliberadamente — H2.

    No es un fallo del sistema y no se reintenta: reintentar una cancelación es
    exactamente lo contrario de lo que pidió quien canceló.
    """

    retryable = False


# ── Aplicación ────────────────────────────────────────────────────────────────

class Denied(SynaptumError):
    """La costura de aplicación no dejó ocurrir el efecto — H4.

    Lleva la ``Decision`` entera para que el bucle sepa qué hacer: ``deny_step``
    admite intentar otra cosa, ``terminate_run`` no, y ``require_approval``
    suspende el run sin que sea un fallo.
    """

    retryable = False

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        detail = decision.message or decision.reason_code or decision.disposition.value
        super().__init__(f"{decision.disposition.value}: {detail}")

    @property
    def disposition(self) -> Disposition:
        return self.decision.disposition

    @property
    def terminal(self) -> bool:
        """``True`` si el run no puede continuar por ninguna vía."""
        return self.decision.disposition is Disposition.TERMINATE_RUN


class LimitExceeded(SynaptumError):
    """Se agotó un límite del bucle: pasos, reintentos, presupuesto de ventana.

    Es corrección, no política: los límites de gasto pertenecen al harness y
    llegan como ``Denied`` con ``terminate_run``.
    """

    retryable = False

    def __init__(self, limit: str, value: int) -> None:
        self.limit = limit
        self.value = value
        super().__init__(f"Límite '{limit}' superado: {value}.")


# ── Herramientas y salida estructurada ────────────────────────────────────────

class InvalidToolCallError(SynaptumError):
    """El modelo pidió una tool que no existe, o con argumentos que no validan.

    No reintentable tal cual: lo que corrige esto es devolverle el error al
    modelo como ``ToolResult`` para que rectifique, no repetir la llamada.
    """

    retryable = False

    def __init__(self, message: str = "", *, tool: str = "", call_id: str = "") -> None:
        self.tool = tool
        self.call_id = call_id
        super().__init__(message)


class ToolExecutionError(SynaptumError):
    """La tool existía, se invocó bien, y falló al ejecutarse.

    Reintentable solo si quien la definió declara que el efecto es idempotente.
    """

    def __init__(self, message: str = "", *, tool: str = "", retryable: bool = False) -> None:
        self.tool = tool
        super().__init__(message, retryable=retryable)


class NoObjectGeneratedError(SynaptumError):
    """Se pidió salida estructurada y no salió un objeto válido.

    Reintentable: el muestreo es estocástico y una segunda pasada suele acertar.
    """

    retryable = True

    def __init__(self, message: str = "", *, raw: str = "") -> None:
        self.raw = raw
        super().__init__(message)


# ── Clasificación ─────────────────────────────────────────────────────────────

_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 413, 422})


def retryable_for_status(status: int | None) -> bool:
    """Aplica la regla de clasificación a un estado HTTP.

    Sin estado devuelve ``True``: un fallo que ni siquiera llegó a tener
    respuesta se parece más a un problema de camino que a una petición
    inválida.

    Un 4xx desconocido, en cambio, devuelve ``False``.  La familia entera
    significa «tu petición es el problema», y repetirla sin cambiarla da el
    mismo resultado — el 429 es la excepción, y está contemplada aparte.
    """
    if status is None:
        return True
    if status in _NON_RETRYABLE_STATUS:
        return False
    if status == 429 or status >= 500:
        return True
    if 400 <= status < 500:
        return False
    return True
