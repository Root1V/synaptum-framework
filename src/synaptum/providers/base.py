"""
SYN-17 · Registro de proveedores, descubierto por *entry points*.

El núcleo no conoce a ningún proveedor. Los encuentra preguntando al entorno,
así que instalar un paquete basta para que su adaptador exista — sin tocar una
lista aquí, sin un `import` condicional, sin que este módulo sepa sus nombres.

Un adaptador **normaliza, no transporta**
------------------------------------------
La separación importa más de lo que parece. Lo que el contrato compartido fija
es la **normalización**: qué objeto unificado sale de un cuerpo nativo concreto.
Eso es una función pura y se puede ejercitar con un fichero.

El transporte —quién abre la conexión, con qué credencial— es otra cosa, y en el
camino gobernado ni siquiera ocurre aquí: lo hace el gateway, porque ahí vive la
credencial. Mezclar las dos en un solo objeto habría hecho que probar la
normalización exigiera un servidor.

Nombrar un modelo
-----------------
``"proveedor:modelo"``. El prefijo dice qué adaptador normaliza; el resto viaja
tal cual y lo interpreta quien ejecuta la llamada.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Protocol, runtime_checkable

from ..core.errors import ConfigurationError
from ..core.types import Request, Response, StreamEvent

__all__ = ["Provider", "ENTRY_POINT_GROUP", "register", "get", "resolve", "available"]


ENTRY_POINT_GROUP = "synaptum.providers"


@runtime_checkable
class Provider(Protocol):
    """Traduce entre el vocabulario unificado y el dialecto de un proveedor.

    Las tres operaciones son **puras**: no abren conexiones y no leen
    credenciales. Por eso el corpus dorado de normalización se puede ejecutar
    sin un servidor delante.
    """

    name: str

    def to_wire(self, request: Request) -> Mapping[str, Any]:
        """Del vocabulario unificado al cuerpo que espera el proveedor."""
        ...

    def from_wire(self, body: Mapping[str, Any]) -> Response:
        """Del cuerpo nativo al vocabulario unificado."""
        ...

    def stream_from_wire(self, chunks: Iterable[Mapping[str, Any]]) -> Iterator[StreamEvent]:
        """De los fragmentos nativos al ciclo de eventos unificado.

        Recibe los fragmentos ya separados del transporte — líneas SSE
        parseadas, por ejemplo — porque cómo llegan por el cable no es asunto de
        la normalización.
        """
        ...


# ── Registro ──────────────────────────────────────────────────────────────────

_MANUAL: dict[str, Provider] = {}
_DISCOVERED: dict[str, Provider] | None = None


def register(provider: Provider) -> None:
    """Registra un adaptador a mano.

    Tiene prioridad sobre lo descubierto, para que un test o un adaptador
    interno puedan sustituir a uno instalado sin desinstalarlo.
    """
    _MANUAL[provider.name] = provider


def _discover() -> dict[str, Provider]:
    """Carga los adaptadores publicados como *entry points*.

    Perezoso y cacheado: el descubrimiento cuesta E/S y no debe pagarse al
    importar el paquete.
    """
    global _DISCOVERED
    if _DISCOVERED is not None:
        return _DISCOVERED

    from importlib.metadata import entry_points

    found: dict[str, Provider] = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            factory = entry.load()
        except Exception as broken:  # noqa: BLE001
            # Un adaptador roto no puede tumbar a los demás: quien lo pida se
            # encontrará con su error, y quien no, ni se entera.
            found[entry.name] = _Broken(entry.name, broken)  # type: ignore[assignment]
            continue
        found[entry.name] = factory() if callable(factory) else factory

    _DISCOVERED = found
    return found


class _Broken:
    """Marcador para un adaptador que no cargó.  Falla al usarse, no al instalarse."""

    def __init__(self, name: str, cause: Exception) -> None:
        self.name = name
        self._cause = cause

    def __getattr__(self, _: str) -> Any:
        raise ConfigurationError(
            f"El adaptador '{self.name}' está instalado pero no carga: {self._cause}"
        ) from self._cause


def get(name: str) -> Provider:
    """Devuelve el adaptador registrado bajo ``name``."""
    if name in _MANUAL:
        return _MANUAL[name]

    discovered = _discover()
    if name in discovered:
        return discovered[name]

    raise ConfigurationError(
        f"No hay adaptador para el proveedor '{name}'. Disponibles: {available()}. "
        f"Instala el paquete que lo publica, o regístralo con providers.register()."
    )


def resolve(spec: str) -> tuple[Provider, str]:
    """Parte ``"proveedor:modelo"`` en su adaptador y el nombre del modelo."""
    provider, _, model = spec.partition(":")
    if not provider or not model:
        raise ConfigurationError(
            f"'{spec}' no nombra un modelo. La forma es 'proveedor:modelo', "
            "por ejemplo 'openai-compatible:llama3-8b-q4'."
        )
    return get(provider), model


def available() -> list[str]:
    """Nombres de los adaptadores alcanzables ahora mismo."""
    return sorted(set(_MANUAL) | set(_discover()))


def _reset_discovery() -> None:
    """Olvida el descubrimiento cacheado.  Solo para tests."""
    global _DISCOVERED
    _DISCOVERED = None
