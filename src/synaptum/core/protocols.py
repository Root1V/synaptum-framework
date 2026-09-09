"""
RM-07 · RM-08 · Los dos protocolos de costura.

Todo el acoplamiento entre Synaptum y el harness cabe en este módulo.  Son dos
protocolos, no uno, porque **vetar y recordar son operaciones de naturaleza
distinta**:

* ``Gateway`` — **aplica**.  Síncrona, denegable, y fuera del proceso del bucle.
* ``Checkpointer`` — **recuerda**.  Persiste; no decide nada.

Fusionarlas obligaría a lo peor de ambas: escrituras de journal síncronas en el
camino crítico de cada paso, y semántica de autorización escondida dentro de un
método de almacenamiento.

Son ``typing.Protocol``, no clases base: quien las implementa no hereda de nada,
no importa nada de aquí en tiempo de ejecución, y puede vivir en otro repositorio
con su propio ciclo de release.  Es la forma de la frontera, no una biblioteca
compartida.

Por qué el ``Gateway`` ejecuta además de decidir
------------------------------------------------
Un motor de decisión puro no basta.  Si el bucle pregunta «¿puedo?», recibe un
sí y **llama él**, entonces la credencial vive en el proceso del bucle — y una
comprobación que el propio proceso gobernado invoca voluntariamente es advisoria,
no aplicada.  Es la misma razón por la que un sandbox no se implementa como una
función que el código encerrado decide llamar.

De modo que el ``Gateway`` termina la conexión: decide y ejecuta en la misma
operación.  El coste es un salto de proceso, y no es sobrecarga evitable — es el
precio de no entregarle credenciales al framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, Protocol, Sequence, runtime_checkable

from .errors import SeamVersionError
from .events import Risk, StepEvent
from .types import Request, Response, StreamEvent, ToolCall, ToolDefinition, ToolResult

__all__ = [
    "SEAM_VERSION",
    "SUPPORT_WINDOW",
    "supported_versions",
    "negotiate",
    "CallContext",
    "Hello",
    "Welcome",
    "Gateway",
    "RunState",
    "Checkpointer",
]


# ── Versionado de la costura — RM-12 / H6 ─────────────────────────────────────

SEAM_VERSION = "0.1"
"""Versión que este paquete habla."""

SUPPORT_WINDOW = 2
"""Cuántas versiones menores se aceptan **en total**, contando la actual.

Barato de fijar ahora y carísimo después.  Con N = 2 los tres proyectos pueden
desplegar de forma independiente sin coordinar una ventana de corte, y sin
arrastrar compatibilidad indefinida.
"""


def supported_versions(current: str = SEAM_VERSION, window: int = SUPPORT_WINDOW) -> tuple[str, ...]:
    """Versiones que este extremo acepta, de la más nueva a la más vieja."""
    major_s, minor_s = current.split(".", 1)
    major, minor = int(major_s), int(minor_s)
    return tuple(f"{major}.{m}" for m in range(minor, max(minor - window, -1), -1))


def negotiate(peer: Sequence[str], *, current: str = SEAM_VERSION) -> str:
    """Elige la versión más alta que ambos extremos hablan.

    Recorre las nuestras de más nueva a más vieja, de modo que dos extremos al
    día no se quedan atascados en una antigua solo porque ambos la soportan.

    Raises:
        SeamVersionError: si no hay ninguna en común.  Fallar aquí es mucho
            mejor que descubrir la incompatibilidad campo a campo.
    """
    ours = supported_versions(current)
    theirs = set(peer)
    for version in ours:
        if version in theirs:
            return version
    raise SeamVersionError(
        f"Sin versión de costura común. Nosotros: {list(ours)}; el otro extremo: {list(peer)}."
    )


# ── Contexto de llamada — RM-11 ───────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CallContext:
    """Lo que acompaña a cada cruce de la costura.

    ``traceparent`` y ``tracestate`` son W3C y no son decorativos: el harness
    emite los spans de ``chat`` y ``execute_tool`` porque es quien ejecuta la
    E/S, mientras el bucle emite los suyos de estructura.  Sin propagar el
    contexto, unos y otros acaban en árboles de traza distintos y nadie puede
    seguir un run de punta a punta.
    """

    run_id: str
    step_id: str
    traceparent: str | None = None
    tracestate: str | None = None
    deadline_s: float | None = None
    """Plazo restante.  Permite al otro extremo cortar antes de empezar algo
    que no va a caber."""
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Acompañantes opacos.  Se propagan sin interpretarse."""


# ── Handshake de sesión — RM-12 + RM-13 ───────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Hello:
    """Apertura de sesión, enviada por el bucle.

    Combina la negociación de versión con el registro de herramientas en un solo
    viaje, y no por ahorrar una llamada: **las dos cosas tienen que ocurrir
    antes del primer turno y ninguna puede repetirse a mitad de sesión.**
    Cambiar el catálogo de tools con la sesión abierta reescribe el prefijo del
    prompt y tira la caché del proveedor.
    """

    versions: tuple[str, ...] = field(default_factory=lambda: supported_versions())
    tools: tuple[ToolDefinition, ...] = ()
    client: str = "synaptum"


@dataclass(frozen=True, slots=True)
class Welcome:
    """Respuesta del harness al ``Hello``.

    ``tool_refs`` mapea nombre de tool a **referencia versionada** (H5).  A
    partir de aquí el esquema no viaja en cada llamada: se resolvió una vez y el
    prefijo cacheado se mantiene estable turno tras turno.
    """

    version: str
    tool_refs: Mapping[str, str] = field(default_factory=dict)
    session_id: str = ""


# ── Costura 1 · aplicación — RM-08 ────────────────────────────────────────────

@runtime_checkable
class Gateway(Protocol):
    """Decide y ejecuta.  Vive fuera del proceso del bucle.

    Cada método puede lanzar ``Denied`` con la ``Decision`` que corresponda —
    ``deny_step`` admite que el bucle intente otra cosa, ``terminate_run`` no, y
    ``require_approval`` suspende el run sin que sea un fallo.

    Toda respuesta devuelve ``Usage`` (H3).  El span de la operación lo emite
    este lado, porque es quien la realiza; pero los contadores vuelven, porque
    la economía de contexto del bucle depende de ellos y no son derivables desde
    fuera.
    """

    async def handshake(self, hello: Hello) -> Welcome:
        """Negocia versión y registra el catálogo de tools.  Una vez por sesión."""
        ...

    async def invoke_model(self, request: Request, ctx: CallContext) -> Response:
        """Ejecuta una llamada al modelo y devuelve la respuesta completa."""
        ...

    def stream_model(self, request: Request, ctx: CallContext) -> AsyncIterator[StreamEvent]:
        """Ejecuta una llamada al modelo en streaming — H2.

        No es ``async def``: devuelve el iterador directamente, para que la
        cancelación tenga a quién dirigirse.  Cerrar el iterador
        (``aclose()``, salir de un ``async with``, o cancelar la tarea que lo
        consume) **debe** propagar la cancelación hasta el proveedor.

        Sin eso no hay corte de presupuesto en caliente: una generación que ya
        se decidió abandonar se sigue pagando hasta el final.

        El último evento es siempre ``Finish``, con la ``Response`` acumulada,
        para que quien consumió los deltas no tenga que reconstruirla.
        """
        ...

    async def invoke_tool(
        self, call: ToolCall, ctx: CallContext, *, risk: Risk, tool_ref: str | None = None
    ) -> ToolResult:
        """Ejecuta una herramienta.

        ``risk`` lo **declara** el framework; qué hacer con él lo **decide** este
        lado.  ``tool_ref`` es la referencia devuelta en el ``Welcome``.
        """
        ...


# ── Costura 2 · memoria — RM-07 ───────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RunState:
    """Lo que ``load`` devuelve: el journal de un run, en orden.

    El bucle lo usa para adelantar — ``fast-forward`` — sobre los pasos que ya
    tienen resultado registrado, en vez de volver a ejecutarlos.  Ahí está el
    valor entero de la ejecución durable: **al reanudar, una inferencia ya
    pagada no se paga otra vez.**
    """

    run_id: str
    events: tuple[StepEvent, ...] = ()

    @property
    def next_seq(self) -> int:
        """Número de secuencia que le toca al siguiente paso."""
        return max((e.seq for e in self.events), default=-1) + 1

    @property
    def final(self) -> StepEvent | None:
        """El evento de cierre, si el run ya terminó.

        Un run cerrado no se reabre: reanudarlo devuelve lo que pasó, no lo
        intenta otra vez.
        """
        for event in reversed(self.events):
            if event.kind == "final":
                return event
        return None

    def result_of(self, step_id: str) -> StepEvent | None:
        """Resultado ya registrado de un paso, si lo hay.

        Que exista significa que el efecto ocurrió y no debe repetirse.
        """
        for event in reversed(self.events):
            if event.step_id == step_id and event.phase.value == "result":
                return event
        return None

    def completed(self, step_id: str) -> bool:
        return self.result_of(step_id) is not None

    def attempted(self, step_id: str) -> bool:
        """``True`` si hay intención registrada, con o sin resultado.

        Una intención sin resultado es el caso incierto: el proceso cayó entre
        el registro y el efecto, así que el efecto **pudo haber ocurrido**.  Si
        no es idempotente, repetirlo a ciegas es lo peor que se puede hacer.
        """
        return any(e.step_id == step_id for e in self.events)


@runtime_checkable
class Checkpointer(Protocol):
    """Persiste el journal.  No decide nada.

    Es lo único que el harness tiene que implementar de forma obligatoria: dos
    métodos contra el almacén que ya use.  No necesita entender el bucle del
    agente, ni la lógica de replay, ni cómo se acuñan los identificadores.
    """

    async def append(self, run_id: str, event: StepEvent) -> None:
        """Añade un evento al journal.

        **Debe ser idempotente por ``(run_id, step_id, phase)``.**  Un ``append``
        repetido con la misma clave es un no-op, nunca un error.

        No es una cortesía: un motor de workflows puede reintentar una unidad de
        trabajo que ya escribió, y sin deduplicación el journal se corrompe con
        duplicados justo en el camino de recuperación — el peor momento posible.

        La urgencia con la que persistir la dicta ``event.durability``.  Un
        evento ``DURABLE`` no puede diferirse: si se pierde, tras una caída es
        imposible saber si el efecto ocurrió.
        """
        ...

    async def load(self, run_id: str) -> RunState | None:
        """Reconstruye el estado de un run.  ``None`` si no existe.

        Los eventos vuelven en orden de ejecución.
        """
        ...
