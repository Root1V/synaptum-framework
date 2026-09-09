"""
RM-22 · RM-23 · RM-25 · Journal, almacén de referencia y replay.

Tres piezas pequeñas que juntas dan la propiedad que justifica toda la
arquitectura: **al reanudar, una inferencia ya pagada no se paga otra vez.**

``Journal``
    Aplica la política de durabilidad por clase de evento.  Un evento
    ``DURABLE`` se persiste de forma síncrona antes de continuar; uno
    ``DEFERRABLE`` se acumula y se escribe por lotes.

``MemoryCheckpointer``
    Implementación de referencia en memoria, con la deduplicación que el
    contrato exige.  Para tests, notebooks y uso autónomo — nunca producción.

``Replay``
    Decide, paso a paso, si hay que ejecutar o si ya está hecho.
"""

from __future__ import annotations

from ..core.errors import UncertainEffect
from ..core.events import Durability, StepEvent
from ..core.protocols import Checkpointer, RunState

__all__ = ["Journal", "MemoryCheckpointer", "Replay"]


# ── RM-23 · Almacén de referencia ─────────────────────────────────────────────

class MemoryCheckpointer:
    """``Checkpointer`` en memoria.  Cumple el contrato completo.

    Existe para dos cosas: que Synaptum sea utilizable sin harness, y que haya
    una implementación contra la que contrastar cualquier otra.  No es un
    motor de producción y no pretende serlo.
    """

    def __init__(self) -> None:
        self._runs: dict[str, list[StepEvent]] = {}
        self._keys: set[tuple[str, str, str]] = set()

    async def append(self, run_id: str, event: StepEvent) -> None:
        """Idempotente por ``(run_id, step_id, phase)``.  Un duplicado es no-op."""
        if event.key in self._keys:
            return
        self._keys.add(event.key)
        self._runs.setdefault(run_id, []).append(event)

    async def load(self, run_id: str) -> RunState | None:
        events = self._runs.get(run_id)
        return RunState(run_id, tuple(events)) if events is not None else None

    # Conveniencias para tests; no forman parte del protocolo.
    def event_count(self, run_id: str) -> int:
        return len(self._runs.get(run_id, ()))


# ── RM-22 · Journal ───────────────────────────────────────────────────────────

class Journal:
    """Escribe eventos respetando la durabilidad que cada uno declara.

    El orden importa más de lo que parece.  Los eventos diferidos se vacían
    **antes** de escribir uno durable, de modo que el journal conserva el orden
    de ejecución: si no, un evento estructural anterior aparecería después del
    efecto que lo siguió, y el registro dejaría de describir lo que pasó.
    """

    def __init__(self, checkpointer: Checkpointer, run_id: str) -> None:
        self._cp = checkpointer
        self._run_id = run_id
        self._pending: list[StepEvent] = []

    async def record(self, event: StepEvent) -> None:
        if event.durability is Durability.DURABLE:
            await self._drain()
            await self._cp.append(self._run_id, event)
        else:
            self._pending.append(event)

    async def flush(self) -> None:
        """Vacía lo diferido.  Se llama al cerrar el run, y ante cualquier salida."""
        await self._drain()

    async def _drain(self) -> None:
        while self._pending:
            await self._cp.append(self._run_id, self._pending.pop(0))


# ── RM-25 · Replay ────────────────────────────────────────────────────────────

class Replay:
    """Consulta al journal si un paso ya ocurrió.

    Tres respuestas posibles para cada paso, y la tercera es la interesante:

    * **Hecho** — hay resultado registrado.  Se devuelve y no se ejecuta nada.
      Aquí es donde se ahorra la inferencia.
    * **Nuevo** — no hay rastro.  Se ejecuta con normalidad.
    * **Incierto** — hay intención sin resultado.  El proceso cayó en medio, así
      que el efecto **pudo haber ocurrido**.
    """

    def __init__(self, state: RunState | None) -> None:
        self._state = state
        self.replayed = 0
        """Cuántos pasos se han saltado.  Métrica, no lógica."""

    @property
    def next_seq(self) -> int:
        return self._state.next_seq if self._state else 0

    @property
    def active(self) -> bool:
        """``True`` si hay algo que reproducir."""
        return self._state is not None and bool(self._state.events)

    def resolve(self, step_id: str, *, idempotent: bool = True) -> StepEvent | None:
        """Devuelve el resultado ya registrado, o ``None`` si toca ejecutar.

        Args:
            step_id: paso a resolver.
            idempotent: si el efecto de este paso puede repetirse sin
                consecuencias.  Solo se consulta en el caso incierto.

        Raises:
            UncertainEffect: si el paso quedó intentado sin resultado y su
                efecto no es idempotente.  El bucle no tiene la información
                para decidir, así que no la inventa: la levanta.
        """
        if self._state is None:
            return None

        done = self._state.result_of(step_id)
        if done is not None:
            self.replayed += 1
            return done

        if self._state.attempted(step_id) and not idempotent:
            raise UncertainEffect(
                step_id,
                detail="Decide el harness si se reintenta o si el run se marca fallido.",
            )

        return None
