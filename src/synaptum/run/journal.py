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

import re

from ..core.errors import ConfigurationError, UncertainEffect
from ..core.events import Durability, Phase, StepEvent
from ..core.types import dumps
from ..core.protocols import AppendResult, Checkpointer, RunState

__all__ = ["Journal", "MemoryCheckpointer", "Replay"]


# ── RM-23 · Almacén de referencia ─────────────────────────────────────────────

#: Un identificador de paso derivado del ordinal del bucle: `000003-model`.
_POSICIONAL = re.compile(r"^\d{6}-")


class MemoryCheckpointer:
    """``Checkpointer`` en memoria.  Cumple el contrato completo.

    Existe para dos cosas: que Synaptum sea utilizable sin harness, y que haya
    una implementación contra la que contrastar cualquier otra.  No es un
    motor de producción y no pretende serlo.
    """

    def __init__(self) -> None:
        self._runs: dict[str, list[StepEvent]] = {}
        self._index: dict[tuple[str, str, str], tuple[int, str]] = {}

    async def append(self, run_id: str, event: StepEvent) -> AppendResult:
        """Idempotente por ``(run_id, step_id, phase)``.  Un duplicado es no-op.

        La divergencia de payload se compara de forma **semántica**, no de
        bytes: ``dumps`` ordena las claves, así que reserializar el mismo objeto
        en otro orden no cuenta como divergencia.  Comparar bytes convertiría en
        falsa alarma cada reintento desde un lenguaje sin orden garantizado.
        """
        payload = dumps(event)
        seen = self._index.get(event.key)
        if seen is not None:
            position, stored = seen
            return AppendResult(seq=position, duplicate=True, payload_diverged=payload != stored)

        entries = self._runs.setdefault(run_id, [])
        position = len(entries)
        entries.append(event)
        self._index[event.key] = (position, payload)
        return AppendResult(seq=position)

    async def load(self, run_id: str) -> RunState:
        return RunState(run_id, tuple(self._runs.get(run_id, ())))

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
    * **Denegado y reintentable** — la costura no dejó ocurrir el efecto, o
      nadie contestó a tiempo.  Se vuelve a intentar: entre una reanudación y
      otra la política pudo cambiar, y a una expiración se puede volver a
      preguntar porque es la **ausencia** de una decisión.
    * **Denegado y cerrado** — **una persona dijo que no.**  No se reintenta ni
      se vuelve a preguntar: volver a preguntar tras una negativa es ir de
      compras a por un sí.  Vuelve al modelo como evidencia, que es lo que hace
      que rectifique en vez de insistir.
    * **Incierto** — hay intención sin resultado.  El proceso cayó en medio, así
      que el efecto **pudo haber ocurrido**.
    """

    def __init__(self, state: RunState) -> None:
        self._state = state
        self.replayed = 0
        """Cuántos pasos se han saltado.  Métrica, no lógica."""
        self.denied = 0
        """Cuántos se reintentan por haber sido denegados antes."""
        self.closed_by_decision = 0
        """Cuántos quedaron cerrados porque una persona dijo que no."""
        self._preguntados: set[str] = set()

    @property
    def active(self) -> bool:
        """``True`` si hay algo que reproducir."""
        return bool(self._state.events)

    @property
    def closed(self) -> StepEvent | None:
        """El cierre del run, si ya lo hubo."""
        return self._state.final

    def ocupado_por_otro(self, step_id: str) -> StepEvent | None:
        """El paso que el diario tiene en **esa misma posición**, si es de otra clase.

        Es la señal inequívoca de que el diario y este bucle no hablan de la
        misma secuencia. Los identificadores son posicionales —`000003-model`—
        así que un bucle que emite un paso más, o uno menos, que el que escribió
        el diario pregunta por posiciones desplazadas: cada consulta falla por
        separado, ninguna sabe de las otras, y **todo se reejecuta**.

        Se compara la **clase** y no la mera ausencia, y esa distinción es todo:
        un hueco es normal —las escrituras diferidas se agrupan, así que un
        registro puede no estar todavía— pero una posición ocupada por otra
        clase de paso no la produce ningún retraso. El paso 0 de este bucle es
        siempre de modelo; si el diario tiene ahí una herramienta, lo escribió
        otra secuencia.

        Lo que **no** caza, dicho para que nadie lo dé por cubierto: un
        desplazamiento que conserve las clases en cada posición. Ahí las dos
        secuencias son indistinguibles desde el diario.
        """
        ordinal, _, clase = step_id.partition("-")
        if not _POSICIONAL.match(step_id):
            return None
        for evento in self._state.events:
            if evento.phase is not Phase.COMPLETED:
                continue
            suyo_ordinal, _, suya_clase = evento.step_id.partition("-")
            if suyo_ordinal == ordinal and suya_clase != clase:
                return evento
        return None

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
        self._preguntados.add(step_id)
        done = self._state.result_of(step_id)

        if done is None:
            # Nada en esta posición. ¿La tiene ocupada otra clase de paso?
            intruso = self.ocupado_por_otro(step_id)
            if intruso is not None:
                raise ConfigurationError(
                    f"El diario del run '{self._state.run_id}' tiene un paso "
                    f"'{intruso.step_id}' donde este bucle pide '{step_id}'.\n"
                    "La identidad de paso es posicional, así que un bucle que emite "
                    "un paso más —o uno menos— que el que escribió el diario pregunta "
                    "por identificadores desplazados: los registros viejos no se "
                    "consultan y **todo se reejecuta**, incluido lo que no es "
                    "idempotente.\n"
                    "Suele significar que el run se grabó con otra versión del "
                    "framework. Ese run es de aquella secuencia: dale un `run_id` "
                    "nuevo, o reanúdalo con la versión que lo escribió."
                )

        if done is not None:
            if done.outcome.final:
                # Una persona dijo que no.  Es un desenlace **cerrado**: se
                # devuelve para que el bucle lo cuente, no para reejecutarlo.
                #
                # Hasta `SYN-80` esto caía en la rama de abajo y se reintentaba
                # igual que una expiración — colapsando exactamente lo que
                # pedimos no colapsar a quien escribe el diario.  Una negativa
                # es una decisión y cierra el paso; una expiración es su
                # ausencia.
                self.closed_by_decision += 1
                return done

            if done.decision is not None and not done.decision.allowed:
                # Denegado antes de ejecutar: desenlace conocido, efecto que no
                # ocurrió.  No es el caso incierto, y se puede reintentar.
                self.denied += 1
                return None

            if not done.outcome.happened:
                # Sin resultado y sin decisión: una expiración, o una política
                # que denegó sin dejar `Decision`.  Reintentable por lo mismo.
                self.denied += 1
                return None

            self.replayed += 1
            return done

        if self._state.attempted(step_id) and not idempotent:
            raise UncertainEffect(
                step_id,
                detail="Decide el harness si se reintenta o si el run se marca fallido.",
            )

        return None
