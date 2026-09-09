"""
SYN-24 · ``SqliteCheckpointer`` — el journal sobrevive al proceso.

Implementación de referencia persistente.  Con el almacén en memoria, reanudar
solo funciona mientras el proceso viva, que es justamente el caso que la
ejecución durable no cubre.  Con este, un run se retoma después de reiniciar.

**No es un motor de producción y no pretende serlo.**  Retención, cifrado,
multi-tenencia y escalado pertenecen al harness, y ese es el reparto acordado.
Esto existe para tests, notebooks y uso autónomo — y para que haya una
implementación completa contra la que contrastar cualquier otra.

La idempotencia no se programa: **es la clave primaria**.  Un ``append``
repetido con el mismo ``(run_id, step_id, phase)`` no hace nada, y no porque
alguien se acordara de comprobarlo antes de insertar, sino porque la tabla no
admite el duplicado.  Es la diferencia entre una garantía y una intención.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..core.codec import decode_event
from ..core.events import StepEvent
from ..core.protocols import AppendResult, RunState
from ..core.types import dumps

__all__ = ["SqliteCheckpointer"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    run_id   TEXT    NOT NULL,
    step_id  TEXT    NOT NULL,
    phase    TEXT    NOT NULL,
    seq      INTEGER NOT NULL,
    step_seq INTEGER NOT NULL,
    payload  TEXT    NOT NULL,
    PRIMARY KEY (run_id, step_id, phase)
);
CREATE INDEX IF NOT EXISTS journal_order ON journal (run_id, seq);
"""


class SqliteCheckpointer:
    """``Checkpointer`` sobre SQLite.  Cumple el contrato completo."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ── Contrato ──────────────────────────────────────────────────────────────

    async def append(self, run_id: str, event: StepEvent) -> AppendResult:
        """Añade un evento.  Un duplicado es no-op, garantizado por la clave.

        ``INSERT OR IGNORE`` sobre la clave primaria: no hay ventana entre
        comprobar y escribir, así que dos procesos reintentando la misma unidad
        de trabajo no pueden colarse a la vez.
        """
        payload = dumps(event)
        existing = self._db.execute(
            "SELECT seq, payload FROM journal WHERE run_id=? AND step_id=? AND phase=?",
            (run_id, event.step_id, event.phase.value),
        ).fetchone()
        if existing is not None:
            return AppendResult(seq=existing[0], duplicate=True, payload_diverged=existing[1] != payload)

        position = self._db.execute(
            "SELECT COUNT(*) FROM journal WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        cursor = self._db.execute(
            "INSERT OR IGNORE INTO journal "
            "(run_id, step_id, phase, seq, step_seq, payload) VALUES (?,?,?,?,?,?)",
            (run_id, event.step_id, event.phase.value, position, event.step_seq, payload),
        )
        self._db.commit()
        if cursor.rowcount == 0:
            # Carrera perdida: otro escritor insertó entre el SELECT y aquí.
            # La clave primaria es el respaldo, no la comprobación previa.
            row = self._db.execute(
                "SELECT seq, payload FROM journal WHERE run_id=? AND step_id=? AND phase=?",
                (run_id, event.step_id, event.phase.value),
            ).fetchone()
            return AppendResult(seq=row[0], duplicate=True, payload_diverged=row[1] != payload)
        return AppendResult(seq=position)

    async def load(self, run_id: str) -> RunState:
        """Reconstruye el estado del run, o ``None`` si no existe.

        Los eventos vuelven en **orden de escritura**, no de número de paso.  El
        journal describe lo que pasó y cuándo, y con pasos concurrentes el orden
        de llegada y el de numeración dejan de coincidir.
        """
        rows = self._db.execute(
            "SELECT payload FROM journal WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return RunState(run_id, tuple(decode_event(json.loads(row[0])) for row in rows))

    # ── Operación ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "SqliteCheckpointer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def runs(self) -> list[str]:
        """Identificadores de los runs almacenados.  Fuera del protocolo."""
        rows = self._db.execute("SELECT DISTINCT run_id FROM journal ORDER BY run_id").fetchall()
        return [row[0] for row in rows]
