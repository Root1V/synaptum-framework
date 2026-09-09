"""Conformidad · costura de durabilidad.

Corre los **casos dorados compartidos** contra nuestras dos implementaciones del
``Checkpointer``. Los casos los publica Aeon en `contratos/costura-durabilidad`;
aquí no se copian ni se ajustan — se leen del fichero y se ejecutan tal cual.

El runner lo trae cada proyecto, que es la parte del acuerdo que importa: si dos
implementaciones sin una línea de código en común reproducen los mismos
resultados observables, la equivalencia deja de ser una afirmación. Por eso los
casos describen resultados y nunca internos.

Si los contratos no están montados, la suite se salta con un aviso — no se
inventa un veredicto verde.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from synaptum import MemoryCheckpointer, ModelStep, Phase, SqliteCheckpointer

_CONTRACTS = Path(
    "/Users/emericespiritusantiago/Documents/Victor/coordinacion_project/contratos"
)
_FIXTURES = _CONTRACTS / "costura-durabilidad" / "fixtures"

pytestmark = pytest.mark.skipif(
    not _FIXTURES.exists(), reason="carpeta de contratos compartidos no disponible"
)


def _load_cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    for path in sorted(_FIXTURES.glob("*.json")):
        document = json.loads(path.read_text())
        for case in document.get("cases", []):
            cases.append((f"{path.stem}::{case['name']}", case))
    return cases


CASES = _load_cases()


def _event(step_id: str, phase: str, payload: dict | None) -> ModelStep:
    """Traduce una operación del caso a nuestro evento tipado.

    El caso habla de ``step_id``, ``phase`` y un ``payload`` opaco.  Nuestro
    evento es más rico, así que el payload viaja en ``meta`` — que es
    precisamente el campo que se propaga sin interpretarse.
    """
    return ModelStep(
        run_id="conformance",
        step_id=step_id,
        step_seq=0,
        phase=Phase(phase),
        meta=payload if payload is not None else {},
    )


async def _run_case(store, case: dict, run_id: str) -> None:
    for index, operation in enumerate(case["operations"]):
        where = f"{case['name']} · operación {index}"
        kind = operation["op"]

        if kind == "append":
            if operation.get("expect_error"):
                with pytest.raises(ValueError):
                    _event(operation["step_id"], operation["phase"], operation.get("payload"))
                continue

            result = await store.append(
                run_id,
                _event(operation["step_id"], operation["phase"], operation.get("payload")),
            )
            expected = operation["expect"]
            assert result.seq == expected["seq"], f"{where}: seq"
            assert result.duplicate is expected["duplicate"], f"{where}: duplicate"
            assert result.payload_diverged is expected["payload_diverged"], (
                f"{where}: payload_diverged"
            )

        elif kind == "load":
            state = await store.load(run_id)
            expected = operation["expect"]
            assert state.next_seq == expected["next_seq"], f"{where}: next_seq"
            if "records" in expected:
                actual = [
                    {
                        "step_id": event.step_id,
                        "phase": event.phase.value,
                        "seq": position,
                        "payload": dict(event.meta),
                    }
                    for position, event in enumerate(state.events)
                ]
                wanted = [
                    {**record, "payload": record.get("payload", {})}
                    for record in expected["records"]
                ]
                assert actual == wanted, f"{where}: records"

        elif kind == "query":
            state = await store.load(run_id)
            expected = operation["expect"]
            step_id = operation["step_id"]
            assert state.completed(step_id) is expected["completed"], f"{where}: completed"
            assert state.attempted(step_id) is expected["attempted"], f"{where}: attempted"

        else:  # pragma: no cover
            pytest.fail(f"{where}: operación desconocida {kind!r}")


@pytest.mark.parametrize("name,case", CASES, ids=[name for name, _ in CASES])
def test_memory_checkpointer_matches_the_golden_case(name: str, case: dict):
    asyncio.run(_run_case(MemoryCheckpointer(), case, "conformance"))


@pytest.mark.parametrize("name,case", CASES, ids=[name for name, _ in CASES])
def test_sqlite_checkpointer_matches_the_golden_case(name: str, case: dict):
    with SqliteCheckpointer() as store:
        asyncio.run(_run_case(store, case, "conformance"))


def test_the_shared_cases_are_actually_being_read():
    """Si la carpeta se mueve, la suite debe fallar en vez de pasar vacía."""
    assert CASES, "no se leyó ningún caso dorado"
