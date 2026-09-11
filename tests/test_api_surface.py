"""SYN-46 · La superficie pública queda fijada, no recordada.

Aeon congeló su DSL de autoría apoyándose en el compromiso de estabilidad. Una
política que nadie comprueba se erosiona — la lección ya salió tres veces en
este proyecto —, así que la superficie vive en un fichero y cualquier cambio
rompe la suite.

Fallar aquí **no significa que el cambio esté mal**. Significa que hay que
mirarlo: si es aditivo, se actualiza el fichero y sigue; si quita o renombra
algo, va por el canal de coordinación antes de entrar.

    uv run python -c "import synaptum, pathlib, json; \
      pathlib.Path('tests/superficie-publica.json').write_text( \
      json.dumps(sorted(synaptum.__all__), indent=2, ensure_ascii=False) + chr(10))"
"""

from __future__ import annotations

import json
from pathlib import Path

import synaptum

_PINNED = Path(__file__).parent / "superficie-publica.json"

# Lo que Aeon congeló su DSL para usar. Va aparte del resto porque quitar o
# renombrar cualquiera de estos seis no es un cambio de API: es romper una
# decisión que otro equipo ya tomó y no puede revertir.
_AUTHORING = {"Agent", "tool", "Tool", "Session", "Limits", "Risk"}


def test_the_public_surface_matches_what_is_pinned():
    pinned = set(json.loads(_PINNED.read_text()))
    current = set(synaptum.__all__)

    removed = sorted(pinned - current)
    added = sorted(current - pinned)

    assert not removed, (
        f"Símbolos retirados de la superficie pública: {removed}. "
        "Quitar o renombrar va por el canal de coordinación antes de entrar (API.md)."
    )
    assert not added, (
        f"Símbolos nuevos en la superficie pública: {added}. "
        "Si es deliberado, regenera tests/superficie-publica.json."
    )


def test_the_authoring_api_is_all_there():
    """Los seis que sostienen la decisión de Aeon."""
    missing = sorted(_AUTHORING - set(synaptum.__all__))
    assert not missing, f"Falta de la API de autoría: {missing}"


def test_everything_exported_actually_resolves():
    """Un símbolo en `__all__` que no existe rompe `from synaptum import *`."""
    broken = [name for name in synaptum.__all__ if not hasattr(synaptum, name)]
    assert not broken, f"Exportados pero inexistentes: {broken}"


def test_the_permissive_defaults_stay_asymmetric():
    """Permisivo en riesgo, conservador en durabilidad.

    Relajar cualquiera de los dos cambiaría en silencio el comportamiento de
    código ya escrito: una tool declarada ayer pasaría a ejecutarse con menos
    control o a perder su garantía de durabilidad sin que nadie tocara nada.
    """
    from synaptum import Risk, ToolDefinition

    default = ToolDefinition(name="cualquiera")
    assert default.risk is Risk.READ, "el riesgo por defecto es el más inocuo"
    assert default.idempotent is False, "quien calla paga durabilidad"
