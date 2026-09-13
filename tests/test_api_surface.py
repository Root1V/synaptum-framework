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


def test_the_declared_version_matches_the_packaged_one():
    """``__version__`` y el `pyproject` tienen que decir lo mismo.

    Son dos sitios y se actualizan a mano, así que divergen en cuanto alguien
    toca uno. La consecuencia es fea y silenciosa: un paquete publicado como
    ``1.0.0rc1`` cuyo ``__version__`` dice otra cosa manda a quien depure a la
    versión equivocada, y no falla nunca.
    """
    import tomllib

    raiz = Path(__file__).resolve().parents[1]
    declarada = tomllib.loads((raiz / "pyproject.toml").read_text())["project"]["version"]

    import synaptum

    assert synaptum.__version__ == declarada, (
        f"__version__ dice {synaptum.__version__!r} y el pyproject {declarada!r}"
    )


def test_the_changelog_mentions_the_version_about_to_be_published():
    """Publicar sin entrada de changelog deja a quien actualiza sin saber qué cambió.

    Solo se exige para versiones publicables: mientras se trabaja en una `.devN`
    la entrada vive bajo «No publicado», que es donde debe estar.
    """
    import synaptum

    raiz = Path(__file__).resolve().parents[1]
    changelog = (raiz / "CHANGELOG.md").read_text()

    if ".dev" in synaptum.__version__:
        assert "[No publicado]" in changelog
        return

    assert synaptum.__version__ in changelog, (
        f"{synaptum.__version__} no aparece en CHANGELOG.md"
    )
