"""La documentación no puede mentir, y las dos versiones no pueden diverger.

Dos riesgos distintos y los dos son reales:

* **Divergencia.** El HTML se genera del Markdown. Escribir los dos a mano
  garantiza que se separen, y el que diverge siempre es el que nadie mira.
* **Documentación caducada.** Una guía que nombra símbolos que ya no existen es
  peor que no tenerla: manda a quien la lee a buscar algo que no está, y le hace
  dudar de lo que sí está.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "docs"

pytestmark = pytest.mark.skipif(not DOCS.exists(), reason="sin docs/")


def paginas() -> list[Path]:
    return sorted(DOCS.glob("*.md"))


def _simbolos_publicos() -> set[str]:
    """Todo lo que el paquete expone, no solo el `__all__` de la raíz.

    Los dobles viven en `synaptum.testing` y el cliente MCP en `synaptum.mcp`:
    una lista de excepciones a mano crecería con cada módulo nuevo y acabaría
    tapando justamente lo que este test busca.
    """
    import builtins
    import importlib

    nombres = set(dir(builtins))
    for modulo in ("synaptum", "synaptum.testing", "synaptum.mcp"):
        try:
            cargado = importlib.import_module(modulo)
        except ImportError:      # un extra que no está instalado
            continue
        nombres |= set(getattr(cargado, "__all__", ())) | set(dir(cargado))
    return nombres


#: Nombres propios que aparecen en prosa y nunca serán símbolos del paquete.
_NO_SON_SIMBOLOS = {
    "Markdown", "MCP", "Python", "JSON", "OpenAI", "HTTP", "GPU", "CI", "SQLite",
    "Postgres", "Redis", "LangGraph", "Temporal", "Synaptum", "Retry", "Annotated",
    "Protocol", "Provider", "Ollama", "Docker",
}


def test_the_html_matches_the_markdown():
    """Si falla, regenera: `uv run python scripts/render_docs.py`."""
    resultado = subprocess.run(
        [sys.executable, "scripts/render_docs.py", "--check"],
        cwd=RAIZ, capture_output=True, text=True,
    )
    assert resultado.returncode == 0, resultado.stderr


@pytest.mark.parametrize("pagina", paginas(), ids=lambda p: p.stem)
def test_every_symbol_the_docs_name_actually_exists(pagina: Path):
    """Los nombres entre comillas invertidas que parecen símbolos, existen.

    Se mira solo lo que tiene forma de símbolo público nuestro —`Agent`,
    `Risk.DESTRUCTIVE`, `MemoryCheckpointer`— y no cada trozo de código, porque
    la alternativa sería un analizador y esto es un guardarraíl.
    """
    publicos = _simbolos_publicos()
    texto = pagina.read_text(encoding="utf-8")

    # Fuera los bloques de código: ahí hay nombres de ejemplo que no son nuestros.
    sin_bloques = re.sub(r"```.*?```", "", texto, flags=re.S)

    sospechosos = {
        m.group(1)
        for m in re.finditer(r"`([A-Z][A-Za-z]+)(?:\.[A-Z_]+)?`", sin_bloques)
    }
    desconocidos = {s for s in sospechosos - _NO_SON_SIMBOLOS if s not in publicos}

    assert not desconocidos, (
        f"{pagina.name} nombra símbolos que no exporta el paquete: {sorted(desconocidos)}"
    )


def test_every_page_is_linked_from_the_index():
    """Una página que no se enlaza no existe para quien lee."""
    indice = (DOCS / "index.md").read_text(encoding="utf-8")
    huerfanas = [
        p.name for p in paginas() if p.name != "index.md" and f"({p.name})" not in indice
    ]
    assert not huerfanas, f"páginas sin enlazar desde el índice: {huerfanas}"


def test_the_roadmap_items_the_docs_cite_are_still_open():
    """Documentar algo como pendiente cuando ya está hecho envía a nadie a ninguna parte."""
    roadmap = (RAIZ / "roadmap.md").read_text(encoding="utf-8")
    citados = {
        m.group(1)
        for pagina in paginas()
        for m in re.finditer(r"`(SYN-\d+)`", pagina.read_text(encoding="utf-8"))
    }
    assert citados, "ninguna página cita un elemento del roadmap"

    ya_hechos = [
        codigo for codigo in sorted(citados)
        if re.search(rf"\|\s*{codigo}\s*\|\s*`HECHO`", roadmap)
    ]
    assert not ya_hechos, (
        f"la documentación los da por pendientes y el roadmap dice HECHO: {ya_hechos}"
    )
