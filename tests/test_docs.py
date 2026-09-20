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
    for modulo in ("synaptum", "synaptum.testing", "synaptum.mcp", "synaptum.a2a"):
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


#: La referencia se genera **de** los símbolos, así que no puede nombrar uno que
#: no exista — y cita en prosa tipos de la biblioteca estándar que este test
#: tomaría por símbolos nuestros. Comprobarla aquí sería comprobar el generador.
_ESCRITAS_A_MANO = [p for p in paginas() if p.stem != "08-referencia"]


@pytest.mark.parametrize("pagina", _ESCRITAS_A_MANO, ids=lambda p: p.stem)
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
    """Una página que no se enlaza no existe para quien lee.

    Una página de sección —`ejemplos-04-…`— cuenta como enlazada desde la
    portada de su sección: pedir que las dieciséis estén en `index.md` haría de
    la portada un listado, que es justo lo que la sección evita.
    """
    huerfanas = []
    for pagina in paginas():
        if pagina.name == "index.md":
            continue
        grupo = pagina.stem.split("-")[0]
        portada = DOCS / f"{grupo}.md"
        desde = portada if grupo != pagina.stem and portada.exists() else DOCS / "index.md"
        if f"({pagina.name})" not in desde.read_text(encoding="utf-8"):
            huerfanas.append(f"{pagina.name} (no la enlaza {desde.name})")
    assert not huerfanas, f"páginas sin enlazar: {huerfanas}"


def _fuentes_que_pueden_caducar() -> list[Path]:
    """Documentación y ejemplos: los dos envejecen igual.

    Los ejemplos se quedaron fuera de esta comprobación al principio, y tres de
    ellos acabaron afirmando que `delegate()` no existía **después** de que
    existiera. Un ejemplo es documentación que además se ejecuta, así que
    caduca por las mismas dos vías — y la que no se ejecuta no la ve nadie.
    """
    ejemplos = sorted((RAIZ / "examples").rglob("*.py")) if (RAIZ / "examples").exists() else []
    readmes = sorted((RAIZ / "examples").rglob("README.md")) if (RAIZ / "examples").exists() else []
    return [*paginas(), *ejemplos, *readmes]


def test_the_roadmap_items_the_docs_cite_are_still_open():
    """Documentar algo como pendiente cuando ya está hecho envía a nadie a ninguna parte."""
    roadmap = (RAIZ / "roadmap.md").read_text(encoding="utf-8")
    citados = {
        m.group(1)
        for fuente in _fuentes_que_pueden_caducar()
        for m in re.finditer(r"`(SYN-\d+)`", fuente.read_text(encoding="utf-8"))
    }
    assert citados, "ninguna página cita un elemento del roadmap"

    ya_hechos = [
        codigo for codigo in sorted(citados)
        if re.search(rf"\|\s*{codigo}\s*\|\s*`HECHO`", roadmap)
    ]
    assert not ya_hechos, (
        f"la documentación los da por pendientes y el roadmap dice HECHO: {ya_hechos}"
    )


def test_every_example_the_docs_point_to_exists():
    """Una ruta de ejemplo que no existe manda a alguien a un 404 con su primer comando.

    El README llevaba tiempo diciendo `examples/01_agente.py`, que fue cierto
    hasta que los ejemplos se repartieron en dos pistas. Nadie lo vio porque
    ningún test mira las rutas que la documentación promete: el comando de
    «pruébalo» del README era el único que no se ejecutaba nunca.
    """
    fuentes = [RAIZ / "README.md", *paginas(), RAIZ / "examples" / "README.md"]
    rotas = {
        f"{fuente.name}: {ruta}"
        for fuente in fuentes
        if fuente.exists()
        for ruta in re.findall(r"examples/[\w/]+\.py", fuente.read_text(encoding="utf-8"))
        if not (RAIZ / ruta).exists()
    }
    assert not rotas, f"la documentación apunta a ejemplos que no existen: {sorted(rotas)}"


def test_the_reference_matches_the_code():
    """La referencia se genera del código; si el código cambia, se regenera.

    Escribirla a mano sería escribir algo que caduca — y una referencia caducada
    manda a buscar un símbolo que ya no está.

    Si falla: `uv run python scripts/render_reference.py`
    """
    generado = DOCS / "08-referencia.md"
    antes = generado.read_text(encoding="utf-8") if generado.exists() else ""

    resultado = subprocess.run(
        [sys.executable, "scripts/render_reference.py"],
        cwd=RAIZ, capture_output=True, text=True,
    )
    assert resultado.returncode == 0, resultado.stderr

    despues = generado.read_text(encoding="utf-8")
    if antes != despues:
        generado.write_text(antes, encoding="utf-8")   # no dejar el árbol tocado
        pytest.fail(
            "docs/08-referencia.md no corresponde al código actual. "
            "Regenera con: uv run python scripts/render_reference.py"
        )


def test_the_example_pages_match_the_examples():
    """Las páginas de la sección Ejemplos se generan de `examples/`.

    Copiar un ejemplo a una página crea dos originales que envejecen por
    separado, y el que envejece es el que nadie ejecuta. Aquí la fuente es el
    fichero que corre.

    Si falla: `uv run python scripts/render_examples.py`
    """
    resultado = subprocess.run(
        [sys.executable, "scripts/render_examples.py", "--check"],
        cwd=RAIZ, capture_output=True, text=True,
    )
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr


def test_every_public_symbol_appears_in_the_reference():
    """Si algo es público y no está documentado, no es usable sin leer el código."""
    import synaptum

    texto = (DOCS / "08-referencia.md").read_text(encoding="utf-8")
    ausentes = [n for n in synaptum.__all__ if f"### `{n}`" not in texto]
    assert not ausentes, f"exportados y sin documentar: {ausentes}"
