"""Genera la versión HTML de `docs/` a partir del Markdown.

**El Markdown es la fuente; el HTML se deriva.** Escribir los dos a mano
garantiza que diverjan, y el que diverge siempre es el que nadie mira. Por eso
hay un test que falla si el HTML no corresponde al Markdown actual.

Las dos salidas son para dos lectores distintos y los dos importan: un agente lee
el `.md` —una página por tema, sin navegación que estorbe— y una persona lee el
`.html`, que sí la necesita.

    uv run python scripts/render_docs.py          # genera
    uv run python scripts/render_docs.py --check  # ¿está al día?

`markdown-it-py` vive en el grupo `dev`: es una herramienta de construcción, no
una dependencia del paquete. Quien instala synaptum no la ve.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ / "docs"
SALIDA = FUENTE / "html"

PLANTILLA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo} · Synaptum</title>
<style>{estilo}</style>
</head>
<body>
<a class="saltar" href="#contenido">Saltar al contenido</a>
<div class="marco">
  <nav aria-label="Secciones">
    <p class="marca"><a href="index.html">Synaptum</a></p>
    <p class="version">{version}</p>
    <ol>{indice}</ol>
    <p class="fuente">También en <a href="{md}">Markdown</a>.</p>
  </nav>
  <main id="contenido">
{cuerpo}
  </main>
</div>
</body>
</html>
"""

ESTILO = """
:root {
  color-scheme: light dark;
  --tinta: #1a1c1e; --fondo: #fdfdfc; --tenue: #5b6066; --linea: #e3e4e6;
  --codigo-fondo: #f5f5f3; --acento: #0b5cad; --aviso: #8a4b00; --aviso-fondo: #fff8ee;
}
@media (prefers-color-scheme: dark) {
  :root {
    --tinta: #e6e6e4; --fondo: #17181a; --tenue: #9aa0a6; --linea: #2c2e31;
    --codigo-fondo: #202225; --acento: #74b0f0; --aviso: #e0a458; --aviso-fondo: #241f17;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--fondo); color: var(--tinta);
  font: 17px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.saltar { position: absolute; left: -9999px; }
.saltar:focus { left: 1rem; top: 1rem; background: var(--fondo); padding: .5rem 1rem; z-index: 9; }
.marco { display: grid; grid-template-columns: 17rem minmax(0, 1fr); gap: 3.5rem;
         max-width: 68rem; margin: 0 auto; padding: 2.5rem 1.5rem 6rem; }
nav { position: sticky; top: 2.5rem; align-self: start; font-size: .92rem; }
.marca { margin: 0; font-weight: 640; letter-spacing: -.01em; font-size: 1.05rem; }
.marca a { color: var(--tinta); text-decoration: none; }
.version { margin: .1rem 0 1.4rem; color: var(--tenue); font-size: .82rem; }
nav ol { list-style: none; margin: 0; padding: 0; counter-reset: s; }
nav li { counter-increment: s; margin: .1rem 0; }
nav a { display: block; padding: .3rem .6rem; margin-left: -.6rem; border-radius: .4rem;
        color: var(--tenue); text-decoration: none; }
nav a:hover { background: var(--codigo-fondo); color: var(--tinta); }
nav a[aria-current] { color: var(--tinta); font-weight: 600; background: var(--codigo-fondo); }
.fuente { margin-top: 1.6rem; color: var(--tenue); font-size: .82rem; }
main { min-width: 0; }
h1 { font-size: 2.1rem; line-height: 1.15; letter-spacing: -.022em; margin: 0 0 1.6rem; }
h2 { font-size: 1.32rem; letter-spacing: -.012em; margin: 3rem 0 .9rem;
     padding-top: 1.4rem; border-top: 1px solid var(--linea); }
h3 { font-size: 1.06rem; margin: 2rem 0 .6rem; }
p, li { max-width: 40rem; }
a { color: var(--acento); }
code { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
       font-size: .875em; background: var(--codigo-fondo); padding: .12em .38em;
       border-radius: .25rem; }
pre { background: var(--codigo-fondo); padding: 1rem 1.15rem; border-radius: .6rem;
      overflow-x: auto; border: 1px solid var(--linea); }
pre code { background: none; padding: 0; font-size: .855rem; line-height: 1.6; }
blockquote { margin: 1.4rem 0; padding: .85rem 1.1rem; border-left: 3px solid var(--aviso);
             background: var(--aviso-fondo); border-radius: 0 .4rem .4rem 0; }
blockquote p { margin: .3rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1.3rem 0; font-size: .93rem;
        display: block; overflow-x: auto; }
th, td { text-align: left; padding: .55rem .8rem; border-bottom: 1px solid var(--linea);
         vertical-align: top; }
th { font-weight: 620; }
hr { border: 0; border-top: 1px solid var(--linea); margin: 2.5rem 0; }
@media (max-width: 62rem) {
  .marco { grid-template-columns: 1fr; gap: 1.5rem; padding-top: 1.5rem; }
  nav { position: static; border-bottom: 1px solid var(--linea); padding-bottom: 1rem; }
  nav ol { columns: 2; }
}
"""


def paginas() -> list[Path]:
    return sorted(p for p in FUENTE.glob("*.md"))


def titulo_de(md: Path) -> str:
    """El primer `# ` del fichero.  Si no hay, el nombre."""
    for linea in md.read_text(encoding="utf-8").splitlines():
        if linea.startswith("# "):
            return linea[2:].strip()
    return md.stem


def _version() -> str:
    texto = (RAIZ / "pyproject.toml").read_text(encoding="utf-8")
    encontrado = re.search(r'^version = "([^"]+)"', texto, re.M)
    return encontrado.group(1) if encontrado else ""


def render(md: Path, todas: list[Path]) -> str:
    from markdown_it import MarkdownIt

    motor = MarkdownIt("commonmark", {"html": False, "linkify": False})
    motor.enable("table").enable("strikethrough")
    cuerpo = motor.render(md.read_text(encoding="utf-8"))

    # Los enlaces entre páginas apuntan al `.md` —que es lo correcto leyendo el
    # repositorio— y aquí se reescriben al `.html` vecino.
    cuerpo = re.sub(r'href="([^":/]+)\.md(#[^"]*)?"', r'href="\1.html\2"', cuerpo)

    indice = "".join(
        '<li><a href="{destino}"{actual}>{nombre}</a></li>'.format(
            destino=f"{otra.stem}.html",
            actual=' aria-current="page"' if otra == md else "",
            nombre=html.escape(titulo_de(otra)),
        )
        for otra in todas
    )

    return PLANTILLA.format(
        titulo=html.escape(titulo_de(md)),
        estilo=ESTILO,
        indice=indice,
        cuerpo=cuerpo,
        md=f"../{md.name}",
        version=html.escape(_version()),
    )


def main() -> int:
    comprobar = "--check" in sys.argv
    todas = paginas()
    if not todas:
        print("No hay páginas en docs/", file=sys.stderr)
        return 1

    SALIDA.mkdir(parents=True, exist_ok=True)
    desfasadas: list[str] = []

    for md in todas:
        destino = SALIDA / f"{md.stem}.html"
        generado = render(md, todas)
        if comprobar:
            actual = destino.read_text(encoding="utf-8") if destino.exists() else ""
            if actual != generado:
                desfasadas.append(destino.name)
        else:
            destino.write_text(generado, encoding="utf-8")

    sobrantes = {p.name for p in SALIDA.glob("*.html")} - {f"{m.stem}.html" for m in todas}
    for nombre in sorted(sobrantes):
        if comprobar:
            desfasadas.append(f"{nombre} (sobra)")
        else:
            (SALIDA / nombre).unlink()

    if comprobar and desfasadas:
        print(
            "El HTML no corresponde al Markdown: " + ", ".join(desfasadas) + "\n"
            "Regenera con: uv run python scripts/render_docs.py",
            file=sys.stderr,
        )
        return 1

    print(f"{'comprobadas' if comprobar else 'generadas'} {len(todas)} páginas en {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
