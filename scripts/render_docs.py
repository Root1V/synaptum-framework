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
    <p class="fuente"><a href="{md}">Leer en Markdown&nbsp;↗</a></p>
  </nav>
  <main id="contenido">
{cuerpo}
  </main>
</div>
</body>
</html>
"""

ESTILO = """
/* Paleta ───────────────────────────────────────────────────────────────────
   Los colores hacen un trabajo, no decoran: el acento marca lo navegable, el
   ámbar marca lo que hay que leer despacio, y el código tiene su propia escala
   para que la sintaxis se distinga sin gritar. Todo con contraste suficiente
   en claro y en oscuro — una documentación que solo se ve bien en un tema es
   una documentación que la mitad de la gente lee peor. */
:root {
  color-scheme: light dark;
  --tinta: #16181d; --fondo: #fcfcfb; --papel: #ffffff;
  --tenue: #5c6672; --linea: #e4e6ea; --linea-suave: #eef0f3;
  --acento: #0a5ac4; --acento-suave: #eef4fd;
  --aviso: #9a5b00; --aviso-borde: #e8a33d; --aviso-fondo: #fff8ec;
  --codigo-fondo: #f7f8fa; --codigo-borde: #e6e9ee; --codigo-tinta: #2a2f3a;
  --etiqueta: #8b94a0;
  --sx-clave: #9226a8;   /* def, class, return, async  */
  --sx-cadena: #0a6b3d;  /* "texto"                    */
  --sx-func: #1a55c4;    /* nombres de función          */
  --sx-tipo: #0b6d8c;    /* int, str, builtins          */
  --sx-num: #a8471a;     /* 42, 0.5                     */
  --sx-com: #6a7482;     /* # comentarios               */
  --sx-dec: #a8471a;     /* @tool                       */
}
@media (prefers-color-scheme: dark) {
  :root {
    --tinta: #e4e6ea; --fondo: #14161a; --papel: #191c21;
    --tenue: #98a1ad; --linea: #2a2e35; --linea-suave: #23262c;
    --acento: #6fa8f5; --acento-suave: #1a2433;
    --aviso: #e7ab5a; --aviso-borde: #8a6a2f; --aviso-fondo: #221d14;
    --codigo-fondo: #1b1e24; --codigo-borde: #2a2e35; --codigo-tinta: #d4d8de;
    --etiqueta: #6e7682;
    --sx-clave: #d79bec; --sx-cadena: #7ec99b; --sx-func: #7fb0f2;
    --sx-tipo: #5fc0dd; --sx-num: #e0a07a; --sx-com: #79828f; --sx-dec: #e0a07a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--fondo); color: var(--tinta);
  font: 17px/1.68 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
  -webkit-text-size-adjust: 100%; -webkit-font-smoothing: antialiased;
}
.saltar { position: absolute; left: -9999px; }
.saltar:focus { left: 1rem; top: 1rem; background: var(--papel); padding: .5rem 1rem;
                z-index: 9; border-radius: .4rem; border: 1px solid var(--acento); }
.marco { display: grid; grid-template-columns: 17.5rem minmax(0, 1fr); gap: 3.5rem;
         max-width: 70rem; margin: 0 auto; padding: 2.5rem 1.5rem 7rem; }

/* Navegación ─────────────────────────────────────────────────────────────── */
nav { position: sticky; top: 2.5rem; align-self: start; font-size: .92rem; }
.marca { margin: 0; font-weight: 650; letter-spacing: -.015em; font-size: 1.12rem; }
.marca a { color: var(--tinta); text-decoration: none; }
.marca a::before {
  content: ""; display: inline-block; width: .5rem; height: .5rem; margin-right: .5rem;
  border-radius: 50%; background: var(--acento); vertical-align: .08em;
}
.version { margin: .15rem 0 1.5rem 1rem; color: var(--etiqueta); font-size: .78rem;
           letter-spacing: .02em; font-variant-numeric: tabular-nums; }
nav ol { list-style: none; margin: 0; padding: 0; }
nav li { margin: .08rem 0; }
nav a { display: block; padding: .34rem .7rem; margin-left: -.7rem; border-radius: .45rem;
        color: var(--tenue); text-decoration: none; border-left: 2px solid transparent;
        transition: background .12s, color .12s; }
nav a:hover { background: var(--linea-suave); color: var(--tinta); }
nav a[aria-current] { color: var(--acento); font-weight: 600; background: var(--acento-suave);
                      border-left-color: var(--acento); }
.fuente { margin-top: 1.8rem; padding-top: 1rem; border-top: 1px solid var(--linea);
          font-size: .82rem; }
.fuente a { color: var(--tenue); text-decoration: none; }
.fuente a:hover { color: var(--acento); text-decoration: underline; }

/* Texto ──────────────────────────────────────────────────────────────────── */
main { min-width: 0; }
h1 { font-size: 2.25rem; line-height: 1.12; letter-spacing: -.028em; margin: 0 0 1.7rem;
     font-weight: 700; }
h2 { font-size: 1.36rem; letter-spacing: -.016em; margin: 3.2rem 0 1rem; font-weight: 650;
     padding-top: 1.5rem; border-top: 1px solid var(--linea); }
h3 { font-size: 1.07rem; margin: 2.1rem 0 .6rem; font-weight: 640; color: var(--tinta); }
p, li { max-width: 41rem; }
li { margin: .3rem 0; }
strong { font-weight: 640; }
a { color: var(--acento); text-decoration-thickness: 1px; text-underline-offset: .15em; }
a:hover { text-decoration-thickness: 2px; }

/* Código ─────────────────────────────────────────────────────────────────── */
code { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
       font-size: .87em; background: var(--codigo-fondo); color: var(--codigo-tinta);
       padding: .13em .4em; border-radius: .3rem; border: 1px solid var(--codigo-borde); }
pre { position: relative; background: var(--codigo-fondo); color: var(--codigo-tinta);
      padding: 1.1rem 1.2rem; border-radius: .7rem; overflow-x: auto;
      border: 1px solid var(--codigo-borde); margin: 1.3rem 0; }
pre code { background: none; padding: 0; border: 0; font-size: .855rem; line-height: 1.62; }
pre[data-lenguaje]::after {
  content: attr(data-lenguaje); position: absolute; top: .55rem; right: .8rem;
  font: 600 .66rem/1 ui-monospace, monospace; letter-spacing: .07em;
  text-transform: uppercase; color: var(--etiqueta); pointer-events: none;
}
.p-k, .p-kn, .p-kd, .p-kc, .p-ow { color: var(--sx-clave); font-weight: 600; }
.p-s, .p-s1, .p-s2, .p-sd, .p-se, .p-sa, .p-si { color: var(--sx-cadena); }
.p-nf, .p-fm { color: var(--sx-func); font-weight: 600; }
.p-nb, .p-nc, .p-ne, .p-nn { color: var(--sx-tipo); }
.p-m, .p-mi, .p-mf, .p-kt { color: var(--sx-num); }
.p-c, .p-c1, .p-cm, .p-ch, .p-cs { color: var(--sx-com); font-style: italic; }
.p-nd { color: var(--sx-dec); font-weight: 600; }
.p-err { color: inherit; }

/* Aviso ──────────────────────────────────────────────────────────────────── */
blockquote { margin: 1.5rem 0; padding: .95rem 1.2rem; border-left: 3px solid var(--aviso-borde);
             background: var(--aviso-fondo); border-radius: 0 .5rem .5rem 0; color: var(--tinta); }
blockquote p { margin: .3rem 0; max-width: none; }
blockquote strong { color: var(--aviso); }

/* Tablas ─────────────────────────────────────────────────────────────────── */
.tabla { overflow-x: auto; margin: 1.4rem 0; border: 1px solid var(--linea);
         border-radius: .6rem; background: var(--papel); }
table { border-collapse: collapse; width: 100%; font-size: .93rem; }
th, td { text-align: left; padding: .6rem .9rem; border-bottom: 1px solid var(--linea-suave);
         vertical-align: top; }
thead th { font-weight: 640; background: var(--linea-suave); color: var(--tinta);
           border-bottom: 1px solid var(--linea); }
tbody tr:last-child td { border-bottom: 0; }
td code, th code { background: transparent; border: 0; padding: 0; color: var(--sx-func); }

hr { border: 0; border-top: 1px solid var(--linea); margin: 2.8rem 0; }

@media (max-width: 62rem) {
  .marco { grid-template-columns: 1fr; gap: 1.5rem; padding-top: 1.5rem; }
  nav { position: static; border-bottom: 1px solid var(--linea); padding-bottom: 1rem; }
  nav ol { columns: 2; column-gap: 1.5rem; }
  h1 { font-size: 1.85rem; }
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


def _resaltar(codigo: str, lenguaje: str, _atributos: str) -> str:
    """Resalta un bloque con Pygments, en tiempo de construcción.

    Se hace aquí y no en el navegador a propósito: nada de JavaScript ni de CDN,
    así que la página se lee igual sin red y no hay un recurso de terceros
    decidiendo cómo se ve la documentación.

    Un lenguaje que no conocemos no es un error — se devuelve escapado y
    legible, que es mejor que romper la página por un bloque mal etiquetado.
    """
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound

    escapado = html.escape(codigo)
    if not lenguaje:
        return f"<pre><code>{escapado}</code></pre>"
    try:
        lexer = get_lexer_by_name(lenguaje, stripall=False)
    except ClassNotFound:
        return f'<pre><code class="lenguaje-{html.escape(lenguaje)}">{escapado}</code></pre>'

    marcado = highlight(codigo, lexer, HtmlFormatter(nowrap=True, classprefix="p-"))
    return f'<pre data-lenguaje="{html.escape(lenguaje)}"><code>{marcado}</code></pre>'


def render(md: Path, todas: list[Path]) -> str:
    from markdown_it import MarkdownIt

    motor = MarkdownIt(
        "commonmark", {"html": False, "linkify": False, "highlight": _resaltar}
    )
    motor.enable("table").enable("strikethrough")
    cuerpo = motor.render(md.read_text(encoding="utf-8"))

    # Los enlaces entre páginas apuntan al `.md` —que es lo correcto leyendo el
    # repositorio— y aquí se reescriben al `.html` vecino.
    cuerpo = re.sub(r'href="([^":/]+)\.md(#[^"]*)?"', r'href="\1.html\2"', cuerpo)

    # Una tabla ancha necesita su propio scroll, y el scroll necesita un borde
    # que diga dónde acaba. Sin envoltorio, la tabla desborda la página entera.
    cuerpo = cuerpo.replace("<table>", '<div class="tabla"><table>').replace(
        "</table>", "</table></div>"
    )

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
