"""Genera la sección **Ejemplos** de la documentación a partir de `examples/`.

Los ejemplos ya eran la mejor documentación que hay en el repositorio, y estaban
solo en el repositorio. Esto los pone en el sitio.

**Se generan, no se copian.** Copiar un ejemplo a una página crea dos originales
que envejecen por separado, y el que envejece es siempre el que nadie ejecuta:
pasó exactamente eso con tres ejemplos que afirmaron durante días que
`delegate()` no existía. Aquí la fuente sigue siendo el fichero que corre, y un
test falla si las páginas no corresponden a él.

Qué hace con cada fichero:

* la **cadena de documentación** del módulo pasa a ser la entradilla de la
  página — es donde cada ejemplo dice de qué va y sobre qué dominio;
* los separadores `# ── Título ──` se convierten en secciones, así que el código
  llega troceado y no como un muro de doscientas líneas;
* el bloque de comentarios del final —«lo que esto enseña»— sube a prosa, que es
  lo que es. Dejarlo dentro de un `<pre>` lo escondía en gris;
* y **cada ejemplo se ejecuta aquí**, para que la página enseñe lo que imprime
  de verdad. Una salida copiada a mano es una captura de pantalla vieja: se
  queda igual mientras el programa cambia, y nadie la compara nunca.

Todo lo demás se transcribe **tal cual**. Una página que reordena o resume el
código deja de servir para lo único que importa aquí: leerlo y correrlo.

    uv run python scripts/render_examples.py
    uv run python scripts/render_examples.py --check
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLOS = RAIZ / "examples"
DOCS = RAIZ / "docs"

BLOB = "https://github.com/Root1V/synaptum-framework/blob/main"
ARBOL = "https://github.com/Root1V/synaptum-framework/tree/main"

#: Las dos pistas, en orden de lectura, y numeradas de corrido.
#:
#: La numeración continúa entre pistas a propósito: son direcciones de páginas y
#: tienen que ser estables. Reiniciarla en la segunda daría dos `ejemplos-01`.
PISTAS = [
    ("agentes", "Construir agentes"),
    ("propiedades", "Qué garantiza el runtime"),
]

SEPARADOR = re.compile(r"^# ── (.+?) ─+\s*$")

#: Ejemplos que necesitan un extra para correr, y cuál.
#:
#: Donde el extra no está —en el CI, que instala solo el núcleo— el ejemplo no
#: se ejecuta y se **conserva** la salida que ya tenía la página. Regenerarla a
#: ciegas la borraría; volver a ejecutarla sin el extra sería inventarla.
REQUIERE_EXTRA = {"08_herramientas_mcp.py": "mcp"}

#: Variables que se quitan antes de ejecutar un ejemplo.
#:
#: Con credenciales en el entorno, los ejemplos hablan con un modelo de verdad,
#: y entonces la salida cambia en cada ejecución y cuesta dinero. La página
#: enseña lo que imprime el doble, que es lo que verá quien lo clone.
DEL_MODELO = (
    "AXONIUM_CLIENT_ID", "AXONIUM_CLIENT_SECRET",
    "SYNAPTUM_BASE_URL", "SYNAPTUM_API_KEY", "SYNAPTUM_MODEL",
)

#: Cuánto se le da a un ejemplo para terminar. Ninguno pasa de un segundo con el
#: doble; el tope está para que un cuelgue falle y no deje el CI colgado.
PLAZO = 120


def salida_de(fichero: Path) -> str | None:
    """Ejecuta el ejemplo y devuelve lo que imprimió.

    ``None`` cuando no se puede ejecutar aquí por un extra que falta — entonces
    quien llama conserva lo que ya había.
    """
    extra = REQUIERE_EXTRA.get(fichero.name)
    if extra and importlib.util.find_spec(extra) is None:
        return None

    entorno = {k: v for k, v in os.environ.items() if k not in DEL_MODELO}
    proceso = subprocess.run(
        [sys.executable, str(fichero)],
        capture_output=True, text=True, timeout=PLAZO, env=entorno, cwd=RAIZ,
    )
    if proceso.returncode != 0:
        # Ruidoso a propósito: un ejemplo que no corre es un ejemplo roto, y
        # publicar su página sin salida lo escondería.
        raise SystemExit(
            f"{fichero.relative_to(RAIZ)} falló al ejecutarse "
            f"(código {proceso.returncode}):\n{proceso.stdout}{proceso.stderr}"
        )
    return proceso.stdout.strip("\n")


def salida_previa(pagina: Path) -> str:
    """Rescata el bloque de salida de una página ya generada."""
    if not pagina.exists():
        return ""
    encontrado = re.search(
        r"## Lo que imprime\n\n```text\n(.*?)\n```", pagina.read_text(encoding="utf-8"), re.S
    )
    return encontrado.group(1) if encontrado else ""


def fuentes() -> list[tuple[int, Path, str]]:
    """`(número, fichero, pista)` en orden de lectura."""
    salida: list[tuple[int, Path, str]] = []
    for carpeta, pista in PISTAS:
        for fichero in sorted((EJEMPLOS / carpeta).glob("[0-9]*.py")):
            salida.append((len(salida) + 1, fichero, pista))
    return salida


def pagina_de(numero: int, fichero: Path) -> str:
    """`ejemplos-04-agente-que-gasta.md` — el nombre de la página de un fichero."""
    apodo = fichero.stem.split("_", 1)[1].replace("_", "-")
    return f"ejemplos-{numero:02d}-{apodo}.md"


def dominio_de(doc: str) -> str:
    """El dominio real sobre el que está montado el ejemplo, si lo declara.

    Se saca de la entradilla —`**Dominio: Argus** — …`— en vez de mantener una
    tabla aparte: la tabla se quedaría atrás en cuanto un ejemplo cambiara de
    dominio, y nadie compara una tabla con doce docstrings.
    """
    marca = re.search(r"\*\*Dominio: (.+?)\*\*", doc)
    if marca is None:
        return ""
    return marca.group(1).strip().rstrip(".:")


def titulo_de(numero: int, doc: str) -> str:
    """La primera línea de la cadena de documentación, con su número delante.

    Los de `agentes/` ya lo traen (`01 · …`) y los de `propiedades/` no. Se
    normaliza aquí para que el menú no mezcle dos formas.
    """
    primera = doc.splitlines()[0].strip().rstrip(".")
    sin_numero = re.sub(r"^\d+\s*·\s*", "", primera)
    return f"{numero:02d} · {sin_numero}"


# ── Del fichero a la página ───────────────────────────────────────────────────

def _entradilla(doc: str, enlaces: dict[str, str]) -> list[str]:
    """La cadena de documentación, sin su primera línea, ya en Markdown.

    La invocación (`uv run python …`) se saca de aquí: tiene su propia sección,
    con la salida pegada debajo. Dentro del docstring iba sangrada, que en
    Markdown es un bloque sin lenguaje — ni resaltado ni botón de copiar, justo
    en la línea que más se copia.
    """
    salida = [
        _enlaces(linea, enlaces)
        for linea in doc.splitlines()[1:]
        if not linea.strip().startswith("uv run ")
    ]
    return _sin_bordes(salida)


def _enlaces(texto: str, enlaces: dict[str, str]) -> str:
    """Los enlaces entre ejemplos apuntan al fichero vecino; aquí, a su página."""
    for fichero, pagina in enlaces.items():
        texto = texto.replace(f"]({fichero})", f"]({pagina})")
    return texto


def _secciones(cuerpo: str) -> list[tuple[str | None, str]]:
    """Trocea el código por los separadores `# ── … ──`.

    Un ejemplo sin separadores —los de `propiedades/` no los usan— devuelve un
    solo trozo sin título, y la página sale con un único bloque. Forzar
    secciones donde el autor no las puso sería inventarse una estructura.
    """
    trozos: list[tuple[str | None, str]] = []
    titulo: str | None = None
    actual: list[str] = []

    for linea in cuerpo.splitlines():
        marca = SEPARADOR.match(linea)
        if marca:
            if "".join(actual).strip():
                trozos.append((titulo, "\n".join(actual).strip("\n")))
            titulo, actual = marca.group(1).strip(), []
        else:
            actual.append(linea)

    if "".join(actual).strip():
        trozos.append((titulo, "\n".join(actual).strip("\n")))
    return trozos


def _cabecera(codigo: str) -> tuple[list[str], str]:
    """Separa el comentario que abre una sección del código que viene detrás.

    Los ejemplos explican cada sección justo debajo del separador, antes de la
    primera línea de código. Eso es prosa, no código: dejarlo dentro del `<pre>`
    lo pinta en gris al lado de lo que describe.
    """
    lineas = codigo.splitlines()
    corte = 0
    while corte < len(lineas) and (
        lineas[corte].startswith("#") or not lineas[corte].strip()
    ):
        corte += 1

    comentario = [linea for linea in lineas[:corte] if linea.startswith("#")]
    if len(comentario) < 2:               # una línea suelta se queda con su código
        return [], codigo
    return comentario, "\n".join(lineas[corte:]).strip("\n")


def _cola(codigo: str) -> tuple[str, list[str]]:
    """Separa el bloque de comentarios final del código que lo precede.

    Es el trozo más valioso de cada ejemplo —lo que enseña, dicho después de
    verlo funcionar— y dentro de un bloque de código se lee como un comentario
    más. Devuelve `(código, líneas de la cola)`.
    """
    lineas = codigo.splitlines()
    try:
        fin = next(i for i, linea in enumerate(lineas) if linea.startswith("if __name__"))
    except StopIteration:
        return codigo, []

    inicio = fin
    while inicio > 0 and not lineas[inicio - 1].strip():
        inicio -= 1
    cola_inicio = inicio
    while cola_inicio > 0 and lineas[cola_inicio - 1].startswith("#"):
        cola_inicio -= 1

    if inicio - cola_inicio < 2:          # una línea suelta no es prosa
        return codigo, []

    resto = lineas[:cola_inicio] + lineas[inicio:]
    return "\n".join(resto).strip("\n"), lineas[cola_inicio:inicio]


def _prosa(comentarios: list[str], enlaces: dict[str, str]) -> list[str]:
    """Convierte un bloque de comentarios en Markdown.

    Lo sangrado dentro del comentario —una lista de piezas, una regla citada—
    se queda en un bloque preformateado: en prosa, Markdown junta esas líneas
    en un párrafo y lo que estaba alineado deja de estarlo.
    """
    salida: list[str] = []
    preformateado: list[str] = []

    def volcar() -> None:
        while preformateado and not preformateado[-1].strip():
            preformateado.pop()
        if not preformateado:
            return
        # Se quita el margen **común**, no el de cada línea: recortar una a una
        # alinearía a la izquierda lo que el autor había escalonado.
        margen = min(
            len(t) - len(t.lstrip()) for t in preformateado if t.strip()
        )
        salida.extend(["```text", *(t[margen:] for t in preformateado), "```", ""])
        preformateado.clear()

    for linea in comentarios:
        texto = linea[1:]
        texto = texto[1:] if texto.startswith(" ") else texto
        if texto.startswith("  ") and texto.strip():
            preformateado.append(texto)
            continue
        if texto.strip() and preformateado:
            volcar()
        if not texto.strip() and preformateado:
            preformateado.append(texto)
            continue
        salida.append(_enlaces(texto, enlaces))

    volcar()
    return _sin_bordes(salida)


def _compacto(codigo: str) -> str:
    """Quita los huecos de tres líneas que deja recortar los comentarios."""
    return re.sub(r"\n{3,}", "\n\n", codigo).strip("\n")


def _sin_bordes(lineas: list[str]) -> list[str]:
    while lineas and not lineas[0].strip():
        lineas.pop(0)
    while lineas and not lineas[-1].strip():
        lineas.pop()
    return lineas


def render(numero: int, fichero: Path, enlaces: dict[str, str], salida: str) -> str:
    fuente = fichero.read_text(encoding="utf-8")
    modulo = ast.parse(fuente)
    doc = ast.get_docstring(modulo, clean=True) or fichero.stem
    relativa = fichero.relative_to(RAIZ).as_posix()

    fin_doc = modulo.body[0].end_lineno if modulo.body else 0
    cuerpo = "\n".join(fuente.splitlines()[fin_doc:]).strip("\n")

    lineas = [
        f"# {titulo_de(numero, doc)}",
        "",
        f"> **Esto es un fichero que se ejecuta:** [`{relativa}`]({BLOB}/{relativa}) ↗",
        "> Esta página lo transcribe y enseña lo que imprime. Si dejan de coincidir, falla un test.",
        "",
        *_entradilla(doc, enlaces),
        "",
        "## Cómo correrlo",
        "",
    ]

    extra = REQUIERE_EXTRA.get(fichero.name)
    lineas += ["```bash"]
    if extra:
        lineas += [f"uv sync --extra {extra}"]
    lineas += [f"uv run python {relativa}", "```", ""]
    lineas += [
        "No hace falta configurar nada: sin modelo, las respuestas van guionizadas y **todo lo",
        "demás es real** — las herramientas se ejecutan, el journal se escribe, el consumo se mide.",
        "Con `AXONIUM_CLIENT_ID` o `SYNAPTUM_BASE_URL` en el entorno, **el mismo fichero sin tocar**",
        "habla con un modelo de verdad; lo que cambia entonces es lo que diga el modelo, no el",
        "código. Ver [Modelos](04-modelos.md).",
        "",
    ]

    if salida:
        lineas += ["## Lo que imprime", "", "```text", salida, "```", ""]

    for titulo, codigo in _secciones(cuerpo):
        cabecera, codigo = _cabecera(codigo)
        codigo, cola = _cola(codigo)
        if titulo:
            lineas += [f"## {titulo}", ""]
        if cabecera:
            lineas += [*_prosa(cabecera, enlaces), ""]
        if codigo.strip():
            lineas += ["```python", _compacto(codigo), "```", ""]
        if cola:
            lineas += ["## Lo que esto enseña", "", *_prosa(cola, enlaces), ""]

    lineas += [
        "---",
        "",
        f"**El fichero entero, para clonarlo y tocarlo:** [`{relativa}`]({BLOB}/{relativa}) ↗",
        "",
        f"Está en [`examples/`]({ARBOL}/examples) con los otros quince, y todos corren igual.",
    ]
    return "\n".join(lineas).rstrip() + "\n"


# ── El índice de la sección ───────────────────────────────────────────────────

def render_indice(entradas: list[tuple[int, Path, str, str, str]]) -> str:
    lineas = [
        "# Ejemplos",
        "",
        "**Dieciséis ficheros que se ejecutan.** Corren sin inferencia y sin configurar nada: las",
        "respuestas van guionizadas y todo lo demás es real — las herramientas se ejecutan, el",
        "journal se escribe, el consumo se mide. Con dos variables de entorno, **el mismo fichero",
        "sin tocar** habla con un modelo de verdad.",
        "",
        "Cada uno está sobre un proyecto real, no sobre un dominio inventado, porque un ejemplo con",
        "`foo` y `bar` enseña la sintaxis y esconde la decisión.",
        "",
        "Cada página trae el fichero entero, **lo que imprime al correrlo** —capturado ejecutándolo,",
        "no escrito a mano— y el enlace a GitHub para clonarlo.",
        "",
        "```bash",
        "uv run python examples/agentes/01_triaje.py",
        "```",
        "",
    ]

    for _, titulo_pista in PISTAS:
        del_pista = [e for e in entradas if e[2] == titulo_pista]
        if not del_pista:
            continue
        lineas += [f"## {titulo_pista}", ""]
        if titulo_pista == "Construir agentes":
            lineas += ["Cada uno añade **una** idea sobre el anterior.", ""]
        else:
            lineas += [
                "Estos no enseñan a escribir un agente: enseñan qué hay debajo cuando ya lo has",
                "escrito.",
                "",
            ]
        lineas += ["| | Dominio |", "|---|---|"]
        lineas += [
            f"| [{titulo}]({pagina_de(numero, fichero)}) | {dominio or '—'} |"
            for numero, fichero, _, titulo, dominio in del_pista
        ]
        lineas += [""]

    lineas += [
        "## Lo que estos ejemplos no enseñan",
        "",
        "`LocalGateway` **no aplica nada**. Corre dentro del proceso que gobernaría, así que sus",
        "comprobaciones son advisorias: las cumple un bucle correcto y se las salta uno con un",
        "fallo. Que el `04` deniegue un pago aquí no dice nada sobre si lo denegaría en producción",
        "— para eso la decisión tiene que tomarse fuera del proceso, que es lo que hace un arnés.",
    ]
    return "\n".join(lineas).rstrip() + "\n"


def main() -> int:
    comprobar = "--check" in sys.argv
    if not EJEMPLOS.exists():
        print("No hay examples/", file=sys.stderr)
        return 1

    lista = fuentes()
    enlaces = {
        fichero.name: pagina_de(numero, fichero) for numero, fichero, _ in lista
    }
    # También la forma con carpeta, que es como se citan desde el README.
    enlaces.update({
        f"{fichero.parent.name}/{fichero.name}": pagina_de(numero, fichero)
        for numero, fichero, _ in lista
    })

    entradas = []
    for numero, fichero, pista in lista:
        doc = ast.get_docstring(ast.parse(fichero.read_text(encoding="utf-8")))
        doc = doc or fichero.stem
        entradas.append((numero, fichero, pista, titulo_de(numero, doc), dominio_de(doc)))

    generadas = {"ejemplos.md": render_indice(entradas)}
    for numero, fichero, _ in lista:
        nombre = pagina_de(numero, fichero)
        salida = salida_de(fichero)
        if salida is None:
            salida = salida_previa(DOCS / nombre)
            print(
                f"  · {fichero.name}: no se ejecuta aquí "
                f"(falta el extra '{REQUIERE_EXTRA[fichero.name]}'), se conserva su salida",
                file=sys.stderr,
            )
        generadas[nombre] = render(numero, fichero, enlaces, salida)

    desfasadas: list[str] = []
    for nombre, contenido in generadas.items():
        destino = DOCS / nombre
        if comprobar:
            actual = destino.read_text(encoding="utf-8") if destino.exists() else ""
            if actual != contenido:
                desfasadas.append(nombre)
        else:
            destino.write_text(contenido, encoding="utf-8")

    sobrantes = {p.name for p in DOCS.glob("ejemplos*.md")} - set(generadas)
    for nombre in sorted(sobrantes):
        if comprobar:
            desfasadas.append(f"{nombre} (sobra)")
        else:
            (DOCS / nombre).unlink()

    if comprobar and desfasadas:
        print(
            "Las páginas de ejemplos no corresponden a examples/: "
            + ", ".join(desfasadas)
            + "\nRegenera con: uv run python scripts/render_examples.py",
            file=sys.stderr,
        )
        return 1

    print(f"{'comprobadas' if comprobar else 'generadas'} {len(generadas)} páginas de ejemplos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
