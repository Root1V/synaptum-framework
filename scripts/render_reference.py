"""Genera `docs/08-referencia.md` **a partir del código**.

Escribir una referencia a mano es escribir algo que caduca. Y una referencia
caducada es peor que no tenerla: manda a buscar un símbolo que ya no está, y
hace dudar de los que sí están.

Aquí se introspecciona lo que el paquete exporta de verdad — firmas, campos de
`dataclass`, métodos públicos y docstrings— así que la referencia no puede
describir algo que no existe. Si el código cambia y esto no se regenera, el test
de documentación lo dice.

**Lo que no hace:** no sustituye a las páginas escritas. Una referencia dice qué
hay; las páginas dicen por qué y cuándo usarlo, y eso no se deriva del código.

    uv run python scripts/render_reference.py
"""

from __future__ import annotations

import enum
import inspect
import re
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "docs" / "08-referencia.md"

#: El orden importa: primero lo que se usa al escribir un agente, luego lo que
#: se necesita al operarlo. Alfabético sería más fácil y menos útil.
GRUPOS: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "Escribir un agente",
        "Lo que se toca en el primer fichero.",
        ("Agent", "Session", "Limits", "tool", "Tool", "Risk", "ToolDefinition", "ToolChoice",
         "cap_tool_output"),
    ),
    (
        "Lo que cede el bucle",
        "Los pasos que llegan por `async for`, y sus fases.",
        ("StepEvent", "ModelStep", "ToolStep", "ApprovalStep", "DelegateStep", "FinalStep",
         "Phase", "Durability", "Event", "make_step_id", "idempotency_key"),
    ),
    (
        "Vocabulario del modelo",
        "La forma que viaja por la costura. Contrato versionado, no detalle interno.",
        ("Message", "Role", "AUTO", "ContentPart", "Text", "Image", "Audio", "Document", "ToolCall",
         "ToolResult", "Thinking", "RedactedThinking", "Request", "Response", "Usage",
         "FinishReason", "ResponseFormat"),
    ),
    (
        "Streaming",
        "El ciclo start/delta/end de cada clase de contenido.",
        ("StreamEvent", "StreamStart", "TextStart", "TextDelta", "TextEnd", "ReasoningStart",
         "ReasoningDelta", "ReasoningEnd", "ToolCallStart", "ToolCallDelta", "ToolCallEnd",
         "Finish"),
    ),
    (
        "Las dos costuras",
        "Protocolos estructurales: quien los implemente no hereda ni importa nada.",
        ("Gateway", "Checkpointer", "CallContext", "Decision", "Disposition", "ALLOW",
         "Hello", "Welcome", "RunState", "SEAM_VERSION", "SUPPORT_WINDOW", "negotiate",
         "supported_versions"),
    ),
    (
        "Runtime",
        "Implementaciones de referencia. Ninguna es para producción a escala.",
        ("LocalGateway", "Check", "MemoryCheckpointer", "SqliteCheckpointer", "Journal",
         "Replay", "HttpModel"),
    ),
    (
        "Errores",
        "La reintentabilidad viaja **en el tipo**, no en una tabla de códigos.",
        ("SynaptumError", "ConfigurationError", "ProviderError", "RequestTimeoutError",
         "NetworkError", "AbortError", "Denied", "InvalidToolCallError", "ToolExecutionError",
         "NoObjectGeneratedError", "LimitExceeded", "UncertainEffect", "SeamVersionError",
         "retryable_for_status"),
    ),
    (
        "Esquemas y prompts",
        "",
        ("Schema", "schema_for", "json_schema_for", "PromptTemplate", "PromptProvider",
         "InMemoryPrompts", "FilePrompts", "PromptRegistry", "fmt_dict", "fmt_list",
         "fmt_records"),
    ),
    (
        "Proveedores y utilidades",
        "",
        ("Provider", "dumps", "to_jsonable", "b64", "__version__"),
    ),
]

#: Módulos opcionales, documentados aparte porque no todo el mundo los instala.
EXTRAS: list[tuple[str, str, str]] = [
    ("synaptum.testing", "Dobles de desarrollo",
     "Infraestructura, no una utilidad de test: es la vía principal para construir sin gastar."),
    ("synaptum.mcp", "Cliente MCP",
     "Extra `[mcp]`. El núcleo no lo carga."),
]


def _firma(obj: Any) -> str:
    """La firma, legible.

    Con `from __future__ import annotations` las anotaciones son cadenas, así que
    `inspect` las devuelve entrecomilladas: `task: 'str'`. Es ruido — quien lee
    una referencia quiere el tipo, no saber cómo se evaluó.
    """
    try:
        texto = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    return re.sub(r"'([\w\[\]., |]+)'", r"\1", texto).replace("synaptum.core.types.", "")


def _resumen(obj: Any) -> str:
    """La primera línea del docstring, que es la que responde «¿qué es esto?»."""
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n")[0].replace("\n", " ").strip()


def _cuerpo_doc(obj: Any) -> str:
    """El resto del docstring, si lo hay y aporta algo."""
    doc = inspect.getdoc(obj) or ""
    partes = doc.split("\n\n")
    return "\n\n".join(p.strip() for p in partes[1:] if p.strip())


def _tipo(anotacion: Any) -> str:
    if anotacion is inspect.Parameter.empty:
        return ""
    texto = anotacion if isinstance(anotacion, str) else getattr(anotacion, "__name__", str(anotacion))
    return _en_celda(str(texto).replace("typing.", "").replace("synaptum.core.types.", ""))


def _en_celda(texto: str) -> str:
    """Escapa lo que rompería una tabla Markdown.

    Una barra dentro de una celda se lee como separador de columna, así que
    `int | None` partía la fila en cinco — y las uniones de tipos están por
    todas partes en este paquete.
    """
    return texto.replace("|", "\\|").replace("\n", " ")


def _campos(clase: type) -> list[str]:
    lineas = []
    for campo in fields(clase):
        if campo.default is not MISSING:
            defecto = f"`{campo.default!r}`"
        elif campo.default_factory is not MISSING:  # type: ignore[misc]
            defecto = "*(fábrica)*"
        else:
            defecto = "**obligatorio**"
        lineas.append(f"| `{campo.name}` | `{_tipo(campo.type)}` | {defecto} |")
    return lineas


def _metodos(clase: type) -> list[tuple[str, str, str]]:
    salida = []
    for nombre, miembro in inspect.getmembers(clase):
        if nombre.startswith("_"):
            continue
        if not (inspect.isfunction(miembro) or inspect.ismethod(miembro) or isinstance(miembro, property)):
            continue
        if isinstance(miembro, property):
            salida.append((nombre, "*(propiedad)*", _resumen(miembro.fget)))
        else:
            asincrona = "async " if inspect.iscoroutinefunction(miembro) else ""
            salida.append((nombre, f"{asincrona}{nombre}{_firma(miembro)}", _resumen(miembro)))
    return salida


def _render_clase(nombre: str, obj: type) -> list[str]:
    fuera = [f"### `{nombre}`", ""]
    resumen = _resumen(obj)
    if resumen:
        fuera += [resumen, ""]

    if issubclass(obj, enum.Enum):
        fuera += ["| Valor | |", "|---|---|"]
        fuera += [f"| `{nombre}.{m.name}` | `{m.value!r}` |" for m in obj]
        fuera.append("")
    elif is_dataclass(obj):
        filas = _campos(obj)
        if filas:
            fuera += ["| Campo | Tipo | Por defecto |", "|---|---|---|", *filas, ""]
    else:
        fuera += ["```python", f"{nombre}{_firma(obj)}", "```", ""]

    metodos = _metodos(obj)
    if metodos:
        fuera += ["| Miembro | Firma | |", "|---|---|---|"]
        fuera += [
            f"| `{n}` | {'—' if f.startswith('*') else f'`{_en_celda(f)}`'} | {_en_celda(d)} |"
            for n, f, d in metodos
        ]
        fuera.append("")

    cuerpo = _cuerpo_doc(obj)
    if cuerpo:
        fuera += [cuerpo, ""]
    return fuera


def _render_funcion(nombre: str, obj: Any) -> list[str]:
    asincrona = "async " if inspect.iscoroutinefunction(obj) else ""
    fuera = [f"### `{nombre}`", "", "```python", f"{asincrona}def {nombre}{_firma(obj)}", "```", ""]
    resumen = _resumen(obj)
    if resumen:
        fuera += [resumen, ""]
    cuerpo = _cuerpo_doc(obj)
    if cuerpo:
        fuera += [cuerpo, ""]
    return fuera


def _render_simbolo(nombre: str, obj: Any) -> list[str]:
    if inspect.isclass(obj):
        return _render_clase(nombre, obj)
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return _render_funcion(nombre, obj)
    return [f"### `{nombre}`", "", f"`{nombre} = {obj!r}`", "", _resumen(obj), ""]


def main() -> int:
    import synaptum

    lineas = [
        "# Referencia",
        "",
        "**Generada del código.** Escribir esto a mano sería escribir algo que caduca, y una",
        "referencia caducada manda a buscar un símbolo que ya no está.",
        "",
        "Dice **qué hay**. El **por qué** y el **cuándo** están en las páginas anteriores, y eso no",
        "se deriva de una firma.",
        "",
        "> La superficie pública es lo que `synaptum/__init__.py` exporta, y nada más. Lo que no",
        "> aparece aquí puede cambiar sin aviso, aunque se pueda importar.",
        "",
    ]

    publicos = set(synaptum.__all__)
    cubiertos: set[str] = set()

    for titulo, nota, nombres in GRUPOS:
        presentes = [n for n in nombres if n in publicos]
        if not presentes:
            continue
        lineas += [f"## {titulo}", ""]
        if nota:
            lineas += [nota, ""]
        for nombre in presentes:
            lineas += _render_simbolo(nombre, getattr(synaptum, nombre))
            cubiertos.add(nombre)

    sueltos = sorted(publicos - cubiertos)
    if sueltos:
        lineas += ["## Sin agrupar", "",
                   "Exportados y todavía sin sitio en esta página. Que aparezcan aquí es un aviso "
                   "para quien mantiene la referencia, no para quien la lee.", ""]
        for nombre in sueltos:
            lineas += _render_simbolo(nombre, getattr(synaptum, nombre))

    for modulo, titulo, nota in EXTRAS:
        try:
            cargado = __import__(modulo, fromlist=["x"])
        except ImportError:
            continue
        lineas += [f"## {titulo}", "", f"`{modulo}` — {nota}", ""]
        for nombre in getattr(cargado, "__all__", ()):
            lineas += _render_simbolo(nombre, getattr(cargado, nombre))

    DESTINO.write_text("\n".join(lineas).rstrip() + "\n", encoding="utf-8")
    print(f"{DESTINO.relative_to(RAIZ)}: {len(publicos)} símbolos públicos, "
          f"{len(sueltos)} sin agrupar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
