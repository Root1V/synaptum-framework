"""Servidor MCP mínimo sobre un repositorio — el papel de un servidor de terceros.

En un caso real esto sería `uvx mcp-server-git`, `mcp-server-filesystem` o el
servidor MCP de tu propia plataforma. Va aquí para que el ejemplo corra sin
instalar nada, y está escrito **como lo escribiría otro equipo**: anota unas
herramientas y otras no, que es exactamente lo que uno se encuentra.
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

RAIZ = Path(__file__).resolve().parents[3]

servidor = MCPServer("repo")


@servidor.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True))
def listar_ficheros(subcarpeta: str = ".") -> str:
    """Lista los ficheros de una carpeta del repositorio."""
    destino = (RAIZ / subcarpeta).resolve()
    if RAIZ not in destino.parents and destino != RAIZ:
        return f"Fuera del repositorio: {subcarpeta}"
    return "\n".join(sorted(p.name for p in destino.iterdir() if not p.name.startswith(".")))


@servidor.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True))
def leer_fichero(ruta: str, lineas: int = 40) -> str:
    """Lee las primeras líneas de un fichero del repositorio."""
    destino = (RAIZ / ruta).resolve()
    if not destino.is_file():
        return f"No existe: {ruta}"
    return "\n".join(destino.read_text(errors="replace").splitlines()[:lineas])


@servidor.tool(annotations=ToolAnnotations(readOnlyHint=True))
def buscar(patron: str, extension: str = ".py") -> str:
    """Busca un patrón en los ficheros del repositorio."""
    aciertos = []
    for fichero in sorted((RAIZ / "src").rglob(f"*{extension}")):
        for numero, linea in enumerate(fichero.read_text(errors="replace").splitlines(), 1):
            if patron in linea:
                aciertos.append(f"{fichero.relative_to(RAIZ)}:{numero}: {linea.strip()[:80]}")
                if len(aciertos) >= 12:
                    return "\n".join(aciertos) + "\n… (recortado)"
    return "\n".join(aciertos) or f"sin aciertos para {patron!r}"


# Y una que **no anota nada**, como pasa con la mitad de los servidores reales.
@servidor.tool()
def aplicar_parche(ruta: str, diff: str) -> str:
    """Aplica un parche a un fichero del repositorio."""
    return f"[simulado] parche de {len(diff)} bytes sobre {ruta}"


if __name__ == "__main__":
    servidor.run()
