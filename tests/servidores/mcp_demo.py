"""Servidor MCP de verdad, para los tests del cliente.

Se arranca como proceso hijo por stdio, igual que uno real. Un doble de mi
propio cliente comprobaría que llamo a mis funciones, que es lo único que no
hace falta comprobar.
"""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

servidor = MCPServer("demo")


@servidor.tool(
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def leer(ruta: str) -> str:
    """Lee un fichero."""
    return f"contenido de {ruta}"


@servidor.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True),
)
def borrar(ruta: str) -> str:
    """Borra un fichero."""
    return f"borrado {ruta}"


@servidor.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
def anotar(texto: str) -> str:
    """Añade una nota."""
    return f"anotado: {texto}"


@servidor.tool()
def sin_anotaciones(x: str) -> str:
    """Una herramienta que no declara nada sobre su efecto."""
    return f"eco {x}"


@servidor.tool()
def revienta() -> str:
    """Siempre falla."""
    raise RuntimeError("el servidor no pudo")


if __name__ == "__main__":
    servidor.run()
