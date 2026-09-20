"""Las herramientas del agente — lo primero que vas a cambiar.

El esquema que ve el modelo **sale de la firma**: los tipos, los obligatorios,
las descripciones de `Annotated` y la primera línea del docstring. No hay que
escribirlo aparte, y por eso no puede desincronizarse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from synaptum import Risk, tool


@tool(idempotent=True)
async def listar_ficheros(
    carpeta: Annotated[str, "Ruta de la carpeta, relativa al directorio actual"] = ".",
) -> str:
    """Lista los ficheros de una carpeta."""
    destino = Path(carpeta)
    if not destino.is_dir():
        # Un error se devuelve como texto, no se lanza: el modelo lo lee y suele
        # corregir. Uno que no ve el error no puede corregirlo.
        return f"No es una carpeta: {carpeta}"
    return "\n".join(sorted(p.name for p in destino.iterdir())) or "(vacía)"


@tool(idempotent=True)
async def contar_lineas(
    fichero: Annotated[str, "Ruta del fichero"],
) -> str:
    """Cuenta las líneas de un fichero de texto."""
    destino = Path(fichero)
    if not destino.is_file():
        return f"No existe: {fichero}"
    return f"{fichero}: {len(destino.read_text(errors='replace').splitlines())} líneas"


@tool(risk=Risk.DESTRUCTIVE)
async def borrar(
    fichero: Annotated[str, "Ruta del fichero a borrar"],
) -> str:
    """Borra un fichero del disco."""
    return f"[simulado] se borraría {fichero}"


# `risk` e `idempotent` se **declaran**: ninguna anotación de tipos puede saber
# que una función que devuelve `str` mueve dinero o borra un disco.
#
# Quien no declara nada obtiene `Risk.READ` e `idempotent=False` — la clase más
# inocua y la garantía más cara. Y eso tiene una consecuencia práctica: una
# política que deniegue por `Risk.DESTRUCTIVE` **no detendrá** tu herramienta si
# no lo declaraste, aunque borre discos.
