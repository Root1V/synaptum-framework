"""Dónde están los contratos compartidos, si es que están.

Los corpus dorados **no viven en este repositorio**: son artefactos conjuntos de
varios proyectos, y copiarlos aquí los convertiría en una copia que se
desincroniza. Pero Synaptum no depende de ellos para nada — son evidencia
adicional, no parte de la suite.

Por eso la ruta sale del entorno y nunca de una ruta absoluta escrita en el
código. Una ruta a la máquina de quien lo escribió hace fallar el repositorio en
cualquier otra, y no por un motivo que tenga que ver con el código.

    export SYNAPTUM_CONTRACTS=/ruta/a/contratos

Sin la variable, las suites que los usan se saltan con un aviso. **No se
inventa un veredicto verde**, pero tampoco se falla: quien clona esto sin tener
acceso a los contratos de otros equipos no ha hecho nada mal.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["CONTRACTS", "corpus", "SIN_CONTRATOS"]


def _raiz() -> Path | None:
    declarada = os.environ.get("SYNAPTUM_CONTRACTS")
    if declarada:
        ruta = Path(declarada).expanduser()
        return ruta if ruta.is_dir() else None

    # Conveniencia para quien tenga los contratos al lado del repositorio, que
    # es el reparto habitual cuando varios proyectos comparten artefactos.
    for candidata in (
        Path(__file__).resolve().parents[2] / "coordinacion_project" / "contratos",
        Path(__file__).resolve().parents[1] / "contratos",
    ):
        if candidata.is_dir():
            return candidata
    return None


CONTRACTS = _raiz()

SIN_CONTRATOS = (
    "contratos compartidos no disponibles — define SYNAPTUM_CONTRACTS si los tienes"
)


def corpus(*partes: str) -> Path | None:
    """Una carpeta dentro de los contratos, o ``None`` si no hay contratos."""
    if CONTRACTS is None:
        return None
    ruta = CONTRACTS.joinpath(*partes)
    return ruta if ruta.exists() else None
