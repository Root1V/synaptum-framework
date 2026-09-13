"""Lo único que cambia entre correr con un modelo real y correr sin él.

Vive aquí para que los ejemplos no lo repitan, y porque separarlo enseña la
idea: **el agente no sabe quién hay al otro lado de la costura.** Cambiar de
doble a modelo real no toca una línea del agente.
"""

from __future__ import annotations

import os
from pathlib import Path

from synaptum import HttpModel, LocalGateway
from synaptum.testing import FakeGateway

CORPUS = Path(
    "/Users/emericespiritusantiago/Documents/Victor/coordinacion_project/"
    "contratos/gateway-prometheus/fixtures"
)


def hay_modelo_real() -> bool:
    return bool(os.environ.get("SYNAPTUM_BASE_URL"))


def nombre_del_modelo() -> str:
    if hay_modelo_real():
        return f"openai-compatible:{os.environ.get('SYNAPTUM_MODEL', 'qwen3-0.6b')}"
    return "openai-compatible:doble"


def gateway_real(tools=()):
    """`LocalGateway` sobre transporte HTTP.  Solo si hay endpoint configurado."""
    modelo = HttpModel(
        os.environ["SYNAPTUM_BASE_URL"],
        api_key=os.environ.get("SYNAPTUM_API_KEY"),
    )
    return LocalGateway(model=modelo, stream=modelo.stream, tools=tools, warn=False)


def gateway(guion, tools=()):
    """El real si está configurado; si no, el doble con el guion que se le pase.

    El doble no es un mock: ejecuta las herramientas de verdad, hace streaming
    de verdad y produce `Usage` de verdad.  Lo único que no hace es inferir.
    """
    if hay_modelo_real():
        return gateway_real(tools)
    return FakeGateway(*guion, tools=tools)


def encabezado(titulo: str) -> None:
    print(f"\n{titulo}")
    print("─" * len(titulo))
    if hay_modelo_real():
        print(f"modelo real · {os.environ['SYNAPTUM_BASE_URL']}\n")
    else:
        print("sin inferencia · respuestas guionizadas "
              "(exporta SYNAPTUM_BASE_URL para usar un modelo real)\n")
