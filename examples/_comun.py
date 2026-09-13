"""Lo único que cambia entre correr con un modelo real y correr sin él.

Vive aquí para que los ejemplos no lo repitan, y porque separarlo enseña la
idea: **el agente no sabe quién hay al otro lado de la costura.** Cambiar de
doble a modelo real no toca una línea del agente.
"""

from __future__ import annotations

import os
from synaptum import HttpModel, LocalGateway
from synaptum.testing import FakeGateway


def hay_modelo_real() -> bool:
    return bool(os.environ.get("SYNAPTUM_BASE_URL"))


def nombre_del_modelo() -> str:
    if hay_modelo_real():
        return f"openai-compatible:{os.environ.get('SYNAPTUM_MODEL', 'qwen3-0.6b')}"
    return "openai-compatible:doble"


def gateway_real(tools=(), policy=None):
    """`LocalGateway` sobre transporte HTTP.  Solo si hay endpoint configurado."""
    modelo = HttpModel(
        os.environ["SYNAPTUM_BASE_URL"],
        api_key=os.environ.get("SYNAPTUM_API_KEY"),
    )
    return _Contado(
        LocalGateway(
            model=modelo, stream=modelo.stream, tools=tools, policy=policy, warn=False
        )
    )


class _Contado:
    """Envuelve un gateway para contar sus llamadas al modelo.

    Existe para que el ejemplo de durabilidad pueda **medir** contra un modelo
    real lo mismo que mide contra el doble. Una reanudación que no vuelve a
    pagar es una afirmación comprobable, y comprobarla solo con el doble sería
    comprobar el doble.

    Delega todo lo demás: el gateway de dentro es el que decide y ejecuta.
    """

    def __init__(self, interior) -> None:
        self._interior = interior
        self.model_calls = 0

    async def invoke_model(self, request, ctx):
        self.model_calls += 1
        return await self._interior.invoke_model(request, ctx)

    def stream_model(self, request, ctx):
        self.model_calls += 1
        return self._interior.stream_model(request, ctx)

    def __getattr__(self, nombre):
        return getattr(self._interior, nombre)


def gateway(guion, tools=(), *, policy=None, deny_tools=None):
    """El real si está configurado; si no, el doble con el guion que se le pase.

    El doble no es un mock: ejecuta las herramientas de verdad, hace streaming
    de verdad y produce `Usage` de verdad.  Lo único que no hace es inferir.

    ``policy`` y ``deny_tools`` expresan la misma denegación de las dos formas
    que cada gateway entiende. Van juntas a propósito: si un ejemplo solo
    supiera denegar con el doble, correría sobre el doble mientras la cabecera
    dice «modelo real», que es peor que no poder correrlo.
    """
    if hay_modelo_real():
        return gateway_real(tools, policy=policy)
    return FakeGateway(*guion, tools=tools, deny_tools=deny_tools or {})


def encabezado(titulo: str) -> None:
    print(f"\n{titulo}")
    print("─" * len(titulo))
    if hay_modelo_real():
        print(f"modelo real · {os.environ['SYNAPTUM_BASE_URL']}\n")
    else:
        print("sin inferencia · respuestas guionizadas "
              "(exporta SYNAPTUM_BASE_URL para usar un modelo real)\n")
