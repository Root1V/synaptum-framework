"""Lo único que cambia entre correr con un modelo real y correr sin él.

Vive aquí para que los ejemplos no lo repitan, y porque separarlo enseña la
idea: **el agente no sabe quién hay al otro lado de la costura.** Cambiar de
doble a modelo real no toca una línea del agente.
"""

from __future__ import annotations

import os
from synaptum import HttpModel, LocalGateway
from synaptum.testing import FakeGateway


def por_sdk() -> bool:
    """Hay un SDK de plataforma configurado, y entonces manda él.

    Cuando existe un SDK que es la **única puerta** a la inferencia de una
    plataforma, ir al endpoint por detrás no es una alternativa: se salta las
    credenciales, el catálogo, la taxonomía de errores y la facturación. Por eso
    tiene prioridad sobre el transporte directo.
    """
    # Se mira la **credencial**, no la URL: desde rc3 el SDK trae sus
    # direcciones por defecto, así que un despliegue perfectamente configurado
    # no tiene ninguna URL en el entorno.  Detectarlo por la URL hacía que
    # cayera al doble en silencio, diciendo que no había SDK cuando sí lo había.
    return bool(os.environ.get("AXONIUM_CLIENT_ID"))


def por_http() -> bool:
    return bool(os.environ.get("SYNAPTUM_BASE_URL"))


def hay_modelo_real() -> bool:
    return por_sdk() or por_http()


def nombre_del_modelo() -> str:
    """El agente nombra su modelo igual sea cual sea la puerta.

    Con el SDK va a secas —él conoce su catálogo—; por HTTP lleva el prefijo del
    adaptador que tiene que normalizar el cable.
    """
    if por_sdk():
        return os.environ.get("SYNAPTUM_MODEL", "gpt-oss-20b-mxfp4")
    if por_http():
        return f"openai-compatible:{os.environ.get('SYNAPTUM_MODEL', 'qwen3-0.6b')}"
    return "openai-compatible:doble"


def gateway_real(tools=(), policy=None):
    """`LocalGateway` sobre la puerta que haya configurada.

    Las dos encajan en la misma ranura y el agente no nota la diferencia: una
    habla el cable y la otra habla un SDK que ya normaliza. Eso **es** la
    propiedad — el bucle no sabe quién hay al otro lado de la costura.
    """
    if por_sdk():
        # Import perezoso: el extra es opcional y no puede hacer falta para
        # correr los ejemplos sin él.
        from synaptum.providers.axonium import AxoniumModel

        puente = AxoniumModel()
        return _Contado(
            LocalGateway(
                model=puente.complete, stream=puente.stream,
                tools=tools, policy=policy, warn=False,
            )
        )

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
    if por_sdk():
        print(f"vía SDK de plataforma · {nombre_del_modelo()}\n")
    elif por_http():
        print(f"HTTP directo · {os.environ['SYNAPTUM_BASE_URL']}\n")
    else:
        print("sin inferencia · respuestas guionizadas "
              "(exporta SYNAPTUM_BASE_URL para usar un modelo real)\n")
