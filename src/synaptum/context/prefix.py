"""
SYN-32 · El prefijo estable, y qué pasa cuando deja de serlo.

Los proveedores cachean **prefijos exactos**: si los primeros N tokens de la
petición coinciden byte a byte con los de la anterior, no se vuelven a procesar.
De ahí sale el `cache_read` que hace barato un run largo — y de ahí sale también
que cualquier cambio temprano tire toda la caché posterior, porque un prefijo que
cambia en el token 10 invalida los 10.000 siguientes.

Qué es el prefijo estable
--------------------------
Lo que **no debe cambiar dentro de un run**, y en este orden:

1. el modelo,
2. las instrucciones de sistema,
3. el catálogo de herramientas — nombres, descripciones y esquemas, en su orden,
4. el formato de salida pedido.

Los mensajes vienen después y crecen; eso es lo normal y no invalida nada,
porque se añaden al final.

Pero el motivo de fondo no es el dinero
----------------------------------------
Se puede reanudar un run con **otro agente** —otro modelo, otras instrucciones,
otras herramientas— y hoy nada avisa. El resultado es peor que una caché fallada:
la primera mitad del run la ejecutó una configuración y la segunda otra, y el
journal lo registra como un solo run. Una auditoría de «qué hizo el agente»
devuelve entonces una historia que **ninguna configuración produjo nunca**.

Por eso esto no es un aviso: es un error. Reanudar es continuar *ese* run, y una
configuración distinta es otro run — con su propio ``run_id``.
"""

from __future__ import annotations

import hashlib

from ..core.types import Request, dumps

__all__ = ["prefix_fingerprint", "describe_prefix_change"]


def prefix_fingerprint(request: Request) -> str:
    """Huella de lo que no debe cambiar dentro de un run.

    Es sensible al **orden** de las herramientas a propósito: reordenarlas
    produce otro prefijo por el cable, así que tira la caché igual que
    cambiarlas. Fingir que da lo mismo sería mentir sobre lo que cuesta.

    No entra en la huella nada que crezca —los mensajes— ni nada que dependa del
    reloj: sería una huella distinta en cada turno y no diría nada.
    """
    return hashlib.sha256(dumps(_partes_estables(request)).encode()).hexdigest()[:16]


def _partes_estables(request: Request) -> dict:
    return {
        "model": request.model,
        "system": request.system,
        "tools": [
            # El esquema entra entero: un parámetro nuevo en una herramienta
            # cambia lo que el modelo ve, y cambia el prefijo.
            {"name": t.name, "description": t.description, "parameters": dict(t.parameters)}
            for t in request.tools
        ],
        "response_format": (
            None
            if request.response_format is None
            else {
                "kind": request.response_format.kind,
                "schema": dict(request.response_format.schema),
            }
        ),
    }


def describe_prefix_change(antes: Request, ahora: Request) -> str:
    """Qué cambió, en términos de quien lo tiene que arreglar.

    «La huella no coincide» no sirve para nada. «Las herramientas pasaron de
    [leer] a [leer, borrar]» se arregla sin abrir el journal.
    """
    viejo, nuevo = _partes_estables(antes), _partes_estables(ahora)
    cambios: list[str] = []

    if viejo["model"] != nuevo["model"]:
        cambios.append(f"modelo: {viejo['model']!r} → {nuevo['model']!r}")

    if viejo["system"] != nuevo["system"]:
        cambios.append("instrucciones de sistema: cambiaron")

    nombres_viejos = [t["name"] for t in viejo["tools"]]
    nombres_nuevos = [t["name"] for t in nuevo["tools"]]
    if nombres_viejos != nombres_nuevos:
        cambios.append(f"herramientas: {nombres_viejos} → {nombres_nuevos}")
    elif viejo["tools"] != nuevo["tools"]:
        # Mismos nombres y distinto contenido: una firma o una descripción
        # cambió. Es el caso que más cuesta ver a ojo.
        distintas = [
            v["name"] for v, n in zip(viejo["tools"], nuevo["tools"]) if v != n
        ]
        cambios.append(f"esquema o descripción de: {distintas}")

    if viejo["response_format"] != nuevo["response_format"]:
        cambios.append("formato de salida: cambió")

    return " · ".join(cambios) or "algo del prefijo estable"
