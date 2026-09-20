"""
SYN-40 · Un catálogo grande sin inflar el prefijo.

Cuarenta herramientas son **3.857 tokens de catálogo en cada turno** — el 94 % de
una ventana de 4.096. Y el modelo tiene que elegir entre cuarenta, que es la otra
mitad del problema y la que no se ve en la factura.

La trampa, medida antes de escribir nada
-----------------------------------------
Lo obvio sería cargar herramientas cuando hagan falta. **Es peor que no hacer
nada**, y esto está medido contra un despliegue real con un historial de run
normal:

===========================  ==========  =============
Petición                     ``input``   ``cache_read``
===========================  ==========  =============
10 herramientas, 2ª llamada       1.444   1.443 (100 %)
11 — una sola añadida             1.467       0 (0 %)
===========================  ==========  =============

**Añadir una herramienta a mitad de run destruye la caché entera**, historial
incluido. El catálogo vive en el prefijo estable, y un prefijo que cambia en el
token 10 invalida los 10.000 siguientes. Cargar bajo demanda cuesta más que
mandarlo todo.

La salida: mover el catálogo al historial
------------------------------------------
El prefijo se queda con **dos herramientas fijas** y no cambia nunca:

* ``buscar_herramientas(consulta)`` — devuelve nombres, descripciones **y
  esquemas** de las que encajan;
* ``usar_herramienta(nombre, argumentos)`` — ejecuta la que sea.

Lo que el modelo necesita —el esquema de la que va a usar— llega como
**resultado de herramienta**, y un resultado se añade al final de los mensajes.
El final crece; crecer al final no invalida nada.

Es el mismo principio que el prefijo estable, al revés: en vez de proteger lo de
delante, se mueve lo variable hacia atrás.

Lo que cuesta, dicho entero
----------------------------
* **Dos turnos extra como mínimo**: buscar y luego usar.
* **El modelo pierde el esquema tipado en la llamada.** Llama a
  ``usar_herramienta`` con un objeto libre, así que los errores de argumentos
  suben. Se compensa validando contra el esquema real al despachar y
  devolviendo un error del que se puede corregir — pero es una compensación, no
  una solución.
* **Por debajo de ~15 herramientas esto es estrictamente peor.** Dos turnos de
  más no los paga un catálogo pequeño, y hay una función que lo dice en vez de
  dejar que se descubra.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ..core.errors import InvalidToolCallError
from ..core.types import Risk, ToolResult, dumps
from .decorator import Tool, tool

__all__ = ["deferred", "merece_la_pena", "UMBRAL_RAZONABLE"]

#: Por debajo de esto, un catálogo diferido sale más caro que uno completo.
#:
#: Sale de la aritmética y no de una intuición: dos turnos extra cuestan una
#: inferencia cada uno, y el catálogo que ahorran solo compensa cuando es
#: grande.  Con quince herramientas de descripción media son unos 1.400 tokens
#: de prefijo — que además se sirven de caché a partir del segundo turno.
UMBRAL_RAZONABLE = 15


def merece_la_pena(herramientas: Sequence[Any]) -> bool:
    """¿Vale la pena diferir este catálogo?

    Está expuesto a propósito: es mejor que alguien pueda preguntar a que lo
    descubra midiendo su factura.
    """
    return len(herramientas) >= UMBRAL_RAZONABLE


def _riesgo_mayor(herramientas: Sequence[Tool]) -> Risk:
    """El despachador es tan peligroso como la peor del catálogo.

    Sin esto, esconder cuarenta herramientas detrás de una las **blanquearía a
    todas**: el gateway vería `usar_herramienta` con riesgo de lectura y dejaría
    pasar la que borra el disco.

    Es la misma regla que en la delegación, y por la misma razón: quien llama no
    sabe qué hay detrás, pero el framework sí.
    """
    orden = (Risk.READ, Risk.SOFT_WRITE, Risk.HARD_WRITE, Risk.DESTRUCTIVE)
    mayor = Risk.READ
    for h in herramientas:
        if orden.index(h.definition.risk) > orden.index(mayor):
            mayor = h.definition.risk
    return mayor


def _puntuar(consulta: str, definicion: Any) -> int:
    """Cuánto encaja una herramienta con la consulta.

    Búsqueda léxica, sin embeddings — el núcleo no tiene dependencias y un
    modelo de embeddings sería una. Es peor que una semántica y tiene dos
    virtudes que aquí pesan: es **determinista**, así que el contexto
    reconstruido al reanudar es idéntico, y es explicable.
    """
    palabras = {p for p in re.split(r"\W+", consulta.lower()) if len(p) > 2}
    if not palabras:
        return 0

    nombre = definicion.name.lower()
    descripcion = (definicion.description or "").lower()
    parametros = " ".join(dumps(definicion.parameters).lower().split())

    puntos = 0
    for palabra in palabras:
        if palabra in nombre:
            puntos += 10          # el nombre es lo que el autor eligió que la describa
        if palabra in descripcion:
            puntos += 4
        if palabra in parametros:
            puntos += 1
    return puntos


def deferred(
    herramientas: Sequence[Tool], *, max_resultados: int = 5
) -> list[Tool]:
    """Convierte un catálogo grande en **dos** herramientas de prefijo fijo.

    Args:
        herramientas: el catálogo entero, decorado con ``@tool``.
        max_resultados: cuántas devuelve una búsqueda. Más no ayuda: el modelo
            elige peor cuantas más ve, que es medio problema que esto resuelve.

    Returns:
        ``[buscar_herramientas, usar_herramienta]``, listas para ``Agent(tools=…)``.

    Ejemplo::

        agente = Agent("a", model=…, tools=deferred(las_cuarenta))
    """
    por_nombre = {h.definition.name: h for h in herramientas}
    riesgo = _riesgo_mayor(list(herramientas))

    async def buscar_herramientas(consulta: str) -> str:
        """Busca herramientas disponibles por lo que quieres hacer.

        Devuelve su nombre, para qué sirven y qué argumentos aceptan. Úsala
        antes de `usar_herramienta` cuando no sepas cuál necesitas.
        """
        puntuadas = [
            (_puntuar(consulta, h.definition), nombre, h)
            for nombre, h in por_nombre.items()
        ]
        # Orden estable: por puntos y luego por nombre. Sin el segundo criterio,
        # dos empatadas saldrían en orden de diccionario y el contexto
        # reconstruido al reanudar podría no ser idéntico.
        encajan = sorted(
            (p for p in puntuadas if p[0] > 0), key=lambda p: (-p[0], p[1])
        )[:max_resultados]

        if not encajan:
            return (
                f"Ninguna de las {len(por_nombre)} herramientas encaja con "
                f"{consulta!r}. Prueba con otras palabras, o di que no se puede hacer."
            )

        bloques = []
        for _, nombre, h in encajan:
            d = h.definition
            bloques.append(
                f"{nombre}: {d.description}\n  argumentos: {dumps(d.parameters)}"
            )
        return "\n\n".join(bloques)

    async def usar_herramienta(nombre: str, argumentos: str = "{}") -> str:
        """Ejecuta una herramienta encontrada con `buscar_herramientas`.

        `argumentos` es un objeto JSON con los argumentos que pide su esquema.
        """
        objetivo = por_nombre.get(nombre)
        if objetivo is None:
            cercanas = sorted(por_nombre)[:8]
            return (
                f"No existe '{nombre}'. Busca primero con buscar_herramientas. "
                f"Algunas disponibles: {cercanas}"
            )

        try:
            valores = argumentos if isinstance(argumentos, Mapping) else json.loads(argumentos or "{}")
        except json.JSONDecodeError as roto:
            # Se devuelve al modelo en vez de romper: un JSON mal formado es una
            # muestra mala, y el modelo corrige si ve el error.
            return f"Los argumentos de '{nombre}' no son JSON válido: {roto}"

        if not isinstance(valores, Mapping):
            return f"Los argumentos de '{nombre}' deben ser un objeto JSON, no {type(valores).__name__}."

        faltan = [
            campo
            for campo in objetivo.definition.parameters.get("required", ())
            if campo not in valores
        ]
        if faltan:
            # El esquema completo, otra vez, porque el modelo ya no lo tiene
            # delante: vive en el historial, varios turnos atrás.
            return (
                f"Faltan argumentos de '{nombre}': {faltan}. "
                f"Su esquema es: {dumps(objetivo.definition.parameters)}"
            )

        try:
            resultado = await objetivo.invoke(f"deferred:{nombre}", dict(valores))
        except InvalidToolCallError as desajuste:
            return f"Argumentos inválidos para '{nombre}': {desajuste}"

        return _texto(resultado)

    return [
        tool(buscar_herramientas),
        # El despachador hereda el riesgo de la peor del catálogo, y **nunca** se
        # declara idempotente: detrás puede haber cualquier cosa.
        tool(usar_herramienta, risk=riesgo, idempotent=False),
    ]


def _texto(resultado: ToolResult) -> str:
    from ..core.types import Text

    texto = "".join(p.text for p in resultado.content if isinstance(p, Text))
    return f"[error] {texto}" if resultado.is_error else texto
