"""
SYN-33 · Recortar la salida de una herramienta antes de que entre en el contexto.

El problema, medido y no supuesto: una herramienta que lee un fichero de log
mete **39.000 tokens** en el contexto y nada los detiene. Con una ventana de
4.096 es un fallo inmediato; con una de 131.072 es solo caro — y caro en cada
turno posterior, porque ese bloque se reenvía entero cada vez.

Las cuatro decisiones que hacen que esto funcione
--------------------------------------------------

**1 · El journal guarda entero; el contexto lleva recortado.** El paso registra
lo que la herramienta devolvió de verdad — si mañana hay que auditar qué leyó el
agente, la respuesta está ahí. Lo que se recorta es la copia que viaja al
modelo. Es la misma separación que en el consumo: el diario cuenta lo que pasó y
el total cuenta lo que se pagó.

**2 · Determinista, o rompe la reanudación.** Al reanudar, el contexto se vuelve
a derivar de los mismos resultados. Si el recorte dependiera del reloj, del
azar o de cuánto espacio quede, el prompt reconstruido sería distinto del
original: caché fallada, y potencialmente otra respuesta. Por eso el tope es un
número fijo y el recorte una función pura.

**3 · Cabeza y cola, no solo cabeza.** Lo que importa suele estar en los dos
extremos: el encabezado dice qué es, y el final trae el total, el resumen o la
excepción. Cortar por el final tira justo la parte que el modelo necesita para
corregir. En un resultado de error el reparto se inclina hacia la cola, porque
el mensaje de una traza está abajo.

**4 · Dice que recortó, y cuánto.** Un texto truncado en silencio hace que el
modelo concluya sobre datos incompletos creyéndolos completos. Con la marca,
puede pedir el resto — o decir que no puede responder, que también es correcto.

Sobre medir en caracteres y no en tokens
-----------------------------------------
Contar tokens exige un tokenizador, y el núcleo no tiene dependencias. Los
caracteres son deterministas, no dependen del modelo y sobreestiman por el lado
seguro. La regla práctica es **~4 caracteres por token** en texto latino; menos
en código y en CJK. Si necesitas un tope exacto en tokens, recórtalo tú en la
herramienta, que es donde vive el conocimiento de qué se puede tirar.
"""

from __future__ import annotations

from ..core.types import Text, ToolResult

__all__ = ["cap_tool_output", "DEFAULT_MAX_CHARS"]

#: Tope por defecto, en caracteres (~4.000 tokens).
#:
#: Generoso a propósito: la mayoría de las herramientas devuelven mucho menos y
#: nunca lo notan. Está activado por defecto porque **el fallo de no recortar es
#: peor y silencioso** — nadie descubre que su agente paga de más hasta que mira
#: la factura, y con una ventana pequeña ni siquiera llega a eso.
DEFAULT_MAX_CHARS = 16_000

#: Fracción que se queda en la cabeza cuando el resultado es normal.
_CABEZA = 0.7

#: Y cuando es un error, porque el mensaje que importa está al final de la traza.
_CABEZA_ERROR = 0.25


def cap_tool_output(
    resultado: ToolResult,
    *,
    max_chars: int | None = DEFAULT_MAX_CHARS,
) -> ToolResult:
    """Devuelve el resultado recortado, o el mismo si cabe.

    Args:
        resultado: lo que la herramienta devolvió.
        max_chars: tope en caracteres. ``None`` desactiva el recorte.

    Returns:
        Un ``ToolResult`` nuevo si hubo que recortar; **el mismo objeto** si no
        —así quien compare por identidad puede saber si se tocó algo.
    """
    if max_chars is None or max_chars <= 0:
        return resultado

    texto = "".join(p.text for p in resultado.content if isinstance(p, Text))
    if len(texto) <= max_chars:
        return resultado

    # Las partes que no son texto no se tocan: una imagen no se recorta por la
    # mitad, y fingir que sí sería peor que dejarla.
    otras = tuple(p for p in resultado.content if not isinstance(p, Text))

    return ToolResult(
        call_id=resultado.call_id,
        content=(Text(_recortar(texto, max_chars, resultado.is_error)), *otras),
        is_error=resultado.is_error,
    )


def _recortar(texto: str, tope: int, es_error: bool) -> str:
    """Cabeza + marca + cola, cabiendo en ``tope``.

    La marca cuenta dentro del tope: un recorte que se pasa del límite por
    explicarse no es un recorte.
    """
    quitados = len(texto) - tope
    marca = (
        f"\n\n[… recortado: {quitados:,} de {len(texto):,} caracteres. "
        f"Pide un rango concreto si necesitas lo que falta …]\n\n"
    ).replace(",", ".")

    disponible = tope - len(marca)
    if disponible <= 0:
        # Un tope tan pequeño que no cabe ni el aviso. Se dice lo esencial y se
        # respeta el tope: quien puso el número manda.
        return f"[… {len(texto)} caracteres recortados …]"[:tope]

    fraccion = _CABEZA_ERROR if es_error else _CABEZA
    cabeza = int(disponible * fraccion)
    cola = disponible - cabeza
    return texto[:cabeza] + marca + (texto[-cola:] if cola else "")
