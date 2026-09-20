"""Economía de contexto — qué ve el modelo y cuánto cuesta.

El journal registra **lo que ocurrió**; el contexto lleva **lo que el modelo
necesita ver**. No son lo mismo, y confundirlos es lo que hace que un agente
correcto sea carísimo.
"""

from .cap import cap_tool_output

__all__ = ["cap_tool_output"]
