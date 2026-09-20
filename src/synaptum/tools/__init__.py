"""Herramientas: el esquema sale de la firma."""

from .decorator import Tool, json_schema_for, tool
from .deferred import UMBRAL_RAZONABLE, deferred, merece_la_pena

__all__ = [
    "Tool",
    "tool",
    "json_schema_for",
    "deferred",
    "merece_la_pena",
    "UMBRAL_RAZONABLE",
]
