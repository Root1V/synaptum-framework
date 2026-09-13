"""Adaptadores de proveedor: normalización pura, descubierta por entry points."""

from .base import ENTRY_POINT_GROUP, Provider, available, get, register, resolve
from .openai_compatible import OpenAICompatible

# El dialecto OpenAI-compatible viene registrado, que es el que habla casi todo: es el
# caso base, no un plugin que haya que instalar aparte.
register(OpenAICompatible())

__all__ = [
    "Provider",
    "register",
    "get",
    "resolve",
    "available",
    "ENTRY_POINT_GROUP",
    "OpenAICompatible",
]
