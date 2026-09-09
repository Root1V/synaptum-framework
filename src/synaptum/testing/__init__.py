"""Dobles de desarrollo.

Por SYN-65 esta es la vía principal para trabajar sin inferencia real, no una
utilidad marginal: en modo autónomo no hay modelos locales disponibles.
"""

from .fake import DEFAULT_USAGE, FakeGateway, calls, says

__all__ = ["FakeGateway", "says", "calls", "DEFAULT_USAGE"]
