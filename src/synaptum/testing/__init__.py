"""Dobles de desarrollo.

Por `SYN-65` esta es la vía principal para trabajar sin inferencia real, no una
utilidad marginal: en modo autónomo no hay modelos locales disponibles.

Dos formas, con virtudes distintas:

* ``FakeGateway`` — guion escrito a mano. Directo y suficiente para la mayoría.
* ``ReplayGateway`` — respuestas **reales grabadas**, normalizadas por el
  adaptador de verdad. Recorre los caminos que nadie escribe a mano porque no
  se le ocurren.
"""

from .fake import DEFAULT_USAGE, FakeGateway, calls, says
from .replay import ReplayGateway, split_sse

__all__ = [
    "FakeGateway",
    "ReplayGateway",
    "says",
    "calls",
    "split_sse",
    "DEFAULT_USAGE",
]
