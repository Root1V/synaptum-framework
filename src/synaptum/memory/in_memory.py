from collections import defaultdict
from typing import Any, DefaultDict, List

from .base import MemoryStore


class InMemoryMemoryStore(MemoryStore):
    def __init__(self):
        self._store: DefaultDict[str, List[Any]] = defaultdict(list)

    async def append(self, key: str, value: Any) -> None:
        self._store[key].append(value)

    async def read(self, key: str) -> List[Any]:
        return list(self._store.get(key, []))
