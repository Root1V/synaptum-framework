from abc import ABC, abstractmethod
from typing import Any, List


class MemoryStore(ABC):
    @abstractmethod
    async def append(self, key: str, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def read(self, key: str) -> List[Any]:
        raise NotImplementedError
