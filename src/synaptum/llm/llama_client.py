from typing import Any, Dict, List
from axonium import LlamaAdapter

from .client import LLMClient, LLMResponse


class LlamaClient(LLMClient):

    def __init__(self):
        self._adapter = LlamaAdapter(model="Emeric/axonium-2-13B-Instruct-v0.1-Q4_0.gguf")

    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        raw = await self._adapter.async_chat(messages, **kwargs)
        return LLMResponse(content=raw.choices[0].message.content)
