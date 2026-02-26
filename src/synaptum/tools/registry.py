from typing import Dict, List
from .base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    def describe_for_llm(self) -> List[dict]:
        out = []
        for t in self._tools.values():
            out.append({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema(),
            })
        return out

