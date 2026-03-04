"""Backward-compatibility shim. Use ``LLMAgent`` instead."""
from .llm_agent import LLMAgent

# Deprecated alias — kept for backward compatibility
SimpleAgent = LLMAgent

__all__ = ["SimpleAgent"]
