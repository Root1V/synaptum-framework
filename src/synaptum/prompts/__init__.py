"""Prompts como configuración versionada, no como literales en el código."""

from .providers import FilePrompts, InMemoryPrompts, PromptProvider, PromptRegistry
from .template import PromptTemplate, fmt_dict, fmt_list, fmt_records

__all__ = [
    "PromptTemplate",
    "PromptProvider",
    "InMemoryPrompts",
    "FilePrompts",
    "PromptRegistry",
    "fmt_dict",
    "fmt_list",
    "fmt_records",
]
