"""
Formatting utilities for prompt construction.

These helpers serialise Python data structures into plain-text strings
suitable for injection into PromptTemplate variables.  They produce raw
data only — no structural headers, bullets, or framing text; all such
text belongs in the YAML templates.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def fmt_dict(d: Dict[str, Any], max_value_len: int = 500) -> str:
    """Serialise a dict as ``key: value`` lines.

    Nested dicts and lists are JSON-serialised and truncated to
    *max_value_len* characters.

    Example::

        fmt_dict({"name": "Acme", "revenue": 1_000_000})
        # "name: Acme\\nrevenue: 1000000"
    """
    lines: List[str] = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:max_value_len]}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def fmt_list(items: List[str], prefix: str = "· ") -> str:
    """Serialise a list as prefixed lines.

    Example::

        fmt_list(["risk A", "risk B"])
        # "· risk A\\n· risk B"
    """
    return "\n".join(f"{prefix}{item}" for item in items)


def fmt_records(items: List[Dict[str, Any]], template: str) -> str:
    """Serialise a list of dicts as lines using a format-string template.

    Each item is expanded with :meth:`str.format_map`, so the template can
    reference any key present in the dict.

    Example::

        fmt_records(
            [{"name": "Alice", "score": 9.2}, {"name": "Bob", "score": 7.8}],
            "{name}: {score:.1f}",
        )
        # "Alice: 9.2\\nBob: 7.8"
    """
    return "\n".join(template.format_map(item) for item in items)
