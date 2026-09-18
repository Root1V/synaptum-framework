"""SYN-39 · Cliente MCP — herramientas que no escribimos nosotros.

Extra opcional: ``pip install synaptum[mcp]``.  El núcleo no lo conoce.
"""

from .client import MCPTools, mcp_risk

__all__ = ["MCPTools", "mcp_risk"]
