from typing import Any, Dict, List, Optional


class AgentContext:
    """
    Contexto de ejecución desacoplado de la plataforma.
    La plataforma puede inyectar:
      - tenant_id, trace_id, scopes, budgets, user, etc.
    """
    def __init__(self, run_id: str, metadata: Optional[Dict[str, Any]] = None, agent_registry: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.metadata: Dict[str, Any] = metadata or {}
        self._agent_registry: Dict[str, Any] = agent_registry or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def agent_ids(self, prefix: Optional[str] = None) -> List[str]:
        """Returns IDs of all registered agents, optionally filtered by prefix."""
        ids = list(self._agent_registry.keys())
        if prefix:
            ids = [aid for aid in ids if aid.startswith(prefix)]
        return ids
