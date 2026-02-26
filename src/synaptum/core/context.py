from typing import Any, Dict, Optional


class AgentContext:
    """
    Contexto de ejecución desacoplado de la plataforma.
    La plataforma puede inyectar:
      - tenant_id, trace_id, scopes, budgets, user, etc.
    """
    def __init__(self, run_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.metadata: Dict[str, Any] = metadata or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.metadata[key] = value
