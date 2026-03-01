from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AgentState:
    """
    Per-agent mutable key/value store.

    Handlers use ``agent.state`` to persist data between message invocations
    (e.g. tracking pending replies, accumulating results across a fan-out).
    State is typically keyed by ``run_key`` (derived from ``msg.id``) to
    safely support concurrent requests on the same agent instance.

    Supports both dict-style access and explicit get/set methods::

        agent.state[run_key] = {"caller": msg.sender, "pending": 3}
        state = agent.state[run_key]
        state = agent.state.get(run_key, {})
        del agent.state[run_key]
    """
    data: Dict[Any, Any] = field(default_factory=dict)

    # ── dict-style access ─────────────────────────────────────────────────────
    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __delitem__(self, key: Any) -> None:
        del self.data[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.data

    # ── explicit access ───────────────────────────────────────────────────────
    def get(self, key: Any, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def clear(self) -> None:
        """Remove all entries."""
        self.data.clear()

    def __repr__(self) -> str:
        return f"AgentState({self.data!r})"
