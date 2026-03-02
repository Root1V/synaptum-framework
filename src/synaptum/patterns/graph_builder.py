"""
GraphBuilder — declarative, node-first graph construction for agent state machines.

Design
------
Following the LangGraph pattern, the builder is the single point of contact:
you register nodes (SimpleAgent instances), wire edges, then call ``build()``
which returns a ``GraphAgent`` that owns and drives all nodes.

The **run state** is a plain dict seeded from the submit-message payload.
Each stage writes its output under a normalised key (hyphens → underscores).
Conditional router lambdas receive the **full accumulated state**, not just
the last stage's output — this mirrors LangGraph's ``StateGraph`` approach
and makes routing logic self-documenting.

Usage
-----
::

    from typing import TypedDict
    from synaptum.patterns.graph_builder import END, GraphBuilder
    from synaptum.agents.simple_agent import SimpleAgent

    class AppState(TypedDict, total=False):
        application: dict
        credit_check: dict

    check  = SimpleAgent("credit-check", output_model=CreditResult, ...)
    decide = SimpleAgent("decision", ...)

    processor = (
        GraphBuilder("processor", state=AppState)
        .submit("app.submitted")
        .result("app.decision")
        .add_node(check)
        .add_node(decide)
        .set_entry(check)
        .add_conditional_edges(
            check,
            lambda state: state["credit_check"]["band"].upper(),
            {"GOOD": decide, "POOR": decide},
        )
        .add_edge(decide, END)
        .verbose()
        .build()                   # → GraphAgent (users never import GraphAgent)
    )

    runtime.register(processor)   # registers processor + all child nodes
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


# ── Internal type aliases ─────────────────────────────────────────────────────

_NodeRef = Any   # SimpleAgent instance or plain string name
_Target  = Any   # _NodeRef | _EndSentinel


class _EndSentinel:
    """
    Marks the terminal node of the graph — analogous to LangGraph's ``END``.
    Singleton: there is only ever one instance.
    """

    _instance: Optional["_EndSentinel"] = None

    def __new__(cls) -> "_EndSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "END"


END = _EndSentinel()


class ParallelNode:
    """
    A synthetic graph node that executes multiple child agents concurrently.

    All children receive the same accumulated state and their results are
    written to the state dict independently (each under its own normalised
    key).  From the graph's perspective this is a single node — it has one
    name, one incoming edge, and one outgoing edge.

    Use the ``parallel()`` factory instead of instantiating directly::

        checks = parallel("risk-checks", credit_check, employment_verify, fraud_scan)

        GraphBuilder(...)
            .add_node(checks)
            .add_edge(intake, checks)
            .add_edge(checks, decision)
    """

    def __init__(self, name: str, *agents: Any) -> None:
        if not agents:
            raise ValueError("parallel() requires at least one agent.")
        self.name   = name
        self.agents = list(agents)

    def __repr__(self) -> str:  # pragma: no cover
        names = ", ".join(a.name for a in self.agents)
        return f"ParallelNode({self.name!r}, [{names}])"


def parallel(name: str, *agents: Any) -> ParallelNode:
    """
    Create a fork/join parallel node.

    Example::

        risk_checks = parallel("risk-checks", credit_check, employment_verify, fraud_scan)

    Parameters
    ----------
    name : str
        Logical name for this parallel stage (used as graph node key).
    *agents
        Two or more ``SimpleAgent`` instances to run concurrently.
    """
    return ParallelNode(name, *agents)


class Graph:
    """
    Compiled, immutable graph topology.  Produced by ``GraphBuilder.build()``.

    Attributes
    ----------
    entry : str
        Name of the entry (start) node.
    nodes : dict[str, Agent]
        Registered node agents keyed by name.
    state_type : type | None
        Optional TypedDict class describing the run-state shape.
    """

    def __init__(
        self,
        entry: str,
        edges: Dict[str, "str | _EndSentinel"],
        conditional_edges: Dict[str, "tuple[Callable[[dict], str], Dict[str, str | _EndSentinel]]"],
        *,
        state_type: Optional[type] = None,
        nodes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._entry             = entry
        self._edges             = edges
        self._conditional_edges = conditional_edges
        self._state_type        = state_type
        self._nodes: Dict[str, Any] = nodes or {}

    @property
    def entry(self) -> str:
        return self._entry

    @property
    def nodes(self) -> Dict[str, Any]:
        """Node agents keyed by name."""
        return self._nodes

    @property
    def state_type(self) -> Optional[type]:
        """TypedDict class for the run state (informational; not enforced at runtime)."""
        return self._state_type

    def next(self, current: str, state: Dict[str, Any]) -> Optional[str]:
        """
        Return the next stage name, or ``None`` if ``current`` is terminal.

        Parameters
        ----------
        current : str
            Name of the stage that just finished.
        state : dict
            Full accumulated run state (original payload + all stage outputs).
            Conditional router lambdas receive this entire dict.

        Raises
        ------
        KeyError
            If the conditional router returns a key not in the mapping,
            or if ``current`` has no outgoing edge.
        """
        if current in self._conditional_edges:
            router_fn, mapping = self._conditional_edges[current]
            key = router_fn(state)
            target = mapping.get(key)
            if target is None:
                raise KeyError(
                    f"No edge for routing key '{key}' from '{current}'. "
                    f"Available keys: {list(mapping.keys())}"
                )
            return None if target is END else str(target)

        if current in self._edges:
            target = self._edges[current]
            return None if target is END else str(target)

        raise KeyError(f"Stage '{current}' has no outgoing edge defined in the graph.")


class GraphBuilder:
    """
    Fluent builder for agent graph state machines.

    Typical usage::

        processor = (
            GraphBuilder("name", state=MyState)
            .submit("task.submitted")
            .result("task.result")
            .add_node(agent_a)
            .add_node(agent_b)
            .set_entry(agent_a)
            .add_edge(agent_a, agent_b)
            .add_edge(agent_b, END)
            .verbose()
            .build()
        )

    All edge methods accept agent instances **or** plain name strings.
    Conditional router lambdas receive the **full accumulated state dict**.
    Call ``build()`` to get the executable ``GraphAgent``.
    """

    def __init__(self, name: str, *, state: type) -> None:
        self._name         = name
        self._state_type   = state
        self._submit_type  = "graph.submitted"
        self._result_type  = "graph.result"
        self._verbose_flag = False
        self._nodes: Dict[str, Any]                    = {}
        self._entry: Optional[str]                     = None
        self._edges: Dict[str, "str | _EndSentinel"]  = {}
        self._conditional_edges: Dict[
            str, "tuple[Callable[[dict], str], Dict[str, str | _EndSentinel]]"
        ] = {}

    # ── Configuration ─────────────────────────────────────────────────────────

    def submit(self, msg_type: str) -> "GraphBuilder":
        """Message type that triggers a new graph run (default: ``'graph.submitted'``)."""
        self._submit_type = msg_type
        return self

    def result(self, msg_type: str) -> "GraphBuilder":
        """Message type used to deliver the final result (default: ``'graph.result'``)."""
        self._result_type = msg_type
        return self

    def verbose(self, enabled: bool = True) -> "GraphBuilder":
        """Print stage-transition progress to stdout during execution."""
        self._verbose_flag = enabled
        return self

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_name(node: _NodeRef) -> str:
        """Extract the string name from an agent instance or a bare string."""
        return node.name if hasattr(node, "name") else str(node)

    @staticmethod
    def _resolve_target(node: _Target) -> "str | _EndSentinel":
        """Resolve an edge target to a name string or END."""
        return node if isinstance(node, _EndSentinel) else GraphBuilder._resolve_name(node)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def add_node(self, agent: _NodeRef) -> "GraphBuilder":
        """
        Register an agent as a graph node.

        The agent's ``name`` is used as the node key and as the state key
        (hyphens are converted to underscores when writing to the run state).
        """
        self._nodes[self._resolve_name(agent)] = agent
        return self

    # ── Topology ──────────────────────────────────────────────────────────────

    def set_entry(self, node: _NodeRef) -> "GraphBuilder":
        """Set the entry (start) node — accepts an agent instance or name string."""
        self._entry = self._resolve_name(node)
        return self

    def add_edge(self, from_node: _NodeRef, to_node: _Target) -> "GraphBuilder":
        """
        Add an unconditional edge.

        Both endpoints accept agent instances or name strings.
        ``to_node`` may be ``END`` to mark a terminal node.
        Raises ``ValueError`` if ``from_node`` already has a conditional edge.
        """
        from_name = self._resolve_name(from_node)
        if from_name in self._conditional_edges:
            raise ValueError(
                f"'{from_name}' already has conditional edges; "
                "a node cannot have both unconditional and conditional edges."
            )
        self._edges[from_name] = self._resolve_target(to_node)
        return self

    def add_conditional_edges(
        self,
        from_node: _NodeRef,
        router_fn: Callable[[Dict[str, Any]], str],
        mapping: Dict[str, _Target],
    ) -> "GraphBuilder":
        """
        Add conditional edges from ``from_node``.

        Parameters
        ----------
        from_node : agent or str
            Node whose output determines the next stage.
        router_fn : callable
            ``(state: dict) → str`` — receives the **full accumulated run state**
            and returns the routing key.  Example::

                lambda state: state["credit_check"]["band"].upper()

        mapping : dict
            Maps routing keys to the next node (agent instance, name string,
            or ``END``).

        Raises ``ValueError`` if ``from_node`` already has an unconditional edge.
        """
        from_name = self._resolve_name(from_node)
        if from_name in self._edges:
            raise ValueError(
                f"'{from_name}' already has an unconditional edge; "
                "a node cannot have both unconditional and conditional edges."
            )
        resolved = {k: self._resolve_target(v) for k, v in mapping.items()}
        self._conditional_edges[from_name] = (router_fn, resolved)
        return self

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self) -> Any:
        """
        Validate and return a ``GraphAgent`` that owns and drives all nodes.

        When registered with an ``AgentRuntime``, the returned ``GraphAgent``
        automatically registers its child node agents — a single
        ``runtime.register(processor)`` suffices.

        Raises ``ValueError`` for a missing entry point, no registered nodes,
        or an entry node that was not registered via ``add_node()``.
        """
        from ..agents.graph_agent import GraphAgent

        if self._entry is None:
            raise ValueError("No entry point set. Call set_entry() first.")
        if not self._nodes:
            raise ValueError(
                "No nodes registered. Call add_node(agent) for each stage agent."
            )
        if self._entry not in self._nodes:
            raise ValueError(
                f"Entry node '{self._entry}' was not registered. "
                "Call add_node() before set_entry()."
            )

        graph = Graph(
            entry             = self._entry,
            edges             = dict(self._edges),
            conditional_edges = dict(self._conditional_edges),
            state_type        = self._state_type,
            nodes             = dict(self._nodes),
        )
        return GraphAgent(
            self._name,
            graph       = graph,
            submit_type = self._submit_type,
            result_type = self._result_type,
            verbose     = self._verbose_flag,
        )
