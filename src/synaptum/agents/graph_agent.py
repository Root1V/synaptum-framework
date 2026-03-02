"""
GraphAgent — drives a compiled Graph as a sequential state machine.

Created exclusively by ``GraphBuilder.build()``. Users never instantiate
this class directly; import only ``GraphBuilder`` and ``END``.

Execution model
---------------
When a submit message arrives, ``_execute_run`` calls each node agent's
``think()`` method directly (no message-bus round-trips for internal stages).
Each stage output is written into the run-state dict under a normalised key
(hyphens → underscores).  The conditional router lambdas defined in
``GraphBuilder.add_conditional_edges`` receive the **full accumulated state**
so they can read any prior stage output.

Child-node registration
-----------------------
When the ``GraphAgent`` is registered with an ``AgentRuntime``, it
automatically calls ``runtime.register()`` for every child node agent,
so a single ``runtime.register(processor)`` is sufficient.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..patterns.graph_builder import Graph, ParallelNode
from ..agents.agent_ref import AgentRef


class GraphAgent(Agent):
    """
    Executes a ``Graph`` by calling each node agent's ``think()`` directly.

    Created by ``GraphBuilder.build()`` — do not instantiate manually.

    When registered with a runtime, automatically registers all child node
    agents so they receive prompt injection and LLM setup.
    """

    def __init__(
        self,
        name: str,
        graph: Graph,
        *,
        submit_type: str = "graph.submitted",
        result_type: str = "graph.result",
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.graph       = graph
        self.submit_type = submit_type
        self.result_type = result_type
        self.verbose     = verbose
        self._ref: Optional[AgentRef] = None

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        for node_agent in self.graph.nodes.values():
            if isinstance(node_agent, ParallelNode):
                for child in node_agent.agents:
                    runtime.register(child)
            else:
                runtime.register(node_agent)

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute_run(message)

    async def _execute_run(self, message: Message) -> None:
        """Drive the graph: call each node's think() in sequence, route via state."""
        if self._ref is None:
            raise RuntimeError(
                f"GraphAgent '{self.name}' has not been bound to a runtime."
            )

        input_payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender
        state: Dict[str, Any] = dict(input_payload)

        stage_name: Optional[str] = self.graph.entry
        last_key: str = stage_name.replace("-", "_")

        if self.verbose:
            print(f"\n\u2500\u2500 [{self.name}]: starting at '{stage_name}' \u2500\u2500")

        while stage_name is not None:
            node = self.graph.nodes.get(stage_name)
            if node is None:
                raise KeyError(
                    f"No node registered for '{stage_name}'. "
                    "Add it with add_node() in GraphBuilder."
                )

            last_key = stage_name.replace("-", "_")
            prompt   = f"Current state:\n{self._format_state(state)}\n\nProcess this stage."

            if isinstance(node, ParallelNode):
                # ── Fork/join: run all children concurrently ──────────────────
                child_names = ", ".join(c.name for c in node.agents)
                if self.verbose:
                    print(f"  {stage_name} [parallel: {child_names}]")

                results = await asyncio.gather(
                    *[child.think(prompt) for child in node.agents]
                )
                merged: Dict[str, Any] = {}
                for child, result_obj in zip(node.agents, results):
                    child_key = child.name.replace("-", "_")
                    child_dict = (
                        result_obj.model_dump()
                        if hasattr(result_obj, "model_dump")
                        else result_obj
                    )
                    state[child_key] = child_dict
                    merged[child_key] = child_dict
                state[last_key] = merged          # also accessible as one aggregate
            else:
                # ── Sequential node ───────────────────────────────────────────
                result_obj  = await node.think(prompt)
                result_dict = (
                    result_obj.model_dump()
                    if hasattr(result_obj, "model_dump")
                    else result_obj
                )
                state[last_key] = result_dict

            next_stage = self.graph.next(stage_name, state)

            if self.verbose and not isinstance(node, ParallelNode):
                print(f"  {stage_name} \u2192 {next_stage if next_stage is not None else '[terminal]'}")
            elif self.verbose:
                print(f"  {stage_name} \u2192 {next_stage if next_stage is not None else '[terminal]'}")

            stage_name = next_stage

        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = {
                "result": state[last_key],
                "state":  state,
                "input":  input_payload,
            },
            metadata = {"in_reply_to": message.id},
        )

    @staticmethod
    def _format_state(state: Dict[str, Any]) -> str:
        """Format run state as structured text for the next stage agent."""
        lines: list[str] = []
        for key, value in state.items():
            if isinstance(value, dict):
                lines.append(f"\n[{key}]")
                lines += [f"  {k}: {v}" for k, v in value.items()]
            elif isinstance(value, list):
                lines.append(f"{key}: {', '.join(str(i) for i in value)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
