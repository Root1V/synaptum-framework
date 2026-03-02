"""
MapReduceAgent — fan-out / fan-in coordination pattern.

Design
------
Implements the classic Map-Reduce pattern for LLM agent workflows:

  1. **Split**  — a user-supplied ``splitter`` function breaks the input
                  payload into a list of independent chunks.
  2. **Map**    — a single ``mapper`` agent processes each chunk concurrently
                  via ``asyncio.gather()``.  Every chunk gets the same agent
                  with the same prompt, but sees different data.
  3. **Reduce** — a ``reducer`` agent receives **all** mapped results at once
                  and produces the final aggregated output.

This is distinct from ``parallel()`` / ``ParallelNode``:

  - ``parallel()``  → N *different* agents, **same** input (fork/join)
  - ``MapReduceAgent`` → **same** agent, N *different* chunks (fan-out/fan-in)

Usage
-----
::

    from synaptum.patterns.map_reduce import MapReduceAgent

    analyst  = SimpleAgent("loan-analyst",  output_model=LoanAssessment, ...)
    manager  = SimpleAgent("portfolio-mgr", output_model=PortfolioReport, ...)

    processor = MapReduceAgent(
        "portfolio-processor",
        mapper   = analyst,
        reducer  = manager,
        splitter = lambda payload: payload["loans"],   # list[dict]
        submit_type = "portfolio.submitted",
        result_type = "portfolio.report",
        verbose  = True,
    )

    runtime.register(processor)   # also registers mapper and reducer
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

from ..core.agent import Agent
from ..core.context import AgentContext
from ..core.message import Message
from ..agents.agent_ref import AgentRef


class MapReduceAgent(Agent):
    """
    Splits an input payload into chunks, maps a single agent over them
    concurrently, then reduces all results with an aggregator agent.

    Created directly by the user — unlike ``GraphAgent`` this is a public
    pattern class.

    Parameters
    ----------
    name : str
        Bus address for this agent.
    mapper : Agent
        Agent whose ``think()`` is called once per chunk (concurrently).
    reducer : Agent
        Agent whose ``think()`` is called once with all mapped results.
    splitter : callable
        ``(payload: dict) → list[dict]``.  Extracts the list of items to
        process from the raw submit payload.
    submit_type : str
        Message type that triggers a new run (default: ``'mapreduce.submitted'``).
    result_type : str
        Message type used to deliver the final result (default: ``'mapreduce.result'``).
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        name: str,
        *,
        mapper: Agent,
        reducer: Agent,
        splitter: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        submit_type: str = "mapreduce.submitted",
        result_type: str = "mapreduce.result",
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.mapper      = mapper
        self.reducer     = reducer
        self.splitter    = splitter
        self.submit_type = submit_type
        self.result_type = result_type
        self.verbose     = verbose
        self._ref: Optional[AgentRef] = None

    # ── Runtime binding ───────────────────────────────────────────────────────

    def _bind_runtime(self, runtime) -> None:
        self._ref = AgentRef(self.name, runtime._bus)
        runtime.register(self.mapper)
        runtime.register(self.reducer)

    # ── Message handling ──────────────────────────────────────────────────────

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == self.submit_type:
            await self._execute(message)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute(self, message: Message) -> None:
        """Run split → map (parallel) → reduce, deliver result to caller."""
        if self._ref is None:
            raise RuntimeError(
                f"MapReduceAgent '{self.name}' has not been bound to a runtime."
            )

        payload = (
            message.payload
            if isinstance(message.payload, dict)
            else {"data": message.payload}
        )
        caller = message.reply_to or message.sender

        # ── 1. Split ──────────────────────────────────────────────────────────
        chunks: List[Dict[str, Any]] = self.splitter(payload)
        if not chunks:
            raise ValueError(
                f"MapReduceAgent '{self.name}': splitter returned an empty list."
            )

        if self.verbose:
            print(
                f"\n── [{self.name}]: {len(chunks)} chunks "
                f"→ map({self.mapper.name}) → reduce({self.reducer.name}) ──"
            )

        # ── 2. Map (concurrent) ───────────────────────────────────────────────
        map_prompts = [
            self._map_prompt(chunk, index=i, total=len(chunks))
            for i, chunk in enumerate(chunks)
        ]

        map_results_raw = await asyncio.gather(
            *[self.mapper.think(p) for p in map_prompts]
        )

        map_results: List[Dict[str, Any]] = []
        for raw in map_results_raw:
            map_results.append(
                raw.model_dump() if hasattr(raw, "model_dump") else raw
            )

        if self.verbose:
            print(f"  map phase complete — {len(map_results)} results")

        # ── 3. Reduce ─────────────────────────────────────────────────────────
        reduce_prompt = self._reduce_prompt(payload, chunks, map_results)
        reduce_raw    = await self.reducer.think(reduce_prompt)
        reduce_result = (
            reduce_raw.model_dump()
            if hasattr(reduce_raw, "model_dump")
            else reduce_raw
        )

        if self.verbose:
            print(f"  reduce phase complete → sending '{self.result_type}' to '{caller}'")

        # ── 4. Deliver ────────────────────────────────────────────────────────
        await self._ref.send(
            to      = caller,
            type    = self.result_type,
            payload = {
                "result":      reduce_result,
                "map_results": map_results,
                "chunks":      chunks,
                "input":       payload,
            },
            metadata = {"in_reply_to": message.id},
        )

    # ── Prompt helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _map_prompt(chunk: Dict[str, Any], *, index: int, total: int) -> str:
        """Format a single chunk as a map-phase prompt."""
        body = "\n".join(f"  {k}: {v}" for k, v in chunk.items())
        return (
            f"Item {index + 1} of {total}:\n"
            f"{body}\n\n"
            "Analyse this item and respond with the required JSON schema."
        )

    @staticmethod
    def _reduce_prompt(
        payload: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        map_results: List[Dict[str, Any]],
    ) -> str:
        """Format all mapped results as a reduce-phase prompt."""
        context_lines: List[str] = []

        # Include any top-level payload fields that are not the chunked list
        for k, v in payload.items():
            if not isinstance(v, list):
                context_lines.append(f"{k}: {v}")

        context_lines.append(
            f"\nTotal items processed: {len(map_results)}\n"
        )

        for i, (chunk, result) in enumerate(zip(chunks, map_results)):
            item_id = chunk.get("id") or chunk.get("loan_id") or f"item-{i + 1}"
            context_lines.append(f"\n── Item {i + 1}  ({item_id}) ──")
            for k, v in result.items():
                if isinstance(v, list):
                    context_lines.append(f"  {k}: {', '.join(str(x) for x in v)}")
                else:
                    context_lines.append(f"  {k}: {v}")

        return (
            "You have received the individual analysis for each item below.\n"
            "Synthesise all results into a single aggregate report.\n\n"
            + "\n".join(context_lines)
            + "\n\nRespond with the required JSON schema."
        )
