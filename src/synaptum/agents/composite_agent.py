"""
CompositeAgent — base class for agents that compose a topology of sub-agents.
=============================================================================

Inherits ``_ref`` and ``_bind_runtime`` from ``MessageAgent`` — every
composite agent needs bus access to fan out its first message, so the wiring
comes for free.

``sub_agents()`` declares the internal agents that form the topology.
``AgentRuntime.register()`` automatically registers all of them when this
composite is registered — no manual ``runtime.register()`` calls needed.

Use this as the base class whenever an agent's primary job is to wire a group
of simpler agents together (e.g. SagaAgent, HITLAgent, ConsensusAgent).
The composite itself is also a full Agent: it can receive messages via
``on_message`` and act as the public entry point for the topology.
"""

from __future__ import annotations

from typing import List

from .message_agent import MessageAgent


class CompositeAgent(MessageAgent):
    """
    Base for agents that own and fan-out to a topology of sub-agents.

    Subclasses must implement ``on_message`` (entry point) and may override
    ``sub_agents()`` to declare their internal topology.

    ``_ref`` and ``_bind_runtime`` are inherited from ``MessageAgent`` —
    no boilerplate needed in subclasses.
    """

    def sub_agents(self) -> List[MessageAgent]:
        """
        Return the list of agents that make up this topology.
        Called once by the runtime during registration.
        Override in subclasses to declare the internal topology.
        """
        return []
