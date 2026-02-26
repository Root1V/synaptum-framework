from dataclasses import dataclass

from ..core.agent import Agent
from ..core.message import Message
from ..core.context import AgentContext


@dataclass
class GraphNode:
    agent_id: str
    prompt_builder: Callable[[str], str]
    next_node: Callable[[str], Optional[str]]  # output_text -> next agent_id or None


class GraphPattern(Agent):
    """
    Graph/state-machine basado en mensajes:
    - Recibe user.input
    - Envía prompt al nodo inicial
    - Recibe agent.output, decide transición, y continúa hasta None
    - Entrega el último output al caller
    """
    def __init__(self, agent_id: str, start_agent_id: str, nodes: Dict[str, GraphNode]):
        super().__init__(agent_id=agent_id)
        self.start_agent_id = start_agent_id
        self.nodes = nodes
        self._runs: Dict[str, Dict] = {}  # run_key -> state

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "user.input":
            await self._start(message)
            return

        if message.type == "agent.output":
            await self._handle_output(message)
            return

    async def _start(self, message: Message) -> None:
        task = str(message.payload.get("text")) if isinstance(message.payload, dict) else str(message.payload)
        caller = message.metadata.get("reply_to") or message.sender
        run_key = message.id

        self._runs[run_key] = {
            "caller": caller,
            "original_msg_id": message.id,
            "task": task,
            "current": self.start_agent_id,
        }

        await self._send_to_current(run_key)

    async def _send_to_current(self, run_key: str) -> None:
        run = self._runs[run_key]
        current = run["current"]
        node = self.nodes[current]
        prompt = node.prompt_builder(run["task"])

        await self.runtime.publish(Message(
            sender=self.agent_id,
            recipient=node.agent_id,
            type="user.input",
            payload={"text": prompt},
            metadata={"reply_to": self.agent_id, "run_key": run_key, "node": current},
        ))

    async def _handle_output(self, message: Message) -> None:
        run_key = message.metadata.get("run_key") or message.metadata.get("in_reply_to")
        if not isinstance(run_key, str) or run_key not in self._runs:
            return

        run = self._runs[run_key]
        node_name = message.metadata.get("node")
        if not isinstance(node_name, str) or node_name not in self.nodes:
            return

        out_text = str(message.payload.get("text", "")).strip() if isinstance(message.payload, dict) else str(message.payload).strip()

        # transición
        nxt = self.nodes[node_name].next_node(out_text)
        if nxt is None:
            # fin -> devolver al caller
            caller = run["caller"]
            await self.runtime.publish(Message(
                sender=self.agent_id,
                recipient=caller,
                type="agent.output",
                payload={"text": out_text},
                metadata={"in_reply_to": run["original_msg_id"]},
            ))
            del self._runs[run_key]
            return

        if nxt not in self.nodes:
            caller = run["caller"]
            await self.runtime.publish(Message(
                sender=self.agent_id,
                recipient=caller,
                type="agent.output",
                payload={"text": out_text},
                metadata={"in_reply_to": run["original_msg_id"]},
            ))
            del self._runs[run_key]
            return

        # avanzar
        run["current"] = nxt
        await self._send_to_current(run_key)
