from dataclasses import dataclass
from typing import Dict, List

from ..core.agent import Agent
from ..core.message import Message
from ..core.context import AgentContext


@dataclass
class RouterPatternConfig:
    name: str
    router_llm_name: str
    specialists: List[str]  # agent names


class RouterPattern(Agent):
    """
    Router bus-driven:
    - Recibe user.input
    - Envía una consulta al router (otro agente LLM) para elegir specialist
    - Reenvía la tarea al specialist
    - El specialist responde al router -> router responde al caller

    Mensajes:
    - Entrada: user.input {text} reply_to=<caller>
    - Router pregunta al router_llm: user.input + lista de specialists
      reply_to=self.agent_id, metadata.route_req_id=<id>
    - Router recibe agent.output del router_llm y reenvía a specialist
      reply_to=self.agent_id, metadata.route_req_id=<id>
    - Router recibe agent.output del specialist y lo entrega al caller original
    """
    def __init__(self, cfg: RouterPatternConfig):
        super().__init__(cfg.name)
        self.cfg = cfg
        self._pending: Dict[str, Dict] = {}  # route_req_id -> {"caller":..., "original_msg_id":...}

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "user.input":
            await self._handle_user_input(message)
            return

        if message.type == "agent.output":
            await self._handle_agent_output(message)
            return

    async def _handle_user_input(self, message: Message) -> None:
        text = ""
        if isinstance(message.payload, dict):
            text = str(message.payload.get("text", ""))
        else:
            text = str(message.payload)

        caller = message.reply_to or message.sender
        route_req_id = message.id  # simple: usar id original como correlación
        self._pending[route_req_id] = {"caller": caller, "original_msg_id": message.id, "text": text}

        prompt = (
            "Elige el mejor especialista para esta tarea.\n"
            f"Especialistas disponibles: {self.cfg.specialists}\n"
            "Responde SOLO con JSON {\"final\":\"<name>\"}.\n"
            f"Tarea:\n{text}"
        )

        await self.runtime.publish(Message(
            sender=self.name,
            recipient=self.cfg.router_llm_name,
            type="user.input",
            payload={"text": prompt},
            reply_to=self.name,
            metadata={"route_req_id": route_req_id},
        ))

    async def _handle_agent_output(self, message: Message) -> None:
        route_req_id = message.metadata.get("route_req_id", "")
        if not route_req_id or route_req_id not in self._pending:
            return

        pending = self._pending[route_req_id]
        txt = ""
        if isinstance(message.payload, dict):
            txt = str(message.payload.get("text", "")).strip()
        else:
            txt = str(message.payload).strip()

        # Caso A: viene del router_llm => txt debe ser el name del specialist
        if message.sender == self.cfg.router_llm_name:
            chosen = txt
            if chosen not in self.cfg.specialists:
                chosen = self.cfg.specialists[0]

            await self.runtime.publish(Message(
                sender=self.name,
                recipient=chosen,
                type="user.input",
                payload={"text": pending["text"]},
                reply_to=self.name,
                metadata={"route_req_id": route_req_id},
            ))
            return

        # Caso B: viene del specialist => entregar al caller y cerrar
        caller = pending["caller"]
        await self.runtime.publish(Message(
            sender=self.name,
            recipient=caller,
            type="agent.output",
            payload={"text": txt},
            metadata={"in_reply_to": pending["original_msg_id"]},
        ))
        del self._pending[route_req_id]
