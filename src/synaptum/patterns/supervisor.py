import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.agent import Agent
from ..core.message import Message
from ..core.context import AgentContext


@dataclass
class SupervisorPatternConfig:
    agent_id: str
    supervisor_llm_agent_id: str
    workers: List[str]  # agent_ids


class SupervisorPattern(Agent):
    """
    Supervisor/Worker bus-driven (sin threads):
    - Recibe user.input
    - Pide al supervisor_llm un plan (tasks con worker+input)
    - Publica cada task al worker, recolecta resultados
    - Pide al supervisor_llm síntesis final y la entrega al caller
    """
    def __init__(self, cfg: SupervisorPatternConfig):
        super().__init__(agent_id=cfg.agent_id)
        self.cfg = cfg
        self._runs: Dict[str, Dict] = {}  # run_key -> state

    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "user.input":
            await self._start_run(message)
            return

        if message.type == "agent.output":
            await self._handle_output(message)
            return

    async def _start_run(self, message: Message) -> None:
        text = str(message.payload.get("text")) if isinstance(message.payload, dict) else str(message.payload)
        caller = message.metadata.get("reply_to") or message.sender
        run_key = message.id

        self._runs[run_key] = {
            "caller": caller,
            "original_msg_id": message.id,
            "task_text": text,
            "phase": "planning",
            "plan": [],
            "pending_workers": set(),
            "worker_results": [],
        }

        prompt = (
            "Crea un plan de subtareas y asigna cada subtarea a un worker.\n"
            f"Workers disponibles (agent_id): {self.cfg.workers}\n"
            "Responde SOLO con JSON {\"final\":\"PLAN_JSON:<json>\"}.\n"
            "El <json> debe ser: {\"tasks\":[{\"worker\":\"agent_id\",\"input\":\"...\"}, ...]}.\n"
            f"Tarea:\n{text}"
        )

        await self.runtime.publish(Message(
            sender=self.agent_id,
            recipient=self.cfg.supervisor_llm_agent_id,
            type="user.input",
            payload={"text": prompt},
            metadata={"reply_to": self.agent_id, "run_key": run_key},
        ))

    async def _handle_output(self, message: Message) -> None:
        run_key = message.metadata.get("run_key") or message.metadata.get("route_req_id") or message.metadata.get("in_reply_to")
        if not isinstance(run_key, str) or run_key not in self._runs:
            return

        run = self._runs[run_key]
        txt = str(message.payload.get("text", "")).strip() if isinstance(message.payload, dict) else str(message.payload).strip()

        if run["phase"] == "planning" and message.sender == self.cfg.supervisor_llm_agent_id:
            plan = self._parse_plan(txt, fallback_task=run["task_text"])
            run["plan"] = plan
            run["phase"] = "executing"

            # dispatch tasks
            for t in plan:
                worker = t["worker"]
                inp = t["input"]
                run["pending_workers"].add(worker)
                await self.runtime.publish(Message(
                    sender=self.agent_id,
                    recipient=worker,
                    type="user.input",
                    payload={"text": inp},
                    metadata={"reply_to": self.agent_id, "run_key": run_key, "worker": worker},
                ))
            return

        if run["phase"] == "executing":
            # worker output
            worker = message.metadata.get("worker") or message.sender
            run["worker_results"].append({"worker": worker, "output": txt})
            if worker in run["pending_workers"]:
                run["pending_workers"].remove(worker)

            if not run["pending_workers"]:
                run["phase"] = "synthesizing"
                synth_prompt = self._build_synth_prompt(run["task_text"], run["plan"], run["worker_results"])
                await self.runtime.publish(Message(
                    sender=self.agent_id,
                    recipient=self.cfg.supervisor_llm_agent_id,
                    type="user.input",
                    payload={"text": synth_prompt},
                    metadata={"reply_to": self.agent_id, "run_key": run_key},
                ))
            return

        if run["phase"] == "synthesizing" and message.sender == self.cfg.supervisor_llm_agent_id:
            caller = run["caller"]
            await self.runtime.publish(Message(
                sender=self.agent_id,
                recipient=caller,
                type="agent.output",
                payload={"text": txt},
                metadata={"in_reply_to": run["original_msg_id"]},
            ))
            del self._runs[run_key]
            return

    def _parse_plan(self, text: str, fallback_task: str) -> List[Dict[str, str]]:
        prefix = "PLAN_JSON:"
        if prefix in text:
            json_part = text.split(prefix, 1)[1].strip()
            try:
                data = json.loads(json_part)
                tasks = data.get("tasks", [])
                if isinstance(tasks, list):
                    out: List[Dict[str, str]] = []
                    for t in tasks:
                        if isinstance(t, dict):
                            w = t.get("worker")
                            i = t.get("input")
                            if isinstance(w, str) and isinstance(i, str) and i.strip():
                                if w not in self.cfg.workers:
                                    w = self.cfg.workers[0]
                                out.append({"worker": w, "input": i})
                    if out:
                        return out
            except Exception:
                pass
        # fallback: 1 task al primer worker
        return [{"worker": self.cfg.workers[0], "input": fallback_task}]

    def _build_synth_prompt(self, task: str, plan: List[Dict[str, str]], results: List[Dict[str, str]]) -> str:
        lines = []
        lines.append("Sintetiza una respuesta final basada en resultados de workers.")
        lines.append("Responde SOLO con JSON {\"final\":\"...\"}.")
        lines.append("")
        lines.append("TAREA ORIGINAL:")
        lines.append(task)
        lines.append("")
        lines.append("PLAN:")
        for t in plan:
            lines.append(f"- worker={t['worker']} input={t['input']}")
        lines.append("")
        lines.append("RESULTADOS:")
        for r in results:
            lines.append(f"- worker={r['worker']} output={r['output']}")
        return "\n".join(lines)
