import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AgentAction:
    tool: str
    args: Dict[str, Any]


@dataclass
class AgentDecision:
    action: Optional[AgentAction] = None
    final: Optional[str] = None


class DecisionParser:
    """
    Protocolo LLM -> decision:
      {"action":{"tool":"name","args":{...}}}
      {"final":"texto"}
    """
    def parse(self, text: str) -> AgentDecision:
        text = (text or "").strip()
        if not text:
            return AgentDecision(final="ERROR: empty model output")

        data = self._try_json(text)
        if data is None:
            return AgentDecision(final=text)

        if isinstance(data, dict) and isinstance(data.get("final"), str):
            return AgentDecision(final=data["final"])

        if isinstance(data, dict) and isinstance(data.get("action"), dict):
            act = data["action"]
            tool = act.get("tool")
            args = act.get("args", {})
            if isinstance(tool, str) and isinstance(args, dict):
                return AgentDecision(action=AgentAction(tool=tool, args=args))

        return AgentDecision(final="ERROR: invalid decision JSON")

    def _try_json(self, text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            snippet = text[i:j+1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None

