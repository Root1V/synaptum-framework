"""
Synaptum — framework de agentes y runtime durable, agnóstico al proveedor.

Es dueño de la *semántica* de ejecución: qué es un paso, dónde puede cortarse,
qué puede repetirse y cómo se re-deriva el contexto.  El *sustrato* — dónde se
persiste, con qué retención y bajo qué política — pertenece al harness.

Estado: v1.0 en construcción.  Fase 0 (contratos) en curso; ver ``roadmap.md``.
"""

from .core import (
    ALLOW,
    AUTO,
    ApprovalStep,
    Audio,
    ContentPart,
    Decision,
    DelegateStep,
    Disposition,
    Document,
    Durability,
    Event,
    FinalStep,
    Finish,
    FinishReason,
    Image,
    Message,
    ModelStep,
    Phase,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    RedactedThinking,
    Request,
    Response,
    ResponseFormat,
    Risk,
    Role,
    StepEvent,
    StreamEvent,
    StreamStart,
    Text,
    TextDelta,
    TextEnd,
    TextStart,
    Thinking,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolChoice,
    ToolDefinition,
    ToolResult,
    ToolStep,
    Usage,
    b64,
    dumps,
    idempotency_key,
    make_step_id,
    to_jsonable,
)
from .core import __all__ as _core_all

__version__ = "1.0.0.dev0"
__all__ = [*_core_all, "__version__"]
