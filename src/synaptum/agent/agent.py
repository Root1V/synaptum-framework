"""
RM-19 · El bucle del agente como stream de eventos.

El bucle no es un ``while`` oculto: es un generador asíncrono que **cede el
control en cada frontera significativa**.  Quien itera puede mirar, medir,
aprobar, interrumpir o guardar — sin que el bucle sepa quién está al otro lado::

    async for step in agent.run(tarea, session=session):
        match step:
            case ModelStep(phase=Phase.RESULT, usage=u):  ...
            case ToolStep(phase=Phase.INTENT, risk=Risk.DESTRUCTIVE): ...
            case FinalStep(output=salida): ...

Cuatro propiedades salen de esa forma, y ninguna otra estructura las da a la vez:

1. **El harness obtiene sus puntos de enganche** sin que Synaptum sepa que
   existe.  Aprobaciones, guardarraíles y métricas son consumidores del stream.
2. **Cada ``yield`` es una frontera de checkpoint natural.**
3. **Interrumpir es dejar de iterar**; reanudar es volver a llamar con el mismo
   ``run_id``.
4. **Probar es iterar una lista.**

Qué ejecuta el bucle y qué no
------------------------------
En modo gobernado, el bucle **no ejecuta nada**.  Decide qué hacer y le pide al
``Gateway`` que lo haga, porque ahí es donde vive la credencial.  Ni la llamada
al modelo ni la ejecución de una tool ocurren en este proceso.

Reanudación
-----------
Al empezar, el bucle carga el journal y recorre los pasos desde el principio.
Cada paso con resultado registrado se resuelve leyendo, no ejecutando — y así
**una inferencia ya pagada no se paga otra vez**.  La ventana de contexto no se
almacena: se vuelve a derivar de los mismos resultados, en el mismo orden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from ..core.errors import Denied, LimitExceeded, SynaptumError
from ..core.events import (
    ApprovalStep,
    Disposition,
    FinalStep,
    ModelStep,
    Phase,
    StepEvent,
    ToolStep,
    make_step_id,
)
from ..core.protocols import CallContext, Checkpointer, Gateway
from ..core.types import (
    Message,
    Request,
    Response,
    Risk,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from ..run.journal import Journal, MemoryCheckpointer, Replay

__all__ = ["Limits", "Session", "Agent"]


# ── Límites — RM-27 ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Limits:
    """Topes del bucle.

    Son **corrección, no política**: evitan que un bucle mal formado no termine
    nunca.  Los límites de gasto pertenecen al harness y llegan por la costura
    como ``Denied`` con ``terminate_run``.
    """

    max_steps: int = 50
    max_retries: int = 2
    reserved_output: float = 0.25
    """Fracción de la ventana reservada para la salida.  Informativa hasta que
    entre el ensamblador de contexto (RM-32)."""


# ── Sesión ────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Session:
    """Un run: su identidad, por dónde sale y dónde se recuerda.

    Reanudar es construir una ``Session`` con el mismo ``run_id`` y el mismo
    ``checkpointer``.  No hay nada más.
    """

    run_id: str
    gateway: Gateway
    checkpointer: Checkpointer = field(default_factory=MemoryCheckpointer)


# ── Agente ────────────────────────────────────────────────────────────────────

class Agent:
    """Composición, no herencia.  Un agente es su configuración más el bucle."""

    def __init__(
        self,
        name: str,
        *,
        model: str,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] = (),
        limits: Limits | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tuple(tools)
        self.limits = limits or Limits()
        self._by_name = {t.name: t for t in self.tools}

    # ── Bucle ─────────────────────────────────────────────────────────────────

    async def run(self, task: str, *, session: Session) -> AsyncIterator[StepEvent]:
        """Ejecuta el agente cediendo cada paso.

        La cancelación se propaga: cerrar el generador o cancelar la tarea que
        lo consume interrumpe el paso en vuelo y vacía lo pendiente del journal.
        """
        state = await session.checkpointer.load(session.run_id)
        journal = Journal(session.checkpointer, session.run_id)
        replay = Replay(state)

        messages: list[Message] = [Message.user(task)]
        total = Usage.zero()
        seq = 0
        turns = 0

        try:
            while True:
                if turns >= self.limits.max_steps:
                    raise LimitExceeded("max_steps", self.limits.max_steps)
                turns += 1

                # ── Paso de modelo ────────────────────────────────────────────
                step_id = make_step_id(seq, "model")
                seq += 1
                request = Request(
                    model=self.model,
                    system=self.instructions,
                    messages=tuple(messages),
                    tools=self.tools,
                )

                # Una llamada al modelo no tiene efecto externo más allá de su
                # coste, así que repetirla tras una caída es caro pero correcto.
                done = replay.resolve(step_id, idempotent=True)
                if done is not None:
                    assert isinstance(done, ModelStep) and done.response is not None
                    response = done.response
                    yield done
                else:
                    intent = ModelStep(
                        run_id=session.run_id, step_id=step_id, seq=seq - 1,
                        phase=Phase.INTENT, at=time.time(), request=request,
                    )
                    await journal.record(intent)
                    yield intent

                    try:
                        response = await self._call_model(session, request, step_id)
                    except Denied as denial:
                        async for event in self._on_denial(
                            denial, session, journal, seq, total, subject="llamada al modelo"
                        ):
                            yield event
                        return

                    result = ModelStep(
                        run_id=session.run_id, step_id=step_id, seq=seq - 1,
                        phase=Phase.RESULT, at=time.time(),
                        response=response, usage=response.usage,
                    )
                    await journal.record(result)
                    yield result

                total += response.usage
                messages.append(response.message)

                calls = response.tool_calls
                if not calls:
                    break

                # ── Pasos de herramienta ──────────────────────────────────────
                results: list[ToolResult] = []
                for call in calls:
                    step_id = make_step_id(seq, "tool")
                    seq += 1
                    spec = self._by_name.get(call.name)
                    risk = spec.risk if spec else Risk.READ
                    idempotent = spec.idempotent if spec else False

                    done = replay.resolve(step_id, idempotent=idempotent)
                    if done is not None:
                        assert isinstance(done, ToolStep) and done.result is not None
                        results.append(done.result)
                        yield done
                        continue

                    intent = ToolStep(
                        run_id=session.run_id, step_id=step_id, seq=seq - 1,
                        phase=Phase.INTENT, at=time.time(),
                        call=call, risk=risk, idempotent=idempotent,
                    )
                    await journal.record(intent)
                    yield intent

                    try:
                        outcome = await self._call_tool(session, call, step_id, spec)
                    except Denied as denial:
                        if denial.disposition is Disposition.DENY_STEP:
                            # El bucle puede intentar otra cosa: se le devuelve
                            # la negativa al modelo para que rectifique.
                            outcome = ToolResult.of(
                                call.id,
                                f"Denegado: {denial.decision.reason_code or 'política'}. "
                                f"{denial.decision.message}".strip(),
                                is_error=True,
                            )
                        else:
                            async for event in self._on_denial(
                                denial, session, journal, seq, total,
                                subject=f"herramienta '{call.name}'",
                            ):
                                yield event
                            return

                    tool_result = ToolStep(
                        run_id=session.run_id, step_id=step_id, seq=seq - 1,
                        phase=Phase.RESULT, at=time.time(),
                        call=call, result=outcome, risk=risk, idempotent=idempotent,
                    )
                    await journal.record(tool_result)
                    yield tool_result
                    results.append(outcome)

                messages.append(Message.tool_results(*results))

            final = FinalStep(
                run_id=session.run_id, step_id=make_step_id(seq, "final"), seq=seq,
                phase=Phase.RESULT, at=time.time(),
                output=messages[-1].text, usage=total,
                meta={"replayed_steps": replay.replayed} if replay.replayed else {},
            )
            await journal.record(final)
            yield final
        finally:
            # Se vacía también si alguien deja de iterar a mitad: lo diferido no
            # puede quedarse en memoria cuando el run se interrumpe.
            await journal.flush()

    # ── Efectos, todos a través de la costura ─────────────────────────────────

    async def _call_model(self, session: Session, request: Request, step_id: str) -> Response:
        return await self._with_retries(
            lambda: session.gateway.invoke_model(request, self._ctx(session, step_id))
        )

    async def _call_tool(
        self, session: Session, call: ToolCall, step_id: str, spec: ToolDefinition | None
    ) -> ToolResult:
        ctx = self._ctx(session, step_id)
        risk = spec.risk if spec else Risk.READ
        ref = spec.ref if spec else None

        async def once() -> ToolResult:
            return await session.gateway.invoke_tool(call, ctx, risk=risk, tool_ref=ref)

        # Solo se reintenta lo que puede repetirse sin consecuencias.
        if spec is not None and spec.idempotent:
            return await self._with_retries(once)
        return await once()

    async def _with_retries(self, operation: Any) -> Any:
        """Reintenta mientras el error se declare reintentable.

        La decisión no es del bucle: viaja en el tipo del error, que la trae de
        quien habló con el proveedor.
        """
        attempt = 0
        while True:
            try:
                return await operation()
            except SynaptumError as error:
                if not error.retryable or attempt >= self.limits.max_retries:
                    raise
                attempt += 1

    def _ctx(self, session: Session, step_id: str) -> CallContext:
        return CallContext(run_id=session.run_id, step_id=step_id)

    # ── Cierre por denegación ─────────────────────────────────────────────────

    async def _on_denial(
        self,
        denial: Denied,
        session: Session,
        journal: Journal,
        seq: int,
        total: Usage,
        *,
        subject: str,
    ) -> AsyncIterator[StepEvent]:
        """Traduce una denegación terminal en el evento de cierre que le toca.

        ``require_approval`` suspende: emite un ``ApprovalStep`` y termina el
        stream **sin** cerrar el run.  Volver a llamar con el mismo ``run_id``
        retoma donde quedó.  ``terminate_run`` cierra de verdad.
        """
        if denial.disposition is Disposition.REQUIRE_APPROVAL:
            pause = ApprovalStep(
                run_id=session.run_id, step_id=make_step_id(seq, "approval"), seq=seq,
                phase=Phase.INTENT, at=time.time(),
                subject=subject, decision=denial.decision,
            )
            await journal.record(pause)
            yield pause
            return

        closing = FinalStep(
            run_id=session.run_id, step_id=make_step_id(seq, "final"), seq=seq,
            phase=Phase.RESULT, at=time.time(),
            output=None, usage=total,
            meta={
                "disposition": denial.disposition.value,
                "reason_code": denial.decision.reason_code,
                "message": denial.decision.message,
            },
        )
        await journal.record(closing)
        yield closing
