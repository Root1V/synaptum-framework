"""
RM-19 · El bucle del agente como stream de eventos.

El bucle no es un ``while`` oculto: es un generador asíncrono que **cede el
control en cada frontera significativa**.  Quien itera puede mirar, medir,
aprobar, interrumpir o guardar — sin que el bucle sepa quién está al otro lado::

    async for step in agent.run(tarea, session=session):
        match step:
            case ModelStep(phase=Phase.COMPLETED, usage=u):  ...
            case ToolStep(phase=Phase.ATTEMPTED, risk=Risk.DESTRUCTIVE): ...
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

import json
import time
from dataclasses import dataclass, field, replace
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
from ..core.errors import NoObjectGeneratedError
from ..core.protocols import CallContext, Checkpointer, Gateway
from ..core.types import (
    Message,
    Request,
    Response,
    ResponseFormat,
    Risk,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from ..schema.protocol import Schema, schema_for
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
        instructions: Any = None,
        tools: Sequence[Any] = (),
        output: Any = None,
        limits: Limits | None = None,
    ) -> None:
        self.name = name
        self.model = model
        # Acepta una cadena o cualquier cosa con `render()` — una plantilla
        # versionada, sin que el bucle tenga que importar el sistema de prompts.
        self.instructions: str | None = (
            instructions.render() if hasattr(instructions, "render") else instructions
        )
        # Acepta ToolDefinition o cualquier objeto que la exponga — un `@tool`,
        # sin que el bucle tenga que importar el decorador.
        self.tools: tuple[ToolDefinition, ...] = tuple(
            t.definition if hasattr(t, "definition") else t for t in tools
        )
        self.limits = limits or Limits()
        self.output: Schema | None = schema_for(output) if output is not None else None
        self._format = (
            ResponseFormat(
                kind="json_schema",
                schema=self.output.json_schema(),
                name=getattr(self.output, "name", "output"),
            )
            if self.output is not None
            else None
        )
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

        closed = replay.closed
        if closed is not None:
            # Un run que ya terminó devuelve lo que pasó, no lo intenta otra vez.
            yield self._rehydrate(closed)
            return

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
                    response_format=self._format,
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
                        run_id=session.run_id, step_id=step_id, step_seq=seq - 1,
                        phase=Phase.ATTEMPTED, at=time.time(), request=request,
                    )
                    await journal.record(intent)
                    yield intent

                    try:
                        response = await self._call_model(session, request, step_id)
                    except Denied as denial:
                        async for event in self._close_denied(
                            denial, session, journal, seq, total,
                            step=ModelStep(
                                run_id=session.run_id, step_id=step_id, step_seq=seq - 1,
                                phase=Phase.COMPLETED, at=time.time(),
                                decision=denial.decision,
                            ),
                            subject="llamada al modelo",
                        ):
                            yield event
                        return

                    result = ModelStep(
                        run_id=session.run_id, step_id=step_id, step_seq=seq - 1,
                        phase=Phase.COMPLETED, at=time.time(),
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
                        run_id=session.run_id, step_id=step_id, step_seq=seq - 1,
                        phase=Phase.ATTEMPTED, at=time.time(),
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
                            async for event in self._close_denied(
                                denial, session, journal, seq, total,
                                step=ToolStep(
                                    run_id=session.run_id, step_id=step_id, step_seq=seq - 1,
                                    phase=Phase.COMPLETED, at=time.time(),
                                    call=call, risk=risk, idempotent=idempotent,
                                    decision=denial.decision,
                                ),
                                subject=f"herramienta '{call.name}'",
                            ):
                                yield event
                            return

                    tool_result = ToolStep(
                        run_id=session.run_id, step_id=step_id, step_seq=seq - 1,
                        phase=Phase.COMPLETED, at=time.time(),
                        call=call, result=outcome, risk=risk, idempotent=idempotent,
                    )
                    await journal.record(tool_result)
                    yield tool_result
                    results.append(outcome)

                messages.append(Message.tool_results(*results))

            typed = self._final_output(messages[-1].text)
            final = FinalStep(
                run_id=session.run_id, step_id=make_step_id(seq, "final"), step_seq=seq,
                phase=Phase.COMPLETED, at=time.time(),
                output=self.output.dump(typed) if self.output is not None else typed,
                usage=total,
                meta={"replayed_steps": replay.replayed} if replay.replayed else {},
            )
            # El journal guarda la forma serializable; quien itera recibe el objeto.
            # La salida tipada se **deriva**, no se almacena — lo mismo que el
            # contexto, y por la misma razón: guardar las dos arriesga que
            # discrepen.
            await journal.record(final)
            yield replace(final, output=typed)
        finally:
            # Se vacía también si alguien deja de iterar a mitad: lo diferido no
            # puede quedarse en memoria cuando el run se interrumpe.
            await journal.flush()

    # ── Efectos, todos a través de la costura ─────────────────────────────────

    async def _call_model(self, session: Session, request: Request, step_id: str) -> Response:
        async def once() -> Response:
            response = await session.gateway.invoke_model(
                request, self._ctx(session, step_id)
            )
            # La validación entra **dentro** del reintento a propósito: un objeto
            # mal formado es reintentable —el muestreo es estocástico— y la
            # taxonomía de errores ya lo dice, así que basta con levantarlo aquí.
            if self.output is not None and not response.tool_calls:
                self._validate(response.message.text)
            return response

        return await self._with_retries(once)

    # ── Salida estructurada — SYN-16 ──────────────────────────────────────────

    def _validate(self, text: str) -> Any:
        """Parsea y valida.  Levanta ``NoObjectGeneratedError``, que es reintentable."""
        assert self.output is not None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as broken:
            raise NoObjectGeneratedError(
                f"Se pidió salida estructurada y no volvió JSON: {broken}", raw=text
            ) from broken
        return self.output.validate(data)

    def _final_output(self, text: str) -> Any:
        """Lo que llega al ``FinalStep``: el objeto tipado, o el texto tal cual."""
        return self._validate(text) if self.output is not None else text

    def _rehydrate(self, closed: StepEvent) -> StepEvent:
        """Reconstruye la salida tipada de un run que ya estaba cerrado.

        Sin esto, reanudar devolvería un diccionario donde la primera vuelta
        devolvió un objeto — la misma llamada dando tipos distintos según
        hubiera corrido antes o no.
        """
        if self.output is None or getattr(closed, "output", None) is None:
            return closed
        return replace(closed, output=self.output.validate(closed.output))

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

    async def _close_denied(
        self,
        denial: Denied,
        session: Session,
        journal: Journal,
        seq: int,
        total: Usage,
        *,
        step: StepEvent,
        subject: str,
    ) -> AsyncIterator[StepEvent]:
        """Registra el desenlace del paso denegado y cierra como corresponda.

        El ``RESULT`` del paso se escribe **con la decisión y sin efecto**.  Es
        lo que permite que una reanudación sepa que ahí no pasó nada, en vez de
        encontrarse una intención huérfana y tratarla como el caso incierto.

        ``require_approval`` suspende: emite un ``ApprovalStep`` y termina el
        stream **sin** cerrar el run.  Volver a llamar con el mismo ``run_id``
        retoma donde quedó.  ``terminate_run`` cierra de verdad.
        """
        await journal.record(step)
        yield step

        if denial.disposition is Disposition.REQUIRE_APPROVAL:
            pause = ApprovalStep(
                run_id=session.run_id, step_id=make_step_id(seq, "approval"), step_seq=seq,
                phase=Phase.ATTEMPTED, at=time.time(),
                subject=subject, decision=denial.decision,
            )
            await journal.record(pause)
            yield pause
            return

        closing = FinalStep(
            run_id=session.run_id, step_id=make_step_id(seq, "final"), step_seq=seq,
            phase=Phase.COMPLETED, at=time.time(),
            output=None, usage=total,
            meta={
                "disposition": denial.disposition.value,
                "reason_code": denial.decision.reason_code,
                "message": denial.decision.message,
            },
        )
        await journal.record(closing)
        yield closing
