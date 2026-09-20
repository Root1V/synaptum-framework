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

import asyncio
import json
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Mapping, Sequence

from ..core.errors import (
    ConfigurationError,
    Denied,
    LimitExceeded,
    ProviderError,
    SynaptumError,
)
from ..core.events import (
    ApprovalStep,
    DelegateStep,
    Disposition,
    FinalStep,
    ModelStep,
    Phase,
    StepEvent,
    ToolStep,
    make_step_id,
)
from ..core.errors import NoObjectGeneratedError
from ..core.protocols import CallContext, Checkpointer, Gateway, RunState
from ..core.types import (
    Message,
    Request,
    Response,
    ResponseFormat,
    Risk,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
    dumps,
)
from ..schema.protocol import Schema, schema_for
from ..context.cap import DEFAULT_MAX_CHARS, cap_tool_output
from ..context.prefix import describe_prefix_change, prefix_fingerprint
from .delegation import Delegate, delegate_risk, sub_run_id
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
    max_delegation_depth: int = 3
    """Cuántos niveles de subagente se permiten.

    ``max_steps`` no lo cubre: cada nivel tiene su propio contador, así que dos
    agentes que se deleguen mutuamente no terminarían nunca."""
    max_tool_chars: int | None = DEFAULT_MAX_CHARS
    """Tope de la salida de una herramienta **en el contexto**, en caracteres.

    ``None`` lo desactiva.  El journal guarda el resultado entero pase lo que
    pase: esto solo decide qué se le reenvía al modelo en cada turno.

    Está activado por defecto porque no recortar falla **en silencio**: con una
    ventana pequeña revienta, y con una grande solo cuesta dinero en cada turno
    posterior, que es peor porque nadie lo mira."""
    max_retries: int = 2
    retry_base: float = 0.5
    """Espera inicial entre reintentos, en segundos.  Se dobla en cada intento."""
    max_retry_wait: float = 30.0
    """Techo de la espera, y también de un ``Retry-After`` desmedido: quien
    gobierna decide cuánto puede tardar un run, no el otro extremo."""
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
        delegates: Sequence[Any] = (),
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
        # Un subagente se presenta al modelo como una herramienta de un solo
        # parámetro —el brief—, así que el catálogo del padre no crece con el
        # del hijo. Eso es lo que hace barato delegar.
        self.delegates: tuple[Delegate, ...] = tuple(
            d if isinstance(d, Delegate) else Delegate(d) for d in delegates
        )
        self._delegates_by_name = {d.name: d for d in self.delegates}

        propias = tuple(t.definition if hasattr(t, "definition") else t for t in tools)
        self.tools: tuple[ToolDefinition, ...] = propias + tuple(
            d.definition for d in self.delegates
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
        # El esquema va **también en las instrucciones**, y no solo en
        # `response_format`.
        #
        # No todo proveedor admite el campo.  Hay gateways que lo descartan y
        # cuyo SDK avisa por warning y sigue — medido contra uno real.  El
        # resultado era el peor posible — la restricción no viajaba, el modelo contestaba en prosa, y el
        # error culpaba al JSON («no volvió JSON») en vez de decir que nadie se lo
        # había pedido.  Un fallo que aparece **después** de pagar la inferencia.
        #
        # Decirlo en el prompt cuesta unos cientos de tokens una vez por turno y
        # funciona en cualquier proveedor.  Donde `response_format` sí se admite,
        # las dos cosas dicen lo mismo y no estorban.
        if self.output is not None:
            self.instructions = _con_esquema(self.instructions, self.output.json_schema())
        self._by_name = {t.name: t for t in self.tools}

    # ── Bucle ─────────────────────────────────────────────────────────────────

    def run(self, task: str, *, session: Session) -> AsyncIterator[StepEvent]:
        """Ejecuta el agente cediendo cada paso.

        La cancelación se propaga: cerrar el generador o cancelar la tarea que
        lo consume interrumpe el paso en vuelo y vacía lo pendiente del journal.

        **No es `async def`**, por el mismo motivo que `Gateway.stream_model`:
        devuelve el iterador del bucle en vez de envolverlo. Un envoltorio que
        hiciera `async for ... yield` parece inocuo y no lo es — al cerrarse
        recibe el `GeneratorExit` y deja el generador de dentro abierto, así que
        la cancelación llega cuando pase el recolector. Sin envoltorio, cerrar
        esto **es** cerrar el bucle.
        """
        return self._loop(task, session, stream=False)

    def stream(
        self, task: str, *, session: Session
    ) -> AsyncIterator[StepEvent | StreamEvent]:
        """Lo mismo, entregando además los fragmentos del modelo según llegan.

        Es el mismo bucle y el mismo journal: lo único que cambia es que la
        llamada al modelo sale por ``stream_model`` y sus eventos se ceden
        intercalados entre la intención del paso y su resultado.

        Va aparte de ``run`` y no como bandera porque **cambia el tipo de lo que
        se cede**. Quien consume ``run`` recibe pasos y puede hacer `match` sobre
        ellos sin una rama para lo que nunca va a llegar.

        Dos cosas que conviene saber antes de usarlo:

        * **Un paso que se reanuda no vuelve a emitir fragmentos.** Ya se pagó, y
          reproducir sus tokens como si estuvieran ocurriendo sería teatro.
        * **Un reintento vuelve a abrir el ciclo.** Los fragmentos ya entregados
          no se retiran —se generaron y se pagaron—, así que un fallo a mitad
          deja lo parcial y el reintento empieza con otro ``stream_start``.
        """
        return self._loop(task, session, stream=True)

    async def _loop(
        self, task: str, session: Session, *, stream: bool, depth: int = 0
    ) -> AsyncIterator[Any]:
        # La profundidad viaja por parámetro y no en el objeto: un `Agent` es
        # configuración reutilizable, y el mismo puede estar en mil runs a la
        # vez. Guardarla dentro haría que dos delegaciones concurrentes se
        # pisaran el contador.
        state = await session.checkpointer.load(session.run_id)
        journal = Journal(session.checkpointer, session.run_id)
        replay = Replay(state)

        closed = replay.closed
        if closed is not None:
            # Un run que ya terminó devuelve lo que pasó, no lo intenta otra vez.
            yield self._rehydrate(closed)
            return

        # El prefijo estable del run: se fija en el primer paso y no cambia.
        # Al reanudar se comprueba contra el que quedó en el journal.
        self._check_prefix(state, replay)

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
                        if stream:
                            # El adaptador ya acumula la respuesta completa en el
                            # `Finish`: quien consumió los fragmentos no debería
                            # tener que reconstruirla.
                            response = None
                            fragments = self._stream_model(session, request, step_id)
                            try:
                                async for fragment in fragments:
                                    if fragment.kind == "finish":
                                        response = fragment.response
                                    yield fragment
                            finally:
                                # Cerrar **aquí** y no dejarlo al recolector: si
                                # quien consume se va a mitad, el iterador de la
                                # costura tiene que cerrarse ya, porque cerrarlo
                                # es lo que para la generación arriba.  Fiarlo al
                                # GC hace que la cancelación llegue tarde, o en
                                # otro momento cada vez — y el modelo sigue
                                # generando, y facturando, mientras tanto.
                                await fragments.aclose()
                            if response is None:
                                raise ProviderError(
                                    "el stream terminó sin evento de cierre",
                                    retryable=True,
                                )
                        else:
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

                # Una respuesta servida desde una clave de idempotencia **no se
                # generó ahora**: su consumo describe la generación original, que
                # ya se contó.  Sumarlo otra vez no falla ni avisa — solo hace
                # que el total del run sea mayor que lo que costó.
                #
                # El paso sí lo registra tal cual: el journal cuenta lo que el
                # proveedor dijo, y el total cuenta lo que se pagó.  Son cosas
                # distintas y conviene que no se mezclen.
                if not response.provider_metadata.get("idempotent_replay"):
                    total += response.usage
                messages.append(response.message)

                calls = response.tool_calls
                if not calls:
                    break

                # ── Pasos de herramienta ──────────────────────────────────────
                results: list[ToolResult] = []
                for call in calls:
                    # Una llamada a un subagente no es un paso de herramienta:
                    # tiene su propio diario, su propio consumo y su propio punto
                    # de reanudación. Por eso se despacha aparte.
                    sub = self._delegates_by_name.get(call.name)
                    if sub is not None:
                        step_id = make_step_id(seq, "delegate")
                        seq += 1
                        async for evento, resultado in self._delegate(
                            sub, call, session, journal, replay, step_id, seq - 1, depth
                        ):
                            if evento is not None:
                                yield evento
                            if resultado is not None:
                                results.append(resultado[0])
                                total += resultado[1]
                        continue

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

                # El recorte se aplica **aquí** y no al ejecutar, y el sitio
                # importa: por este punto pasan tanto los resultados recién
                # ejecutados como los que vienen del journal al reanudar. Así el
                # contexto reconstruido es idéntico al original — si se recortara
                # solo en la ejecución, reanudar produciría otro prompt, fallaría
                # la caché y podría cambiar la respuesta.
                #
                # El paso ya se registró con el resultado **entero**: el diario
                # cuenta lo que ocurrió, el contexto lleva lo que el modelo
                # necesita ver.
                messages.append(
                    Message.tool_results(
                        *(
                            cap_tool_output(r, max_chars=self.limits.max_tool_chars)
                            for r in results
                        )
                    )
                )

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

    def _stream_model(
        self, session: Session, request: Request, step_id: str
    ) -> AsyncIterator[StreamEvent]:
        """Lo mismo por la costura de streaming.

        **No es `async def` a propósito**, igual que `Gateway.stream_model`:
        devuelve el iterador para que cerrarlo *sea* la señal de cancelación. Un
        canal que se está cerrando no es sitio para mandar el aviso de que se
        cierra.

        Aquí no hay reintento envolviendo el generador: reintentar por dentro
        obligaría a decidir qué hacer con los fragmentos ya cedidos, y la única
        respuesta honesta —no se retiran— hace que el reintento sea visible de
        todas formas. Que lo decida quien consume.
        """
        return session.gateway.stream_model(request, self._ctx(session, step_id))

    # ── Salida estructurada — SYN-16 ──────────────────────────────────────────

    def _validate(self, text: str) -> Any:
        """Parsea y valida.  Levanta ``NoObjectGeneratedError``, que es reintentable."""
        assert self.output is not None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as broken:
            # Qué volvió, no solo que no parseaba.  «Expecting value: line 1
            # column 1» sobre una cadena vacía no dice nada, y la cadena vacía es
            # justo el caso frecuente: un modelo de razonamiento que se quedó
            # pensando y no llegó a responder.
            detalle = (
                "no volvió texto — el modelo pudo quedarse en la fase de razonamiento"
                if not text.strip()
                else f"volvió: {text.strip()[:200]!r}"
            )
            raise NoObjectGeneratedError(
                f"Se pidió salida estructurada y {detalle} ({broken})", raw=text
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
        """Reintenta mientras el error se declare reintentable, **esperando entre medias**.

        La decisión de *si* reintentar no es del bucle: viaja en el tipo del
        error, que la trae de quien habló con el proveedor. Lo que sí es del
        bucle es *cuándo*.

        Antes reintentaba al instante, y eso no es reintentar: es repetir. Tres
        intentos en dos milisegundos golpean el mismo estado roto y agotan el
        presupuesto antes de que nada haya podido cambiar. Se vio contra una
        plataforma real con un `500` transitorio; el doble no podía enseñarlo
        porque devuelve sus errores sin tardar.

        * Si el otro extremo dijo cuánto esperar, se espera eso. Un `429` con
          `Retry-After` sabe cuándo estará libre, y quien reintenta antes
          empeora la cola en la que está.
        * Si no, espera creciente con **jitter**. El jitter no es adorno: sin
          él, N agentes que fallan por la misma caída reintentan todos a la vez
          y reconstruyen el pico que los tiró.
        """
        attempt = 0
        while True:
            try:
                return await operation()
            except SynaptumError as error:
                if not error.retryable or attempt >= self.limits.max_retries:
                    raise
                await asyncio.sleep(self._espera(attempt, error.retry_after))
                attempt += 1

    def _espera(self, intento: int, pedida: float | None) -> float:
        """Segundos antes del siguiente intento."""
        if pedida is not None:
            return max(0.0, min(pedida, self.limits.max_retry_wait))
        base = min(self.limits.retry_base * (2 ** intento), self.limits.max_retry_wait)
        return base * (0.5 + random.random() / 2)

    def _check_prefix(self, state: RunState, replay: Replay) -> None:
        """Reanudar con otra configuración no es reanudar: es otro run.

        Se compara la petición del primer paso de modelo —que el journal guarda
        entera— contra la que produciría este agente ahora. Si el prefijo estable
        difiere, la primera mitad del run la ejecutó una configuración y la
        segunda otra, y el journal lo registraría como uno solo: una auditoría
        devolvería una historia que ninguna configuración produjo nunca.

        Que además tire la caché del proveedor es lo de menos, y es lo único que
        se nota sin buscarlo.
        """
        primera = next(
            (
                e.request
                for e in state.events
                if isinstance(e, ModelStep) and e.phase is Phase.ATTEMPTED and e.request
            ),
            None,
        )
        if primera is None:      # run nuevo: no hay nada con lo que comparar
            return

        ahora = Request(
            model=self.model,
            system=self.instructions,
            tools=self.tools,
            response_format=self._format,
        )
        if prefix_fingerprint(primera) == prefix_fingerprint(ahora):
            return

        raise ConfigurationError(
            f"El run '{state.run_id}' se creó con otra configuración y reanudarlo "
            f"con esta mezclaría dos agentes en un mismo diario "
            f"({describe_prefix_change(primera, ahora)}). "
            "Reanudar es continuar ese run; una configuración distinta es otro "
            "run — usa un run_id nuevo."
        )

    async def _delegate(
        self,
        sub: Delegate,
        call: ToolCall,
        session: Session,
        journal: Journal,
        replay: Replay,
        step_id: str,
        step_seq: int,
        depth: int,
    ):
        """Ejecuta un subagente y cede los eventos del padre.

        Cede pares ``(evento, resultado)``: el evento se reenvía a quien consume
        el run del padre, y el resultado —cuando llega— trae lo que hay que meter
        en el contexto y el consumo que hay que sumar.

        Los pasos del subagente **no se reenvían**. Quien mira el run del padre
        ve una delegación, no la conversación ajena; la de dentro está entera en
        su propio diario, que es donde se audita sin pagarla en cada turno.
        """
        brief = str(call.arguments.get("brief", "")).strip()

        hecho = replay.resolve(step_id, idempotent=False)
        if hecho is not None:
            assert isinstance(hecho, DelegateStep)
            yield hecho, (
                ToolResult.of(call.id, str(hecho.result or "")),
                hecho.usage,
            )
            return

        if depth >= self.limits.max_delegation_depth:
            # Un ciclo entre dos agentes que se delegan mutuamente no termina
            # solo, y `max_steps` no lo ve: cada nivel tiene su propio contador.
            yield None, (
                ToolResult.of(
                    call.id,
                    f"No se puede delegar más: se alcanzó la profundidad máxima "
                    f"({self.limits.max_delegation_depth}).",
                    is_error=True,
                ),
                Usage.zero(),
            )
            return

        intencion = DelegateStep(
            run_id=session.run_id, step_id=step_id, step_seq=step_seq,
            phase=Phase.ATTEMPTED, at=time.time(),
            agent=sub.name, brief=brief,
        )
        await journal.record(intencion)
        yield intencion, None

        salida: Any = None
        consumo = Usage.zero()
        # Mismo almacén, otro `run_id`: un solo diario guarda el árbol entero, y
        # reanudar al padre encuentra el sub-run donde lo dejó.
        sesion_hija = Session(
            sub_run_id(session.run_id, step_id), session.gateway, session.checkpointer
        )
        async for paso in sub.agent._loop(
            brief, sesion_hija, stream=False, depth=depth + 1
        ):
            if isinstance(paso, FinalStep):
                salida, consumo = paso.output, paso.usage

        resultado = DelegateStep(
            run_id=session.run_id, step_id=step_id, step_seq=step_seq,
            phase=Phase.COMPLETED, at=time.time(),
            agent=sub.name, result=salida,
            # El consumo agregado del subagente sube aquí, y de aquí al total del
            # padre. Es lo que hace que el coste de orquestar deje de ser
            # invisible.
            usage=consumo,
        )
        await journal.record(resultado)
        yield resultado, (ToolResult.of(call.id, str(salida or "")), consumo)

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


def _con_esquema(instrucciones: str | None, esquema: Mapping[str, Any]) -> str:
    """Añade el esquema de salida a las instrucciones del sistema.

    Va al final y en un bloque marcado para que el modelo lo lea como una
    restricción y no como parte de la tarea.
    """
    base = (instrucciones or "").rstrip()
    return (
        f"{base}\n\n"
        "Responde ÚNICAMENTE con un objeto JSON que valide contra este esquema. "
        "Sin texto antes ni después, sin vallas de código.\n\n"
        f"{dumps(esquema)}"
    ).lstrip()
