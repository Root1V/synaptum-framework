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
from typing import Any, AsyncIterator, Mapping, Sequence, Union

from ..core.errors import (
    ConfigurationError,
    Denied,
    InvalidToolCallError,
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
    Outcome,
    Phase,
    StepEvent,
    ToolStep,
    make_step_id,
)
from ..core.errors import NoObjectGeneratedError
from ..core.protocols import CallContext, Checkpointer, Gateway, RunState
from ..core.types import (
    AUTO,
    Message,
    Role,
    Request,
    Response,
    ResponseFormat,
    Risk,
    StreamEvent,
    ToolCall,
    ToolChoice,
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

__all__ = ["Limits", "Sampling", "Session", "Agent"]

#: Lo que se le puede pedir a un agente: texto, un mensaje ya armado, o las
#: partes de uno.
#:
#: Las tres formas existen porque una tarea no siempre es una frase. Un agente
#: que mira una página recibe texto **y** la imagen, y obligar a quien llama a
#: construir el `Message` entero para eso convierte el caso común en el caso
#: raro.
Entrada = Union[str, "Message", Sequence[Any]]


def _como_mensaje(task: "Entrada") -> "Message":
    """Normaliza la tarea a un mensaje de usuario."""
    if isinstance(task, Message):
        if task.role is not Role.USER:
            raise ConfigurationError(
                f"La tarea llegó como un mensaje de rol {task.role.value!r}. "
                "Una tarea la pide quien llama, así que es un mensaje de usuario; "
                "el prompt de sistema va en `instructions`."
            )
        return task
    if isinstance(task, str):
        return Message.user(task)
    return Message(role=Role.USER, content=tuple(task))


#: Nombre de la herramienta con la que un agente entrega su salida final.
#:
#: Fijo y no configurable: el modelo lo ve en el catálogo y nada más del bucle
#: lo necesita. Dejarlo elegir añadiría una forma de que el agente y quien lo
#: lee hablen de herramientas distintas, a cambio de nada.
SUBMIT = "submit"


# ── Muestreo ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Sampling:
    """Cómo muestrea el modelo, y qué puede o no llamar.

    Va aparte de ``Limits`` porque son dos cosas distintas: los límites son
    **corrección** —que un bucle mal formado termine— y esto es **conducta**.
    Mezclarlos haría que subir un tope pareciera del mismo orden que bajar la
    temperatura.

    Todo a ``None`` significa «lo que decida quien ejecuta». No se inventa un
    valor por defecto: un ``temperature=0.7`` nuestro pisaría el del proveedor
    sin que nadie lo hubiera pedido, y la diferencia solo se vería en la
    conducta del modelo, que es donde menos se busca.

    **Forma parte del prefijo estable.** Cambiar la temperatura a mitad de un
    run cambia cómo responde el modelo, así que reanudar con otra se rechaza
    igual que reanudar con otro modelo: el diario describiría un run que
    ninguna configuración produjo.
    """

    temperature: float | None = None
    """``0`` es un valor medido, no la ausencia de uno: es lo que hace
    reproducible una extracción."""
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] = ()
    tool_choice: ToolChoice = AUTO
    """Si el modelo elige herramienta, debe usar una, o no puede usar ninguna."""
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    """La válvula de escape para lo que un proveedor concreto expone y el
    vocabulario común no cubre.  Usarla para algo que sí es común es señal de
    que falta un campo en la especificación, no de que esto sobre."""

    def as_request_fields(self) -> dict[str, Any]:
        """Lo que hay que pasarle al ``Request``, sin los que nadie fijó."""
        campos: dict[str, Any] = {"tool_choice": self.tool_choice}
        if self.stop:
            campos["stop"] = tuple(self.stop)
        if self.provider_options:
            campos["provider_options"] = dict(self.provider_options)
        for nombre in ("temperature", "top_p", "max_output_tokens"):
            valor = getattr(self, nombre)
            if valor is not None:
                campos[nombre] = valor
        return campos


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
        sampling: "Sampling | None" = None,
        submit_tool: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        # Acepta una cadena o cualquier cosa con `render()` — una plantilla
        # versionada, sin que el bucle tenga que importar el sistema de prompts.
        self.instructions: str | None = (
            instructions.render() if hasattr(instructions, "render") else instructions
        )
        #: Qué prompt produjo estas instrucciones, si vino de uno versionado.
        #:
        #: El agente renderizaba la plantilla y se quedaba solo con el texto, así
        #: que la versión se perdía justo donde más falta hace: cuando una
        #: respuesta sale mal en producción y la primera pregunta es con qué
        #: prompt se generó. Viaja en el `meta` de cada paso de modelo, que es
        #: donde queda **en el diario** y sobrevive al proceso.
        self.prompt: Mapping[str, str] = {
            k: v for k, v in (
                ("prompt.name", getattr(instructions, "name", "") or ""),
                ("prompt.version", getattr(instructions, "version", "") or ""),
            ) if v
        }
        # Acepta ToolDefinition o cualquier objeto que la exponga — un `@tool`,
        # sin que el bucle tenga que importar el decorador.
        # Un subagente se presenta al modelo como una herramienta de un solo
        # parámetro —el brief—, así que el catálogo del padre no crece con el
        # del hijo. Eso es lo que hace barato delegar.
        # Se acepta cualquier cosa que **ya cumpla el contrato** —`name`,
        # `definition`, `execute`— y se envuelve un `Agent` suelto por comodidad.
        # Comprobar el tipo en vez del contrato dejaría fuera a un delegado
        # remoto, que es exactamente lo que no debe distinguirse de uno local.
        self.delegates: tuple[Any, ...] = tuple(
            d if hasattr(d, "execute") else Delegate(d) for d in delegates
        )
        self._delegates_by_name = {d.name: d for d in self.delegates}

        propias = tuple(t.definition if hasattr(t, "definition") else t for t in tools)
        self.tools: tuple[ToolDefinition, ...] = propias + tuple(
            d.definition for d in self.delegates
        )
        self.limits = limits or Limits()
        # El muestreo es **del prefijo**, no de la llamada: cambiarlo a mitad de
        # un run cambia la conducta del modelo, así que entra en la huella y
        # reanudar con otro se rechaza como se rechaza cambiar de modelo.
        self.sampling = sampling or Sampling()
        self.output: Schema | None = schema_for(output) if output is not None else None

        #: La salida final se entrega **llamando a una herramienta**, no
        #: devolviendo texto.
        #:
        #: Sustituye a ``response_format`` en vez de acompañarlo, y no es una
        #: preferencia: dos restricciones que piden lo mismo pueden divergir, y
        #: con algunos motores una gramática de salida en la misma petición que
        #: un catálogo de herramientas **impide que el modelo emita tool calls**
        #: — con lo que la forma de entregar y la de trabajar se estorban.
        #:
        #: Lo que gana a cambio: un objeto que no valida vuelve al modelo como
        #: resultado de herramienta, con el error dentro, y el modelo corrige en
        #: el turno siguiente. Con texto plano no hay dónde poner esa respuesta.
        self.submit_tool = submit_tool and self.output is not None

        self._format = (
            ResponseFormat(
                kind="json_schema",
                schema=self.output.json_schema(),
                name=getattr(self.output, "name", "output"),
            )
            if self.output is not None and not self.submit_tool
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
        if self.output is not None and not self.submit_tool:
            self.instructions = _con_esquema(self.instructions, self.output.json_schema())

        if self.submit_tool:
            assert self.output is not None
            # El esquema viaja como los **parámetros** de la herramienta, así que
            # no hace falta repetirlo en las instrucciones: el catálogo ya lo
            # lleva, y repetirlo sería la segunda restricción que puede divergir.
            self.tools = self.tools + (
                ToolDefinition(
                    name=SUBMIT,
                    description=(
                        "Entrega el resultado final. Llámala **una sola vez**, cuando "
                        "tengas todos los datos. Si algo no valida, te lo devolveré "
                        "con el error para que lo corrijas."
                    ),
                    parameters=self.output.json_schema(),
                    risk=Risk.READ,
                    idempotent=True,
                ),
            )
        self._by_name = {t.name: t for t in self.tools}

    # ── Bucle ─────────────────────────────────────────────────────────────────

    def run(self, task: Entrada, *, session: Session) -> AsyncIterator[StepEvent]:
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
        self, task: Entrada, session: Session, *, stream: bool, depth: int = 0
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

        messages: list[Message] = [_como_mensaje(task)]
        total = Usage.zero()
        seq = 0
        turns = 0
        entregas = 0
        """Cuántas veces el modelo intentó entregar. Distingue «nunca lo intentó»
        de «lo intentó y nunca validó», que son dos diagnósticos distintos para
        quien lea un run agotado."""
        aceptado: Any = None
        ultimo_fallo = ""
        """Lo último que no validó, para que el error lo lleve dentro."""

        try:
            while True:
                if turns >= self.limits.max_steps:
                    if self.output is not None and entregas:
                        # «Nunca lo intentó» y «lo intentó y nunca validó» son
                        # dos diagnósticos distintos, y se separan **por tipo**
                        # y no solo por el texto: quien atrapa el error suele
                        # querer actuar distinto —uno se arregla con el prompt y
                        # el otro con el esquema— y leer un mensaje para decidir
                        # es la clase de contrato que se rompe al reescribirlo.
                        raise NoObjectGeneratedError(
                            f"El modelo entregó {entregas} vez/veces en "
                            f"{self.limits.max_steps} turnos y ninguna validó. "
                            f"Lo último que devolvió: {ultimo_fallo[:200]!r}",
                            raw=ultimo_fallo,
                        )
                    if self.output is not None:
                        donde = f"`{SUBMIT}`" if self.submit_tool else "una salida válida"
                        raise LimitExceeded(
                            f"max_steps · el modelo nunca llegó a entregar {donde}",
                            self.limits.max_steps,
                        )
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
                    **self.sampling.as_request_fields(),
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
                        meta=dict(self.prompt),
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
                if not calls and self.output is not None and not self.submit_tool:
                    entregas += 1
                    try:
                        aceptado = self._validate(response.message.text)
                    except NoObjectGeneratedError as invalido:
                        # Re-preguntar **enseñando qué falló**. La respuesta
                        # fallida ya está en el contexto —se añadió arriba— así
                        # que aquí solo va el porqué.
                        ultimo_fallo = response.message.text
                        messages.append(Message.user(
                            f"Esa salida no vale: {invalido}. Devuelve solo el "
                            "objeto JSON que pide el esquema, corregido."
                        ))
                        continue
                    break

                if not calls:
                    if self.submit_tool:
                        # Con la entrega por herramienta, un turno sin llamadas
                        # no cierra nada: el modelo habló en vez de entregar.
                        messages.append(Message.user(
                            f"No has entregado nada. Llama a `{SUBMIT}` con el "
                            "resultado para terminar."
                        ))
                        continue
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

                    entregando = self.submit_tool and call.name == SUBMIT
                    if entregando:
                        entregas += 1

                    denegado = None
                    done = replay.resolve(step_id, idempotent=idempotent)
                    if done is not None:
                        assert isinstance(done, ToolStep)
                        if entregando and done.result is not None and not done.result.is_error:
                            # Una entrega válida reproducida del diario: el
                            # objeto se **vuelve a derivar** de los argumentos
                            # registrados en vez de guardarse. Guardar las dos
                            # cosas arriesga que discrepen, igual que con el
                            # contexto.
                            aceptado = self.output.validate(dict(done.call.arguments))
                        if done.result is None:
                            # Desenlace cerrado sin resultado: una persona dijo
                            # que no.  No se reejecuta y no se vuelve a
                            # preguntar; vuelve al modelo como evidencia, que es
                            # lo que le hace rectificar en vez de insistir.
                            #
                            # Sin esta rama el bucle daba por hecho que todo
                            # paso resuelto traía resultado, que era cierto
                            # mientras un «no» se reintentaba siempre.
                            results.append(ToolResult.of(
                                call.id,
                                f"Denegado: {done.reason or done.outcome.value}.",
                                is_error=True,
                            ))
                        else:
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

                    if entregando:
                        # No sale por la costura: no hay efecto externo que
                        # gobernar, solo una comprobación contra el esquema que
                        # el propio agente declaró.
                        try:
                            aceptado = self.output.validate(dict(call.arguments))
                            outcome = ToolResult.of(call.id, "Aceptado.")
                        except Exception as invalido:  # noqa: BLE001
                            # El error vuelve **al modelo**, que es quien puede
                            # corregirlo. Esa es la diferencia entre reintentar
                            # —repetir la misma petición y esperar otra suerte—
                            # y re-preguntar enseñando qué falló.
                            ultimo_fallo = dumps(call.arguments)
                            outcome = ToolResult.of(
                                call.id,
                                f"El resultado no valida: {invalido}. "
                                f"Corrígelo y vuelve a llamar a `{SUBMIT}`.",
                                is_error=True,
                            )
                        tool_result = ToolStep(
                            run_id=session.run_id, step_id=step_id, step_seq=seq - 1,
                            phase=Phase.COMPLETED, at=time.time(),
                            call=call, result=outcome, risk=risk, idempotent=True,
                        )
                        await journal.record(tool_result)
                        yield tool_result
                        results.append(outcome)
                        continue

                    try:
                        outcome = await self._call_tool(session, call, step_id, spec)
                    except InvalidToolCallError as desajuste:
                        # El modelo llamó con argumentos que no encajan con la
                        # firma. **Eso no es un fallo del run**: es una muestra
                        # mala, y un modelo que ve el error suele corregirla.
                        #
                        # Lo decía ya el propio tipo —«lo que corrige esto es
                        # devolverle el error al modelo como ToolResult para que
                        # rectifique, no repetir la llamada»— y el bucle hacía
                        # otra cosa: dejaba subir la excepción y el run moría
                        # entero por un argumento mal escrito.
                        outcome = ToolResult.of(
                            call.id,
                            f"{desajuste}. Corrige la llamada y vuelve a intentarlo.",
                            is_error=True,
                        )
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
                            # Y la decisión viaja **en el paso**, no solo dentro
                            # del texto que ve el modelo.  Sin esto, quien
                            # consuma el stream o lea el diario no puede
                            # distinguir «el gobierno lo paró» de «la
                            # herramienta falló», que son cosas distintas para
                            # quien opera: una es el sistema funcionando.
                            denegado = denial.decision
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
                        decision=denegado,
                        outcome=(
                            Outcome.DENIED_BY_POLICY if denegado else Outcome.RESULT
                        ),
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

                # La entrega válida cierra el run **después** de registrar su
                # resultado: el diario cuenta que se entregó y qué se dijo, y no
                # solo que el run terminó.
                if aceptado is not None:
                    break

            typed = aceptado if aceptado is not None else self._final_output(messages[-1].text)
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
            # La validación **ya no entra aquí**, y el motivo es que entrar era
            # el fallo: un objeto mal formado hacía repetir la **misma**
            # petición, byte a byte, esperando otra suerte del muestreo. Eso es
            # repetir, no reintentar — y el modelo nunca llegaba a ver qué había
            # fallado. Ahora la comprueba el bucle y re-pregunta enseñando el
            # error, que es lo único que le permite corregir.
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
            **self.sampling.as_request_fields(),
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

        # Mismo almacén, otro `run_id`: un solo diario guarda el árbol entero, y
        # reanudar al padre encuentra el sub-run donde lo dejó. El `run_id` del
        # hijo es además el `contextId` de A2A cuando el subagente es remoto.
        #
        # `execute` es lo único que sabe **dónde** vive el subagente. Lo demás
        # —paso durable, reanudación, consumo agregado, riesgo— es igual aquí y
        # al otro lado de una red, y por eso vive fuera de él.
        salida, consumo = await replace(sub, _depth=depth + 1).execute(
            brief, session, sub_run_id(session.run_id, step_id)
        )

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
