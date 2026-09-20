# Referencia

**Generada del código.** Escribir esto a mano sería escribir algo que caduca, y una
referencia caducada manda a buscar un símbolo que ya no está.

Dice **qué hay**. El **por qué** y el **cuándo** están en las páginas anteriores, y eso no
se deriva de una firma.

> La superficie pública es lo que `synaptum/__init__.py` exporta, y nada más. Lo que no
> aparece aquí puede cambiar sin aviso, aunque se pueda importar.

## Escribir un agente

Lo que se toca en el primer fichero.

### `Agent`

Composición, no herencia.  Un agente es su configuración más el bucle.

```python
Agent(name: str, *, model: str, instructions: Any = None, tools: Sequence[Any] = (), delegates: Sequence[Any] = (), output: Any = None, limits: Limits | None = None) -> None
```

| Miembro | Firma | |
|---|---|---|
| `run` | `run(self, task: str, *, session: Session) -> AsyncIterator[StepEvent]` | Ejecuta el agente cediendo cada paso. |
| `stream` | `stream(self, task: str, *, session: Session) -> AsyncIterator[StepEvent \| StreamEvent]` | Lo mismo, entregando además los fragmentos del modelo según llegan. |

### `Session`

Un run: su identidad, por dónde sale y dónde se recuerda.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `gateway` | `Gateway` | **obligatorio** |
| `checkpointer` | `Checkpointer` | *(fábrica)* |

Reanudar es construir una ``Session`` con el mismo ``run_id`` y el mismo
``checkpointer``.  No hay nada más.

### `Limits`

Topes del bucle.

| Campo | Tipo | Por defecto |
|---|---|---|
| `max_steps` | `int` | `50` |
| `max_delegation_depth` | `int` | `3` |
| `max_tool_chars` | `int \| None` | `16000` |
| `max_retries` | `int` | `2` |
| `retry_base` | `float` | `0.5` |
| `max_retry_wait` | `float` | `30.0` |
| `reserved_output` | `float` | `0.25` |

Son **corrección, no política**: evitan que un bucle mal formado no termine
nunca.  Los límites de gasto pertenecen al harness y llegan por la costura
como ``Denied`` con ``terminate_run``.

### `Delegate`

Un subagente, tal como lo ve quien delega.

| Campo | Tipo | Por defecto |
|---|---|---|
| `agent` | `'Agent'` | **obligatorio** |
| `description` | `str` | `''` |
| `_depth` | `int` | `0` |

| Miembro | Firma | |
|---|---|---|
| `definition` | — | Lo que el modelo ve: un nombre, cuándo usarlo, y un hueco para el brief. |
| `execute` | `async execute(self, brief: str, session: Any, run_id: str) -> tuple[Any, Any]` | Corre el subagente y devuelve ``(resultado, consumo)``. |
| `name` | — |  |
| `risk` | — |  |

Se presenta al modelo como una herramienta de **un solo parámetro**: el
brief. No se le ofrecen las herramientas del subagente, y eso es lo que hace
barato delegar — el catálogo del padre no crece con el del hijo.

### `tool`

```python
def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None, risk: Risk = <Risk.READ: read>, idempotent: bool = False, parameters: Mapping[str, Any] | None = None) -> Any
```

Convierte una función tipada en una herramienta.

Se puede usar con o sin paréntesis::

@tool
    def buscar(q: str) -> str: ...

@tool(risk=Risk.HARD_WRITE, idempotent=False)
    async def transferir(cuenta: str, importe: float) -> str: ...

Args:
    name: por defecto, el nombre de la función.
    description: por defecto, el primer párrafo del docstring.
    risk: clase de riesgo del efecto.  Synaptum declara; el harness decide.
    idempotent: si el efecto puede repetirse sin consecuencias.  Determina
        si el replay puede reintentar el paso tras una caída, y si su
        registro en el journal puede diferirse.
    parameters: esquema a mano, para el caso raro en el que la firma no
        baste.  Salta la derivación por completo.

### `Tool`

Una función con su contrato.

```python
Tool(fn: Callable[..., Any], *, name: str | None = None, description: str | None = None, risk: Risk = <Risk.READ: read>, idempotent: bool = False, parameters: Mapping[str, Any] | None = None, localns: Mapping[str, Any] | None = None) -> None
```

| Miembro | Firma | |
|---|---|---|
| `invoke` | `async invoke(self, call_id: str, arguments: Mapping[str, Any]) -> ToolResult` | Ejecuta y envuelve el resultado. |
| `name` | — |  |

Sigue siendo invocable con normalidad — el decorador no la esconde — y
además expone ``definition``, que es lo que viaja en el ``Hello`` y lo que
el modelo ve.

### `Risk`

Nivel de riesgo del efecto de una herramienta — RM-21.

| Valor | |
|---|---|
| `Risk.READ` | `'read'` |
| `Risk.SOFT_WRITE` | `'soft_write'` |
| `Risk.HARD_WRITE` | `'hard_write'` |
| `Risk.DESTRUCTIVE` | `'destructive'` |

Synaptum **declara**; el harness **decide**.  El bucle garantiza que un paso
destructivo no se ejecuta antes de tener una decisión; cuál sea esa decisión
no es asunto suyo.

Vive aquí, junto a ``ToolDefinition``, porque es parte del contrato de la
herramienta: viaja con ella en el handshake y el gateway lo necesita para
decidir antes de ejecutar.

### `ToolDefinition`

Definición de una tool.

| Campo | Tipo | Por defecto |
|---|---|---|
| `name` | `str` | **obligatorio** |
| `description` | `str` | `''` |
| `parameters` | `Mapping[str, Any]` | *(fábrica)* |
| `ref` | `str \| None` | `None` |
| `risk` | `Risk` | `<Risk.READ: 'read'>` |
| `idempotent` | `bool` | `False` |

``ref`` es la referencia versionada de H5.  Cuando está presente, el esquema
no viaja en cada llamada: el handshake lo resolvió una vez y el prefijo
cacheado se mantiene estable.  ``parameters`` sigue disponible para el modo
autónomo, donde no hay registro contra el que resolver.

### `ToolChoice`

ToolChoice(mode: "Literal['auto', 'none', 'required', 'named']" = 'auto', name: 'str | None' = None)

| Campo | Tipo | Por defecto |
|---|---|---|
| `mode` | `Literal['auto', 'none', 'required', 'named']` | `'auto'` |
| `name` | `str \| None` | `None` |

## Economía de contexto

Qué ve el modelo y qué costó. `Usage` dice cuánto; esto dice por qué.

### `cap_tool_output`

```python
def cap_tool_output(resultado: ToolResult, *, max_chars: int | None = 16000) -> ToolResult
```

Devuelve el resultado recortado, o el mismo si cabe.

Args:
    resultado: lo que la herramienta devolvió.
    max_chars: tope en caracteres. ``None`` desactiva el recorte.

Returns:
    Un ``ToolResult`` nuevo si hubo que recortar; **el mismo objeto** si no
    —así quien compare por identidad puede saber si se tocó algo.

### `economy`

```python
def economy(state: RunState) -> ContextEconomy
```

Calcula el informe de un run a partir de su journal.

Empareja cada intención de modelo con su resultado: la petición vive en
``ATTEMPTED`` —es donde está el prefijo— y el consumo en ``COMPLETED``.
Un turno sin resultado no se cuenta: se intentó y no se sabe qué costó.

### `ContextEconomy`

El informe de un run.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `turns` | `tuple[TurnEconomy, ...]` | `()` |
| `prefix_changes` | `tuple[str, ...]` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `cache_hit_ratio` | — | Acierto de caché del run entero. |
| `input_growth` | — | Cuánto crece la entrada por turno, en tokens. |
| `prefix_rewrites` | — | Cuántas veces cambió el prefijo estable dentro del run. |
| `report` | `report(self) -> str` | El informe en texto, para leerlo en un terminal o pegarlo en un ticket. |
| `total` | — | Suma de todos los turnos.  ``None`` en un contador se propaga. |

### `TurnEconomy`

Un turno: una llamada al modelo y lo que costó.

| Campo | Tipo | Por defecto |
|---|---|---|
| `step_id` | `str` | **obligatorio** |
| `usage` | `Usage` | **obligatorio** |
| `prefix_rewritten` | `bool` | **obligatorio** |

| Miembro | Firma | |
|---|---|---|
| `cache_hit_ratio` | — | Fracción de la entrada servida desde caché, o ``None`` si no se midió. |

## Lo que cede el bucle

Los pasos que llegan por `async for`, y sus fases.

### `StepEvent`

Base común.  No se instancia directamente.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `step_seq` | `int` | **obligatorio** |
| `phase` | `Phase` | **obligatorio** |
| `at` | `float \| None` | `None` |
| `meta` | `Mapping[str, Any]` | *(fábrica)* |
| `decision` | `Decision \| None` | `None` |
| `kind` | `str` | `'step'` |

| Miembro | Firma | |
|---|---|---|
| `durability` | — |  |
| `key` | — |  |

### `ModelStep`

Una llamada al modelo.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `step_seq` | `int` | **obligatorio** |
| `phase` | `Phase` | **obligatorio** |
| `at` | `float \| None` | `None` |
| `meta` | `Mapping[str, Any]` | *(fábrica)* |
| `decision` | `Decision \| None` | `None` |
| `kind` | `str` | `'model'` |
| `request` | `Request \| None` | `None` |
| `response` | `Response \| None` | `None` |
| `usage` | `Usage` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `durability` | — | La intención es diferible; el resultado, no. |
| `key` | — |  |

Siempre ``DURABLE``: una inferencia cuesta dinero y no es reproducible, así
que perder el registro de que ocurrió es exactamente el fallo que la
durabilidad existe para evitar.

### `ToolStep`

La ejecución de una herramienta.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `step_seq` | `int` | **obligatorio** |
| `phase` | `Phase` | **obligatorio** |
| `at` | `float \| None` | `None` |
| `meta` | `Mapping[str, Any]` | *(fábrica)* |
| `decision` | `Decision \| None` | `None` |
| `kind` | `str` | `'tool'` |
| `call` | `ToolCall \| None` | `None` |
| `result` | `ToolResult \| None` | `None` |
| `risk` | `Risk` | `<Risk.READ: 'read'>` |
| `idempotent` | `bool` | `False` |

| Miembro | Firma | |
|---|---|---|
| `durability` | — | Depende del efecto, y ambas fases por igual. |
| `key` | — |  |

Es el único evento cuya durabilidad depende del efecto: releer un fichero se
puede repetir sin consecuencias, ordenar una transferencia no.  Esa
diferencia la declara el contrato de la tool, dentro del framework — es
justamente lo que un harness no puede deducir desde fuera.

### `ApprovalStep`

Una pausa a la espera de decisión.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `step_seq` | `int` | **obligatorio** |
| `phase` | `Phase` | **obligatorio** |
| `at` | `float \| None` | `None` |
| `meta` | `Mapping[str, Any]` | *(fábrica)* |
| `decision` | `Decision \| None` | `None` |
| `kind` | `str` | `'approval'` |
| `subject` | `str` | `''` |

| Miembro | Firma | |
|---|---|---|
| `durability` | — |  |
| `key` | — |  |

Siempre ``DURABLE``, y por una razón distinta a las demás: no se le pregunta
dos veces a una persona porque el proceso se cayó.

### `DelegateStep`

Delegación a un subagente con contexto aislado.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `step_seq` | `int` | **obligatorio** |
| `phase` | `Phase` | **obligatorio** |
| `at` | `float \| None` | `None` |
| `meta` | `Mapping[str, Any]` | *(fábrica)* |
| `decision` | `Decision \| None` | `None` |
| `kind` | `str` | `'delegate'` |
| `agent` | `str` | `''` |
| `brief` | `str` | `''` |
| `result` | `Any` | `None` |
| `usage` | `Usage` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `durability` | — |  |
| `key` | — |  |

El worker recibe un brief estrecho y devuelve resultado y referencias, nunca
su historial: es lo que hace que delegar aísle contexto en vez de duplicarlo.

### `FinalStep`

Cierre del run.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `step_seq` | `int` | **obligatorio** |
| `phase` | `Phase` | **obligatorio** |
| `at` | `float \| None` | `None` |
| `meta` | `Mapping[str, Any]` | *(fábrica)* |
| `decision` | `Decision \| None` | `None` |
| `kind` | `str` | `'final'` |
| `output` | `Any` | `None` |
| `usage` | `Usage` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `durability` | — |  |
| `key` | — |  |

### `Phase`

Las dos fases de un paso, con los nombres del contrato compartido.

| Valor | |
|---|---|
| `Phase.ATTEMPTED` | `'attempted'` |
| `Phase.COMPLETED` | `'completed'` |

``attempted`` y ``completed`` describen el **estado** del paso, no el
contenido del evento — y coinciden con los nombres de las consultas
derivadas de ``RunState``, que es donde se leen.

### `Durability`

Con qué urgencia debe persistirse un evento — RM-14.

| Valor | |
|---|---|
| `Durability.DURABLE` | `'durable'` |
| `Durability.DEFERRABLE` | `'deferrable'` |

Separar las dos costuras no sirve para que el journal sea asíncrono: sirve
para que cada evento tenga la política que le toca en vez de heredar la peor
de las dos.

### `Event`

`Event = synaptum.core.events.ModelStep | synaptum.core.events.ToolStep | synaptum.core.events.DelegateStep | synaptum.core.events.ApprovalStep | synaptum.core.events.FinalStep`

Represent a PEP 604 union type

### `make_step_id`

```python
def make_step_id(seq: int, kind: str) -> str
```

Acuña un identificador de paso determinista.

Deriva de la posición del paso en el run, nunca de un UUID ni de un reloj:
un run reproducido debe acuñar exactamente los mismos identificadores, o el
replay no puede reconocer lo ya hecho.

El ancho fijo mantiene el orden lexicográfico alineado con el numérico, de
modo que un almacén que ordene por clave devuelve el journal en orden de
ejecución sin índice adicional.

### `idempotency_key`

```python
def idempotency_key(event: "StepEvent") -> tuple[str, str, str]
```

La clave con la que el ``Checkpointer`` deduplica — A3.

Un ``append`` repetido con esta misma clave es un no-op, nunca un error.

## Vocabulario del modelo

La forma que viaja por la costura. Contrato versionado, no detalle interno.

### `Message`

Message(role: 'Role', content: 'tuple[ContentPart, ...]' = (), name: 'str | None' = None)

| Campo | Tipo | Por defecto |
|---|---|---|
| `role` | `Role` | **obligatorio** |
| `content` | `tuple[ContentPart, ...]` | `()` |
| `name` | `str \| None` | `None` |

| Miembro | Firma | |
|---|---|---|
| `assistant` | `assistant(text: str) -> "Message"` |  |
| `developer` | `developer(text: str) -> "Message"` |  |
| `text` | — | Concatena las partes de texto.  Ignora razonamiento y binarios. |
| `tool_calls` | — |  |
| `tool_results` | `tool_results(*results: ToolResult) -> "Message"` |  |
| `user` | `user(text: str, *, name: str \| None = None) -> "Message"` |  |

### `Role`

Los cinco roles que cualquier proveedor sabe expresar.

| Valor | |
|---|---|
| `Role.SYSTEM` | `'system'` |
| `Role.USER` | `'user'` |
| `Role.ASSISTANT` | `'assistant'` |
| `Role.TOOL` | `'tool'` |
| `Role.DEVELOPER` | `'developer'` |

### `AUTO`

`AUTO = ToolChoice(mode='auto', name=None)`

ToolChoice(mode: "Literal['auto', 'none', 'required', 'named']" = 'auto', name: 'str | None' = None)

### `ContentPart`

`ContentPart = synaptum.core.types.Text | synaptum.core.types.Image | synaptum.core.types.Audio | synaptum.core.types.Document | synaptum.core.types.ToolCall | synaptum.core.types.ToolResult | synaptum.core.types.Thinking | synaptum.core.types.RedactedThinking`

Represent a PEP 604 union type

### `Text`

Text(text: 'str', kind: "Literal['text']" = 'text')

| Campo | Tipo | Por defecto |
|---|---|---|
| `text` | `str` | **obligatorio** |
| `kind` | `Literal['text']` | `'text'` |

### `Image`

Image(media_type: 'str', data: 'str | None' = None, url: 'str | None' = None, kind: "Literal['image']" = 'image')

| Campo | Tipo | Por defecto |
|---|---|---|
| `media_type` | `str` | **obligatorio** |
| `data` | `str \| None` | `None` |
| `url` | `str \| None` | `None` |
| `kind` | `Literal['image']` | `'image'` |

### `Audio`

Audio(media_type: 'str', data: 'str | None' = None, url: 'str | None' = None, kind: "Literal['audio']" = 'audio')

| Campo | Tipo | Por defecto |
|---|---|---|
| `media_type` | `str` | **obligatorio** |
| `data` | `str \| None` | `None` |
| `url` | `str \| None` | `None` |
| `kind` | `Literal['audio']` | `'audio'` |

### `Document`

Document(media_type: 'str', data: 'str | None' = None, url: 'str | None' = None, name: 'str | None' = None, kind: "Literal['document']" = 'document')

| Campo | Tipo | Por defecto |
|---|---|---|
| `media_type` | `str` | **obligatorio** |
| `data` | `str \| None` | `None` |
| `url` | `str \| None` | `None` |
| `name` | `str \| None` | `None` |
| `kind` | `Literal['document']` | `'document'` |

### `ToolCall`

Invocación pedida por el modelo.

| Campo | Tipo | Por defecto |
|---|---|---|
| `id` | `str` | **obligatorio** |
| `name` | `str` | **obligatorio** |
| `arguments` | `Mapping[str, Any]` | *(fábrica)* |
| `kind` | `Literal['tool_call']` | `'tool_call'` |

``arguments`` ya viene decodificado.  Los proveedores que lo entregan como
cadena JSON lo parsean en su adaptador: el bucle no debería tener que
adivinar si recibió un objeto o su serialización.

### `ToolResult`

Resultado devuelto al modelo.

| Campo | Tipo | Por defecto |
|---|---|---|
| `call_id` | `str` | **obligatorio** |
| `content` | `tuple['ContentPart', ...]` | `()` |
| `is_error` | `bool` | `False` |
| `kind` | `Literal['tool_result']` | `'tool_result'` |

| Miembro | Firma | |
|---|---|---|
| `of` | `of(call_id: str, text: str, *, is_error: bool = False) -> "ToolResult"` |  |

``is_error`` conserva la evidencia del fallo en vez de ocultarla: un modelo
que no ve el error no puede corregirlo.

### `Thinking`

Razonamiento visible.

| Campo | Tipo | Por defecto |
|---|---|---|
| `text` | `str` | **obligatorio** |
| `signature` | `str \| None` | `None` |
| `kind` | `Literal['thinking']` | `'thinking'` |

``signature`` transporta el token de verificación que algunos proveedores
exigen devolver intacto en el siguiente turno.

### `RedactedThinking`

Razonamiento cifrado por el proveedor.

| Campo | Tipo | Por defecto |
|---|---|---|
| `data` | `str` | **obligatorio** |
| `kind` | `Literal['redacted_thinking']` | `'redacted_thinking'` |

Opaco para nosotros, pero debe reenviarse tal cual o el proveedor pierde el
hilo de su propio razonamiento.

### `Request`

Lo que cruza la costura hacia quien ejecuta la llamada.

| Campo | Tipo | Por defecto |
|---|---|---|
| `model` | `str` | **obligatorio** |
| `messages` | `tuple[Message, ...]` | `()` |
| `system` | `str \| None` | `None` |
| `tools` | `tuple[ToolDefinition, ...]` | `()` |
| `tool_choice` | `ToolChoice` | `ToolChoice(mode='auto', name=None)` |
| `max_output_tokens` | `int \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `top_p` | `float \| None` | `None` |
| `stop` | `tuple[str, ...]` | `()` |
| `response_format` | `ResponseFormat \| None` | `None` |
| `provider_options` | `Mapping[str, Any]` | *(fábrica)* |

``provider_options`` es la válvula de escape para lo que un proveedor
concreto expone y el vocabulario común no cubre.  Existe para que nadie
tenga que bifurcar el tipo; usarla para algo que sí es común es una señal de
que falta un campo en la especificación.

### `Response`

Response(message: 'Message', finish_reason: 'FinishReason', usage: 'Usage' = <factory>, model: 'str' = '', provider_metadata: 'Mapping[str, Any]' = <factory>)

| Campo | Tipo | Por defecto |
|---|---|---|
| `message` | `Message` | **obligatorio** |
| `finish_reason` | `FinishReason` | **obligatorio** |
| `usage` | `Usage` | *(fábrica)* |
| `model` | `str` | `''` |
| `provider_metadata` | `Mapping[str, Any]` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `text` | — |  |
| `tool_calls` | — |  |

### `Usage`

Contadores de tokens — H3.

| Campo | Tipo | Por defecto |
|---|---|---|
| `input` | `int \| None` | `None` |
| `output` | `int \| None` | `None` |
| `reasoning` | `int \| None` | `None` |
| `cache_read` | `int \| None` | `None` |
| `cache_write` | `int \| None` | `None` |
| `estimated` | `bool` | `False` |

| Miembro | Firma | |
|---|---|---|
| `cache_hit_ratio` | — | Fracción de ``input`` servida desde caché, o ``None`` si no se midió. |
| `total` | — | Tokens facturables, o ``None`` si falta alguno por medir. |
| `zero` | `zero() -> "Usage"` | Punto de partida para acumular.  Distinto de ``Usage()``, que es «nada medido». |

Cinco campos, y **ninguno por defecto a cero**.  ``None`` significa «nadie
lo midió»; ``0`` significa «se midió y fue cero».  La distinción no es
purismo: si ``cache_write`` llega como cero cuando en realidad la fuente no
lo reporta, el bucle concluye que escribir en caché es gratis y decide mal
en cada compactación.  Ese fallo no salta — solo cuadra mal.

No es hipotético.  Un backend llama.cpp, por ejemplo, solo puede alimentar
``input``, ``output`` y ``cache_read``; ``reasoning`` y ``cache_write`` no
existen en la fuente.

``estimated`` marca los contadores **derivados** en vez de reportados —
llama.cpp obliga a deducirlos de los ``timings`` del chunk final.  Sin ese
bit, FinOps factura sobre una estimación creyéndola exacta.

``input`` es **inclusivo**: contiene los tokens servidos desde caché, y
``cache_read`` dice cuántos de ellos lo fueron.  Sin fijarlo, dos
implementaciones eligen convenciones distintas y la factura sale mal sin que
nada falle.

### `FinishReason`

str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str

| Valor | |
|---|---|
| `FinishReason.STOP` | `'stop'` |
| `FinishReason.LENGTH` | `'length'` |
| `FinishReason.TOOL_CALLS` | `'tool_calls'` |
| `FinishReason.CONTENT_FILTER` | `'content_filter'` |
| `FinishReason.ERROR` | `'error'` |

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to 'utf-8'.
errors defaults to 'strict'.

### `ResponseFormat`

ResponseFormat(kind: "Literal['text', 'json_object', 'json_schema']" = 'text', schema: 'Mapping[str, Any] | None' = None, name: 'str | None' = None, strict: 'bool' = True)

| Campo | Tipo | Por defecto |
|---|---|---|
| `kind` | `Literal['text', 'json_object', 'json_schema']` | `'text'` |
| `schema` | `Mapping[str, Any] \| None` | `None` |
| `name` | `str \| None` | `None` |
| `strict` | `bool` | `True` |

## Streaming

El ciclo start/delta/end de cada clase de contenido.

### `StreamEvent`

`StreamEvent = synaptum.core.types.StreamStart | synaptum.core.types.TextStart | synaptum.core.types.TextDelta | synaptum.core.types.TextEnd | synaptum.core.types.ReasoningStart | synaptum.core.types.ReasoningDelta | synaptum.core.types.ReasoningEnd | synaptum.core.types.ToolCallStart | synaptum.core.types.ToolCallDelta | synaptum.core.types.ToolCallEnd | synaptum.core.types.Finish`

Represent a PEP 604 union type

### `StreamStart`

StreamStart(model: 'str' = '', kind: "Literal['stream_start']" = 'stream_start')

| Campo | Tipo | Por defecto |
|---|---|---|
| `model` | `str` | `''` |
| `kind` | `Literal['stream_start']` | `'stream_start'` |

### `TextStart`

TextStart(index: 'int' = 0, kind: "Literal['text_start']" = 'text_start')

| Campo | Tipo | Por defecto |
|---|---|---|
| `index` | `int` | `0` |
| `kind` | `Literal['text_start']` | `'text_start'` |

### `TextDelta`

TextDelta(text: 'str' = '', index: 'int' = 0, kind: "Literal['text_delta']" = 'text_delta')

| Campo | Tipo | Por defecto |
|---|---|---|
| `text` | `str` | `''` |
| `index` | `int` | `0` |
| `kind` | `Literal['text_delta']` | `'text_delta'` |

### `TextEnd`

TextEnd(index: 'int' = 0, kind: "Literal['text_end']" = 'text_end')

| Campo | Tipo | Por defecto |
|---|---|---|
| `index` | `int` | `0` |
| `kind` | `Literal['text_end']` | `'text_end'` |

### `ReasoningStart`

ReasoningStart(index: 'int' = 0, kind: "Literal['reasoning_start']" = 'reasoning_start')

| Campo | Tipo | Por defecto |
|---|---|---|
| `index` | `int` | `0` |
| `kind` | `Literal['reasoning_start']` | `'reasoning_start'` |

### `ReasoningDelta`

ReasoningDelta(text: 'str' = '', index: 'int' = 0, kind: "Literal['reasoning_delta']" = 'reasoning_delta')

| Campo | Tipo | Por defecto |
|---|---|---|
| `text` | `str` | `''` |
| `index` | `int` | `0` |
| `kind` | `Literal['reasoning_delta']` | `'reasoning_delta'` |

### `ReasoningEnd`

ReasoningEnd(index: 'int' = 0, signature: 'str | None' = None, kind: "Literal['reasoning_end']" = 'reasoning_end')

| Campo | Tipo | Por defecto |
|---|---|---|
| `index` | `int` | `0` |
| `signature` | `str \| None` | `None` |
| `kind` | `Literal['reasoning_end']` | `'reasoning_end'` |

### `ToolCallStart`

ToolCallStart(id: 'str' = '', name: 'str' = '', index: 'int' = 0, kind: "Literal['tool_call_start']" = 'tool_call_start')

| Campo | Tipo | Por defecto |
|---|---|---|
| `id` | `str` | `''` |
| `name` | `str` | `''` |
| `index` | `int` | `0` |
| `kind` | `Literal['tool_call_start']` | `'tool_call_start'` |

### `ToolCallDelta`

Fragmento de los argumentos, tal como llegan del proveedor.

| Campo | Tipo | Por defecto |
|---|---|---|
| `arguments_delta` | `str` | `''` |
| `index` | `int` | `0` |
| `kind` | `Literal['tool_call_delta']` | `'tool_call_delta'` |

Los argumentos se transmiten troceados y sin garantía de ser JSON válido
hasta el final; quien consume acumula y solo parsea al recibir el
``ToolCallEnd`` correspondiente.

### `ToolCallEnd`

ToolCallEnd(index: 'int' = 0, kind: "Literal['tool_call_end']" = 'tool_call_end')

| Campo | Tipo | Por defecto |
|---|---|---|
| `index` | `int` | `0` |
| `kind` | `Literal['tool_call_end']` | `'tool_call_end'` |

### `Finish`

Cierre del stream con la respuesta acumulada.

| Campo | Tipo | Por defecto |
|---|---|---|
| `response` | `Response` | **obligatorio** |
| `kind` | `Literal['finish']` | `'finish'` |

Lleva el ``Response`` completo para que quien consumió los deltas no tenga
que reconstruirlo, y para que ``Usage`` vuelva por la costura (H3) aunque el
span de ``chat`` lo emita quien ejecutó la llamada (A5).

## Las dos costuras

Protocolos estructurales: quien los implemente no hereda ni importa nada.

### `Gateway`

Decide y ejecuta.  Vive fuera del proceso del bucle.

```python
Gateway(*args, **kwargs)
```

| Miembro | Firma | |
|---|---|---|
| `handshake` | `async handshake(self, hello: Hello) -> Welcome` | Negocia versión y registra el catálogo de tools.  Una vez por sesión. |
| `invoke_model` | `async invoke_model(self, request: Request, ctx: CallContext) -> Response` | Ejecuta una llamada al modelo y devuelve la respuesta completa. |
| `invoke_tool` | `async invoke_tool(self, call: ToolCall, ctx: CallContext, *, risk: Risk, tool_ref: str \| None = None) -> ToolResult` | Ejecuta una herramienta. |
| `stream_model` | `stream_model(self, request: Request, ctx: CallContext) -> AsyncIterator[StreamEvent]` | Ejecuta una llamada al modelo en streaming — H2. |

Cada método puede lanzar ``Denied`` con la ``Decision`` que corresponda —
``deny_step`` admite que el bucle intente otra cosa, ``terminate_run`` no, y
``require_approval`` suspende el run sin que sea un fallo.

Toda respuesta devuelve ``Usage`` (H3).  El span de la operación lo emite
este lado, porque es quien la realiza; pero los contadores vuelven, porque
la economía de contexto del bucle depende de ellos y no son derivables desde
fuera.

### `Checkpointer`

Persiste el journal.  No decide nada.

```python
Checkpointer(*args, **kwargs)
```

| Miembro | Firma | |
|---|---|---|
| `append` | `async append(self, run_id: str, event: StepEvent) -> AppendResult` | Añade un evento al journal. |
| `load` | `async load(self, run_id: str) -> RunState` | Reconstruye el diario de un run, en orden de escritura. |

Es lo único que el harness tiene que implementar de forma obligatoria: dos
métodos contra el almacén que ya use.  No necesita entender el bucle del
agente, ni la lógica de replay, ni cómo se acuñan los identificadores.

### `CallContext`

Lo que acompaña a cada cruce de la costura.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `traceparent` | `str \| None` | `None` |
| `tracestate` | `str \| None` | `None` |
| `deadline_s` | `float \| None` | `None` |
| `extra` | `Mapping[str, Any]` | *(fábrica)* |

``traceparent`` y ``tracestate`` son W3C y no son decorativos: el harness
emite los spans de ``chat`` y ``execute_tool`` porque es quien ejecuta la
E/S, mientras el bucle emite los suyos de estructura.  Sin propagar el
contexto, unos y otros acaban en árboles de traza distintos y nadie puede
seguir un run de punta a punta.

### `Decision`

Decision(disposition: 'Disposition', reason_code: 'str' = '', message: 'str' = '')

| Campo | Tipo | Por defecto |
|---|---|---|
| `disposition` | `Disposition` | **obligatorio** |
| `reason_code` | `str` | `''` |
| `message` | `str` | `''` |

| Miembro | Firma | |
|---|---|---|
| `allowed` | — |  |

### `Disposition`

«No» no es una sola cosa.

| Valor | |
|---|---|
| `Disposition.ALLOW` | `'allow'` |
| `Disposition.DENY_STEP` | `'deny_step'` |
| `Disposition.TERMINATE_RUN` | `'terminate_run'` |
| `Disposition.REQUIRE_APPROVAL` | `'require_approval'` |

Sin distinguir estos casos el bucle no sabe si reintentar con otra cosa,
esperar o parar, y cada implementación acabaría inventando su convención.

### `ALLOW`

`ALLOW = Decision(disposition=<Disposition.ALLOW: 'allow'>, reason_code='', message='')`

Decision(disposition: 'Disposition', reason_code: 'str' = '', message: 'str' = '')

### `Hello`

Apertura de sesión, enviada por el bucle.

| Campo | Tipo | Por defecto |
|---|---|---|
| `versions` | `tuple[str, ...]` | *(fábrica)* |
| `tools` | `tuple[ToolDefinition, ...]` | `()` |
| `client` | `str` | `'synaptum'` |

Combina la negociación de versión con el registro de herramientas en un solo
viaje, y no por ahorrar una llamada: **las dos cosas tienen que ocurrir
antes del primer turno y ninguna puede repetirse a mitad de sesión.**
Cambiar el catálogo de tools con la sesión abierta reescribe el prefijo del
prompt y tira la caché del proveedor.

### `Welcome`

Respuesta del harness al ``Hello``.

| Campo | Tipo | Por defecto |
|---|---|---|
| `version` | `str` | **obligatorio** |
| `tool_refs` | `Mapping[str, str]` | *(fábrica)* |
| `session_id` | `str` | `''` |

``tool_refs`` mapea nombre de tool a **referencia versionada** (H5).  A
partir de aquí el esquema no viaja en cada llamada: se resolvió una vez y el
prefijo cacheado se mantiene estable turno tras turno.

### `RunState`

Lo que ``load`` devuelve: el journal de un run, en orden.

| Campo | Tipo | Por defecto |
|---|---|---|
| `run_id` | `str` | **obligatorio** |
| `events` | `tuple[StepEvent, ...]` | `()` |

| Miembro | Firma | |
|---|---|---|
| `attempted` | `attempted(self, step_id: str) -> bool` | ``True`` si hay intención **y no** resultado. |
| `completed` | `completed(self, step_id: str) -> bool` |  |
| `final` | — | El evento de cierre, si el run ya terminó. |
| `next_seq` | — | Posición que ocupará la siguiente entrada del diario. |
| `result_of` | `result_of(self, step_id: str) -> StepEvent \| None` | Resultado ya registrado de un paso, si lo hay. |

El bucle lo usa para adelantar — ``fast-forward`` — sobre los pasos que ya
tienen resultado registrado, en vez de volver a ejecutarlos.  Ahí está el
valor entero de la ejecución durable: **al reanudar, una inferencia ya
pagada no se paga otra vez.**

### `SEAM_VERSION`

`SEAM_VERSION = '0.1'`

str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str

### `SUPPORT_WINDOW`

`SUPPORT_WINDOW = 2`

int([x]) -> integer int(x, base=10) -> integer

### `negotiate`

```python
def negotiate(peer: Sequence[str], *, current: str = 0.1) -> str
```

Elige la versión más alta que ambos extremos hablan.

Recorre las nuestras de más nueva a más vieja, de modo que dos extremos al
día no se quedan atascados en una antigua solo porque ambos la soportan.

Raises:
    SeamVersionError: si no hay ninguna en común.  Fallar aquí es mucho
        mejor que descubrir la incompatibilidad campo a campo.

### `supported_versions`

```python
def supported_versions(current: str = 0.1, window: int = 2) -> tuple[str, ...]
```

Versiones que este extremo acepta, de la más nueva a la más vieja.

## Runtime

Implementaciones de referencia. Ninguna es para producción a escala.

### `LocalGateway`

``Gateway`` de referencia para modo autónomo.

```python
LocalGateway(*, model: ModelCall, stream: ModelStream | None = None, tools: Sequence[Tool] = (), policy: Policy | None = None, warn: bool = True) -> None
```

| Miembro | Firma | |
|---|---|---|
| `handshake` | `async handshake(self, hello: Hello) -> Welcome` |  |
| `invoke_model` | `async invoke_model(self, request: Request, ctx: CallContext) -> Response` |  |
| `invoke_tool` | `async invoke_tool(self, call: ToolCall, ctx: CallContext, *, risk: Risk = <Risk.READ: read>, tool_ref: str \| None = None) -> ToolResult` |  |
| `report` | `report(self) -> str` | Qué habría decidido un gateway real, en texto. |
| `stream_model` | `stream_model(self, request: Request, ctx: CallContext) -> AsyncIterator[StreamEvent]` |  |

Args:
    model: callable ``(Request) -> Awaitable[Response]``.  Cualquier
        adaptador de proveedor encaja sin que este módulo lo conozca.
    stream: opcional.  Si falta, el streaming se sirve igual entregando la
        respuesta completa como un único fragmento — y el ``Finish`` lo
        declara en ``provider_metadata["streamed"]``.  Ocultar la
        diferencia haría creer a quien corta por presupuesto que está
        ahorrando cuando ya ha pagado la respuesta entera.
    tools: herramientas decoradas con ``@tool``, que se ejecutan de verdad.
    policy: política simulada, opcional.
    warn: pon ``False`` solo si ya sabes que esto no aplica nada.

### `Check`

Una decisión que un gateway real habría tomado.

| Campo | Tipo | Por defecto |
|---|---|---|
| `kind` | `str` | **obligatorio** |
| `name` | `str` | **obligatorio** |
| `step_id` | `str` | **obligatorio** |
| `risk` | `Risk` | `<Risk.READ: 'read'>` |
| `arguments` | `Mapping[str, Any]` | *(fábrica)* |
| `decision` | `Decision \| None` | `None` |
| `enforced` | `bool` | `False` |
| `detail` | `Mapping[str, Any]` | *(fábrica)* |

### `MemoryCheckpointer`

``Checkpointer`` en memoria.  Cumple el contrato completo.

```python
MemoryCheckpointer() -> None
```

| Miembro | Firma | |
|---|---|---|
| `append` | `async append(self, run_id: str, event: StepEvent) -> AppendResult` | Idempotente por ``(run_id, step_id, phase)``.  Un duplicado es no-op. |
| `event_count` | `event_count(self, run_id: str) -> int` |  |
| `load` | `async load(self, run_id: str) -> RunState` |  |

Existe para dos cosas: que Synaptum sea utilizable sin harness, y que haya
una implementación contra la que contrastar cualquier otra.  No es un
motor de producción y no pretende serlo.

### `SqliteCheckpointer`

``Checkpointer`` sobre SQLite.  Cumple el contrato completo.

```python
SqliteCheckpointer(path: str | Path = ':memory:') -> None
```

| Miembro | Firma | |
|---|---|---|
| `append` | `async append(self, run_id: str, event: StepEvent) -> AppendResult` | Añade un evento.  Un duplicado es no-op, garantizado por la clave. |
| `close` | `close(self) -> None` |  |
| `load` | `async load(self, run_id: str) -> RunState` | Reconstruye el estado del run, o ``None`` si no existe. |
| `runs` | `runs(self) -> list[str]` | Identificadores de los runs almacenados.  Fuera del protocolo. |

### `Journal`

Escribe eventos respetando la durabilidad que cada uno declara.

```python
Journal(checkpointer: Checkpointer, run_id: str) -> None
```

| Miembro | Firma | |
|---|---|---|
| `flush` | `async flush(self) -> None` | Vacía lo diferido.  Se llama al cerrar el run, y ante cualquier salida. |
| `record` | `async record(self, event: StepEvent) -> None` |  |

El orden importa más de lo que parece.  Los eventos diferidos se vacían
**antes** de escribir uno durable, de modo que el journal conserva el orden
de ejecución: si no, un evento estructural anterior aparecería después del
efecto que lo siguió, y el registro dejaría de describir lo que pasó.

### `Replay`

Consulta al journal si un paso ya ocurrió.

```python
Replay(state: RunState) -> None
```

| Miembro | Firma | |
|---|---|---|
| `active` | — | ``True`` si hay algo que reproducir. |
| `closed` | — | El cierre del run, si ya lo hubo. |
| `resolve` | `resolve(self, step_id: str, *, idempotent: bool = True) -> StepEvent \| None` | Devuelve el resultado ya registrado, o ``None`` si toca ejecutar. |

Tres respuestas posibles para cada paso, y la tercera es la interesante:

* **Hecho** — hay resultado registrado.  Se devuelve y no se ejecuta nada.
  Aquí es donde se ahorra la inferencia.
* **Nuevo** — no hay rastro.  Se ejecuta con normalidad.
* **Denegado** — hay resultado, y dice que la costura no dejó ocurrir el
  efecto.  Se vuelve a intentar: entre una reanudación y otra alguien pudo
  aprobar lo que antes se denegó.
* **Incierto** — hay intención sin resultado.  El proceso cayó en medio, así
  que el efecto **pudo haber ocurrido**.

### `HttpModel`

Llama a un endpoint HTTP y devuelve respuestas ya normalizadas.

```python
HttpModel(base_url: str, *, api_key: str | None = None, provider: str | None = None, path: str = '/chat/completions', headers: Mapping[str, str] | None = None, timeout: float = 120.0) -> None
```

| Miembro | Firma | |
|---|---|---|
| `stream` | `stream(self, request: Request) -> AsyncIterator[StreamEvent]` |  |

Args:
    base_url: raíz del servicio, por ejemplo ``http://localhost:8080/v1``.
    api_key: si falta, no se manda cabecera de autorización — hay
        despliegues locales que no la piden, y mandar ``Bearer None`` es
        peor que no mandar nada.
    provider: nombre del adaptador. Por defecto se toma del prefijo de
        ``Request.model`` (``"openai-compatible:qwen3-0.6b"``), que es lo
        que el agente ya escribe.
    path: ruta del endpoint de chat, relativa a ``base_url``.
    headers: cabeceras extra. Se mandan tal cual.
    timeout: segundos de espera **por lectura**, no por respuesta completa.
        Un modelo de razonamiento puede tardar mucho en emitir el primer
        token y eso no es un fallo.

## Errores

La reintentabilidad viaja **en el tipo**, no en una tabla de códigos.

### `SynaptumError`

Raíz de todo lo que este framework lanza a propósito.

```python
SynaptumError(message: str = '', *, retryable: bool | None = None, retry_after: float | None = None) -> None
```

``retryable`` es la única pregunta que el bucle le hace a un error.

### `ConfigurationError`

Falta algo, sobra algo, o dos cosas se contradicen.

```python
ConfigurationError(message: str = '', *, retryable: bool | None = None, retry_after: float | None = None) -> None
```

Nunca reintentable: el entorno no se arregla solo entre dos intentos.

### `ProviderError`

Error devuelto por el proveedor, con su estado HTTP si lo hubo.

```python
ProviderError(message: str = '', *, status: int | None = None, provider: str = '', retryable: bool | None = None, retry_after: float | None = None) -> None
```

Cuando no se pasa ``retryable`` explícito, se deduce del estado según la
regla de clasificación.  Un error sin estado se considera transitorio.

### `RequestTimeoutError`

La petición no respondió dentro del plazo.

```python
RequestTimeoutError(message: str = '', *, retryable: bool | None = None, retry_after: float | None = None) -> None
```

### `NetworkError`

Falló el camino, no el destino.

```python
NetworkError(message: str = '', *, retryable: bool | None = None, retry_after: float | None = None) -> None
```

### `AbortError`

Alguien canceló deliberadamente — H2.

```python
AbortError(message: str = '', *, retryable: bool | None = None, retry_after: float | None = None) -> None
```

No es un fallo del sistema y no se reintenta: reintentar una cancelación es
exactamente lo contrario de lo que pidió quien canceló.

### `Denied`

La costura de aplicación no dejó ocurrir el efecto — H4.

```python
Denied(decision: Decision) -> None
```

| Miembro | Firma | |
|---|---|---|
| `disposition` | — |  |
| `terminal` | — | ``True`` si el run no puede continuar por ninguna vía. |

Lleva la ``Decision`` entera para que el bucle sepa qué hacer: ``deny_step``
admite intentar otra cosa, ``terminate_run`` no, y ``require_approval``
suspende el run sin que sea un fallo.

### `InvalidToolCallError`

El modelo pidió una tool que no existe, o con argumentos que no validan.

```python
InvalidToolCallError(message: str = '', *, tool: str = '', call_id: str = '') -> None
```

No reintentable tal cual: lo que corrige esto es devolverle el error al
modelo como ``ToolResult`` para que rectifique, no repetir la llamada.

### `ToolExecutionError`

La tool existía, se invocó bien, y falló al ejecutarse.

```python
ToolExecutionError(message: str = '', *, tool: str = '', retryable: bool = False) -> None
```

Reintentable solo si quien la definió declara que el efecto es idempotente.

### `NoObjectGeneratedError`

Se pidió salida estructurada y no salió un objeto válido.

```python
NoObjectGeneratedError(message: str = '', *, raw: str = '') -> None
```

Reintentable: el muestreo es estocástico y una segunda pasada suele acertar.

### `LimitExceeded`

Se agotó un límite del bucle: pasos, reintentos, presupuesto de ventana.

```python
LimitExceeded(limit: str, value: int) -> None
```

Es corrección, no política: los límites de gasto pertenecen al harness y
llegan como ``Denied`` con ``terminate_run``.

### `UncertainEffect`

Al reanudar hay una intención registrada sin resultado, y el efecto no es idempotente.

```python
UncertainEffect(step_id: str, *, detail: str = '') -> None
```

El proceso cayó entre el registro y el efecto, así que el efecto **pudo
haber ocurrido** y no hay forma de saberlo desde aquí.  Repetirlo a ciegas
es lo peor posible; ignorarlo, tampoco es correcto.

El bucle no puede resolverlo por sí mismo — lo levanta para que lo decida
quien tiene la información: el harness, o una persona.

### `SeamVersionError`

No hay versión de costura común entre los dos extremos — RM-12.

```python
SeamVersionError(message: str = '', *, retryable: bool | None = None, retry_after: float | None = None) -> None
```

### `retryable_for_status`

```python
def retryable_for_status(status: int | None) -> bool
```

Aplica la regla de clasificación a un estado HTTP.

Sin estado devuelve ``True``: un fallo que ni siquiera llegó a tener
respuesta se parece más a un problema de camino que a una petición
inválida.

Un 4xx desconocido, en cambio, devuelve ``False``.  La familia entera
significa «tu petición es el problema», y repetirla sin cambiarla da el
mismo resultado — el 429 es la excepción, y está contemplada aparte.

## Esquemas y prompts

### `Schema`

Un tipo de salida: su esquema y cómo validar lo que vuelve.

```python
Schema(*args, **kwargs)
```

| Miembro | Firma | |
|---|---|---|
| `dump` | `dump(self, obj: Any) -> Any` | Devuelve el objeto como datos JSON, para el journal. |
| `json_schema` | `json_schema(self) -> Mapping[str, Any]` | El JSON Schema que viaja al proveedor. |
| `validate` | `validate(self, data: Any) -> Any` | Convierte datos ya parseados en el objeto tipado. |

### `schema_for`

```python
def schema_for(spec: Any) -> Schema
```

Convierte lo que el usuario tenga a mano en un ``Schema``.

Acepta, por este orden: algo que ya cumple el protocolo, un modelo de
Pydantic, un dataclass, o un JSON Schema como diccionario.

### `json_schema_for`

```python
def json_schema_for(fn: Callable[..., Any], *, localns: Mapping[str, Any] | None = None) -> dict[str, Any]
```

Deriva el JSON Schema de los parámetros de una función tipada.

### `PromptTemplate`

Un prompt con su versión.

| Campo | Tipo | Por defecto |
|---|---|---|
| `content` | `str` | **obligatorio** |
| `version` | `str` | `'1.0'` |
| `description` | `str` | `''` |
| `variables` | `Mapping[str, Any]` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `placeholders` | — | Nombres de las variables que el template usa. |
| `render` | `render(self, **values: Any) -> str` | Interpola las variables.  Los argumentos ganan sobre los del template. |

``version`` no es decoración: cuando una respuesta sale mal en producción, la
primera pregunta es con qué prompt se generó, y un literal en el código no
sabe contestarla.

### `PromptProvider`

Cualquier fuente de prompts.

```python
PromptProvider(*args, **kwargs)
```

| Miembro | Firma | |
|---|---|---|
| `exists` | `exists(self, name: str) -> bool` |  |
| `get` | `get(self, name: str) -> PromptTemplate` | Raises: ``KeyError`` si no existe. |

### `InMemoryPrompts`

Prompts en un diccionario.  Para tests y para sobrescribir en local.

```python
InMemoryPrompts(prompts: Mapping[str, PromptTemplate | str] | None = None) -> None
```

| Miembro | Firma | |
|---|---|---|
| `exists` | `exists(self, name: str) -> bool` |  |
| `get` | `get(self, name: str) -> PromptTemplate` |  |
| `register` | `register(self, name: str, template: PromptTemplate \| str) -> None` |  |

### `FilePrompts`

Prompts en un fichero JSON o YAML.

```python
FilePrompts(path: str | Path) -> None
```

| Miembro | Firma | |
|---|---|---|
| `exists` | `exists(self, name: str) -> bool` |  |
| `get` | `get(self, name: str) -> PromptTemplate` |  |
| `reload` | `reload(self) -> None` |  |

**JSON funciona sin dependencias.** YAML necesita PyYAML, y por eso no es el
camino por defecto: en la v0.4 lo era, y arrastraba una dependencia para todo
el mundo por una comodidad de sintaxis.

Carga perezosa y cacheada. ``reload()`` la invalida, que es lo que hace
utilizable editar prompts sin reiniciar.

### `PromptRegistry`

Encadena proveedores por prioridad: el primero que tenga el prompt, gana.

```python
PromptRegistry(*providers: PromptProvider) -> None
```

| Miembro | Firma | |
|---|---|---|
| `add` | `add(self, provider: PromptProvider) -> "PromptRegistry"` |  |
| `exists` | `exists(self, name: str) -> bool` |  |
| `get` | `get(self, name: str) -> PromptTemplate` |  |
| `prepend` | `prepend(self, provider: PromptProvider) -> "PromptRegistry"` | Lo pone delante de todo: es cómo se sobrescribe sin borrar. |

El orden importa y es el de registro, como un PATH. Así se sobrescribe un
prompt en local anteponiendo un proveedor, sin editar el fichero que
comparte todo el mundo.

### `fmt_dict`

```python
def fmt_dict(data: Mapping[str, Any], *, max_value: int = 500) -> str
```

Serializa un diccionario como líneas ``clave: valor``.

### `fmt_list`

```python
def fmt_list(items: list[Any], *, prefix: str = '· ') -> str
```

Serializa una lista como líneas con prefijo.

### `fmt_records`

```python
def fmt_records(items: list[Mapping[str, Any]], template: str) -> str
```

Serializa una lista de diccionarios con una plantilla por línea.

## Proveedores y utilidades

### `Provider`

Traduce entre el vocabulario unificado y el dialecto de un proveedor.

```python
Provider(*args, **kwargs)
```

| Miembro | Firma | |
|---|---|---|
| `from_wire` | `from_wire(self, body: Mapping[str, Any]) -> Response` | Del cuerpo nativo al vocabulario unificado. |
| `stream_from_wire` | `stream_from_wire(self, chunks: Iterable[Mapping[str, Any]]) -> Iterator[StreamEvent]` | De los fragmentos nativos al ciclo de eventos unificado. |
| `to_wire` | `to_wire(self, request: Request) -> Mapping[str, Any]` | Del vocabulario unificado al cuerpo que espera el proveedor. |

Las tres operaciones son **puras**: no abren conexiones y no leen
credenciales. Por eso el corpus dorado de normalización se puede ejecutar
sin un servidor delante.

### `dumps`

```python
def dumps(obj: Any) -> str
```

Serializa de forma determinista.

Claves ordenadas y separadores fijos: dos objetos iguales producen la misma
cadena byte a byte, en cualquier proceso y en cualquier ejecución.  Es lo
que hace comparables los eventos entre ejecución y replay, y lo que permite
que un prefijo de prompt sea estable frente a la caché del proveedor.

### `to_jsonable`

```python
def to_jsonable(obj: Any) -> Any
```

Convierte a estructuras JSON puras, recursivamente.

Los ``None`` se omiten: reducen el tamaño del journal y evitan que añadir un
campo opcional cambie la serialización de valores que no lo usan, lo que
rompería la comparación de replay entre versiones.

### `b64`

```python
def b64(data: bytes) -> str
```

Codifica binario para los campos ``data`` de las partes de contenido.

### `__version__`

`__version__ = '1.0.0rc1'`

str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str

## Sin agrupar

Exportados y todavía sin sitio en esta página. Que aparezcan aquí es un aviso para quien mantiene la referencia, no para quien la lee.

### `deferred`

```python
def deferred(herramientas: Sequence[Tool], *, max_resultados: int = 5) -> list[Tool]
```

Convierte un catálogo grande en **dos** herramientas de prefijo fijo.

Args:
    herramientas: el catálogo entero, decorado con ``@tool``.
    max_resultados: cuántas devuelve una búsqueda. Más no ayuda: el modelo
        elige peor cuantas más ve, que es medio problema que esto resuelve.

Returns:
    ``[buscar_herramientas, usar_herramienta]``, listas para ``Agent(tools=…)``.

Ejemplo::

agente = Agent("a", model=…, tools=deferred(las_cuarenta))

### `merece_la_pena`

```python
def merece_la_pena(herramientas: Sequence[Any]) -> bool
```

¿Vale la pena diferir este catálogo?

Está expuesto a propósito: es mejor que alguien pueda preguntar a que lo
descubra midiendo su factura.

## Dobles de desarrollo

`synaptum.testing` — Infraestructura, no una utilidad de test: es la vía principal para construir sin gastar.

### `FakeGateway`

``Gateway`` guionizado.  Ejecuta de verdad las tools que se le registran.

```python
FakeGateway(*script: ScriptItem, tools: Sequence[Tool] = (), chunk_size: int = 8, deny_tools: Mapping[str, Decision] | None = None) -> None
```

| Miembro | Firma | |
|---|---|---|
| `handshake` | `async handshake(self, hello: Hello) -> Welcome` |  |
| `invoke_model` | `async invoke_model(self, request: Request, ctx: CallContext) -> Response` |  |
| `invoke_tool` | `async invoke_tool(self, call: ToolCall, ctx: CallContext, *, risk: Risk = <Risk.READ: read>, tool_ref: str \| None = None) -> ToolResult` |  |
| `stream_model` | `stream_model(self, request: Request, ctx: CallContext) -> AsyncIterator[StreamEvent]` | Emite la respuesta troceada, honrando la cancelación. |

El guion se consume en orden.  Cada elemento puede ser:

* ``Response`` — se devuelve tal cual.
* ``str`` — atajo de ``says(...)``.
* ``BaseException`` — se lanza.  Con un ``ProviderError`` se ejercita el
  camino de reintentos.
* ``Decision`` — se lanza como ``Denied``.  Es como se prueban las tres
  disposiciones sin montar un motor de políticas.
* ``callable(request)`` — se llama y se trata su retorno como lo anterior.
  Sirve para responder según lo que el bucle acabe de enviar.

### `ReplayGateway`

``Gateway`` que sirve respuestas grabadas, normalizadas de verdad.

```python
ReplayGateway(*bodies: str | Path, provider: str = 'openai-compatible', tools: Sequence[Tool] = (), deny_tools: Mapping[str, Decision] | None = None) -> None
```

| Miembro | Firma | |
|---|---|---|
| `handshake` | `async handshake(self, hello: Hello) -> Welcome` |  |
| `invoke_model` | `async invoke_model(self, request: Request, ctx: CallContext) -> Response` |  |
| `invoke_tool` | `async invoke_tool(self, call: ToolCall, ctx: CallContext, *, risk: Risk = <Risk.READ: read>, tool_ref: str \| None = None) -> ToolResult` |  |
| `rewind` | `rewind(self) -> None` | Vuelve al primer cuerpo.  Útil para reanudar el mismo run dos veces. |
| `stream_model` | `stream_model(self, request: Request, ctx: CallContext) -> AsyncIterator[StreamEvent]` |  |

Args:
    bodies: rutas a cuerpos grabados.  ``.sse`` se sirve como stream;
        cualquier otra extensión, como respuesta completa.
    provider: adaptador con el que normalizar.  Por defecto, el dialecto
        OpenAI-compatible, que es el que habla casi todo.
    tools: herramientas decoradas, que se ejecutan de verdad.
    deny_tools: denegaciones simuladas por nombre de herramienta.

### `says`

```python
def says(text: str, *, usage: Usage | None = None) -> Response
```

Respuesta de texto que cierra el turno.

### `calls`

```python
def calls(name: str, *, id: str = 'call-1', usage: Usage | None = None, **arguments: Any) -> Response
```

Respuesta que pide una herramienta.

### `split_sse`

```python
def split_sse(text: str) -> list[dict[str, Any]]
```

Separa los fragmentos de un cuerpo SSE.

Vive aquí y no en el adaptador porque **cómo llegan los fragmentos por el
cable no es normalización**: es transporte, y esa separación es lo que
permite ejercitar la normalización con un fichero.

El centinela **termina** el cuerpo; no se salta. Un gateway real llegó a
mandar dos —reenviaba el del backend y añadía el suyo—, y la diferencia
entre saltarlos y parar en el primero no era cosmética: todo lo que
registraba la petición vivía pasado ese punto, así que el cliente correcto
era justo el que no se facturaba. Se arregló allí, y por eso mismo conviene
que esto no dependa de cuántos vengan.

### `DEFAULT_USAGE`

`DEFAULT_USAGE = Usage(input=100, output=20, reasoning=0, cache_read=0, cache_write=0, estimated=False)`

Contadores de tokens — H3.

## Cliente MCP

`synaptum.mcp` — Extra `[mcp]`. El núcleo no lo carga.

### `MCPTools`

Las herramientas de un servidor MCP, listas para pasárselas a un ``Agent``.

```python
MCPTools(session: Any, *, prefix: str = '') -> None
```

| Miembro | Firma | |
|---|---|---|
| `connected` | `connected(session: Any, *, prefix: str = '') -> "AsyncIterator[MCPTools]"` | Sobre una sesión que ya existe — la abre y la cierra quien la creó. |
| `destructivas` | — | Las que entran marcadas como destructivas. |
| `discover` | `async discover(self) -> list[MCPTool]` | Pregunta al servidor qué publica y lo adapta. |
| `stdio` | `stdio(command: str, *args: str, prefix: str = '', env: Mapping[str, str] \| None = None) -> "AsyncIterator[MCPTools]"` | Arranca un servidor MCP por stdio y descubre sus herramientas. |

Args:
    session: sesión MCP ya inicializada.
    prefix: prefijo para los nombres. Con dos servidores que publiquen
        ``search``, el modelo no puede distinguirlos — y el que gana es el
        que se registró último, en silencio.

### `mcp_risk`

```python
def mcp_risk(annotations: Any) -> Risk
```

Traduce las pistas de una herramienta MCP a un nivel de riesgo.

Los valores por defecto son los de la especificación MCP, no los nuestros:
**sin anotaciones, destructiva**. Nuestro ``@tool`` usa ``READ`` por defecto
porque quien escribe la función está delante y puede declarar; aquí el autor
no está, y suponer en su lugar es suponer a favor.

## Agentes remotos (A2A)

`synaptum.a2a` — Solo biblioteca estándar. En el camino gobernado la llamada sale por el proxy del arnés.

### `RemoteDelegate`

Un agente remoto, tal como lo ve quien delega.

| Campo | Tipo | Por defecto |
|---|---|---|
| `name` | `str` | **obligatorio** |
| `url` | `str` | **obligatorio** |
| `risk` | `Risk` | **obligatorio** |
| `description` | `str` | `''` |
| `poll_every` | `float` | `1.0` |
| `timeout` | `float` | `600.0` |
| `headers` | `Any` | `None` |
| `_depth` | `int` | `0` |

| Miembro | Firma | |
|---|---|---|
| `definition` | — |  |
| `execute` | `async execute(self, brief: str, session: Any, run_id: str) -> tuple[Any, Usage]` | Manda el brief, espera, y devuelve el resultado. |

Args:
    name: con qué nombre lo ve el modelo.
    url: el agente, o **el proxy que lo gobierna**. En el camino gobernado
        es lo segundo: el framework apunta al endpoint del arnés y no cambia
        nada más.
    description: cuándo usarlo. Si falta se toma de su tarjeta.
    risk: **obligatorio, sin valor por defecto.** Ver abajo por qué no lo tiene.
    poll_every: segundos entre consultas mientras la tarea trabaja.
    timeout: tope total de espera.

### `A2AClient`

Los cuatro métodos que hacen falta para delegar.

```python
A2AClient(base_url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 120.0) -> None
```

| Miembro | Firma | |
|---|---|---|
| `agent_card` | `async agent_card(self) -> AgentCard` | Lee `/.well-known/agent-card.json`. |
| `cancel_task` | `async cancel_task(self, task_id: str) -> Task` | Cancelar es lo que hacemos al cerrar el iterador, también por red. |
| `get_task` | `async get_task(self, task_id: str) -> Task` |  |
| `list_tasks` | `async list_tasks(self, *, context_id: str) -> list[Task]` | Las tareas de un contexto. |
| `send_message` | `async send_message(self, brief: str, *, context_id: str, message_id: str, task_id: str \| None = None) -> Task` |  |

Args:
    base_url: raíz del agente remoto, o del proxy que lo gobierna.
    headers: cabeceras extra — lo que pida el `securitySchemes` de su tarjeta.
    timeout: segundos por lectura. Un agente puede tardar mucho en el primer
        byte y eso no es un fallo.

### `AgentCard`

Lo que un agente publica en ``/.well-known/agent-card.json``.

| Campo | Tipo | Por defecto |
|---|---|---|
| `name` | `str` | **obligatorio** |
| `description` | `str` | `''` |
| `url` | `str` | `''` |
| `version` | `str` | `''` |
| `skills` | `tuple[Mapping[str, Any], ...]` | `()` |
| `capabilities` | `Mapping[str, Any]` | *(fábrica)* |
| `security_schemes` | `Mapping[str, Any]` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `push_notifications` | — |  |
| `streaming` | — |  |

Declara **qué sabe hacer**, no **qué puede romper**: no hay campo de riesgo
en la especificación. De ahí que un agente remoto entre como destructivo
mientras alguien no diga lo contrario.

### `Task`

Una unidad de trabajo al otro lado.

| Campo | Tipo | Por defecto |
|---|---|---|
| `task_id` | `str` | **obligatorio** |
| `context_id` | `str` | `''` |
| `state` | `TaskState` | `<TaskState.SUBMITTED: 'submitted'>` |
| `message` | `str` | `''` |
| `artifacts` | `tuple[Artifact, ...]` | `()` |
| `raw` | `Mapping[str, Any]` | *(fábrica)* |

| Miembro | Firma | |
|---|---|---|
| `esperando` | — |  |
| `result` | — | Lo que vuelve al agente que delegó. |
| `terminal` | — |  |

``task_id`` lo asigna **el servidor** y no podemos aportarlo — es el hecho
que decide todo nuestro diseño de reanudación. ``context_id`` sí lo ponemos
nosotros, y es por donde se reencuentra.

### `TaskState`

Los ocho estados de una tarea A2A.

| Valor | |
|---|---|
| `TaskState.SUBMITTED` | `'submitted'` |
| `TaskState.WORKING` | `'working'` |
| `TaskState.COMPLETED` | `'completed'` |
| `TaskState.FAILED` | `'failed'` |
| `TaskState.CANCELED` | `'canceled'` |
| `TaskState.REJECTED` | `'rejected'` |
| `TaskState.INPUT_REQUIRED` | `'input_required'` |
| `TaskState.AUTH_REQUIRED` | `'auth_required'` |

Se distinguen tres clases, y la distinción es la que decide qué hace el
bucle: los **terminales** no aceptan más mensajes, los que **esperan a
alguien** sí, y el resto siguen su curso.

### `Artifact`

Un entregable producido por una tarea.

| Campo | Tipo | Por defecto |
|---|---|---|
| `artifact_id` | `str` | `''` |
| `name` | `str` | `''` |
| `parts` | `tuple[Mapping[str, Any], ...]` | `()` |

| Miembro | Firma | |
|---|---|---|
| `text` | — |  |

Distinto de un mensaje: un mensaje es conversación, un artefacto es
resultado.

### `TERMINALES`

`TERMINALES = frozenset({<TaskState.CANCELED: 'canceled'>, <TaskState.COMPLETED: 'completed'>, <TaskState.FAILED: 'failed'>, <TaskState.REJECTED: 'rejected'>})`

Build an immutable unordered collection of unique elements.

### `ESPERANDO`

`ESPERANDO = frozenset({<TaskState.AUTH_REQUIRED: 'auth_required'>, <TaskState.INPUT_REQUIRED: 'input_required'>})`

Build an immutable unordered collection of unique elements.
