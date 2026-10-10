# Registro de cambios

Formato [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Versionado según
[`API.md`](API.md): SemVer desde `1.0.0`, con ventana de deprecación de dos versiones menores.

## [Sin publicar]

### Añadido

- **`HttpCheckpointer`** — el journal deja de vivir en el proceso y pasa a vivir donde un arnés lo
  gobierna, con presupuestos, atribución y durabilidad compartidas. El bucle no se entera: la
  costura es un protocolo y esto es otro relleno.

  La identidad de un sub-run se **parte**: el run padre va en la ruta —un segmento, sin escapes— y
  el resto viaja en el cuerpo como `sub_run_id`. Una barra cruda no enruta, y `%2F` funciona hoy y
  es lo que normalizan o rechazan los proxies: pasa en el test y falla detrás de un gateway.

  Un run de primer nivel **omite** la clave en vez de mandarla vacía, porque «no sé de sub-runs» y
  «digo que no hay» son cosas distintas para quien la recibe.

- **`HttpModel` acepta cabeceras por llamada**, como una función del contexto, y manda la **clave de
  idempotencia** derivada de la identidad del paso más la huella del cuerpo — la misma regla que ya
  usaba el puente de SDK. Sin ella, un reintento tras una caída es una segunda generación
  facturable: el journal no puede taparlo porque el agujero está entre mandar la petición y
  registrar su resultado.

  Es una función y no una lista de nombres nuestros a propósito: cómo se llaman las cabeceras que
  un gateway gobernado necesita lo decide quien las recibe, y escribirlas aquí metería su
  vocabulario dentro del framework. Quien pase su propia clave manda.

- **`thinks()` en el kit de dobles** — un turno que solo razona, que es el que destapó lo de abajo y
  antes solo se alcanzaba con un modelo real y de forma intermitente. En streaming el razonamiento
  también tiene su ciclo: un turno que no emitía **ningún** evento era el doble mintiendo.

- **`calls(..., raw="{roto")`** — una llamada con los argumentos ilegibles, para poder escribir el
  test del caso de abajo sin un modelo real.

- **`ToolCall.unreadable_arguments`** — el texto de los argumentos tal como llegó, cuando no se
  pudieron leer. Entonces `arguments` queda vacío y **la llamada no se ejecuta**.

### Corregido

- **Un turno del asistente con solo razonamiento ya no mata el run** (reportado por Veritium contra
  la `rc4`, visto contra llama-server con `gpt-oss-20b`). El razonamiento no se devuelve al
  proveedor, así que el mensaje salía al cable sin `content` y sin `tool_calls` —
  `{"role": "assistant", "content": null}`, que el propio dialecto declara inválido— y el servidor lo
  rechazaba con un **400 no reintentable**: el run entero perdido, en el turno *siguiente* al que
  razonó y solo cuando el modelo razona sin hablar. Intermitente, y más probable cuanto más largo es
  el run.

  Son dos arreglos. El adaptador **no envuelve** un turno que no lleva nada enviable, y no manda
  `content: ""` en su lugar: hay dialectos que rechazan un bloque de texto vacío, así que la forma
  «válida» dependería de quién esté al otro lado. Y el bucle **vuelve a preguntar** en vez de cerrar
  el run con un turno sin respuesta, que devolvía la cadena vacía sin que nada fallara.

  Lo segundo solo cuando el turno **terminó solo**, y el corte lo puso una grabación real: el caso
  que existe en el corpus es un razonamiento que se queda sin tokens —`finish_reason: length`—, y
  volver a preguntar eso es pedir la misma respuesta con el contexto más largo. Se trunca igual y se
  paga cada intento.

  El turno no se pierde: sigue en el diario, que es de donde lee el replay. Lo que desaparece es un
  envoltorio sin nada dentro.

  Las dos direcciones del adaptador no son independientes, y ahí estaba el hueco: la dirección
  *response* **produce** la parte que la dirección *request* no sabe mandar. El contrato de
  normalización dice que la dirección request «no se normaliza» y que el documento trata la response,
  así que el corpus que ejecutan las dos implementaciones no podía ver esto en ninguno de los dos
  lenguajes.

- **Una tool call con los argumentos ilegibles ya no mata el run** (reportado por Veritium, mismo
  día y mismo modelo). El modelo escribe un `tool_call` con el JSON de `arguments` cortado, y la
  conversión de la respuesta lo convertía en un `ProviderError` **no reintentable**: se perdía el
  turno, el resto del mensaje y el run entero por un argumento mal escrito — y el run se reintentaba
  completo, que cuesta un run en vez de un turno.

  Ahora el texto crudo viaja en la llamada y vuelve al modelo como resultado de error, con lo que
  escribió dentro: el mismo trato que una llamada cuyos argumentos no validan, porque para el modelo
  es el mismo error. **No sale por la costura:** nadie puede ejecutarla, así que no llega al arnés ni
  hace que una persona apruebe algo cuyos argumentos no se entienden.

  Sigue sin convertirse en `{}`: un objeto vacío silencioso ejecutaría la herramienta sin argumentos.
  Y ahora tampoco cuando el texto *parsea* y no es un objeto —`"[1,2]"`—, que era el caso que el
  aviso escrito en esa función decía cubrir y no cubría.

  El criterio lo lee **una sola función** que usan los dos lectores del dialecto. Dos copias escritas
  a mano del mismo criterio es lo que otro equipo acaba de ver fallar, con las dos perdiendo el campo
  nuevo a la vez.

- **Un turno del asistente que dijo la cadena vacía viaja con `content: ""`**, y solo se queda sin
  sobre el que no tiene **ninguna** parte de texto. Las dos cosas salían como `null`, que es la forma
  que el dialecto rechaza. La distinción la midió el equipo del arnés contra su servidor al arreglar
  su mitad del mismo fallo: `content: ""` devuelve 200 y `null` sin `tool_calls` devuelve 400, así
  que colapsarlas cambia el cuerpo de peticiones que hoy funcionan.

  Las dos formas quedan fijadas en el corpus compartido, en **dos casos de dirección `request`** —la
  dirección que el contrato declaraba que no había que normalizar, y donde falló—. El cuerpo de un
  caso así es un `Request` del vocabulario compartido y lo que se afirma es el cuerpo que sale al
  cable; sus claves son nuevas a propósito, para que un runner que solo conozca la dirección
  *response* falle a gritos en vez de saltárselas.

- **La referencia generada ya no desentrecomilla un valor por defecto.** Quitar las comillas es para
  las anotaciones —`task: 'str'` es ruido—, pero se las quitaba también a lo que viene detrás de un
  `=`: `current: str = 0.1` donde el valor es `'0.1'` invita a pasar un número donde va una cadena.

## [1.0.0rc4] — 2026-10-07

**Cortada a petición del segundo consumidor**, cuatro días después de la `rc3` y por un motivo
concreto: tenía cinco capacidades suyas entregadas en `main` y ninguna publicada, y dos de las que
faltan dependen de un tercer equipo. *«No tiene sentido retener cinco capacidades entregadas
esperando a dos que no controlan»* — y tienen razón.

Todo lo de esta versión salió de alguien portando su plataforma encima: lo que encontró al hacerlo
está aquí, y dos de las cosas que pidió no eran carencias nuestras sino fallos.

**Sin rupturas de superficie.** Cambia una conducta, y está dicha abajo.

### Añadido

- **`generate(task, model=…, session=…, output=…)`** — una llamada al modelo gobernada y durable sin
  escribir un agente. Es **un run de un solo turno, no un atajo por fuera**: registra su `ModelStep`,
  así que al reanudar no vuelve a inferir. Existe porque una plataforma tiene llamadas sueltas
  —clasificar, segmentar, enrutar— que si no pasan por la misma puerta quedan sin presupuesto, sin
  atribución y sin diario.

- **Entrada multimodal.** Una imagen viaja como `image_url` data-URI, y `run()` acepta
  `str | Message | Sequence[ContentPart]` además de texto. Un `Document` **se rechaza a propósito**:
  convertirlo sería decidir por quien lo manda cómo se ve una página, y eso lo decide quien la
  recortó.

- **`PromptTemplate.name`**, que el proveedor sella al servir la plantilla por su clave, y que junto
  con la versión viaja en el `meta` de cada paso de modelo y en el span del turno. El agente
  renderizaba la plantilla y se quedaba solo con el texto, así que la versión se perdía justo donde
  más falta hace: cuando una respuesta sale mal y la primera pregunta es con qué prompt se generó.

- **`Agent(submit_tool=True)`** — la salida final se entrega **llamando a una herramienta** cuyo
  esquema es el `output`, y una entrega que no valida vuelve al modelo con el error dentro para que
  corrija en el turno siguiente. **Sustituye a `response_format`** en vez de acompañarlo: dos
  restricciones que piden lo mismo pueden divergir, y con algunos motores una gramática de salida en
  la misma petición que un catálogo de herramientas impide que el modelo emita tool calls.

  Pedido por el segundo consumidor, y las tres decisiones de forma las tomó él.

### Cambia una conducta por defecto

- **Una salida que no valida se vuelve a pedir enseñando el error**, en vez de repetir la misma
  petición. Lo anterior no era reintentar: era repetir byte a byte esperando otra suerte del
  muestreo, con el modelo sin ver nunca qué había fallado.

  Consecuencias para quien ya lo usaba: el reintento consume **turnos** (`max_steps`) y no
  `max_retries`, y un run que se agota distingue los dos fallos **por tipo** — `LimitExceeded` si
  el modelo nunca llegó a entregar, `NoObjectGeneratedError` —con el último fallo dentro— si
  entregó y ninguna validó. Antes las dos acababan igual.

### Corregido

- Nada todavía. Se anota aquí según entra y no al cortar la versión: reconstruir un registro del
`git log` tres semanas después produce una lista de commits, no un registro de cambios — y lo que
se pierde es siempre el *porqué*, que es la mitad que sirve.

## [1.0.0rc3] — 2026-10-07

Tercer candidato, y el primero que sale **porque alguien de fuera lo necesitaba**: un segundo
consumidor —una plataforma documental— fue a portar su loop de extracción y lo que encontró está
casi todo aquí. Dos de las tres cosas que pidió no eran carencias, eran fallos nuestros.

**Ninguna ruptura respecto a `1.0.0rc2`** en la superficie. Sí cambian dos conductas al reanudar, y
están dichas abajo.

### Añadido

- **`Outcome`** y los campos `StepEvent.outcome` y `StepEvent.reason` — cómo terminó un paso, según
  el diario. Un desenlace denegado **es un hecho registrado**, no la ausencia de uno: sin esto, un
  paso que no se ejecutó porque alguien dijo que no se parece demasiado a uno del que no se sabe
  nada. Cinco valores: `result`, `denied_by_policy`, `approval_granted`, `approval_denied` y
  `approval_expired`.

- **`Sampling`** — `temperature`, `top_p`, `max_output_tokens`, `stop`, `tool_choice` y
  `provider_options` desde el `Agent`. Va aparte de `Limits` porque son cosas distintas: los límites
  son **corrección** y esto es **conducta**. Nada tiene valor por defecto: un `temperature` nuestro
  pisaría el del proveedor sin que nadie lo hubiera pedido, y la diferencia solo se vería en cómo
  responde el modelo, que es donde menos se busca.

  **Forma parte del prefijo estable**, así que reanudar un run con otro muestreo se rechaza igual
  que reanudarlo con otro modelo, y el error dice qué cambió: `muestreo: temperature 0.0 → 0.7`.

- **`synaptum.telemetry.traced()`** — la estructura del run como trazas: `agent.run`, `agent.turn`,
  `agent.delegate` y `agent.approval`. Envuelve el iterador de `Agent.run()` en vez de tocar el
  bucle, así que el núcleo sigue sin conocer OpenTelemetry y el extra `[otel]` solo hace falta para
  esto. **No emite la llamada al modelo ni la ejecución de herramientas**: las emite quien las
  ejecuta, y dos spans del mismo hecho traen dos duraciones que nunca coinciden.

  Un rechazo de gobierno viaja con **el tipo de disposición, no un booleano**, y **no** marca el
  span como error: una denegación de política es el sistema funcionando, y contarla como fallo
  enterraría los fallos reales bajo un flujo de rechazos correctos.

  **`describe_tracing()`** dice si el proveedor configurado descarta los spans. Es el fallo que no
  produce ningún error: sin proveedor los spans se emiten, no llegan a ninguna parte y todo
  funciona — solo faltan trazas, que es lo que nadie mira justo después de montarlas.

- **`tool_call_hash()`** — lo que ata una aprobación a lo que se aprobó, y **se niega** a construir
  el hash con un número de magnitud mayor que `2^53`, diciendo cuál y dónde. La canonicalización
  compartida serializa los números como dobles, así que por encima de ese límite dos valores
  distintos producen el mismo hash: una aprobación concedida para un importe validaría otro. El
  corte es de magnitud y no de ida y vuelta — comprobar si el número «sobrevive» aceptaría `2^53+2`
  y rechazaría `2^53+1`, dejando pasar la mitad de los identificadores según su paridad.

  Con un **flotante** el corte es `>=` en vez de `>`, y la asimetría es deliberada: un
  `9007199254740993.0` llega ya plegado a `2^53` —el dígito se pierde al decodificar, antes de que
  nada pueda mirarlo— así que un flotante que vale justo `2^53` es indistinguible de uno que vino
  de más arriba. Con un entero no hay ambigüedad.

- **`ProviderError.request_id` y `.trace_id`.** Son lo único que hace diagnosticable un fallo del
  otro lado, y viajaban dentro del texto del mensaje — donde van las cosas que nadie puede leer con
  un programa. Salió de un caso real: quien opera la plataforma pidió el identificador de una
  respuesta concreta y no lo teníamos, estando delante.

- El adaptador de plataforma recoge **`waited_s` y `attempts`** del SDK. Importan porque hay **dos
  reintentos apilados** —el del SDK dentro de una llamada y el del bucle encima—, y sin ellos una
  espera respetada a propósito es indistinguible de una plataforma lenta.

### Cambia una conducta por defecto

- **Un paso que una persona denegó ya no se reintenta al reanudar.** Antes, cualquier «no»
  registrado volvía a intentarse; ahora solo los que se pueden volver a intentar: una denegación de
  política (que pudo cambiar) y una **expiración** (que es la *ausencia* de una decisión, no una
  decisión). Una negativa humana cierra el paso y vuelve al modelo como evidencia, porque volver a
  preguntar tras un «no» es ir de compras a por un sí.

  Es la distinción que pedimos por el canal que no se colapsara, y que colapsábamos nosotros.

- **Un paso denegado por política llega al stream y al diario con su `Decision`.** Antes la
  negativa solo viajaba dentro del texto que ve el modelo, así que quien consumiera los eventos no
  podía distinguir «el gobierno lo paró» de «la herramienta falló» — y son cosas distintas para
  quien opera: la primera es el sistema funcionando.

- **Reanudar un run cuyo diario describe otra secuencia de pasos se rechaza** con
  `ConfigurationError`. Pasa al actualizar el framework si el bucle emite un paso más o uno menos:
  como los identificadores son posicionales, cada consulta falla por separado, ninguna sabe de las
  otras y **todo se reejecuta** — medido, un pago no idempotente repetido y la inferencia pagada
  otra vez, en silencio. Se detecta por la **clase** del paso que ocupa esa posición y no por su
  ausencia: un hueco es normal, porque las escrituras diferidas se agrupan.

- Un registro **sin `outcome` sigue significando `result`**. No es tolerancia: hay diarios ya
  escritos, y leerlos de otra forma dejaría colgado un paso que sí se ejecutó.

### Documentación

- **Los dieciséis ejemplos son una sección del sitio**, una página por ejemplo, con el fichero
  entero, **lo que imprime al correrlo** —capturado ejecutándolo, no escrito a mano— y el enlace a
  GitHub. Se generan de `examples/`: copiarlos crearía dos originales que envejecen por separado.
- **El CI ejecuta los ejemplos**, como efecto de lo anterior. Hasta ahora nada los corría: un
  ejemplo roto pasaba la suite entera.
- Los nombres de los sistemas sobre los que están montados enlazan a su repositorio, una vez por
  página y una por fila de tabla.
- La documentación **le habla a quien la lee**. Dos ejemplos se dirigían al autor («cualquiera de
  tus repos»), lo que para quien acaba de instalar el paquete significa que el ejemplo es para otro.
- Los datos de los ejemplos son ahora **inequívocamente** inventados —`John Doe`, `ACME`— y la regla
  está escrita en `examples/README.md`. Había un nombre verosímil al lado de un sueldo: sintético no
  es lo mismo que inocuo.

### Publicación

- **Publicar en PyPI exige la aprobación de una persona**, en el entorno de GitHub. Un permiso que
  depende de acordarse de preguntar no es un control: es una costumbre. TestPyPI se queda sin puerta
  a propósito — pedir un clic por cada ensayo acabaría con los ensayos.
- **El job que verifica lo publicado no se estaba ejecutando.** GitHub propaga el salto de un job
  por toda la cadena de `needs`, así que añadir `ensayo` apagó `verify` sin que nada fallara. Dos
  releases salieron en verde sin que nadie comprobara que lo publicado se instala.
- Y `--extra-index-url` invertía la búsqueda: uv mira los índices extra **antes** que el principal,
  así que desde que el paquete existe en PyPI, pedir una versión que solo está en TestPyPI fallaba
  con «no existe».

### Corregido

- **Una tool llamada con argumentos que no encajan ya no mata el run.** Vuelve al modelo como
  `ToolResult(is_error=True)` con el detalle, y el modelo corrige. El docstring de
  `InvalidToolCallError` describía esa conducta desde el principio —«devolverle el error al modelo
  para que rectifique, no repetir la llamada»— mientras el bucle dejaba subir la excepción.

- **El adaptador `openai-compatible` se niega a mandar lo que no sabe transportar** en vez de
  descartarlo en silencio. Una imagen en un mensaje desaparecía del cuerpo, el modelo contestaba sin
  ella y nada fallaba: la respuesta parecía mala y lo que estaba mal era el envío.

- **El cargador de los corpus compartidos suponía la forma por la ubicación**, leyendo todo `*.json`
  de una carpeta como si solo pudiera haber una clase de documento ahí. El día que llegó un vecino
  con otra forma, sus once casos entraron en el test equivocado. Ahora selecciona por lo que el
  documento **declara ser**.

- `ContextEconomy.report()` mezclaba separadores de millares en el mismo informe.
- `validar_ruc`, en el ejemplo `03`, prometía comprobar el dígito verificador y solo comprobaba el
  formato. Una herramienta que declara más de lo que hace es lo peor que puede haber en un catálogo.

### Pendiente antes de `1.0.0`

- **Un segundo consumidor que lo haya usado de verdad.** Es el único criterio que falta y no es de
  código: una API que solo ha usado quien la escribió no está probada, está confirmada. Hay uno
  portando su plataforma ahora; lo que encuentre pesa más que cualquier fila del roadmap.
- **Entrada multimodal** (`SYN-85`) — hoy el adaptador se niega a mandar una imagen en vez de
  descartarla, que es la mitad honesta pero no la útil.
- **Re-preguntar con el error de validación** (`SYN-86`) — hoy un reintento manda un request
  idéntico, así que el modelo nunca ve qué falló. Es repetir, no reintentar.
- **La mitad servidor de A2A**, esperando una decisión de gobierno que no es nuestra, y el cliente
  `Gateway` y el `HttpCheckpointer` contra el arnés, que esperan a que sus endpoints existan.

## [1.0.0rc2] — 2026-09-20

Segundo candidato. Todo lo de la Fase 2 —economía de contexto— y la Fase 3 —multi-agente— que no
depende de nadie más. **Ninguna ruptura respecto a `1.0.0rc1`**: lo que estaba sigue estando y con
la misma forma.

### Añadido

- **`Agent(delegates=[...])`** — delegar a un subagente con contexto aislado, diario propio y
  consumo agregado. El subagente **no se reejecuta al reanudar**: su identidad se deriva de la del
  padre, `{run_id}/{step_id}`. Y el riesgo **se deriva** en vez de declararse — delegar en alguien
  que borra es destructivo, que es lo que envolver un agente en una función blanqueaba a `READ`.

- **`RemoteDelegate`** — un agente en otro despliegue, por A2A, en la misma lista de `delegates`.
  Cliente propio sobre el binding HTTP+JSON, solo biblioteca estándar. Reanudar es una **consulta**
  (`tasks/list` por `contextId`) y no una apuesta: el `taskId` lo asigna el servidor y se pierde en
  una caída. Contra un servidor sin `tasks/list` se baja un peldaño, y se dice.

- **`economy(state)`** — informe de economía de contexto calculado **del journal**, así que sirve
  sobre un run terminado: acierto de caché, reescrituras de prefijo y crecimiento por turno. No
  exporta nada: la forma de la instrumentación la fija quien la consuma.

- **`deferred(herramientas)`** — un catálogo grande sin inflar el prefijo: dos herramientas fijas y
  el esquema por el **historial**, no por el prefijo. Medido contra una plataforma real: añadir una
  sola herramienta a mitad de un run baja `cache_read` de 1.443 a 0. `merece_la_pena()` dice cuándo
  compensa, porque por debajo de quince sale más caro.

- **`MCPTools`** — herramientas servidas por un servidor MCP, adaptadas a lo que el bucle ya
  consume. Una anotación ausente se traduce a `DESTRUCTIVE`: son los defectos de MCP, no los
  nuestros.

- **`cap_tool_output`** y **`prefix_fingerprint` / `describe_prefix_change`**, expuestos para quien
  necesite la pieza suelta.

- **Plantilla de proyecto** (`plantilla/`) y **sitio de documentación** — ocho páginas de una sola
  fuente, con referencia de API generada del código.

### Cambia una conducta por defecto

- **La salida de una herramienta se recorta a 16.000 caracteres antes de entrar en el contexto**
  (`Limits.max_tool_chars`). Activado por defecto porque no recortar falla **en silencio**: con una
  ventana pequeña revienta, y con una grande solo cuesta dinero en cada turno posterior. El journal
  sigue guardando el resultado entero. Se desactiva con `max_tool_chars=None`.

- **Reanudar un run con otra configuración se rechaza** con `ConfigurationError`. Si el modelo, las
  instrucciones, el catálogo de herramientas o el formato de salida cambian, es otro run y necesita
  otro `run_id`. Evita que el journal describa una historia que **ninguna configuración produjo**.

- **`RemoteDelegate.risk` es obligatorio**, sin valor por defecto. Un `AgentCard` declara
  habilidades, no riesgo. Un defecto conservador sería correcto **y silencioso**: nadie se entera
  nunca de que el riesgo no se declaró, y la decisión la toma un valor en vez de una persona.

### Corregido

- El ciclo de razonamiento se cerraba solo al llegar texto, así que `reasoning_end` caía en mitad de
  una llamada a herramienta. Estaba **en los dos adaptadores**, y el segundo solo apareció contra
  una plataforma de verdad.
- La clave de idempotencia era `(run_id, step_id)` y colisionaba cuando el cuerpo cambiaba. Ahora
  lleva una huella del cuerpo.
- Los reintentos no esperaban nada. Ahora hay retroceso exponencial con jitter y `Retry-After`, con
  un techo: quien gobierna decide cuánto puede tardar un run.
- La salida tipada fallaba en silencio contra un gateway que descarta `response_format`. El esquema
  viaja también en las instrucciones.
- `Check` no llevaba `arguments`, así que una política de herramienta no podía ver aquello sobre lo
  que decide.

### Documentación y ejemplos

Doce ejemplos, del agente más simple a delegar en uno que vive en otro contenedor, cada uno sobre un
dominio real y **ejecutable sin inferencia**. La comprobación de caducidad de la documentación mira
ahora también `examples/`: tres ejemplos llegaron a afirmar que `delegate()` no existía después de
que existiera, y ningún test lo veía porque los ejemplos no eran documentación a ojos del
instrumento.

### Pendiente antes de `1.0.0`

- **Instrumentación** (`SYN-37`) — aplazada a propósito: hay una plataforma de observabilidad en
  construcción que va a decidir la forma, y adelantarse sería instrumentar contra un formato que
  habría que tirar.
- **La mitad servidor de A2A** (`SYN-44`) — exponer un agente nuestro como agente A2A. Esperando a
  una decisión de gobierno que no es nuestra: quién autoriza una delegación remota y dónde.
- **Suite de conformidad de la costura** (`SYN-31`) — en curso, y es artefacto conjunto con el
  arnés.
- Artefactos y compactación (`SYN-34`, `SYN-35`, `SYN-36`), grafo declarativo y recetas (`SYN-42`,
  `SYN-43`). Ninguno tiene todavía un caso real que los pida, y construirlos antes es adivinar.

## [1.0.0rc1] — 2026-09-13

Publicado en TestPyPI. Instalable con `--index-url https://test.pypi.org/simple/`.

Primera publicación. **Es un candidato y no una `1.0.0`, a propósito.** La superficie está completa
contra el contrato de hoy y no va a cambiar por gusto, pero la Fase 2 (economía de contexto) todavía
puede tocarla, y el compromiso de estabilidad de [`API.md`](API.md) arranca en `1.0.0`. Prometerlo
antes de tiempo sería peor que esperar.


Primera línea `1.0.x`. La `0.x`, con un diseño distinto, quedó congelada en
[v0.4.0](https://github.com/Root1V/synaptum-framework/releases/tag/v0.4.0) y **no hay ruta de
migración**: la superficie no se parece. El estado por elemento está en [`roadmap.md`](roadmap.md).

### Añadido

- Bucle del agente como stream de eventos tipados — `Agent.run()` y `Agent.stream()`.
- Ejecución durable: al reanudar un run, **una inferencia ya pagada no se paga otra vez**.
  `MemoryCheckpointer` y `SqliteCheckpointer` como implementaciones de referencia.
- Identidad determinista de paso (`000003-model`), publicada como especificación abierta, y que
  además viaja como clave de idempotencia hacia proveedores que la admitan.
- Herramientas con esquema derivado de la firma: `@tool`. `risk` e `idempotent` se declaran.
- Vocabulario unificado de modelo con `Usage` de **tres estados** — medido, sin medir, estimado.
- Adaptador `openai-compatible` y transporte HTTP de desarrollo, ambos sin dependencias.
- Prompts como configuración versionada.
- `FakeGateway` y `ReplayGateway` como infraestructura de desarrollo de primera clase.
- Marcador `py.typed`: los tipos llegan a quien consume el paquete.

### Pendiente antes de `1.0.0`

Economía de contexto (Fase 2) y multi-agente (Fase 3). Ver [`roadmap.md`](roadmap.md).
