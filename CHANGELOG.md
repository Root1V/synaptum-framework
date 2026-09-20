# Registro de cambios

Formato [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Versionado según
[`API.md`](API.md): SemVer desde `1.0.0`, con ventana de deprecación de dos versiones menores.

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
