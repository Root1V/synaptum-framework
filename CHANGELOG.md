# Registro de cambios

Formato [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Versionado según
[`API.md`](API.md): SemVer desde `1.0.0`, con ventana de deprecación de dos versiones menores.

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

### Cambia una conducta por defecto

- **La salida de una herramienta se recorta a 16.000 caracteres antes de entrar en el contexto**
  (`Limits.max_tool_chars`). Activado por defecto porque no recortar falla **en silencio**: con una
  ventana pequeña revienta, y con una grande solo cuesta dinero en cada turno. El journal sigue
  guardando el resultado entero. Se desactiva con `max_tool_chars=None`.

- **Reanudar un run con otra configuración se rechaza** con `ConfigurationError`. Si el modelo, las
  instrucciones, el catálogo de herramientas o el formato de salida cambian, es otro run y necesita
  otro `run_id`. Evita que el journal describa una historia que ninguna configuración produjo.

- **`economy(state)`** — informe de economía de contexto calculado del journal: acierto de caché,
  reescrituras de prefijo y crecimiento por turno.

- **`Agent(delegates=[...])`** — delegar a un subagente con contexto aislado, diario propio y
  consumo agregado. El subagente no se reejecuta al reanudar.

- **`deferred(herramientas)`** — un catálogo grande sin inflar el prefijo: dos herramientas fijas y
  el esquema por el historial. `merece_la_pena()` dice cuándo compensa.

### Pendiente antes de `1.0.0`

Economía de contexto (Fase 2) y multi-agente (Fase 3). Ver [`roadmap.md`](roadmap.md).
