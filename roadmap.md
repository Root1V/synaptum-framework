# Roadmap — Synaptum v1.0

Backlog de implementación del rediseño. Cada elemento tiene identificador `RM-xx`, estado y una línea sobre lo que implica.

**Contexto:** la línea 0.x queda congelada en [v0.4.0](https://github.com/Root1V/synaptum-framework/releases/tag/v0.4.0). La v1.0 se construye desde cero sobre la frontera acordada con Aeon: Synaptum es dueño de la *semántica* de ejecución, Aeon del *sustrato*, y entre ambos hay dos costuras — una que aplica y otra que recuerda.

## Estados

| Estado | Significado |
|---|---|
| `LISTO` | Sin dependencias abiertas, se puede empezar |
| `BLOQUEADO` | Depende de una decisión de contrato aún no cerrada |
| `PENDIENTE` | Depende de otro RM interno todavía no hecho |
| `EN CURSO` | Trabajo iniciado |
| `HECHO` | Completado y con test |
| `EXTERNO` | Propiedad de Aeon; Synaptum depende del resultado |

## Decisiones de contrato

Las seis quedaron resueltas en la cuarta iteración. Ninguna bloquea ya.

| # | Decisión | Resultado |
|---|---|---|
| H1 | Dueño de la capa de modelo | **Opción D** — vocabulario compartido, implementación por lenguaje |
| H2 | Streaming en la costura | Stream con cancelación bidireccional desde v0.1 |
| H3 | `Usage` de vuelta | `input · output · reasoning · cache_read · cache_write` |
| H4 | Disposición de denegación | `deny_step · terminate_run · require_approval` |
| H5 | Esquema de tool | Referencia versionada en el cable; el handshake resuelve o registra |
| H6 | Versionado | N = 2 versiones menores, aviso de obsolescencia una por delante |

---

## Fase 0 · Contratos

Congelar antes de escribir implementación. Es la fase que impide reescrituras posteriores.

| ID | Estado | Feature | Implica |
|---|---|---|---|
| RM-01 | `HECHO` | Vocabulario unificado de modelo | `Message`, `ContentPart` (8 tipos), `Request`, `Response`, `StreamEvent`, `FinishReason`. Contrato compartido versionado, no interno de Synaptum |
| RM-02 | `LISTO` | Especificación de normalización entre proveedores · repo `Root1V/agentic-seam-contracts` | Colocación de resultados de tool, extracción del mensaje de sistema, bloques de razonamiento, reconstrucción de tool calls en streaming. Vive en el repo de contratos, no en ninguno de los dos proyectos |
| RM-03 | `LISTO` | Corpus de fixtures dorados · desbloqueado, el repo lo crea Aeon | Respuestas nativas grabadas por proveedor con su salida unificada esperada. Es lo único que impide que las implementaciones Go y Python diverjan bajo la opción D |
| RM-04 | `HECHO` | Taxonomía de errores | Jerarquía con `retryable`. No reintentar 400/401/403/404/422; sí 429/5xx/timeouts |
| RM-05 | `HECHO` | Taxonomía de eventos del bucle | `ModelStep`, `ToolStep`, `DelegateStep`, `ApprovalStep`, `FinalStep`. Unión tipada, inmutable y ordenada |
| RM-06 | `EN CURSO` | Identidad determinista de paso | Especificación **abierta**, no interna. Cualquier framework que la implemente obtiene durabilidad de Nivel 1. Es también la clave de idempotencia del journal |
| RM-07 | `HECHO` | Protocolo `Checkpointer` | `append(run_id, event)` / `load(run_id)`. Idempotente por `(run_id, step_id, phase)`; un duplicado es no-op |
| RM-08 | `HECHO` | Protocolo de la costura de aplicación | Síncrona, denegable, fuera del proceso del bucle. Separada del `Checkpointer` porque vetar y recordar tienen presupuestos opuestos |
| RM-09 | `HECHO` | Disposición de denegación como tipo | Tres valores más código de razón y mensaje legible. Sin esto el bucle no sabe si replanificar o parar |
| RM-10 | `HECHO` | Contrato de streaming con cancelación | Bidireccional desde v0.1. Habilita el corte de presupuesto en caliente |
| RM-11 | `HECHO` | Propagación de contexto de traza | `traceparent` / `tracestate` W3C en ambos sentidos de la costura, o los spans del bucle y los de E/S quedan en árboles distintos |
| RM-12 | `HECHO` | Handshake de versión de la costura | `Hello` / `Welcome` en un solo viaje con RM-13: ambas cosas ocurren antes del primer turno y ninguna puede repetirse a mitad de sesión. Ventana N = 2 **en total**, contando la actual |
| RM-13 | `HECHO` | Referencia versionada de esquema de tool | El esquema viaja fuera de la llamada para no romper el prefijo estable de caché. Handshake resuelve contra registro existente o registra por sesión |
| RM-14 | `HECHO` | Clases de durabilidad por evento | Intención y resultado de efecto no idempotente: síncronos y durables. Resto: diferible. Corrige el supuesto de `append` asíncrono |

---

## Fase 1 · Núcleo durable

Un agente único, provider-agnóstico, reanudable. Es el mínimo que Aeon puede gobernar.

| ID | Estado | Feature | Implica |
|---|---|---|---|
| RM-15 | `HECHO` | Paquete `core` sin dependencias | Solo stdlib. Verificado: 6 paquetes en el entorno frente a los 108 de la v0.4, y cero módulos de terceros al importar |
| RM-16 | `PENDIENTE` | Protocolo `Schema` y adaptadores | `json_schema()` + `validate()`. Hace opcional a Pydantic; soporta también dataclasses y msgspec |
| RM-17 | `PENDIENTE` | Registro de proveedores por entry points | Resolución de `"provider:modelo"` vía `importlib.metadata`. Sin conocimiento previo de los plugins |
| RM-18 | `PENDIENTE` | Adaptadores Python de proveedor | Subconjunto de desarrollo, **no paridad** con el gateway. Solo los presentes en ambos lados necesitan fixtures de RM-03 |
| RM-19 | `HECHO` | Bucle del agente como stream de eventos | `async for step in agent.run(...)`. El motor es `await`, no una cola. Cada `yield` es frontera de checkpoint |
| RM-20 | `PENDIENTE` | Decorador `@tool` | JSON Schema derivado de la firma tipada. Sin duplicar la descripción a mano |
| RM-21 | `HECHO` | Niveles de riesgo de tool | `read / soft_write / hard_write / destructive`. Synaptum **declara**; Aeon **decide** |
| RM-22 | `HECHO` | Journal con durabilidad por clase | Escritura anticipada de la intención antes de todo efecto no idempotente |
| RM-23 | `HECHO` | `Checkpointer` en memoria | Implementación de referencia para tests y notebooks |
| RM-24 | `PENDIENTE` | `Checkpointer` SQLite | Implementación de referencia persistente. Nunca un motor de producción |
| RM-25 | `HECHO` | Replay con fast-forward | Al reanudar, saltar pasos ya registrados sin repetir inferencia ya pagada. Es la propiedad que justifica toda la arquitectura |
| RM-26 | `PENDIENTE` | Costura de aplicación local permisiva | Ejecuta contra credenciales del entorno y registra lo que habría comprobado. **Con aviso explícito de que no es aplicación real** |
| RM-27 | `HECHO` | Límites del bucle | `max_steps`, `max_retries`, reserva de salida del 20–25 % de la ventana. Corrección, no política |
| RM-28 | `PENDIENTE` | Sistema de prompts | Portado desde v0.4: `PromptTemplate` versionado, providers encadenados, disciplina YAML-first |
| RM-29 | `PENDIENTE` | Suite de tests del núcleo | **Íntegramente sobre `FakeModel`**, sin depender de inferencia real — obligatorio por P10, no preferible. Cierra la contradicción de la v0.4, que vendía testabilidad sin un solo test |
| RM-65 | `PENDIENTE` | `FakeModel` como infraestructura de primera clase | Por P10 es la **vía principal de desarrollo**, no una utilidad. Respuestas guionizadas, simulación de tool calls, de streaming con cancelación, de `Usage` con tokens de caché y razonamiento, e inyección de errores de la taxonomía |
| RM-66 | `PENDIENTE` | `ReplayModel` sobre el corpus de fixtures | Los fixtures dorados de `RM-03` sirven doble: además de probar equivalencia, respaldan un modelo que reproduce respuestas reales grabadas. Da comportamiento realista con cero acceso y cero coste — la mejor respuesta disponible a la limitación de P10 |
| RM-30 | `HECHO` | Empaquetado con extras | Cero dependencias duras. `[pydantic]`, `[anthropic]`, `[openai]`, `[mcp]`, `[otel]`. Elimina el arrastre de torch vía llm-guard. El extra `[axonium]` espera a RM-48: un extra irresoluble rompe `uv lock` entero, no solo su instalación |
| RM-31 | `BLOQUEADO` | Suite de conformidad de la costura | Artefacto **conjunto**. Se escribe tras congelar el contrato. Denegación honrada, idempotencia, traza preservada, `Usage` correcto, cancelación a mitad de stream |

---

## Fase 2 · Contexto y observabilidad

Economía de contexto y visibilidad de producción. Aquí es donde se gana o se pierde el coste por turno.

| ID | Estado | Feature | Implica |
|---|---|---|---|
| RM-32 | `PENDIENTE` | Ensamblador de contexto cache-first | Orden estable del prefijo. Prohibido cambiar tools o modelo a mitad de sesión |
| RM-33 | `PENDIENTE` | Middleware `cap_tool_output` | Capar salidas a tamaño estable. Medido en −38 % de coste por turno sin pérdida de recall |
| RM-34 | `PENDIENTE` | Middleware `artifactize` | Resultados grandes fuera del contexto, con referencia. Umbral de 8–16k tokens |
| RM-35 | `PENDIENTE` | Almacén de artefactos | Direccionable por URI con procedencia. Protocolo, no implementación de producción |
| RM-36 | `PENDIENTE` | Compactación por niveles | Escalonada y **desactivada por defecto**: con caching, conservar todo suele salir más barato que resumir |
| RM-37 | `PENDIENTE` | Instrumentación OTel GenAI | Solo estructura del bucle: fronteras de turno, planificación, delegación. `chat` y `execute_tool` los emite Aeon |
| RM-38 | `PENDIENTE` | Métricas de economía de contexto | Tasa de acierto de caché, reescrituras de prefijo, frecuencia de compactación. `Usage` llega de vuelta por la costura |
| RM-39 | `PENDIENTE` | Cliente MCP | Extra opcional. Vía estándar de herramientas desde su donación a la Linux Foundation |
| RM-40 | `PENDIENTE` | Carga diferida de tools | Búsqueda de tools para catálogos grandes, sin inflar el prefijo |

---

## Fase 3 · Multi-agente

Solo después de que un agente único sea sólido. Regla: *single agent first*.

| ID | Estado | Feature | Implica |
|---|---|---|---|
| RM-41 | `PENDIENTE` | `delegate()` con contexto aislado | Orchestrator-worker. Brief estrecho, devuelve resultado y referencias, nunca historial. `asyncio.gather` real |
| RM-42 | `PENDIENTE` | Grafo declarativo | Máquina de estados tipada sobre las primitivas. Sin duplicar `GraphPattern`/`GraphAgent` como en v0.4 |
| RM-43 | `PENDIENTE` | Recetas de patrones | Los 21 patrones portados como recetas de ~40 líneas, no como núcleo. Saga sobre journal, no sobre coreografía a mano |
| RM-44 | `PENDIENTE` | Adaptador A2A | Delegación a agentes remotos. Extra opcional |
| RM-45 | `PENDIENTE` | Cookbook bancario | Ejemplos y prompts YAML a repo aparte. Mantiene el núcleo ligero y sirve de validación end-to-end |

---

## Transversal

| ID | Estado | Feature | Implica |
|---|---|---|---|
| RM-46 | `LISTO` | Compromiso de estabilidad de API | Contrapartida a que Aeon congele su DSL de autoría nativo y dependa de Synaptum como capa de autoría |
| RM-47 | `LISTO` | Repo de contratos compartido | Solo especificaciones, esquemas y fixtures. **Sin implementación**, o se convierte en el quinto proyecto que descartamos |
| RM-48 | `LISTO` | Publicar o vendorizar Axonium | Deuda heredada: `uv.lock` lo resolvía desde un registro local, así que la instalación desde git nunca funcionó para terceros |

## Axonium · Puerta única a inferencia local

**Regla fijada por el dueño del proyecto:** *toda la inferencia local se resuelve en la plataforma Prometheus, y la única forma de llegar a los modelos que Prometheus expone es el SDK Axonium.*

**Decisiones cerradas (P1 · P2 · P3):**

| # | Decisión |
|---|---|
| P1 | Axonium es **SDK en tres sabores**: Python, Go y Rust. No es un servicio — no hay salto de red añadido |
| P2 | Alcance = **todos los modelos que expone Prometheus**. Toda la inferencia local pasa por ahí |
| P3 | Axonium **normaliza respuestas** y **consume el esquema de request compartido**, dejando de reimplementarlo |

**Frontera de proveedores que resulta:**

| Clase | Quién normaliza |
|---|---|
| Inferencia local (todo Prometheus) | **Axonium**, en cada uno de sus tres sabores |
| Inferencia cloud (Anthropic, OpenAI, Gemini) | Adaptadores nativos del gateway (Go) y adaptadores dev de Synaptum (Python) |

Nadie reimplementa la normalización de otro. Cada proveedor tiene exactamente una implementación por lenguaje, alojada donde vive el conocimiento de ese proveedor.

**Quién usa el SDK:**

| Consumidor | Sabor | Modo |
|---|---|---|
| **Aeon** — Model Gateway | Go | Gobernado, producción. Ahí vive la credencial de Prometheus |
| **Synaptum** | Python | Autónomo — dev, notebooks, tests. **Nunca** en el camino gobernado |
| Terceros | Rust | Fuera del alcance de este acuerdo |

| ID | Estado | Feature | Implica |
|---|---|---|---|
| RM-49 | `EXTERNO` | Axonium consume el esquema de request compartido | Deja de reimplementarlo. Convierte a Axonium en **tercer consumidor del repo de contratos**, junto a Synaptum y Aeon |
| RM-50 | `EXTERNO` | **Axonium-Go — camino crítico** | Contrato definido (catálogo de errores, corpus de fixtures, Python como referencia con 452 tests) pero **no construido**. El camino gobernado de Aeon hacia modelos de Prometheus depende de él. Prioridad por encima de Rust y de publicar en PyPI: es el único que alguien más está esperando |
| RM-51 | `EXTERNO` | Streaming con cancelación en los tres sabores | **Dependencia dura de H2**: sin stream cancelable no hay corte de presupuesto en caliente para ningún modelo de Prometheus. `context.Context` en Go, cancelación de asyncio en Python. Debe entrar en RM-50 desde v0.1, no como añadido posterior |
| RM-64 | `EXTERNO` | Equivalencia entre sabores de Axonium | El corpus de fixtures y los 452 tests de Python ya existen: **son el criterio de aceptación de RM-50**, no trabajo adicional. Go se construye contra ellos |
| RM-52 | `PENDIENTE` | Adaptador Synaptum → Axonium-Python | Entry point `synaptum.providers`, extra `synaptum[axonium]`. Synaptum **no** escribe adaptador de Prometheus: usa Axonium |
| RM-53 | `EXTERNO` | Gateway Go → Axonium-Go | Sustituye al adaptador `prometheus_inference` nativo. Y decidir qué pasa con `openai_compatible`: si servía endpoints locales, pasa a Prometheus; si servía cloud compatible, se queda |
| RM-54 | `EXTERNO` | Propagación de cancelación extremo a extremo | Camino gobernado: Synaptum → costura → gateway con Axonium en proceso → Prometheus. **Tres saltos, no cuatro** — el SDK en proceso ahorra uno |

## Dependencias externas — Aeon

Synaptum depende del resultado; no bloquean la Fase 0 ni la Fase 1.

| ID | Estado | Feature | Implica |
|---|---|---|---|
| RM-55 | `EXTERNO` | Streaming con cancelación en el Model Gateway | Hoy es petición/respuesta puro. Sin esto no hay corte de presupuesto en caliente para ningún cliente |
| RM-60 | `EXTERNO` | Axonium: dos modos de credencial, permanentes | **Gobernado** (Aeon/Go): proveedor inyectado, el SDK nunca ve un secreto, y el proveedor es autoridad completa — Axonium cede su refresco anticipado y se queda con reintento reactivo. **Autónomo** (Synaptum/Python): `client_id`/`client_secret`, el SDK acuña y refresca. Exactamente uno debe suministrarse, validado en construcción |
| RM-61 | `EXTERNO` | `TokenProvider` con señal de invalidación | Con Axonium en proceso, **es Axonium quien ve el 401**, no la capa HTTP de Aeon. `Token(ctx, forceRefresh)` mínimo. Falta fijar la **semántica concurrente**: dos llamadas que reciben 401 del mismo token no deben acuñar dos tokens. Alternativa que lo hace trivialmente correcto: pasar el token rechazado en vez de un booleano |
| RM-62 | `HECHO` | Anclaje de reloj del `TokenSource` — cerrado por Aeon | Ni reloj de servidor ni local: Go usa reloj **monótono**, que mide tiempo transcurrido y es inmune al desfase y a saltos de NTP. Mejor que anclar al `Date`. Residuo declarado: suspensión del sistema, que se resuelve por el camino 401 → reacuñación |
| RM-63 | `EXTERNO` | Regla local→Prometheus como política de routing | En `ModelPolicyBundle`, config-as-code, **no borrando `openai_compatible`**. Condición derivada de forma independiente por Aeon y por Axonium: la política debe poder **negar y fallar en cerrado**, con la denegación observable. Enrutar no basta |
| RM-56 | `EXTERNO` | `Checkpointer` de Aeon con deduplicación | Por `(run_id, step_id, phase)`, resuelve el reintento *at-least-once* de Activities de Temporal |
| RM-57 | `EXTERNO` | Nivel de durabilidad como campo consultable | Campo del run y atributo de traza. Un operador debe poder saber en un incidente si ese run repite inferencia al reanudarse |
| RM-58 | `EXTERNO` | Medición gateway vs proxy de egress local | Con streaming. Excluye el motor de decisión puro, que no cumple la condición de credencial fuera del proceso |
| RM-59 | `EXTERNO` | Implementación Go de la especificación de normalización | Contra el corpus de fixtures de RM-03, para demostrar equivalencia con la implementación Python |

---

## Preguntas abiertas

**Cerradas:** P1 · P2 · P3 por el dueño de Axonium; P4 · P6 · P9 por Aeon.

| # | Respuesta |
|---|---|
| P4 | **Proveedor de tokens inyectado**, no secreto. Patrón `TokenSource` ya funcionando en Aeon → `RM-60` |
| P6 | **No hay usuarios externos de Modo A.** Todos los consumidores de `GraphRunWorkflow` son internos; no hay migración que negociar |
| P9 | **El adaptador se queda; la regla se aplica como política de routing**, no borrando código → `RM-61` |

Queda esto, y nada bloquea la Fase 0.

| # | Pregunta | Decide | Bloquea |
|---|---|---|---|
| P5 | ¿La cancelación propaga los tres saltos hasta Prometheus? | Aeon + Axonium | RM-50 · RM-54 |
| P7 | Forma exacta de `StreamEvent` sobre el cable | Sesión conjunta | RM-01 · RM-10 |
| P8 | Gobierno del repo de contratos, ahora con **tres** consumidores | Sesión conjunta ampliada a Axonium | RM-47 · RM-49 |
**P10 · Cerrada por el dueño del proyecto, en firme.** No hay vía alternativa —la única puerta a Prometheus es el SDK Axonium, también en modo autónomo— **y un desarrollador no tiene acceso real a una instancia de Prometheus.**

Consecuencia, documentada como limitación conocida y no descubierta al clonar: en modo autónomo Synaptum **no tiene inferencia local**. Solo proveedores cloud de pago, o modelos simulados. Esto convierte el modelo falso en la vía principal de desarrollo, no en una utilidad de test → `RM-65` · `RM-66`.

### Hito de dependencia

El **primer run gobernado extremo a extremo de un agente Synaptum contra un modelo de Prometheus** está condicionado a `RM-50` (Axonium-Go con streaming y cancelación). No bloquea la Fase 0 ni la Fase 1 de Synaptum, que corren sobre Axonium-Python en modo autónomo — pero ninguna de las tres partes debe planificar una fecha de integración sin contar con él.

---

*Última actualización: cuarta iteración de la frontera del runtime, más la regla de puerta única a Prometheus vía Axonium. Documentos de referencia: «Dónde vive el runtime» · «Frontera del runtime» (Aeon) · «Acta de convergencia» · «Cierre de Fase 0» (Aeon).*
