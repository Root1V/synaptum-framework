# SYN-44 · Delegar a un agente remoto

**Estado: diseño, sin implementar.** Este documento existe para decidir antes de escribir, porque la
decisión difícil no es el transporte: es qué pasa con la identidad de paso cuando el que la genera
no somos nosotros.

## Qué dice el estado del arte, a septiembre de 2026

| | |
|---|---|
| **A2A** | v1.0 en abril de 2026, v1.0.1 en mayo. Hospedado por la Linux Foundation, +150 organizaciones, uso en producción |
| **MCP** | 97 millones de descargas mensuales de SDK en marzo de 2026 |
| **El consenso** | **MCP + A2A es la pila de facto.** No hay un ganador único: A2A enruta una tarea al especialista, MCP le da a ese especialista sus herramientas |
| **ACP** (IBM) y **AGNTCY/OASF** (Cisco) | Existen y resuelven cosas distintas — negociación y descripción de agentes. No compiten por el mismo hueco |

Ya hablamos MCP (`SYN-39`). **A2A es exactamente la mitad que nos falta**, y elegir otra cosa sería
inventar un protocolo cuando hay uno con 150 organizaciones detrás.

### Lo que A2A define y vamos a usar

- **AgentCard** en `/.well-known/agent-card.json`: identidad, `capabilities`, `skills`,
  `securitySchemes`, `interfaces`, `extensions`.
- **Task con ocho estados**: `submitted`, `working`, `input_required`, `auth_required`,
  `completed`, `failed`, `canceled`, `rejected`. Los terminales no aceptan más mensajes; los
  interrumpidos sí.
- **Métodos**: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`,
  `SubscribeToTask`, y la familia de `PushNotificationConfig`.
- **Tres bindings**: JSON-RPC 2.0, gRPC y HTTP+JSON. Streaming por SSE en el binding HTTP.

---

## El problema que decide el diseño

**A2A genera el `taskId` en el servidor. El cliente no puede aportarlo.**

Nuestra durabilidad se apoya exactamente en lo contrario: la identidad de paso es **determinista y
derivada por el cliente** —`000003-model`, `{run_id}/{step_id}`— y por eso reanudar no vuelve a
pagar. Si el identificador de la unidad de trabajo lo inventa el otro extremo, al reanudar **no
sabemos a qué preguntar.**

No es un detalle de implementación: es el punto donde nuestro modelo y el protocolo se tocan mal.

### Cómo lo resuelve la industria, y qué le falta

La búsqueda devuelve el mismo problema una y otra vez, con la misma conclusión: **persistir el
`taskId` antes de seguir**. Hay implementaciones que devuelven una tarea sin guardarla y luego
fallan con `TaskNotFoundError` sobre ids que ellas mismas acababan de emitir. Y hay stores durables
con arrendamiento —si el trabajador muere, la tarea se vuelve a arrendar— que es la misma idea que
nuestro journal, un nivel más arriba.

Lo que ninguna resuelve del todo es **la ventana entre enviar y registrar**: si el proceso muere
después del `SendMessage` y antes de escribir el `taskId`, nadie sabe si hay una tarea corriendo al
otro lado.

## La resolución que propongo

Tres identificadores, cada uno del lado que puede generarlo bien:

| A2A | Lo pone | Qué le damos |
|---|---|---|
| `contextId` | **El cliente puede** | Nuestro `run_id`. Agrupa todas las delegaciones de un run |
| `messageId` | **El cliente** | Derivado de `(run_id, step_id)`, determinista |
| `taskId` | **El servidor** | Lo recibimos y lo guardamos en el paso |

Y el ciclo queda así:

```
1. DelegateStep ATTEMPTED  →  journal  (DURABLE, antes de nada)
2. SendMessage(contextId=run_id, messageId=det(run_id, step_id))
3. taskId ← respuesta
4. esperar / SubscribeToTask
5. DelegateStep COMPLETED con result + usage + taskId  →  journal
```

**Al reanudar**, tres casos y ninguno inventado:

- **Hay COMPLETED** → se salta. Como cualquier paso.
- **No hay ni ATTEMPTED** → no se envió nada. Se envía.
- **Hay ATTEMPTED sin COMPLETED** → *no se sabe* si la tarea existe al otro lado. Se reenvía **con
  el mismo `messageId`**: la especificación dice que `SendMessage` *puede* detectar duplicados por
  `messageId`. Si el servidor lo hace, devuelve la misma tarea y no se paga dos veces.

### Y lo que hay que decir en voz alta

**«Puede detectar duplicados» no es una garantía.** Contra un servidor que no deduplique, un
reintento es una segunda tarea — y una tarea puede haber movido dinero.

Delegar **no es idempotente** y ya está declarado así, así que este caso cae en la máquina que ya
existe: `UncertainEffect`. El bucle no adivina; levanta y deja que decida quien gobierna. Es
incómodo y es correcto.

Lo que **no** vamos a hacer es meter un contador de intento en el `messageId` para esquivarlo: lo
haría no determinista, y entonces reanudar siempre generaría trabajo nuevo — justo lo que todo esto
existe para evitar.

---

## Riesgo: un agente remoto es destructivo mientras no demuestre lo contrario

Un `AgentCard` declara `skills`, no riesgo. No hay forma de saber qué puede hacer un agente ajeno.

Se aplica **la misma regla que con MCP**: sin declaración, `Risk.DESTRUCTIVE`. Ahí el autor no está
delante para declarar, y suponer en su lugar es suponer a favor. Con un agente remoto es más claro
todavía: no solo no sabemos qué hace, es que **puede cambiar sin avisarnos**.

Si A2A estandariza una extensión de riesgo, se lee. Mientras tanto, se puede declarar a mano al
construir el delegado remoto — explícito y de quien asume la consecuencia.

## La delegación remota **tiene** que cruzar la costura

Con un subagente en proceso, que la delegación no cruce el gateway es una limitación conocida: sus
herramientas sí cruzan, así que el efecto destructivo se detiene igual.

**Con un agente remoto eso deja de ser cierto.** Sus herramientas cruzan *su* gateway, no el
nuestro, y nosotros no sabemos cuál es ni qué política aplica. El efecto sale de nuestro perímetro
en el momento del `SendMessage`.

Por tanto: **`SYN-44` depende de que la costura admita autorizar una delegación sin ejecutarla.** Es
un cambio de contrato y va al canal de coordinación antes que el código.

## Lo que hay que construir, y lo que no

**Cliente** (`synaptum[a2a]`, extra opcional)

- `RemoteDelegate` — mismo contrato que `Delegate`: `name`, `risk`, `definition`. Despacha por red.
- Lectura de `AgentCard` para la descripción y las capacidades.
- Binding **HTTP+JSON primero**. gRPC después, si alguien lo pide.
- Mapeo de los ocho estados a lo nuestro: `completed` → resultado; `failed`/`rejected` → error de
  herramienta que vuelve al modelo; `canceled` → cancelación; `input_required`/`auth_required` →
  **`ApprovalStep`**, que es exactamente el mecanismo que ya tenemos para un run que espera a una
  persona.
- `CancelTask` al cerrar el iterador. Cerrar **es** la señal, también por red.

**Servidor** (`synaptum[a2a]`, el mismo extra)

- Exponer un `Agent` como servidor A2A: `AgentCard`, `SendMessage`, `GetTask`, `CancelTask`.
- El `taskId` se **persiste antes de devolverlo**. Es el fallo que la búsqueda encuentra una y otra
  vez, y evitarlo nos sale gratis: ya tenemos un journal.
- El `contextId` entrante se usa como `run_id`, así que un run distribuido comparte diario si los
  contenedores comparten almacén.

**Lo que no se construye aquí**

- Un registro de agentes. Descubrir quién existe es infraestructura, no framework.
- Autenticación propia: se leen los `securitySchemes` del `AgentCard` y se usa lo que diga.
- El almacén compartido. Es un `Checkpointer`, ya es un protocolo, y contra Postgres lo implementa
  quien opere.

---

## Por qué esto vale la pena, dicho sin adornos

Un montaje A2A normal trata una llamada remota como **opaca**: si se corta, se repite entera.

Aquí no tiene por qué. La identidad es determinista, el diario es un protocolo y el `contextId`
puede ser nuestro `run_id`. **Con almacén compartido, una delegación remota interrumpida se reanuda
sin volver a pagarla.** Eso no lo da el protocolo — lo da tenerlo encima de un runtime durable.

Esa es la única razón por la que merece la pena implementarlo nosotros en vez de recomendar un SDK.

## Preguntas abiertas antes de escribir código

**Planteadas en el canal de coordinación el 2026-09-20. Sin empezar hasta tener respuesta**, porque
de la primera depende que la mitad de esto sea nuestro o no.

1. **¿De quién es la llamada remota?** Tres opciones sobre la mesa, y me inclino por la tercera
   aunque vaya contra mi instinto:
   - la costura gana un método que **autoriza sin ejecutar**;
   - la delegación viaja como una llamada a herramienta más;
   - **el arnés hace la llamada A2A**, porque una delegación remota es *egress* — lleva credencial,
     sale de la red y tiene coste, que es exactamente lo que la costura de aplicación gobierna. Si
     es esta, el cliente A2A solo nos hace falta en modo autónomo, y lo nuestro sigue siendo la
     semántica: paso durable, reanudación sin repetir, consumo agregado, riesgo declarado.

2. **El `taskId` lo asigna el otro lado.** Preguntado a quien lleva un año con Temporal, que resuelve
   esta misma clase de problema. Si hay un patrón que funcione, se copia antes que inventar.

3. **¿Cliente, servidor, o los dos?** Para «cada agente en su contenedor» hacen falta los dos; el
   cliente solo ya sirve para consumir agentes ajenos.

4. **¿Almacén compartido entre contenedores?** Si cada uno lleva el suyo, se pierde la propiedad que
   justifica todo esto y quedamos en un cliente A2A más.
