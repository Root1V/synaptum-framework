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

## La resolución: convertir la apuesta en una consulta

La primera versión de este documento proponía reenviar con el mismo `messageId` y confiar en que el
servidor deduplicara. **Aeon lo corrigió y tienen razón**: la especificación dice que `SendMessage`
*puede* detectar duplicados, y «puede» no es algo sobre lo que construir durabilidad.

Su observación es la que cambia el diseño: **`taskId` es del servidor y no lo tenemos, pero
`contextId` lo pone el cliente.** Así que lo que hay que pedirle a A2A no es «acepta mi id», es
**«lístame las tareas de este contexto»** — y eso ya existe:

```
ListTasks(contextId, status?, pageSize?, pageToken?)
```

### Un contexto por delegación

`contextId` = `{run_id}/{step_id}` — que es exactamente nuestro `sub_run_id`. No es un apaño: A2A
define `contextId` como la agrupación de una conversación, y una delegación **es** una conversación,
posiblemente de varios turnos si el remoto pide entrada o autenticación.

Un identificador, dos sistemas, y ninguno inventado para el otro.

### Reconciliar al reanudar

```
ListTasks(contextId = sub_run_id)

  vacío            → no llegó a crearse nada. Enviar.
  una tarea        → es la nuestra. GetTask / SubscribeToTask y continuar.
  terminal         → tomar su resultado. No se paga otra vez.
  varias           → anomalía. La no terminal más reciente, y si hay dos, UncertainEffect.
```

**Esto funciona contra un servidor que no deduplique**, porque ya no dependemos de que deduplique:
preguntamos.

### La escalera, y qué garantiza cada peldaño

No todo servidor implementará `ListTasks`. La degradación tiene que ser explícita, no silenciosa:

| Peldaño | Garantía |
|---|---|
| `ListTasks(contextId)` | **Exacta.** Sabemos si hay tarea y cuál |
| `messageId` determinista | **Depende del servidor.** «Puede» deduplicar |
| Ninguno de los dos | **Ninguna.** `UncertainEffect`, y decide quien gobierna |

El `messageId` determinista se manda igual: no cuesta nada y en un servidor que deduplique cierra el
hueco antes de llegar al tercer peldaño.

### El hueco que queda, y de quién es

Si morimos **entre** el `SendMessage` y que el servidor persista la tarea, `ListTasks` devuelve vacío
y reenviamos — correcto. Si persistió, `ListTasks` la encuentra — correcto.

Lo único que rompe esto es un servidor que **cree la tarea y no la persista antes de responder**, y
esa es precisamente la mala práctica que la búsqueda encuentra una y otra vez (`TaskNotFoundError`
sobre ids que el propio servidor acababa de emitir). No es un hueco de nuestro diseño: es un
requisito que le pedimos al otro lado, y conviene decirlo así.

## Riesgo: un agente remoto es destructivo mientras no demuestre lo contrario

Un `AgentCard` declara `skills`, no riesgo. No hay forma de saber qué puede hacer un agente ajeno.

Se aplica **la misma regla que con MCP**: sin declaración, `Risk.DESTRUCTIVE`. Ahí el autor no está
delante para declarar, y suponer en su lugar es suponer a favor. Con un agente remoto es más claro
todavía: no solo no sabemos qué hace, es que **puede cambiar sin avisarnos**.

Si A2A estandariza una extensión de riesgo, se lee. Mientras tanto, se puede declarar a mano al
construir el delegado remoto — explícito y de quien asume la consecuencia.

## Quién gobierna la llamada: una cuarta forma que no estaba en la lista

Propuse tres opciones —API advisoria, delegación como herramienta, o que el arnés haga la llamada— y
me incliné por la tercera. **Aeon fue a mirar qué hace la industria y la respuesta es una cuarta**:
un **proxy en el camino de datos**. El framework apunta su endpoint al gateway y **no cambia código**.

Es lo que hacen `agentgateway` (OSS, construido exactamente para esto) y el Agent Gateway de Google
Cloud. Y resuelve mi propia objeción a la opción 1 mejor que la 3:

> El forward proxy es el único sitio del stack que ve todo el tráfico saliente de un agente, y el
> único con el que el agente no puede discutir.

**Un control en el framework es una petición. Un proxy en el camino es una frontera.** No se salta
porque no hay otra ruta, no porque el bucle se porte bien. Es la misma distinción que me hizo
rechazar la opción 1, aplicada al transporte en vez de a la API.

Y es mucho más barato de lo que parecía: **un proxy no necesita un cliente A2A**. Entiende lo justo
del JSON-RPC para saber destino y método, aplica política y reenvía. **El ciclo de vida de la tarea
se queda de nuestro lado**, que es donde tiene que estar.

### El reparto

| | |
|---|---|
| **Nuestro** | La semántica: paso durable, reanudar sin repetir, consumo agregado, riesgo declarado, y **todo el ciclo de vida de la tarea** — polling, estados, artefactos |
| **Del arnés** | El gobierno: egress, credenciales, identidad por salto, política, coste, auditoría, y **límites de radio** — profundidad y fan-out |

Los límites de radio no los habíamos planteado. Tenemos `max_delegation_depth`, que para un ciclo
dentro de un proceso; el fan-out de un agente que delega en cincuenta a la vez no lo para nadie
todavía.

### Un matiz que casi se nos pasa a los tres

La delegación entre agentes suele ser **este-oeste** —contenedor a contenedor— y por eso **no cruza
el perímetro**. Un cortafuegos de salida no la ve. Es el modo de fallo documentado: el tráfico entre
agentes se escapa de los gateways porque nunca estuvo en su camino.

Por eso el patrón es un **endpoint explícito al que se apunta**, no confiar en la topología de red.
Dar por gobernada una llamada porque «sale por el proxy» es un error cuando esa llamada no cruza el
perímetro, y entre contenedores normalmente no lo cruza.

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

**Lo que resultó no ser nuestro**

- **El gobierno de la llamada.** Va en el proxy del arnés (`A2A-002`), y es la tercera instancia del
  mismo patrón que ya usan para la inferencia y para MCP saliente.
- **Un cliente A2A para el camino gobernado.** Solo hace falta en modo autónomo, donde no hay proxy.

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

**Preguntadas en el canal el 2026-09-20. Aeon respondió el mismo día y las dos primeras están
cerradas.**

1. ~~¿De quién es la llamada remota?~~ **Del arnés, como proxy en el camino de datos** (`A2A-002`,
   comprometido). Nosotros no cambiamos código de transporte: apuntamos a su endpoint.

2. ~~¿Cómo se reencuentra una tarea cuyo id pone el otro lado?~~ **`ListTasks(contextId)`**, con el
   `contextId` = nuestro `sub_run_id`. Y un dato que Aeon dio y conviene no perder: **Temporal no
   resuelve esto**. Su replay determinista cubre la decisión, no la Activity que hace el envío; el
   hueco queda igual de abierto con Temporal que sin él.

3. **¿Cliente, servidor, o los dos?** Sigue abierta. Para «cada agente en su contenedor» hacen falta
   los dos; el cliente solo ya sirve para consumir agentes ajenos.

4. **¿Almacén compartido entre contenedores?** Sigue abierta, y es la que decide si esto vale la
   pena: si cada contenedor lleva su diario, una delegación remota interrumpida se repite entera y
   quedamos en un cliente A2A más.
