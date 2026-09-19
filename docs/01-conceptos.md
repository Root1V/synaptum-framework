# Conceptos

Seis piezas. No hay más, y ninguna es opcional de entender.

## `Agent` — la configuración, no el comportamiento

```python
agente = Agent(
    "explorador",                       # nombre, para trazas y registros
    model="openai-compatible:qwen3-8b", # proveedor:modelo
    instructions="Sé breve.",           # el prompt de sistema
    tools=[contar_lineas],
    output=Diagnostico,                 # opcional: salida tipada
    limits=Limits(max_steps=8),         # opcional: topes
)
```

Un agente es **composición, no herencia**: no se hereda de él, no hay métodos que sobrescribir. Es
su configuración más el bucle.

`model` lleva prefijo porque dice **quién normaliza la respuesta**, no solo qué modelo se quiere. El
prefijo es nuestro y no viaja por el cable.

**Qué no hace:** no abre conexiones, no guarda estado entre llamadas y no sabe quién ejecuta. Puedes
reutilizar el mismo `Agent` para mil runs a la vez.

## `Session` — dónde ocurre y dónde se recuerda

```python
Session(run_id, gateway, checkpointer)
```

| | |
|---|---|
| `run_id` | La identidad del run. Llamar otra vez con el mismo id **reanuda** en vez de empezar. |
| `gateway` | Por dónde salen los efectos: el modelo y las herramientas. |
| `checkpointer` | Dónde queda el rastro. Por defecto, memoria. |

Reanudar es construir una `Session` con el mismo `run_id` y el mismo `checkpointer`. No hay nada
más — ni un método `resume()`, ni un flag.

**Qué no hace:** no guarda la conversación. La ventana de contexto se **vuelve a derivar** de los
mismos resultados en el mismo orden. Guardarla sería guardar dos veces lo mismo y arriesgarse a que
discrepen.

## El gateway — la única puerta de salida

Todo efecto externo sale por aquí, y esa es la razón de que exista: **si hay una sola puerta, alguien
puede estar en ella.**

```python
from synaptum import LocalGateway

gateway = LocalGateway(model=mi_modelo, tools=[...], policy=mi_politica)
```

`LocalGateway` es la implementación de referencia para modo autónomo. En producción, el gateway es
un proceso aparte que tiene las credenciales y puede denegar.

> **`LocalGateway` no aplica nada.** Corre dentro del proceso que gobernaría, así que sus
> comprobaciones son advisorias: las cumple un bucle correcto y se las salta uno con un fallo. Marca
> cada decisión con `enforced=False` para que nadie confunda su informe con una auditoría.

`Gateway` es un `typing.Protocol` de cuatro métodos. Quien lo implemente no hereda ni importa nada
nuestro, y puede estar escrito en otro lenguaje al otro lado de un socket.

## El checkpointer — el diario del run

```python
from synaptum import MemoryCheckpointer, SqliteCheckpointer

MemoryCheckpointer()              # por defecto. El run muere con el proceso.
SqliteCheckpointer("runs.db")     # sobrevive. stdlib: sqlite3, json, pathlib.
```

**No es una caché.** Una caché se puede tirar sin cambiar la corrección. Si tiras el journal pierdes
la capacidad de distinguir *«no se ejecutó»* de *«no se sabe si se ejecutó»*, y esa distinción es lo
único que impide cobrar dos veces una transferencia tras una caída.

**No ata a nada externo.** Las dos implementaciones son stdlib pura, y `Checkpointer` es un
`Protocol` de dos métodos: implementa `append` y `load` contra Postgres, Redis o lo que quieras.

Es el mismo reparto que hace la industria —LangGraph tiene sus *checkpoint savers*, Temporal guarda
un historial de eventos y reconstruye reproduciéndolo—. Lo que cambia aquí es que **el sustrato no
es nuestro a propósito**: el framework es dueño de la semántica, no del almacenamiento.

## Los pasos — qué cede el bucle

```python
async for paso in agente.run(tarea, session=sesion):
    match paso:
        case ToolStep(phase=Phase.ATTEMPTED, call=llamada):
            ...
        case ModelStep(phase=Phase.COMPLETED, usage=consumo):
            ...
        case FinalStep(output=salida):
            ...
```

| Paso | Cuándo | Lleva |
|---|---|---|
| `ModelStep` | Una llamada al modelo | `request` / `response` / `usage` |
| `ToolStep` | La ejecución de una herramienta | `call` / `result` / `risk` / `idempotent` |
| `ApprovalStep` | El run se detiene a esperar a una persona | `subject` / `decision` |
| `DelegateStep` | Delegación a un subagente *(tipo definido, aún sin emisor)* | `agent` / `brief` / `result` |
| `FinalStep` | El run terminó | `output` / `usage` acumulado |

## Las fases — un paso tiene dos momentos

`Phase.ATTEMPTED` y `Phase.COMPLETED`. **Todos los pasos tienen las dos.** Un paso no es un instante:
se intenta, y luego se sabe cómo fue.

```
000000-model   model   attempted   durabilidad=deferrable
000000-model   model   completed   durabilidad=durable
000001-tool    tool    attempted   durabilidad=durable      ← la diferencia
000001-tool    tool    completed   durabilidad=durable
```

Lo que cambia entre tipos no son las fases: es **la durabilidad de la intención**.

- La intención de un modelo es **diferible**. Una inferencia cuesta dinero, pero repetirla no rompe
  nada, así que se puede escribir en lote.
- La intención de una herramienta no idempotente es **durable**: tiene que estar en disco *antes* de
  que el efecto ocurra. Si no, tras una caída no hay forma de saber si la transferencia salió. Es la
  única espera bloqueante que el bucle impone en todo un turno.

Esto se llama *escritura anticipada*, y es la razón de que el journal exista. Ver
[Durabilidad](03-durabilidad.md).

## El identificador de paso

`000003-model`. Ordinal con relleno más tipo, derivado de **la posición del paso en el run** — nunca
de un UUID ni de un reloj.

Que sea determinista es lo que permite reanudar: el paso 3 de un run es siempre el paso 3, así que
el replay puede saltárselo sin coordinar nada con nadie. Está publicado como especificación abierta,
y cualquiera puede implementarla sin usar Synaptum ni hablar Python.
