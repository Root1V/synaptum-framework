# Modelos

## Nombrar un modelo

```python
Agent("a", model="openai-compatible:qwen3-8b")
```

El prefijo dice **quién normaliza la respuesta**, no solo qué modelo se quiere. Es nuestro y no
viaja por el cable: al proveedor le llega `qwen3-8b` a secas.

## Las tres puertas

El agente **no sabe cuál tiene**, y eso es la propiedad, no comodidad.

### Sin inferencia — el doble

Por defecto. Ver [Probar](06-probar.md).

### Transporte HTTP directo

```python
from synaptum import HttpModel, LocalGateway

modelo = HttpModel("http://localhost:8080/v1", api_key=None)
gateway = LocalGateway(model=modelo, stream=modelo.stream, tools=[...])
```

Solo stdlib — `urllib` en un hilo, porque el núcleo no tiene dependencias y no las va a tener por
esto. Vale para cualquier endpoint OpenAI-compatible: vLLM, Ollama, LM Studio, llama.cpp o la API de
OpenAI.

> **Es transporte de desarrollo.** En el camino gobernado la llamada al modelo **no ocurre en este
> proceso**: sale por la costura hacia quien tiene las credenciales y puede denegar. Un transporte
> dentro del proceso tiene la clave en su memoria y llama a quien le digan.

### Un SDK de plataforma

Cuando la inferencia está gobernada y un SDK es la única puerta legítima —credenciales, catálogo,
cuotas y facturación viven ahí— tiene prioridad. Ir al endpoint por detrás se salta todo eso.

```python
from synaptum.providers.axonium import AxoniumModel   # pip install synaptum[axonium]

puente = AxoniumModel()
gateway = LocalGateway(model=puente.complete, stream=puente.stream, tools=[...])
```

Esto **no es un `Provider`**: un `Provider` normaliza y no transporta, y un SDK así hace las dos
cosas. Lo que falta es traducir entre dos vocabularios que ya están normalizados.

## Escribir un adaptador

`Provider` es un `Protocol` de tres funciones **puras**: `to_wire`, `from_wire`, `stream_from_wire`.
No abren conexiones ni leen credenciales.

Que la normalización sea separable del transporte es lo que permite ejercitarla **contra un fichero**
—sin servidor, sin red, sin gastar— y es la razón de que el corpus dorado exista.

Los adaptadores se descubren por *entry points*, así que el núcleo no conoce a ninguno.

## `Usage` tiene tres estados

```python
Usage(input=100, output=20, cache_read=80, estimated=True)
#     reasoning=None, cache_write=None   ← nadie los midió
```

| | |
|---|---|
| `None` | **Nadie lo midió.** |
| `0` | Se midió y fue cero. |
| `estimated=True` | Derivado, no reportado. |

Confundir `None` con `0` es lo que hace creer que escribir en caché es gratis, y el bucle decide mal
en cada compactación. Lo desconocido se propaga al sumar: si un tramo no midió un contador, el total
de ese contador es `None` — sumar solo lo conocido daría una cota inferior con aspecto de cifra
exacta.

**`input` es inclusivo**: contiene los tokens servidos desde caché, y `cache_read` dice cuántos.
Confirmado contra grabaciones reales, donde `prompt_tokens == prompt_n + cache_n`.

## Streaming

```python
async for evento in agente.stream(tarea, session=sesion):
    if evento.kind == "text_delta":
        print(evento.text, end="", flush=True)
```

Es el mismo bucle y el mismo journal que `run()`; lo único que cambia es que los fragmentos del
modelo se ceden intercalados entre la intención del paso y su resultado.

Va aparte de `run()` porque **cambia el tipo de lo que se cede**: quien consume `run()` hace `match`
sobre pasos sin una rama para lo que nunca va a llegar.

- **Un paso reanudado no vuelve a emitir fragmentos.** Ya se pagó; reproducir sus tokens como si
  estuvieran ocurriendo sería teatro.
- **Un reintento vuelve a abrir el ciclo.** Los fragmentos ya entregados no se retiran: se generaron
  y se pagaron.

### Cancelar es dejar de iterar

No hay evento de cancelación, y no lo habrá: **un canal que se está cerrando no es sitio para mandar
el aviso de que se cierra.** Cerrar el iterador cierra el cuerpo de la respuesta, y eso es lo que
para la generación arriba.

```python
flujo = agente.stream(tarea, session=sesion)
async for evento in flujo:
    if suficiente(evento):
        break
await flujo.aclose()          # esto es la señal
```

## Reintentos

Un error trae en su **tipo** si es reintentable — la decisión no es del bucle, la sabe quien habló
con el proveedor. Cualquier 4xx salvo 429 no lo es; 429, 5xx, timeouts y fallos de red sí.

El bucle espera entre intentos, con espera creciente y *jitter*, y respeta el `Retry-After` del otro
extremo con un techo. Reintentar al instante no es reintentar: es repetir contra el mismo estado
roto. Y sin jitter, N agentes que caen por la misma razón reintentan a la vez y reconstruyen el pico
que los tiró.

```python
Limits(max_retries=2, retry_base=0.5, max_retry_wait=30.0)
```

## Salida estructurada

```python
@dataclass
class Diagnostico:
    causa: str
    confianza: float

Agent("a", model=..., output=Diagnostico)
```

Una `dataclass` de stdlib basta; Pydantic es un extra para quien ya lo use.

El esquema viaja por dos caminos a la vez: en `response_format` **y** en las instrucciones. No todo
gateway admite el campo —algunos lo descartan avisando por warning— y entonces la restricción no
viaja, el modelo contesta en prosa y el error culpa al JSON. Decirlo también en el prompt cuesta
unos cientos de tokens y funciona con cualquier proveedor.

La validación ocurre **dentro del reintento**: un objeto mal formado no es un fallo del run, es una
muestra mala, y el muestreo es estocástico.
