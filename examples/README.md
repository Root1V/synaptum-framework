# Ejemplos

Dos pistas. **[`agentes/`](agentes/)** enseña a construir; **[`propiedades/`](propiedades/)**
enseña qué garantiza el runtime debajo. Si vienes a escribir un agente, empieza por la primera.

Todos corren **sin inferencia y sin configurar nada**: las respuestas van guionizadas y el resto es
real — las herramientas se ejecutan, el journal se escribe, el consumo se mide.

```bash
uv run python examples/agentes/01_triaje.py
```

## Construir agentes

Cada uno añade **una** idea sobre el anterior, y todos están sobre un proyecto real de este
portafolio, no sobre un dominio inventado.

| | Qué añade | Dominio |
|---|---|---|
| [`01_triaje.py`](agentes/01_triaje.py) | Un agente, una herramienta. Lo mínimo que funciona | Argus · triaje de una señal |
| [`02_causa_raiz.py`](agentes/02_causa_raiz.py) | Varias herramientas y **salida tipada** | Argus · AIOps |
| [`03_extraccion_acotada.py`](agentes/03_extraccion_acotada.py) | `Limits` y **el camino de error**: qué pasa cuando el modelo se equivoca | Plataforma documental |
| [`04_agente_que_gasta.py`](agentes/04_agente_que_gasta.py) | `risk`, denegación y **aprobación humana** con el run suspendido en disco | Aerarium · tesorería |
| [`05_cadena_de_agentes.py`](agentes/05_cadena_de_agentes.py) | Dos agentes en cadena, cada uno con su contexto | Pipeline de doblaje |
| [`06_agentes_en_paralelo.py`](agentes/06_agentes_en_paralelo.py) | Fan-out con `asyncio.gather` y un sintetizador | Argus bajo tormenta |
| [`07_agente_como_herramienta.py`](agentes/07_agente_como_herramienta.py) | Un supervisor que enruta a especialistas | Mesa de entrada del portafolio |

### Sobre multi-agente, dicho antes de que lo descubras

**`delegate()` no existe todavía** — es `SYN-41`. El tipo `DelegateStep` está escrito y nada lo
emite.

Eso **no impide** construir sistemas multi-agente hoy, y los tres últimos ejemplos lo demuestran: el
bucle de un agente es un generador asíncrono de verdad, así que orquestar varios es código asyncio
normal. Lo que `SYN-41` añadirá es que la delegación quede **en el journal** como un paso propio,
con su consumo atribuido y su punto de reanudación.

Las consecuencias de que no esté, hoy, son tres, y están explicadas al final de
[`07`](agentes/07_agente_como_herramienta.py): el coste de un subagente no aparece en el `usage` de
quien delega, un subagente no es un paso durable, y el riesgo no se propaga solo.

La regla que sí se puede seguir desde el primer día:

> **Entre agentes viaja el resultado, nunca el historial.**

Duplicar el contexto de un agente en otro se paga dos veces y hace que el segundo herede los errores
del primero sin poder distinguirlos de sus datos.

## Qué garantiza el runtime

| | |
|---|---|
| [`01_bucle.py`](propiedades/01_bucle.py) | El bucle entero como stream de eventos |
| [`02_durabilidad.py`](propiedades/02_durabilidad.py) | Matar el proceso y reanudar **sin volver a pagar la inferencia** — medido, no afirmado |
| [`03_aprobacion.py`](propiedades/03_aprobacion.py) | Un run suspendido que sobrevive al proceso |
| [`04_streaming.py`](propiedades/04_streaming.py) | Tokens según llegan, y cortar a mitad |

## Contra un modelo de verdad

**El mismo fichero, sin tocar una línea.** Hay dos puertas y el agente no nota la diferencia, que es
justamente la propiedad.

**Por un SDK de plataforma**, cuando la inferencia está gobernada y ese SDK es la única puerta
legítima — credenciales, catálogo, cuotas y facturación viven ahí. Tiene prioridad: ir al endpoint
por detrás se salta todo eso.

```bash
export AXONIUM_CLIENT_ID=…  AXONIUM_CLIENT_SECRET=…
export SYNAPTUM_MODEL=gpt-oss-20b-mxfp4
uv run python examples/agentes/02_causa_raiz.py
```

**Por HTTP directo**, contra cualquier endpoint OpenAI-compatible — vLLM, Ollama, LM Studio,
llama.cpp o la API de OpenAI:

```bash
export SYNAPTUM_BASE_URL=http://localhost:8080/v1
export SYNAPTUM_MODEL=qwen3-8b
uv run python examples/agentes/02_causa_raiz.py
```

Sin ninguna de las dos, el doble responde el guion. **No es un mock**: ejecuta las herramientas de
verdad, hace streaming de verdad y produce `Usage` de verdad. Lo único que no hace es inferir.

## Lo que estos ejemplos no enseñan

`LocalGateway` **no aplica nada**. Corre dentro del proceso que gobernaría, así que sus
comprobaciones son advisorias: las cumple un bucle correcto y se las salta uno con un fallo. Que el
`04` deniegue un pago aquí no dice nada sobre si lo denegaría en producción — para eso la decisión
tiene que tomarse fuera del proceso, que es lo que hace un arnés.
