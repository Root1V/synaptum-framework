# Ejemplos

Dos pistas. **[`agentes/`](agentes/)** enseña a construir; **[`propiedades/`](propiedades/)**
enseña qué garantiza el runtime debajo. Si vienes a escribir un agente, empieza por la primera.

Todos corren **sin inferencia y sin configurar nada**: las respuestas van guionizadas y el resto es
real — las herramientas se ejecutan, el journal se escribe, el consumo se mide.

```bash
uv run python examples/agentes/01_triaje.py
```

También están **[en la documentación](https://root1v.github.io/synaptum-framework/ejemplos.html)**,
una página por ejemplo. Esas páginas se **generan de estos ficheros**: la fuente sigue siendo lo que
se ejecuta, y un test falla si dejan de corresponder.

## Construir agentes

Cada uno añade **una** idea sobre el anterior, y todos están sobre un sistema real —observabilidad,
banca agéntica, inferencia autoalojada, doblaje de vídeo— y no sobre un dominio inventado. Un
ejemplo con `foo` y `bar` enseña la sintaxis y esconde la decisión.

| | Qué añade | Dominio |
|---|---|---|
| [`01_triaje.py`](agentes/01_triaje.py) | Un agente, una herramienta. Lo mínimo que funciona | Argus · triaje de una señal |
| [`02_causa_raiz.py`](agentes/02_causa_raiz.py) | Varias herramientas y **salida tipada** | Argus · AIOps |
| [`03_extraccion_acotada.py`](agentes/03_extraccion_acotada.py) | `Limits` y **el camino de error**: qué pasa cuando el modelo se equivoca | Plataforma documental |
| [`04_agente_que_gasta.py`](agentes/04_agente_que_gasta.py) | `risk`, denegación y **aprobación humana** con el run suspendido en disco | Aerarium · tesorería |
| [`05_cadena_de_agentes.py`](agentes/05_cadena_de_agentes.py) | Dos agentes en cadena, cada uno con su contexto | [Prosodia](https://github.com/Root1V/ai-video-dubbing-pipeline) · doblaje |
| [`06_agentes_en_paralelo.py`](agentes/06_agentes_en_paralelo.py) | Fan-out con `asyncio.gather` y un sintetizador | Argus bajo tormenta |
| [`07_agente_como_herramienta.py`](agentes/07_agente_como_herramienta.py) | Un supervisor que enruta a especialistas | Mesa de entrada de varios sistemas |
| [`08_herramientas_mcp.py`](agentes/08_herramientas_mcp.py) | Herramientas de un **servidor MCP**, y qué cambia cuando no las escribiste tú | Cualquier proyecto de GitHub · servidor incluido |
| [`09_delegar.py`](agentes/09_delegar.py) | `delegates=`: delegar como **primitiva** — coste que sube, riesgo que se deriva, paso durable | Aerarium · cierre de mes |
| [`10_economia_del_contexto.py`](agentes/10_economia_del_contexto.py) | `max_tool_chars`, prefijo estable y `economy()`: qué ve el modelo y qué cuesta | Argus · depurar con logs |
| [`11_catalogo_diferido.py`](agentes/11_catalogo_diferido.py) | `deferred()`: veinte herramientas sin inflar el prefijo | Prometheus · consola del operador |
| [`12_agente_remoto.py`](agentes/12_agente_remoto.py) | `RemoteDelegate`: un agente en **otro contenedor**, por A2A | [Prosodia](https://github.com/Root1V/ai-video-dubbing-pipeline) · revisor ajeno |

### Sobre MCP

El [`08`](agentes/08_herramientas_mcp.py) trae su propio servidor MCP, así que corre sin instalar
nada más allá del extra:

```bash
uv sync --extra mcp
uv run python examples/agentes/08_herramientas_mcp.py
```

Conectarse son cuatro líneas. Lo que el ejemplo enseña de verdad es **qué cambia cuando la
herramienta es ajena**: el riesgo lo insinúa quien no manda, los errores llegan sin explicación, los
nombres chocan entre servidores, y el esquema puede cambiar debajo sin que tu repositorio se entere.

### Sobre multi-agente, dicho antes de que lo descubras

Hay **tres formas** de componer agentes y las tres son correctas en su sitio:

1. **A mano, con asyncio** ([`05`](agentes/05_cadena_de_agentes.py),
   [`06`](agentes/06_agentes_en_paralelo.py)). El bucle de un agente es un generador asíncrono de
   verdad, así que encadenarlos o lanzarlos en paralelo es código normal. Es lo que quieres cuando
   **el orden lo decides tú**.
2. **Envolviendo un agente en un `@tool`** ([`07`](agentes/07_agente_como_herramienta.py)). Sirve
   cuando el modelo elige, y sigue siendo la respuesta cuando lo de dentro **no es un `Agent`** —
   una API ajena, un servicio heredado.
3. **Delegando** ([`09`](agentes/09_delegar.py)). `Agent(delegates=[…])`, y entre agentes es lo que
   quieres: el consumo del subagente sube al total, la delegación es un paso durable con su propio
   diario, y el riesgo **se deriva** — delegar en alguien que borra es destructivo sin que nadie lo
   declare. Ese tercero no se puede conseguir a mano.

Y si el agente está **en otro contenedor**, la forma es la misma:
[`12`](agentes/12_agente_remoto.py) conecta uno por A2A y lo mete en la misma lista de `delegates`.
El bucle no distingue.

La regla vale para todas, incluida la remota:

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

## Los datos son inventados, y tienen que parecerlo

Todo lo que aparece en estos ficheros —nombres, RUC, cuentas, importes, identificadores— está
compuesto. La regla, que costó aprender: **no basta con que el dato sea falso; tiene que ser
inequívocamente falso.**

El `03` llevaba un nombre peruano perfectamente verosímil, con dos de los apellidos más comunes del
país, al lado de un sueldo y una AFP. Nadie lo copió de ningún sitio, y daba igual: es un nombre que
casi con seguridad lleva alguien, publicado en un repositorio, en un sitio web y dentro de los
artefactos de PyPI. Que el dato sea inventado no ayuda a quien se llame así.

Así que: **los marcadores de siempre** —John Doe, ACME— en vez de inventar uno nuevo. Un nombre
que el lector ya reconoce como marcador se lee como «aquí va un nombre» sin pensarlo, y nadie se
pregunta si detrás hay alguien. Además, identificadores que fallen su propio dígito verificador —el
RUC del `03` lo hace a propósito— y cuentas truncadas. No hay un test que lo compruebe, y sería
peor tenerlo mal: una lista negra de apellidos daría verde con el siguiente nombre que a alguien se
le ocurra. Esto se mira al escribir el ejemplo.

## Lo que estos ejemplos no enseñan

`LocalGateway` **no aplica nada**. Corre dentro del proceso que gobernaría, así que sus
comprobaciones son advisorias: las cumple un bucle correcto y se las salta uno con un fallo. Que el
`04` deniegue un pago aquí no dice nada sobre si lo denegaría en producción — para eso la decisión
tiene que tomarse fuera del proceso, que es lo que hace un arnés.
