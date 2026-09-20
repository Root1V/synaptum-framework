# Ejemplos

**Dieciséis ficheros que se ejecutan.** Corren sin inferencia y sin configurar nada: las
respuestas van guionizadas y todo lo demás es real — las herramientas se ejecutan, el
journal se escribe, el consumo se mide. Con dos variables de entorno, **el mismo fichero
sin tocar** habla con un modelo de verdad.

Cada uno está sobre un proyecto real, no sobre un dominio inventado, porque un ejemplo con
`foo` y `bar` enseña la sintaxis y esconde la decisión.

Cada página trae el fichero entero, **lo que imprime al correrlo** —capturado ejecutándolo,
no escrito a mano— y el enlace a GitHub para clonarlo.

```bash
uv run python examples/agentes/01_triaje.py
```

## Construir agentes

Cada uno añade **una** idea sobre el anterior.

| | Dominio |
|---|---|
| [01 · Lo mínimo: un agente, una herramienta](ejemplos-01-triaje.md) | Argus |
| [02 · Varias herramientas y una respuesta tipada](ejemplos-02-causa-raiz.md) | Argus |
| [03 · Un bucle acotado, y qué pasa cuando el modelo se equivoca](ejemplos-03-extraccion-acotada.md) | la plataforma de inteligencia documental |
| [04 · Un agente que mueve dinero, y el runtime que lo frena](ejemplos-04-agente-que-gasta.md) | Aerarium + Mercatus |
| [05 · Varios agentes en cadena, cada uno con su contexto](ejemplos-05-cadena-de-agentes.md) | el pipeline de doblaje (Prosodia) |
| [06 · Varios agentes a la vez, y uno que decide](ejemplos-06-agentes-en-paralelo.md) | Argus bajo tormenta |
| [07 · Un agente como herramienta de otro, y un supervisor que enruta](ejemplos-07-agente-como-herramienta.md) | la mesa de entrada de tu portafolio |
| [08 · Herramientas que no escribiste tú](ejemplos-08-herramientas-mcp.md) | cualquiera de tus repos |
| [09 · Delegar como primitiva, no como patrón](ejemplos-09-delegar.md) | Aerarium |
| [10 · Lo que el modelo ve en cada turno, y lo que cuesta](ejemplos-10-economia-del-contexto.md) | Argus |
| [11 · Un catálogo grande sin pagarlo en cada turno](ejemplos-11-catalogo-diferido.md) | Prometheus |
| [12 · Un agente que vive en otro contenedor](ejemplos-12-agente-remoto.md) | el pipeline de doblaje (Prosodia) |

## Qué garantiza el runtime

Estos no enseñan a escribir un agente: enseñan qué hay debajo cuando ya lo has
escrito.

| | Dominio |
|---|---|
| [13 · El bucle entero, en un fichero](ejemplos-13-bucle.md) | — |
| [14 · Matar el proceso a mitad y reanudar sin volver a pagar la inferencia](ejemplos-14-durabilidad.md) | — |
| [15 · Un paso destructivo se detiene, alguien decide, y el run sigue donde estaba](ejemplos-15-aprobacion.md) | — |
| [16 · Ver los tokens según llegan, y cortar a mitad](ejemplos-16-streaming.md) | — |

## Los sistemas de los que salen

No son dominios inventados: cada ejemplo está montado sobre un sistema que existe, y esa es
la mitad de lo que enseñan. Un ejemplo con `foo` y `bar` enseña la sintaxis; uno sobre una
conciliación de verdad enseña **dónde está la decisión**.

| | |
|---|---|
| [Plataforma de inteligencia documental](https://github.com/Root1V/agentic-doc-intelligence-platform) | extracción agéntica de documentos |
| [Prometheus](https://github.com/Root1V/prometheus-inference-platform) | plataforma de inferencia autoalojada |
| [Aerarium](https://github.com/Root1V/aerarium-agentic-banking) | núcleo bancario para comercio agéntico |
| [Mercatus](https://github.com/Root1V/mercatus-agentic-payments) | pagos entre agentes |
| [Prosodia](https://github.com/Root1V/ai-video-dubbing-pipeline) | doblaje de vídeo multihablante, todo local |
| [Axonium](https://github.com/Root1V/axonium-sdk) | SDK a la plataforma de inferencia |
| [Argus](https://github.com/Root1V/argus-observability-platform) | observabilidad y AIOps agéntico |
| [Aeon](https://github.com/Root1V/aeon-agent-harness) | el arnés: ejecución durable y política fuera del modelo |

## Lo que estos ejemplos no enseñan

`LocalGateway` **no aplica nada**. Corre dentro del proceso que gobernaría, así que sus
comprobaciones son advisorias: las cumple un bucle correcto y se las salta uno con un
fallo. Que el `04` deniegue un pago aquí no dice nada sobre si lo denegaría en producción
— para eso la decisión tiene que tomarse fuera del proceso, que es lo que hace un arnés.
