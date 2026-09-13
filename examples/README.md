# Ejemplos

Todos corren **sin instalar nada** y **sin inferencia**: por defecto usan respuestas grabadas o
guionizadas, que en modo autónomo es la vía principal de desarrollo y no una utilidad de test.

```bash
uv run python examples/01_agente.py
```

Para apuntar a un modelo de verdad, dos variables y **el mismo fichero sin tocar**:

```bash
export SYNAPTUM_BASE_URL=http://localhost:8080/v1     # cualquier endpoint OpenAI-compatible
export SYNAPTUM_MODEL=qwen3-8b
export SYNAPTUM_API_KEY=...                           # opcional: hay despliegues que no la piden
uv run python examples/01_agente.py
```

Con llama.cpp, que es lo más corto si tienes un `.gguf` a mano —`--jinja` hace falta para que el
modelo pueda llamar herramientas—:

```bash
llama-server -m modelo.gguf --port 8899 --jinja -c 8192 -ngl 99 --alias mi-modelo
```

Los cuatro se han ejecutado así contra `Qwen3-8B-Q6_K`. Vale cualquier endpoint OpenAI-compatible:
vLLM, Ollama, LM Studio, un gateway, o la API de OpenAI.

Que el mismo código sirva para las dos cosas no es comodidad: es la propiedad. El agente no sabe
quién hay al otro lado de la costura, así que cambiar de doble a modelo real no le toca una línea.

| Fichero | Qué enseña |
|---|---|
| [`01_agente.py`](01_agente.py) | El bucle entero: herramientas, streaming, consumo, y el gateway como única puerta |
| [`02_durabilidad.py`](02_durabilidad.py) | Matar el proceso a mitad y reanudar **sin volver a pagar la inferencia** |
| [`03_aprobacion.py`](03_aprobacion.py) | Un paso destructivo que se detiene, alguien aprueba, y el run continúa donde estaba |
| [`04_streaming.py`](04_streaming.py) | Los tokens según llegan, y cortar a mitad — cerrar el iterador *es* la señal |

## Lo que estos ejemplos no enseñan

`LocalGateway` **no aplica nada**. Corre dentro del proceso que gobernaría, así que sus
comprobaciones son advisorias: las cumple un bucle correcto y se las salta uno con un fallo. Que un
run pase por aquí sin denegaciones no dice nada sobre si pasaría por el gateway real.

Lo mismo con `HttpModel`: es transporte **de desarrollo**. En el camino gobernado la llamada al
modelo no ocurre en este proceso — sale por la costura hacia quien tiene las credenciales y puede
denegar.
