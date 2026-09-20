# Mi agente

Punto de partida para un agente con Synaptum. Copia esta carpeta, renómbrala y empieza.

```bash
uv sync
uv run python -m mi_agente          # corre sin inferencia, con respuestas guionizadas
uv run pytest                       # los tests tampoco necesitan un modelo
```

## Apuntarlo a un modelo real

**El mismo código, sin tocar una línea.** El agente no sabe quién hay al otro lado.

```bash
export SYNAPTUM_BASE_URL=http://localhost:8080/v1   # cualquier endpoint OpenAI-compatible
export SYNAPTUM_MODEL=qwen3-8b
uv run python -m mi_agente
```

## Qué hay aquí

| | |
|---|---|
| `src/mi_agente/herramientas.py` | Lo primero que vas a cambiar |
| `src/mi_agente/agente.py` | El agente: modelo, instrucciones, herramientas, salida tipada |
| `src/mi_agente/__main__.py` | Cómo se ejecuta y cómo se elige la puerta |
| `tests/test_agente.py` | Probar un agente sin gastar |

## Lo siguiente

1. **Cambia las herramientas.** El esquema sale de la firma, así que basta con tipar los argumentos
   y escribir un docstring.
2. **Declara el riesgo** de las que escriban algo. Sin declararlo entran como `Risk.READ`, y una
   política que deniegue por `Risk.DESTRUCTIVE` no las detendrá.
3. **Persiste el run** cuando quieras reanudar: pasa un `SqliteCheckpointer` a la `Session`.

Documentación: **https://root1v.github.io/synaptum-framework/**
