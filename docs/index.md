# Empezar

Synaptum es un framework de agentes con un runtime durable. Agnóstico al proveedor, cero
dependencias, y con una idea central: **el bucle de un agente no es un `while` escondido, es un
generador asíncrono que cede el control en cada frontera significativa.**

```bash
pip install synaptum
```

## Un agente completo

```python
import asyncio
from typing import Annotated
from synaptum import Agent, FinalStep, Session, tool

@tool
async def contar_lineas(fichero: Annotated[str, "Ruta del fichero"]) -> str:
    """Cuenta las líneas de un fichero."""
    return f"{fichero}: {len(open(fichero).read().splitlines())} líneas"

async def main():
    agente = Agent(
        "explorador",
        model="openai-compatible:gpt-oss-20b",
        instructions="Respondes usando las herramientas. Sé breve.",
        tools=[contar_lineas],
    )
    async for paso in agente.run("¿Cuántas líneas tiene README.md?", session=sesion):
        if isinstance(paso, FinalStep):
            print(paso.output)

asyncio.run(main())
```

Falta `sesion`, y es lo primero que hay que entender: [**Conceptos**](01-conceptos.md).

## Las páginas

| | |
|---|---|
| [Conceptos](01-conceptos.md) | `Agent`, `Session`, gateway, checkpointer, pasos y fases |
| [Herramientas](02-herramientas.md) | `@tool`, riesgo, idempotencia, errores, MCP |
| [Durabilidad](03-durabilidad.md) | El journal, reanudar sin volver a pagar, aprobaciones |
| [Modelos](04-modelos.md) | Proveedores, transporte, `Usage` de tres estados, streaming |
| [Varios agentes](05-multiagente.md) | Cadena, paralelo, agente como herramienta |
| [Probar](06-probar.md) | Construir agentes sin gastar en inferencia |
| [Qué no hace](07-limites.md) | Los límites, dichos en voz alta |
| [Referencia](08-referencia.md) | Cada símbolo público, **generado del código** |

## Empezar un proyecto

```bash
cp -r plantilla mi-agente && cd mi-agente && uv sync && uv run python -m mi_agente
```

Corre sin configurar nada, y trae los tests que enseñan a probar un agente sin gastar en inferencia.

## Ejecutar los ejemplos

Doce ejemplos ejecutables en [`examples/`](https://github.com/Root1V/synaptum-framework/tree/main/examples).
Corren **sin configurar nada**: las respuestas van guionizadas y todo lo demás es real.

```bash
uv run python examples/agentes/01_triaje.py
```

Para apuntar a un modelo de verdad, dos variables y **el mismo fichero sin tocar**. Ver
[Modelos](04-modelos.md).

## Esta documentación, en dos formatos

El Markdown de `docs/` es la fuente; el HTML de `docs/html/` se genera. Un agente lee una página
`.md` sin navegación que estorbe; una persona lee el `.html`, que sí la necesita.

```bash
uv run python scripts/render_docs.py
```

El sitio se publica solo en cada cambio de `docs/`. Un test falla si el HTML del repositorio deja de
corresponder al Markdown. Escribir los dos a mano garantiza que
diverjan, y el que diverge siempre es el que nadie mira.
