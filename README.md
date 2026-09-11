# Synaptum

**Framework de agentes y runtime durable, agnóstico al proveedor.**

Synaptum es dueño de la *semántica* de ejecución de un agente: qué es un paso, dónde puede cortarse,
qué puede repetirse, y cómo se vuelve a derivar el contexto. El *sustrato* —dónde se persiste, con
qué retención, bajo qué política— pertenece al arnés que lo opera.

> **Versión:** `1.0.0.dev0` · **Python:** ≥ 3.13 · **Licencia:** MIT
>
> **En construcción.** La línea 0.x, con un diseño distinto, está congelada en
> [v0.4.0](https://github.com/Root1V/synaptum-framework/releases/tag/v0.4.0). El estado real de esta
> por elemento está en [`roadmap.md`](roadmap.md).

---

## La idea

El bucle de un agente no es un `while` escondido: es un generador asíncrono que **cede el control en
cada frontera significativa**.

```python
async for step in agent.run(tarea, session=session):
    match step:
        case ToolStep(phase=Phase.ATTEMPTED, risk=Risk.DESTRUCTIVE):
            ...
        case ModelStep(phase=Phase.COMPLETED, usage=consumo):
            ...
        case FinalStep(output=salida):
            ...
```

De esa forma salen cuatro propiedades, y ninguna otra estructura las da a la vez:

1. **El arnés obtiene sus puntos de enganche** sin que Synaptum sepa que existe. Aprobaciones,
   guardarraíles y métricas son consumidores del stream.
2. **Cada `yield` es una frontera de checkpoint natural.**
3. **Interrumpir es dejar de iterar**; reanudar es volver a llamar con el mismo `run_id`.
4. **Probar es iterar una lista.**

## Lo que justifica todo lo demás

Al reanudar un run, **una inferencia ya pagada no se paga otra vez.**

```python
store = SqliteCheckpointer("runs.db")

# Primera vuelta: dos llamadas al modelo, una a una herramienta.
async for step in agent.run("lee /x", session=Session("run-1", gateway, store)):
    ...

# El proceso muere. Otro proceso, otra conexión, mismo run_id.
async for step in agent.run("lee /x", session=Session("run-1", otro_gateway, store)):
    ...   # cero llamadas al modelo, cero a la herramienta
```

La ventana de contexto no se almacena: se **vuelve a derivar** de los mismos resultados en el mismo
orden. Guardarla sería guardar dos veces lo mismo y arriesgarse a que discrepen.

Esto se apoya en una sola pieza: la [identidad determinista de paso](../coordinacion_project/contratos/identidad-de-paso/spec.md),
publicada como **especificación abierta**. Quien la implemente obtiene la misma propiedad sin
importar nada de Synaptum ni hablar Python.

## Herramientas

El esquema sale de la firma. Un esquema escrito aparte se desincroniza, y cuando lo hace el modelo
manda argumentos que la función no acepta — lejos del cambio que lo causó y con una inferencia ya
pagada.

```python
@tool(risk=Risk.DESTRUCTIVE)
async def borrar(path: Annotated[str, "Ruta absoluta"], forzar: bool = False) -> str:
    """Borra un fichero del disco."""
    ...
```

`risk` e `idempotent` se declaran y no se deducen: no son propiedades del tipo sino del **efecto**, y
ninguna anotación puede saber que una función que devuelve `str` mueve dinero.

Los defaults son asimétricos a propósito — **permisivo en riesgo, conservador en durabilidad**: quien
calla obtiene la clase más inocua y la garantía más cara.

Un tipo que no sabemos traducir **falla al decorar**, no al invocar.

## Sin dependencias

```bash
pip install synaptum      # 0 dependencias
```

El núcleo es stdlib puro. Pydantic, los proveedores, MCP y OpenTelemetry son extras. Los adaptadores
se descubren por *entry points*, así que el núcleo no conoce a ninguno.

## Desarrollo sin inferencia

No hay modelos locales disponibles en modo autónomo, así que el doble de desarrollo es
infraestructura y no una utilidad de test. Ejercita todo lo que el bucle sabe hacer: llamadas a
herramientas con ejecución real, streaming con cancelación, las tres disposiciones de denegación, la
taxonomía de errores y el consumo de tres estados.

```python
from synaptum.testing import FakeGateway, calls, says

gateway = FakeGateway(calls("leer", path="/x"), says("dice hola"), tools=[leer])
```

## Dónde encaja

```
Aeon         arnés · política, aprobaciones, secretos, retención, escalado
  │          dos costuras: una aplica, otra recuerda
Synaptum     framework + runtime · semántica de ejecución      ← esto
  │
Axonium      SDK en tres sabores · única puerta a inferencia local
  │
Prometheus   plataforma de inferencia
```

Synaptum funciona sin nada de lo de arriba ni lo de abajo: `LocalGateway` y `MemoryCheckpointer` son
implementaciones de referencia completas. `LocalGateway` **avisa de que no aplica política** y marca
cada comprobación con `enforced=False` — una comprobación dentro del proceso gobernado es advisoria,
y que un run pase por ahí sin denegaciones no dice nada sobre si pasaría por el gateway real.

## Contratos compartidos

Tres, en `contratos/`, con casos dorados que cada proyecto ejecuta con su propio runner:

| Contrato | Estado |
|---|---|
| Costura de durabilidad | 8 casos · verdes contra nuestras dos implementaciones |
| Identidad de paso | 10 casos · verdes |
| Normalización entre proveedores | 9 casos · validados, **ejecutables cuando exista el primer adaptador** |

Los casos describen **resultados observables** y nunca estructuras internas: dos implementaciones sin
una línea de código en común tienen que poder reproducirlos.

## Estabilidad

Aeon congeló su DSL de autoría apoyándose en nuestro compromiso de estabilidad, así que está escrito
y comprobado por un test, no recordado. Ver [`API.md`](API.md).

## Estado

| Fase | |
|---|---|
| **0 · Contratos** | completa |
| **1 · Núcleo durable** | completa |
| **2 · Contexto y observabilidad** | sin empezar |
| **3 · Multi-agente** | sin empezar |

`roadmap.md` lleva el detalle por elemento.

---

MIT © 2026 · Emeric Espiritu Santiago
