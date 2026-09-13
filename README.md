# Synaptum

**Framework de agentes y runtime durable, agnóstico al proveedor.**

Synaptum es dueño de la *semántica* de ejecución de un agente: qué es un paso, dónde puede cortarse,
qué puede repetirse, y cómo se vuelve a derivar el contexto. El *sustrato* —dónde se persiste, con
qué retención, bajo qué política— pertenece al arnés que lo opera.

> **Versión:** `1.0.0.dev0` · **Python:** ≥ 3.13 · **Licencia:** MIT
>
> **En construcción.** La línea 0.x, con un diseño distinto, está congelada en
> [v0.4.0](https://github.com/Root1V/synaptum-framework/releases/tag/v0.4.0). El estado real de esta
> por elemento está en [`roadmap.md`](https://github.com/Root1V/synaptum-framework/blob/main/roadmap.md).

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

Esto se apoya en una sola pieza: la **identidad determinista de paso**, publicada como
especificación abierta. Quien la implemente obtiene la misma propiedad sin importar nada de Synaptum
ni hablar Python.

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

## Probarlo

```bash
uv run python examples/01_agente.py
```

Corre **sin inferencia**: las respuestas van guionizadas, y el resto es real —las herramientas se
ejecutan, el journal se escribe, el consumo se mide—. Para apuntar a un modelo de verdad, dos
variables y **el mismo fichero sin tocar**:

```bash
export SYNAPTUM_BASE_URL=http://localhost:8080/v1
export SYNAPTUM_MODEL=qwen3-0.6b
uv run python examples/01_agente.py
```

Que el mismo código sirva para las dos cosas no es comodidad: es la propiedad. Ver
[`examples/`](https://github.com/Root1V/synaptum-framework/tree/main/examples) — el bucle entero, la reanudación **medida** (no afirmada), una aprobación
humana a mitad de run, y el streaming con cancelación.

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
from synaptum.testing import FakeGateway, ReplayGateway, calls, says

# Guion escrito a mano: directo, y suficiente para la mayoría.
gateway = FakeGateway(calls("leer", path="/x"), says("dice hola"), tools=[leer])

# Respuestas reales grabadas, normalizadas por el adaptador de verdad.
gateway = ReplayGateway("fixtures/chat_completion.json", tools=[leer])
```

Un guion a mano dice lo que uno espera; una grabación dice lo que el proveedor hizo — y la diferencia
aparece en los caminos que nadie escribe porque no se le ocurren.

## Dónde encaja

Synaptum se sostiene solo. Habla con dos piezas por **protocolo**, y trae una implementación de
referencia completa de cada una:

```
arnés          política, aprobaciones, secretos, retención, escalado
  │            Gateway — decide y ejecuta          · por defecto: LocalGateway
Synaptum       semántica de ejecución              ← esto
  │            Checkpointer — persiste, no decide  · por defecto: SqliteCheckpointer
almacén
                                    ─────
proveedor      cualquier endpoint OpenAI-compatible · por defecto: HttpModel
```

**Las dos costuras son protocolos estructurales (`typing.Protocol`), no clases base.** Quien las
implemente no hereda ni importa nada nuestro, y puede estar escrito en otro lenguaje al otro lado de
un socket. No hay ningún arnés, SDK ni plataforma de inferencia concretos en el árbol de
dependencias: `pip install synaptum` trae **cero** paquetes.

En el despliegue donde nació, esas dos ranuras las ocupan un arnés llamado Aeon y un SDK llamado
Axonium sobre una plataforma de inferencia local. Nada de eso es un requisito, y el paquete no los
nombra: son **un** relleno posible de un protocolo abierto. El extra `[axonium]` existe para quien
tenga esa combinación, y es opcional como el de Anthropic o el de OpenAI.

`LocalGateway` **avisa de que no aplica política** y marca cada comprobación con `enforced=False` —
una comprobación dentro del proceso gobernado es advisoria, y que un run pase por ahí sin
denegaciones no dice nada sobre si pasaría por un gateway real.

## Contratos compartidos

Tres especificaciones con casos dorados, ejecutables por cualquier implementación con su propio
runner. **Viven fuera de este repositorio** porque son artefactos conjuntos de varios proyectos, y
copiarlos aquí los convertiría en una copia que se desincroniza. Synaptum no depende de ellos: son
evidencia adicional, y sin ellos la suite pasa igual (209 de 277; el resto se salta).

```bash
export SYNAPTUM_CONTRACTS=/ruta/a/contratos    # opcional
```


| Contrato | Estado |
|---|---|
| Costura de durabilidad | 8 casos · verdes contra nuestras dos implementaciones |
| Identidad de paso | 10 casos · verdes |
| Normalización entre proveedores | 14 casos · verdes contra el adaptador Python, todos menos uno **grabaciones reales** |

Los casos describen **resultados observables** y nunca estructuras internas: dos implementaciones sin
una línea de código en común tienen que poder reproducirlos.

Que los cuerpos sean grabaciones reales y no ejemplos escritos a mano ya encontró dos fallos
nuestros. Un stream que era **solo razonamiento** no emitía ni un evento, porque el cuerpo inventado
que teníamos antes solo llevaba deltas de texto. Y en un stream de razonamiento que termina llamando
a una herramienta, el cierre del pensamiento caía **en mitad de la llamada**: cerrábamos el ciclo al
ver texto, y ahí no había texto.

Un caso dorado **no fija cifras**. Una regrabación tiró cinco de los nuestros sin que ninguna
implementación hubiera cambiado: fijaban tokens y un id de réplica, que pertenecen a una grabación y
no a la especificación. Lo que se afirma es si un contador está medido o sin medir, y cómo se
relacionan entre sí.

## Estabilidad

Qué se garantiza, durante cuánto y qué no: está **escrito y comprobado por un test**, no recordado.
Un arnés retiró su DSL de autoría apoyándose en ese compromiso, que es por qué existe por escrito.
Ver [`API.md`](https://github.com/Root1V/synaptum-framework/blob/main/API.md).

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
