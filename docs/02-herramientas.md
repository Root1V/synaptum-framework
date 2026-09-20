# Herramientas

## El esquema sale de la firma

```python
from typing import Annotated
from synaptum import tool

@tool
async def contar_lineas(
    fichero: Annotated[str, "Ruta relativa a la raíz del proyecto"],
    maximo: int = 100,
) -> str:
    """Cuenta las líneas de un fichero del proyecto."""
    ...
```

De ahí sale el JSON Schema que ve el modelo: los tipos, los obligatorios, las descripciones de
`Annotated` y la primera línea del docstring.

**Por qué no se escribe aparte:** un esquema escrito a mano se desincroniza. Y cuando lo hace, el
modelo manda argumentos que la función no acepta — lejos del cambio que lo causó y con una
inferencia ya pagada.

Un tipo que no sabemos traducir **falla al decorar**, no al invocar.

## Riesgo e idempotencia se declaran

```python
@tool(risk=Risk.DESTRUCTIVE)
async def transferir(destino: str, importe: float) -> str:
    """Ordena una transferencia."""
```

| | |
|---|---|
| `Risk.READ` | Lee. No cambia nada. **Es el valor por defecto.** |
| `Risk.SOFT_WRITE` | Escribe algo reversible. |
| `Risk.HARD_WRITE` | Escribe algo que cuesta deshacer. |
| `Risk.DESTRUCTIVE` | Destruye o mueve dinero. |

**Ninguna anotación de tipos puede saber que una función que devuelve `str` mueve dinero.** Por eso
se declara a mano y no se deduce.

`idempotent=True` dice que **repetir la llamada no tiene consecuencias**. No es una optimización: es
lo que permite al runtime reintentar tras una caída sin arriesgarse a duplicar un efecto.

### Qué pasa si no declaras nada

Obtienes `risk=READ` e `idempotent=False`. Asimétrico a propósito: **la clase más inocua y la
garantía más cara.**

La consecuencia práctica, y conviene saberla: una política que deniegue por `Risk.DESTRUCTIVE` **no
detendrá tu herramienta**, aunque borre discos. En riesgo, callar es decir «inofensivo». En
durabilidad, callar es decir «asume lo peor», y por eso su intención sí se escribe antes de
ejecutar.

Endurecer un default sería un cambio MAJOR; relajarlo, directamente no se hace.

## Declarar no es decidir

**Synaptum declara; quien gobierna decide.** Marcar algo `Risk.DESTRUCTIVE` no ejecuta ninguna política:
el riesgo viaja por la costura y el gateway decide. Marcarlo `Risk.READ` tampoco autoriza nada.

```python
def politica(check):
    if check.risk is Risk.DESTRUCTIVE and check.arguments.get("importe", 0) > 100:
        return Decision(disposition=Disposition.REQUIRE_APPROVAL, reason_code="tope")
    return ALLOW
```

`check.arguments` es campo propio y no una clave genérica, porque **es lo que una política decide**:
negar «capturar» sin ver el importe no es una política, es un interruptor.

## Los errores vuelven al modelo

Un fallo dentro de la función **no rompe el run**: vuelve como `ToolResult` con `is_error=True`, y
el modelo lo ve y suele corregir. Un modelo que no ve el error no puede corregirlo.

Lo que **sí** se propaga es una llamada mal formada: argumentos que no encajan en la firma no son
algo que el modelo pueda arreglar leyendo un mensaje, son un desajuste entre el esquema y la
función.

## La salida se recorta antes de entrar en el contexto

Una herramienta que lee un fichero de log puede devolver **39.000 tokens**. Con una ventana de 4.096
es un fallo inmediato; con una de 131.072 es solo caro — y caro **en cada turno posterior**, porque
ese bloque se reenvía entero cada vez.

Por eso hay un tope, **activado por defecto**:

```python
Limits(max_tool_chars=16_000)   # ~4.000 tokens. `None` lo desactiva.
```

Cuatro cosas que conviene entender antes de cambiarlo:

**El journal guarda entero; el contexto lleva recortado.** El paso registra lo que la herramienta
devolvió de verdad — si mañana hay que auditar qué leyó el agente, la respuesta está ahí. Lo que se
recorta es la copia que viaja al modelo.

**El recorte es determinista, o rompería la reanudación.** Al reanudar, el contexto se vuelve a
derivar del journal y se recorta igual. Si dependiera del reloj o del espacio restante, el prompt
reconstruido sería distinto del original: caché fallada, y potencialmente otra respuesta a la misma
pregunta.

**Se conservan los dos extremos.** El encabezado dice qué es y el final trae el total, el resumen o
la excepción. En un resultado de error el reparto se inclina hacia la cola, porque el mensaje de una
traza está abajo.

**Dice lo que quitó.** Un texto truncado en silencio hace que el modelo concluya sobre datos
incompletos creyéndolos completos.

```
[… recortado: 91.700 de 92.000 caracteres. Pide un rango concreto si necesitas lo que falta …]
```

Se mide en **caracteres y no en tokens** porque contar tokens exige un tokenizador y el núcleo no
tiene dependencias. La regla práctica es ~4 caracteres por token en texto latino; menos en código y
en CJK. Si necesitas un tope exacto, recórtalo en la herramienta — que es donde vive el conocimiento
de qué se puede tirar.

## Un catálogo grande: `deferred`

Cuarenta herramientas son **3.737 tokens de catálogo en cada turno**, el 94 % de una ventana de
4.096. Y el modelo tiene que elegir entre cuarenta, que es la otra mitad del problema.

```python
from synaptum import deferred, merece_la_pena

agente = Agent("a", model=…, tools=deferred(las_cuarenta))   # 3.737 → 142 tokens
```

### Lo obvio es peor que no hacer nada

Cargar herramientas cuando hagan falta parece la solución. **Está medido contra un despliegue real,
con un historial de run normal:**

| Petición | `input` | `cache_read` |
|---|---|---|
| 10 herramientas, 2ª llamada | 1.444 | 1.443 (**100 %**) |
| 11 — una sola añadida | 1.467 | 0 (**0 %**) |

Añadir **una** herramienta a mitad de run destruye la caché entera, historial incluido. El catálogo
vive en el prefijo estable, y un prefijo que cambia en el token 10 invalida los 10.000 siguientes.

### Lo que sí funciona: mover el catálogo al historial

El prefijo se queda con **dos herramientas fijas** y no cambia nunca:

- `buscar_herramientas(consulta)` — devuelve nombres, descripciones **y esquemas**;
- `usar_herramienta(nombre, argumentos)` — ejecuta la que sea.

El esquema que el modelo necesita llega como **resultado de herramienta**, y un resultado se añade al
final de los mensajes. El final crece; crecer al final no invalida nada.

Es el mismo principio que el prefijo estable, al revés: en vez de proteger lo de delante, se mueve lo
variable hacia atrás.

### Lo que cuesta

- **Dos turnos extra como mínimo**: buscar y luego usar.
- **El modelo pierde el esquema tipado en la llamada**, así que los errores de argumentos suben. Se
  compensa validando al despachar y devolviendo el esquema en el error — pero es una compensación.
- **Por debajo de ~15 herramientas es estrictamente peor.** `merece_la_pena(catalogo)` lo dice, en
  vez de dejar que se descubra midiendo una factura.

### El riesgo no se blanquea

El despachador hereda **el mayor riesgo del catálogo**. Sin eso, esconder cuarenta herramientas
detrás de una las blanquearía a todas: el gateway vería lectura y dejaría pasar la que borra el
disco. Misma regla que en la delegación, y por lo mismo — quien llama no sabe qué hay detrás, pero el
framework sí.

## Herramientas de un servidor MCP

```bash
pip install synaptum[mcp]
```

```python
from synaptum.mcp import MCPTools

async with MCPTools.stdio("uvx", "mcp-server-git", "--repository", ".", prefix="git.") as git:
    agente = Agent("dev", model=..., tools=list(git))
```

Entran con su esquema y **el bucle no nota que son remotas**: cumplen el mismo contrato que `@tool`.

El `async with` no es decoración: un servidor por stdio es un proceso hijo, y salir del bloque es lo
que lo cierra.

### Qué cambia cuando la herramienta es ajena

**El riesgo lo insinúa quien no manda.** MCP trae `readOnlyHint`, `destructiveHint` e
`idempotentHint`, y su propia especificación dice que un cliente **no debe fiarse de ellas** para
decidir seguridad: un servidor equivocado —o malicioso— puede declarar inocua una herramienta que
borra.

Por eso la traducción sigue **los defectos de MCP y no los nuestros**: sin anotar, `Risk.DESTRUCTIVE`.
Produce más avisos de los que uno espera, y es lo correcto. Nuestro `@tool` es permisivo por defecto
porque el autor está delante y puede declarar; con un servidor ajeno no está.

```python
print([t.name for t in repo.destructivas])   # míralas antes de dárselas a un agente
```

**Los errores llegan sin explicación.** Un servidor MCP no filtra sus internos: llega
`Error executing tool X` y nada más. El modelo sabe *que* falló y casi nunca *por qué*.

**Los nombres chocan.** Dos servidores que publiquen `search` no son distinguibles para el modelo, y
gana el último registrado, en silencio. De ahí el `prefix=`.

**El esquema es de otro.** Si el servidor cambia el suyo, tu agente empieza a mandar argumentos que
ya no encajan sin que nada en tu repositorio haya cambiado. Es justo el problema que `@tool` resuelve
derivando el esquema de la firma, y con MCP vuelve porque la firma vive en otro sitio.
