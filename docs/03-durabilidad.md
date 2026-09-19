# Durabilidad

**Al reanudar un run, una inferencia ya pagada no se paga otra vez.** Es la propiedad que justifica
toda la arquitectura.

```python
store = SqliteCheckpointer("runs.db")

# Primera vuelta: dos llamadas al modelo, una a una herramienta.
async for paso in agente.run("lee /x", session=Session("run-1", gateway, store)):
    ...

# El proceso muere. Otro proceso, otra conexión, mismo run_id.
async for paso in agente.run("lee /x", session=Session("run-1", otro_gateway, store)):
    ...   # cero llamadas al modelo, cero a la herramienta
```

No es una afirmación: [`examples/propiedades/02_durabilidad.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/propiedades/02_durabilidad.py)
cuenta las llamadas y lo mide.

```
1ª vuelta  modelo ×1 · herramienta ×1   ← el proceso muere aquí
2ª vuelta  modelo ×1 · herramienta ×0   ← solo ocurre lo que no llegó a ocurrir
3ª vuelta  modelo ×0                    ← un run cerrado no se reabre
```

## Cómo funciona

El journal guarda **qué pasos ocurrieron**, no la conversación. Al reanudar, el contexto se vuelve a
derivar de los mismos resultados en el mismo orden.

La clave de deduplicación es `(run_id, step_id, phase)`, y en SQLite **es la clave primaria** — no
una comprobación previa que dos procesos puedan cruzar.

## Escritura anticipada

Antes de ejecutar una herramienta no idempotente, su intención está en disco. Eso permite distinguir
tres situaciones que de otro modo serían la misma:

| En el journal | Significa | Qué hace el replay |
|---|---|---|
| Intención **y** resultado | Ocurrió y se sabe cómo fue | Salta el paso |
| Ni intención ni resultado | No llegó a intentarse | Lo ejecuta |
| Intención sin resultado, **no idempotente** | **No se sabe** si ocurrió | Levanta `UncertainEffect` |
| Intención sin resultado, **idempotente** | Da igual: repetir no cuesta | Lo ejecuta |

`UncertainEffect` no es un fallo del framework: es el framework negándose a adivinar. Si tu
herramienta mueve dinero y el proceso murió justo después de mandarla, **nadie sabe si salió**, y
fingir lo contrario es peor que parar.

## Denegación y aprobación humana

Una denegación previa a la ejecución **no es un silencio**: es un desenlace conocido del paso.

```
1ª vuelta → ModelStep · ToolStep(denegado, decision=require_approval) · ApprovalStep
            ← el run queda suspendido en disco

… una persona aprueba …

2ª vuelta → se reanuda donde estaba, el efecto ocurre exactamente una vez
```

Sin esto, un `require_approval` dejaba intención sin resultado, y al reanudar eso parecía el caso
incierto. Pero **aquí no hay incertidumbre**: se denegó *antes* de ejecutar. Por eso `StepEvent`
lleva `decision`, y el replay distingue «no se ejecutó porque se denegó» de «no se sabe».

**El bucle no puede escribir esa aprobación**: no estaba corriendo cuando se tomó. La escribe el
arnés, y eso significa que `append` no lo llama solo el bucle — el checkpointer es el diario del
run, no un almacén privado del framework.

## Idempotencia hacia el proveedor

La identidad del paso viaja como clave de idempotencia hacia proveedores que la admitan:
`(run_id, step_id, huella-del-cuerpo)`.

Cierra un agujero que el journal solo no puede tapar: si el proceso muere **después** de mandar la
petición y **antes** de registrar el resultado, al reanudar el replay reintenta —una llamada al
modelo es repetible— y sin clave esa repetición es **una segunda generación facturable**.

La huella del cuerpo no es prudencia: una clave identifica *una* petición, y reutilizarla para otra
distinta es un rechazo, no un replay. Sin ella, dos runs con el mismo `run_id` y distinta tarea
chocan — y también el mismo run después de cambiar las instrucciones del agente.

## Un run cerrado no se reabre

Si el journal ya tiene un `FinalStep`, reanudar **devuelve lo que pasó** en vez de recorrerlo otra
vez. Reemitir la historia como si estuviera ocurriendo sería engañoso; quien quiera la línea
temporal la tiene en `Checkpointer.load`.
