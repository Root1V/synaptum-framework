# Varios agentes

Tres formas, y la primera ya es una primitiva del bucle.

## Delegar a un subagente

```python
analista = Agent("analista", model=..., instructions="Analizas valores.", tools=[cotizacion])
mesa = Agent("mesa", model=..., instructions="Enrutas al especialista.", delegates=[analista])
```

El subagente se presenta al modelo como **una herramienta de un solo parámetro**: el brief. El
catálogo del padre no crece con el del hijo, y eso es lo que hace barato delegar — el analista puede
tener quince herramientas y la mesa ve una firma de una línea.

Al subagente le llega **el brief y nada más**: no ve la conversación del padre. De vuelta suben el
resultado y el **consumo agregado**. Su historial se queda en su propio diario, donde se audita sin
pagarlo en cada turno.

### Qué cierra respecto a componerlos a mano

**El coste deja de ser invisible.** El `usage` del `FinalStep` del padre incluye lo que gastó el
subagente, porque el `DelegateStep` lo transporta. Antes había que sumarlo a mano, y un sistema que
gastaba cinco veces más parecía igual de barato.

**Un subagente es un paso durable.** Tiene su propio `run_id`, derivado del padre:
`{run_id}/{step_id}`. Si el proceso muere a mitad, al reanudar **no se reejecuta**:

```
1ª vuelta  modelo ×3 · herramienta del hijo ×1   ← muere aquí
2ª vuelta  modelo ×1 · herramienta del hijo ×0   ← solo lo que faltaba
```

**El riesgo se deriva.** `delegate_risk` es el mayor de las herramientas del subagente, y de los
suyos. Aquí sí se puede derivar —quien delega no sabe qué tiene el otro, pero el framework sí— al
contrario que en `@tool`, donde ninguna anotación puede saber que una función que devuelve `str`
mueve dinero.

> **Hasta dónde llega eso hoy.** El riesgo **se declara** —el modelo lo ve, el arnés lo ve en el
> handshake— pero la delegación **no cruza la costura**: el bucle arranca al subagente sin
> preguntar. Las herramientas del subagente sí cruzan, así que un efecto destructivo se detiene
> igual; lo que se pierde es detenerlo *antes* de pagar la inferencia del hijo. Denegar la
> delegación en sí exigiría un método de la costura que autorice sin ejecutar, y eso es un cambio de
> contrato.

**Los ciclos se paran.** `Limits.max_delegation_depth` (3 por defecto). `max_steps` no lo cubre:
cada nivel tiene su propio contador, así que dos agentes que se deleguen mutuamente no terminarían
nunca.

### Un agente remoto declara su riesgo, o no se conecta

```python
RemoteDelegate(name="analista", url="https://…", risk=Risk.READ)   # `risk` es obligatorio
```

Un `AgentCard` declara `skills`, no riesgo: la especificación no tiene ese campo. Poner un defecto
conservador sería correcto **y silencioso** — nadie se enteraría de que nunca se declaró.

La diferencia con `@tool` es quién está delante. Ahí el autor de la función puede declarar, y por eso
`Risk.READ` por defecto es razonable. Con un agente ajeno el autor no está, pero **quien lo conecta
sí** — así que la decisión es suya y se le pide que la tome.

### Lo que sigue siendo válido componer a mano

Delegar sirve cuando **el modelo decide** a quién llamar. Cuando el orden lo decides tú —una cadena
fija, un fan-out— componer con asyncio sigue siendo lo correcto y más simple. Las dos formas están
abajo.

## Cadena — uno detrás de otro

Cuando cada etapa necesita criterio distinto: traducir y luego revisar que quepa en el hueco.

```python
traduccion = await ejecutar(traductor(), "Traduce el segmento seg-0042.", "dub-tr")

# Al revisor le llega el RESULTADO, no la conversación del traductor.
brief = f"Traducción propuesta: «{traduccion.texto_es}». El original dura 5.5s."
revision = await ejecutar(revisor(), brief, "dub-rev")
```

Cada agente tiene **su propio `run_id`**: son runs distintos, se reanudan por separado, y el journal
de uno no contamina el del otro.

## Paralelo — fan-out y síntesis

Cuando varias hipótesis compiten y son independientes.

```python
resultados = await asyncio.gather(
    *(investigar(nombre, brief, herramienta) for nombre, brief, herramienta in LINEAS)
)
```

Aislar contexto es **lo que hace barato paralelizar**: si los investigadores compartieran historial,
habría que serializarlos. En reloj cuesta una fracción; en tokens, lo mismo que en serie — los
tokens no saben de concurrencia.

El sintetizador recibe los hallazgos, no las transcripciones. Es lo que impide que el coste crezca
con el número de trabajadores.

Con `return_exceptions=True` decides qué hacer con la línea que reventó.

## Un agente como herramienta de otro

El patrón más usado en la práctica: un supervisor que enruta.

```python
@tool
async def especialista_observabilidad(pregunta: Annotated[str, "La pregunta, tal cual"]) -> str:
    """Responde sobre incidentes, latencias y trazas."""
    async for paso in agente_obs.run(pregunta, session=Session("sup-obs", puerta)):
        if isinstance(paso, FinalStep):
            return str(paso.output)
```

**A favor:** el supervisor no necesita conocer las herramientas de nadie. Sus tres herramientas
tienen una firma de una línea, mientras cada especialista puede tener quince. Añadir un dominio es
añadir una función, no reescribir un prompt.

**En contra, y es lo que hay que saber antes de usarlo:**

1. **El coste desaparece de la vista.** `FinalStep.usage` del supervisor mide *sus* llamadas, no las
   de dentro. Hay que sumarlo a mano; si solo miras la del supervisor, un sistema que gasta cinco
   veces más parece igual de barato. Es exactamente lo que `DelegateStep.usage` existe para resolver.

2. **Un subagente no es un paso durable.** Si el proceso muere a mitad de un especialista, al
   reanudar la herramienta se reejecuta entera — el journal del supervisor la ve como una llamada, no
   como un run con sus propios pasos. Para lectura da igual; para algo que mueva dinero, no.

3. **El riesgo no se propaga.** El envoltorio es `Risk.READ` por defecto aunque por dentro llame a algo
   que escribe. Hay que declararlo a mano; el framework no puede deducirlo, y fingir que sí sería
   peor.

## Cuál elegir

| Situación | Forma |
|---|---|
| Etapas con criterios distintos, una depende de la anterior | Cadena |
| Hipótesis independientes, el reloj importa | Paralelo + síntesis |
| Dominios separados, una entrada única | Agente como herramienta |
| Un solo dominio y muchas herramientas | **Un agente.** Varios no arreglan un prompt malo. |
