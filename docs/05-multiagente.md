# Varios agentes

**`delegate()` no existe todavía** — es `SYN-41`. El tipo `DelegateStep` está escrito y nada lo
emite.

Eso **no impide** construir sistemas multi-agente hoy: el bucle de un agente es un generador
asíncrono de verdad, así que orquestar varios es código asyncio normal. Lo que `SYN-41` añadirá es
que la delegación quede **en el journal** como un paso propio, con su consumo atribuido y su punto de
reanudación.

La regla que sí se puede seguir desde el primer día:

> **Entre agentes viaja el resultado, nunca el historial.**

Duplicar el contexto de un agente en otro se paga dos veces, y hace que el segundo herede los errores
del primero sin poder distinguirlos de sus datos.

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
