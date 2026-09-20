# Qué no hace

Dicho en voz alta, porque un framework que no dice dónde acaba deja que le supongan el resto.

## No aplica política

`LocalGateway` **no aplica nada**. Corre dentro del proceso que gobernaría, así que sus
comprobaciones son advisorias: las cumple un bucle correcto y se las salta uno con un fallo — o uno
comprometido. Es la misma razón por la que un sandbox no se implementa como una función que el código
encerrado decide llamar.

Marca cada decisión con `enforced=False` y su informe lleva una cabecera diciéndolo, porque un
informe que no lo dice acaba pegado en un ticket como si fuera una auditoría.

**Que un run pase por aquí sin denegaciones no dice nada** sobre si pasaría por un gateway real.

## No guarda secretos ni gobierna credenciales

`HttpModel` tiene la clave en la memoria del proceso y llama a quien le digan. En el camino gobernado
la llamada al modelo **no ocurre en este proceso**.

## No decide cuánto puede gastar un run

`Limits` es **corrección, no política**: evita que un bucle mal formado no termine nunca. Los topes
de gasto pertenecen al arnés y llegan por la costura como una denegación.

## No elige dónde se persiste

`Checkpointer` es un protocolo. Traemos dos implementaciones de referencia —memoria y SQLite— y
ninguna es para producción a escala. La retención, el cifrado y el borrado son del sustrato, y el
sustrato no es nuestro.

## No es un bus de mensajes

El motor es `await`, no una cola. No hay broker, ni suscriptores, ni entrega garantizada entre
agentes. El stream de eventos es **una sola cosa cediendo control**, no N cosas hablando entre sí.

Si necesitas agentes en procesos distintos comunicándose de forma asíncrona y persistente, eso es un
arnés, y está bien que lo sea.

## No reduce el historial cuando crece

Dos piezas de economía de contexto **sí** existen, y conviene no confundirlas con la que falta:

- La salida de una herramienta se recorta antes de entrar en el contexto.
- El prefijo estable está protegido: reanudar con otra configuración se rechaza en vez de mezclar
  dos agentes en un mismo diario.

Lo que **no** hay es compactación por niveles (`SYN-36`). En un run largo, el historial completo se
reenvía en cada turno. Con caché de prefijo eso suele salir más barato que resumir —por eso la
compactación va desactivada por defecto cuando llegue— pero hay un punto en el que deja de serlo, y
hoy nadie lo detecta por ti.

## No delega con contexto aislado

`delegate()` es `SYN-41`. Se puede componer a mano —y funciona— con las tres consecuencias listadas
en [Varios agentes](05-multiagente.md).

## No emite trazas todavía

`SYN-37`. Las convenciones GenAI de OpenTelemetry están en el plan; hoy no hay spans.

## No reintenta lo que no debe

Y esto es un «no hace» deliberado, no una carencia: **cualquier 4xx salvo 429 no se reintenta.** La
familia entera significa «tu petición es el problema»: repetirla sin cambiarla da el mismo resultado
y cuesta lo mismo.

## No adivina cuando no sabe

Si el proceso muere justo después de lanzar una herramienta no idempotente y antes de registrar el
resultado, **nadie sabe si el efecto ocurrió**. El replay levanta `UncertainEffect` en vez de elegir
por ti. Es el framework negándose a adivinar, y es la conducta correcta aunque sea la incómoda.
