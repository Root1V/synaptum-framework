# Política de estabilidad de API

`SYN-46`. Existe porque **Aeon congeló su DSL de autoría nativo** apoyándose en que Synaptum sería
la capa de autoría. Esa decisión es difícil de revertir, así que la contrapartida no puede quedarse
en una frase del acuerdo tripartito: tiene que decir qué está garantizado, durante cuánto, y qué no.

## Qué es superficie pública

**Lo que `synaptum/__init__.py` exporta, y nada más.**

Un símbolo alcanzable solo por su módulo —`synaptum.run.journal.Replay`, por ejemplo— es interno
aunque no lleve guion bajo, y puede cambiar sin aviso. Si algo interno te hace falta, pídelo y lo
promovemos; usarlo desde fuera es aceptar que se romperá.

La superficie está **fijada por un test**: cualquier cambio en `__all__` rompe la suite y obliga a
actualizar el fichero de referencia a mano. No es burocracia — es que una política que nadie
comprueba se erosiona, y ya hemos visto esa lección tres veces en este proyecto.

## Qué queda fuera a propósito

| Fuera | Por qué |
|---|---|
| Representación interna de los eventos | Lo que se garantiza es el **contrato de la costura**, no cómo se guardan los campos en memoria |
| Mensajes de error y sus textos | Los códigos de razón sí son estables; la prosa no |
| El módulo `synaptum.testing` | Es andamiaje de desarrollo y evoluciona con lo que haga falta probar |
| Cualquier cosa marcada `experimental` en su docstring | Declarado a la entrada, sin sorpresa |

## Versionado

SemVer **a partir de `1.0.0`**:

- **MAJOR** — se quita algo de la superficie pública, o se cambia lo que significa.
- **MINOR** — se añade. Añadir un campo con valor por defecto, un parámetro opcional o un símbolo
  nuevo es minor.
- **PATCH** — correcciones que no cambian la superficie.

**Antes de `1.0.0` la superficie puede cambiar**, y decirlo claro vale más que fingir lo contrario.
Estamos en `1.0.0.dev0`. Lo que sí se compromete desde hoy está en la sección siguiente.

## Qué se garantiza ya, antes de 1.0.0

Dos bloques, con garantías distintas y conviene no confundirlos:

### Lo que viaja por la costura — garantía del contrato, no de este paquete

El vocabulario del modelo, la taxonomía de eventos y los dos protocolos de costura están
especificados en `contratos/` y versionados aparte, con una ventana de **dos versiones menores
vivas**. Esa garantía es más fuerte que la de esta política y no depende de ella: alguien puede
implementar la costura sin usar Synaptum.

### La API de autoría — lo que Aeon congeló su DSL para usar

```python
Agent · tool · Tool · Session · Limits · Risk
```

Sobre estos seis:

1. **No se quitan ni se renombran antes de `1.0.0`** sin un cambio acordado en el canal de
   coordinación, con la razón por escrito.
2. **Los parámetros existentes no cambian de significado.** Añadir parámetros opcionales sí es
   posible.
3. **Los valores por defecto no se relajan en la dirección permisiva.** `risk=READ` e
   `idempotent=False` son deliberadamente asimétricos —permisivo en riesgo, conservador en
   durabilidad— y cambiarlos alteraría en silencio el comportamiento de código ya escrito. Endurecer
   un default es un cambio MAJOR; relajarlo, directamente no se hace.

## Deprecación

Un símbolo que va a desaparecer:

1. Emite `DeprecationWarning` con el sustituto nombrado en el mensaje.
2. **Sigue funcionando durante al menos dos versiones menores.** Misma ventana que la costura, por
   coherencia y porque permite a los tres proyectos desplegar sin coordinar una fecha de corte.
3. Solo entonces se quita, y en una MAJOR.

Un aviso de deprecación que no nombra el sustituto es un aviso incompleto: el que lo recibe tiene
que poder actuar sin abrir un issue.

## Qué hacer si esto se rompe

Si un cambio nuestro rompe a Aeon o a Axonium **fuera de lo que esta política permite**, es un fallo
nuestro y se revierte — no se negocia una migración a posteriori. Ese es el sentido de haberlo
escrito antes.
