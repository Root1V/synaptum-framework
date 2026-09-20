# 11 · Un catálogo grande sin pagarlo en cada turno

> **Generada de [`examples/agentes/11_catalogo_diferido.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/11_catalogo_diferido.py).** El fichero corre; esta página lo
> transcribe. Si los dos no coinciden, falla un test.

**Dominio: Prometheus** — la consola del operador de la plataforma de inferencia
local. Veinte herramientas: backends, modelos cargados, colas, cuotas, claves.
Todas legítimas, y en un turno cualquiera hacen falta una o dos.

El catálogo entero viaja en **el prefijo** de cada petición. Veinte
herramientas con su esquema son varios miles de tokens que se envían siempre —
aunque se sirvan de caché, y **dejan de servirse en cuanto el catálogo cambia**.
Y hay un coste que no es dinero: cuantas más ve el modelo, peor elige.

`deferred()` cambia veinte por dos:

    buscar_herramientas(consulta)      →  las que encajan, con su esquema
    usar_herramienta(nombre, argumentos)  →  ejecuta la que se eligió

```bash
uv run python examples/agentes/11_catalogo_diferido.py
```

```python
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import (
    Agent,
    FinalStep,
    MemoryCheckpointer,
    Phase,
    Risk,
    Role,
    Session,
    ToolStep,
    deferred,
    dumps,
    merece_la_pena,
    tool,
)
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo
```

## El catálogo

En la consola de verdad son veinte funciones escritas a mano, cada una con su
firma y su docstring. Aquí salen de una tabla para que el fichero se pueda
leer de una sentada: lo que importa es **cuántas** hay y qué riesgo tienen,
no qué hace cada una.

```python
CONSOLA = [
    # (nombre, qué hace, argumento, riesgo)
    ("listar_backends", "Lista los backends de inferencia y su estado.", "filtro", Risk.READ),
    ("estado_backend", "Salud, latencia y cola de un backend concreto.", "backend", Risk.READ),
    ("listar_modelos", "Modelos disponibles en el catálogo de la plataforma.", "familia", Risk.READ),
    ("modelos_cargados", "Qué modelos tiene un backend en memoria ahora mismo.", "backend", Risk.READ),
    ("uso_de_vram", "VRAM ocupada y libre por GPU.", "backend", Risk.READ),
    ("cola_de_peticiones", "Peticiones en espera y tiempo medio de cola.", "backend", Risk.READ),
    ("latencias", "p50, p95 y p99 de un modelo en una ventana.", "modelo", Risk.READ),
    ("errores_recientes", "Últimos errores agrupados por causa.", "ventana", Risk.READ),
    ("consumo_por_cliente", "Tokens consumidos por cliente en un periodo.", "periodo", Risk.READ),
    ("cuota_de_cliente", "Cuota asignada y consumida de un cliente.", "cliente", Risk.READ),
    ("listar_claves", "Claves de acceso emitidas, sin el secreto.", "cliente", Risk.READ),
    ("auditar_accesos", "Quién llamó a qué en una ventana de tiempo.", "ventana", Risk.READ),
    ("precargar_modelo", "Carga un modelo en un backend antes de que haga falta.", "modelo", Risk.SOFT_WRITE),
    ("fijar_concurrencia", "Cambia la concurrencia máxima de un backend.", "backend", Risk.SOFT_WRITE),
    ("ajustar_cuota", "Cambia la cuota de un cliente.", "cliente", Risk.HARD_WRITE),
    ("rotar_clave", "Rota la clave de acceso de un cliente.", "cliente", Risk.HARD_WRITE),
    ("drenar_backend", "Deja de enviarle peticiones nuevas y espera a las vivas.", "backend", Risk.HARD_WRITE),
    ("descargar_modelo", "Saca un modelo de memoria. Las peticiones en vuelo fallan.", "modelo", Risk.DESTRUCTIVE),
    ("reiniciar_backend", "Reinicia el proceso. Corta todo lo que esté en vuelo.", "backend", Risk.DESTRUCTIVE),
    ("revocar_clave", "Revoca una clave. Irreversible.", "clave", Risk.DESTRUCTIVE),
]

def _construir(nombre: str, que_hace: str, argumento: str, riesgo: Risk):
    async def fn(**kwargs) -> str:
        return _RESPUESTAS.get(nombre, f"{nombre}({kwargs}) · ok")

    return tool(
        fn,
        name=nombre,
        description=que_hace,
        risk=riesgo,
        idempotent=riesgo is Risk.READ,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {argumento: {"type": "string", "description": f"{argumento}."}},
            "required": [argumento],
        },
    )

_RESPUESTAS = {
    "modelos_cargados": (
        "llama-4-scout        38,2 GB  · 412 peticiones/h\n"
        "qwen3-32b            21,7 GB  ·  18 peticiones/h\n"
        "gpt-oss-20b-mxfp4    12,1 GB  · 980 peticiones/h\n"
        "whisper-large-v3      3,1 GB  ·   0 peticiones/h  (última hace 9 días)"
    ),
}

HERRAMIENTAS = [_construir(*fila) for fila in CONSOLA]

def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")
```

## Qué ve el modelo, con el catálogo y sin él

```python
def comparar() -> list:
    diferidas = deferred(HERRAMIENTAS)

    entero = sum(len(dumps(h.definition.parameters)) + len(h.definition.description or "")
                 for h in HERRAMIENTAS)
    dos = sum(len(dumps(h.definition.parameters)) + len(h.definition.description or "")
              for h in diferidas)

    print(f"  el catálogo entero en el prefijo: {len(HERRAMIENTAS):>2} herramientas, "
          f"{_miles(entero):>5} caracteres")
    print(f"  diferido:                          {len(diferidas):>2} herramientas, "
          f"{_miles(dos):>5} caracteres")
    print(f"\n  ¿merece la pena con este catálogo? {merece_la_pena(HERRAMIENTAS)}")
    print(
        "  (con menos de quince, no: los dos turnos extra de buscar y usar\n"
        "  cuestan una inferencia cada uno, y solo los amortiza un catálogo grande)\n"
    )

    despachador = next(h for h in diferidas if h.definition.name == "usar_herramienta")
    print(f"  riesgo de `usar_herramienta`: {despachador.definition.risk.value}")
    print(
        "  — el despachador hereda el riesgo de **la peor del catálogo**. Sin eso,\n"
        "  esconder veinte herramientas detrás de una las blanquearía a todas: el\n"
        "  arnés vería una lectura y dejaría pasar `revocar_clave`.\n"
    )
    return diferidas
```

## El run

```python
async def main() -> None:
    encabezado("11 · Un catálogo grande, diferido")

    diferidas = comparar()

    agente = Agent(
        "consola",
        model=nombre_del_modelo(),
        instructions=(
            "Operas una plataforma de inferencia local. No conoces las "
            "herramientas de antemano: búscalas por lo que quieres hacer y "
            "después úsalas. Responde con datos, no con generalidades."
        ),
        tools=diferidas,
    )

    sesion = Session(
        "consola-1",
        gateway([_guion] * 4, tools=diferidas),
        MemoryCheckpointer(),
    )

    async for paso in agente.run(
        "¿Qué modelos están cargados en el backend 3 y cuánta VRAM ocupan?",
        session=sesion,
    ):
        match paso:
            case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                print(f"  → {llamada.name}({dumps(llamada.arguments)})")
            case ToolStep(phase=Phase.COMPLETED, result=resultado) if resultado:
                primera = "".join(
                    p.text for p in resultado.content if hasattr(p, "text")
                ).splitlines()[0]
                print(f"    ← {primera}")
            case FinalStep(output=salida):
                print(f"\n  {salida}\n")

def _guion(peticion):
    resultados = sum(1 for m in peticion.messages if m.role is Role.TOOL)
    if resultados == 0:
        return calls("buscar_herramientas", id="b1", consulta="modelos cargados vram backend")
    if resultados == 1:
        return calls(
            "usar_herramienta", id="u1",
            nombre="modelos_cargados", argumentos='{"backend": "backend-3"}',
        )
    return says(
        "El backend 3 tiene cuatro modelos en memoria y 75,1 GB ocupados. "
        "whisper-large-v3 lleva nueve días sin una sola petición y ocupa 3,1 GB: "
        "es lo primero que descargaría si hace falta sitio."
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Lo que esto enseña

Las dos cosas que hay que saber antes de usar esto:

**El catálogo acaba en el historial, no en el prefijo.** Es deliberado, y es
lo contrario del primer diseño. Medido contra la plataforma real: **añadir una
sola herramienta a mitad de un run bajó `cache_read` de 1.443 a 0.** Cualquier
cosa que modifique el catálogo entre turnos invalida toda la caché posterior,
así que el resultado de una búsqueda entra como un mensaje más —al final, que
es donde crecer no invalida nada— y el prefijo se queda igual todo el run.

**La búsqueda es léxica, no semántica.** Peor que una con embeddings y con dos
virtudes que aquí pesan más: no añade una dependencia al núcleo, y es
**determinista** — al reanudar, la misma consulta devuelve exactamente las
mismas herramientas en el mismo orden, así que el contexto se reconstruye
idéntico. Una búsqueda que ordenase distinto en cada proceso rompería la
reanudación sin decirlo.

Y lo que esto **no** arregla: dos turnos de más. Por eso `merece_la_pena()`
está expuesto — es mejor poder preguntarlo que descubrirlo en la factura.
