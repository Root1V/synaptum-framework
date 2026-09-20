"""10 · Lo que el modelo ve en cada turno, y lo que cuesta.

**Dominio: Argus** — depurar un incidente mirando logs. La herramienta devuelve
lo que devuelve: 150.000 caracteres de una ventana de cinco minutos. El modelo
necesita saber qué hay dentro; no necesita releerlo **en cada turno posterior**,
que es lo que ocurre si nadie lo impide.

Tres piezas que son la misma idea vista desde tres sitios:

    Limits(max_tool_chars=…)   qué parte del resultado entra en el contexto
    prefijo estable            qué **no** debe cambiar, para que haya caché
    economy(estado)            qué costó el run y **por qué**

Ninguna de las tres inventa nada: si nadie midió la caché, el informe lo dice en
vez de escribir un cero.

    uv run python examples/agentes/10_economia_del_contexto.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import (
    Agent,
    Limits,
    MemoryCheckpointer,
    Request,
    Role,
    Session,
    Text,
    ToolResult,
    cap_tool_output,
    economy,
    tool,
)
from synaptum.context import describe_prefix_change, prefix_fingerprint
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

# ── La herramienta que devuelve demasiado ─────────────────────────────────────
#
# No es un caso rebuscado.  Un `kubectl logs`, un `SELECT *`, la respuesta de
# una API paginada que no pagina: lo normal es que una herramienta devuelva
# muchísimo más de lo que hacía falta.

LINEA = (
    "2026-09-18T02:14:{s:02d}.{ms:03d}Z  prometheus-gateway  INFO  "
    "req_id=8f3a{n:04d} POST /v1/chat/completions upstream=llama-4 "
    "model_load_ms={carga} total_ms={total} status=200\n"
)


#: Trescientas líneas son ~45.000 caracteres. Un `kubectl logs` de cinco
#: minutos devuelve bastante más; aquí se queda corto a propósito, para que el
#: ejemplo se pueda correr contra un modelo de verdad sin ocupar la plataforma
#: un minuto entero. La proporción es lo que importa, y no cambia.
LINEAS = 300


def _volcado() -> str:
    lineas = [
        LINEA.format(
            s=n % 60, ms=n % 1000, n=n,
            carga=5900 if n % 7 == 0 else 3,
            total=6400 if n % 7 == 0 else 510,
        )
        for n in range(LINEAS)
    ]
    return "".join(lineas)


@tool(idempotent=True)
async def leer_logs(
    servicio: Annotated[str, "Servicio del que leer"],
    ventana: Annotated[str, "Ventana temporal, por ejemplo '02:14-02:19'"],
) -> str:
    """Logs crudos de un servicio en una ventana de tiempo."""
    return _volcado()


# ── 1 · Qué entra en el contexto ──────────────────────────────────────────────

def parte_uno() -> None:
    entero = ToolResult.of("call-1", _volcado())
    recortado = cap_tool_output(entero, max_chars=16_000)

    largo = len("".join(p.text for p in entero.content if isinstance(p, Text)))
    corto = len("".join(p.text for p in recortado.content if isinstance(p, Text)))

    print(f"  la herramienta devolvió   {largo:>9,} caracteres".replace(",", "."))
    print(f"  al modelo le llegan       {corto:>9,} caracteres".replace(",", "."))
    print("\n  y el recorte lo dice, en vez de terminar en mitad de una línea:\n")

    texto = "".join(p.text for p in recortado.content if isinstance(p, Text))
    marca = next(linea for linea in texto.splitlines() if linea.startswith("["))
    print(f"    {marca}")
    print(
        "\n  Cabeza **y** cola, no los primeros 16.000 caracteres: en un volcado\n"
        "  de logs el final es donde está el fallo, y cortar por delante lo tira.\n"
        "  El journal guarda el resultado entero pase lo que pase — esto solo\n"
        "  decide qué se le **reenvía** al modelo en cada turno posterior.\n"
    )


# ── 2 · Lo mismo, medido sobre un run ─────────────────────────────────────────

class Medido:
    """Cuenta los caracteres que salen hacia el modelo en cada turno.

    Envuelve el gateway que haya —el doble o el real— porque la medida tiene que
    ser la misma en los dos casos. Medir solo contra el doble sería medir el
    doble.
    """

    def __init__(self, interior) -> None:
        self._interior = interior
        self.por_turno: list[int] = []

    async def invoke_model(self, request, ctx):
        self.por_turno.append(_caracteres(request))
        return await self._interior.invoke_model(request, ctx)

    def stream_model(self, request, ctx):
        self.por_turno.append(_caracteres(request))
        return self._interior.stream_model(request, ctx)

    def __getattr__(self, nombre):
        return getattr(self._interior, nombre)


def _caracteres(request) -> int:
    """Todo el texto de una petición: sistema, mensajes y resultados."""
    total = len(request.system or "")
    for mensaje in request.messages:
        for parte in mensaje.content:
            if isinstance(parte, Text):
                total += len(parte.text)
            elif isinstance(parte, ToolResult):
                total += sum(
                    len(p.text) for p in parte.content if isinstance(p, Text)
                )
    return total


async def correr(*, tope: int | None, run_id: str) -> tuple[list[int], object]:
    agente = Agent(
        "depurador",
        model=nombre_del_modelo(),
        instructions=(
            "Depuras incidentes de una plataforma de inferencia. Lee los logs "
            "del servicio afectado y di en una frase dónde se va el tiempo."
        ),
        tools=[leer_logs],
        limits=Limits(max_tool_chars=tope),
    )

    puerta = Medido(gateway([_guion] * 4, tools=[leer_logs]))
    store = MemoryCheckpointer()
    sesion = Session(run_id, puerta, store)

    async for _ in agente.run(
        "¿Por qué el p95 del gateway pasó de 820ms a 6.4s a las 02:14?",
        session=sesion,
    ):
        pass
    return puerta.por_turno, await store.load(run_id)


def _guion(peticion):
    if any(m.role is Role.TOOL for m in peticion.messages):
        return says(
            "Una de cada siete peticiones recarga el modelo: 5.9s de los 6.4s "
            "son model_load. El resto responde en ~510ms. No es el gateway, es "
            "el desalojo de la caché de pesos."
        )
    return calls("leer_logs", servicio="prometheus-gateway", ventana="02:14-02:19")


# ── 3 · El prefijo que se rompe solo ──────────────────────────────────────────

class ConLaHoraDentro(Agent):
    """Un agente cuyas instrucciones llevan el reloj.

    Parece inofensivo y es de lo más común: dar la hora al modelo para que sepa
    qué es «ayer». El problema es que el prompt de sistema **abre** el prefijo,
    así que cambiarlo en cada turno invalida todo lo que viene detrás — y el
    síntoma es una caché que nunca arranca, no un error.
    """

    @property
    def instructions(self) -> str:
        return (
            "Depuras incidentes de una plataforma de inferencia. "
            f"Ahora son las {datetime.now():%H:%M:%S.%f}."
        )

    @instructions.setter
    def instructions(self, _valor) -> None:
        pass          # el constructor intenta asignarlas; aquí se ignoran


async def parte_tres() -> None:
    agente = ConLaHoraDentro(
        "depurador", model=nombre_del_modelo(), tools=[leer_logs]
    )

    # El mecanismo, sin depender de que el modelo haga nada: dos peticiones
    # construidas con un instante de diferencia, y sus huellas.
    antes = Request(model=agente.model, system=agente.instructions, tools=agente.tools)
    ahora = Request(model=agente.model, system=agente.instructions, tools=agente.tools)

    print(f"  huella del prefijo: {prefix_fingerprint(antes)} → {prefix_fingerprint(ahora)}")
    print(f"  qué cambió:         {describe_prefix_change(antes, ahora)}")
    print(
        "\n  Eso no dice «la huella no coincide», que no sirve para nada: dice\n"
        "  qué arreglar. Y así es como sale en el informe de un run:\n"
    )

    store = MemoryCheckpointer()
    sesion = Session("con-reloj", gateway([_guion] * 4, tools=[leer_logs]), store)
    async for _ in agente.run("¿Qué pasó a las 02:14?", session=sesion):
        pass

    informe = economy(await store.load("con-reloj"))
    print(informe.report())

    if informe.prefix_rewrites:
        print(
            "\n  Nadie escribió un error y nada falló. Lo único que pasó es que la\n"
            "  factura es más alta de lo que debería, y eso no se nota mirando un run.\n"
        )
    else:
        print(
            "\n  Este run se resolvió en un solo turno —el modelo contestó sin usar\n"
            "  la herramienta— y con uno no hay nada con lo que comparar: el\n"
            "  informe no cuenta el estreno como una regresión. El daño aparece\n"
            "  cuando el run es largo, que es justo cuando la caché importaría.\n"
        )


# ── Todo junto ────────────────────────────────────────────────────────────────

async def main() -> None:
    encabezado("10 · Economía del contexto")

    parte_uno()

    print("  ── el mismo run, con el tope y sin él ──\n")
    sin_tope, _ = await correr(tope=None, run_id="sin-tope")
    con_tope, estado = await correr(tope=16_000, run_id="con-tope")

    for turno, (a, b) in enumerate(zip(sin_tope, con_tope), start=1):
        print(f"    turno {turno}: sin tope {a:>9,}  ·  con tope {b:>9,}".replace(",", "."))
    print(
        "\n  El turno 1 es idéntico: el volcado todavía no ha vuelto. La\n"
        "  diferencia aparece en el 2 — y se pagaría **otra vez** en el 3, en el\n"
        "  4 y en todos los demás, porque el historial se reenvía entero.\n"
    )

    print("  ── el informe del run ──\n")
    print(economy(estado).report())
    print(
        "\n  Con el doble los números son los del guion —declara `cache_read=0`,\n"
        "  un cero **medido**— así que lo que enseña el informe es su forma. Un\n"
        "  proveedor que no reporte caché saldría «sin medir», que es distinto de\n"
        "  cero y el informe no los confunde. Contra la plataforma real estas\n"
        "  cifras son reales: exporta las credenciales y vuelve a correrlo.\n"
    )

    print("  ── y el mismo agente, con la hora en las instrucciones ──\n")
    await parte_tres()


# Por qué esto no es una optimización, dicho una vez:
#
# **No recortar falla en silencio.** Con una ventana pequeña el run revienta y
# te enteras; con una grande solo cuesta dinero en cada turno posterior, que es
# peor, porque nadie lo mira. Por eso `max_tool_chars` viene activado.
#
# **Reescribir el prefijo tampoco avisa.** `economy()` existe porque el síntoma
# —una caché que no arranca— solo se ve agregando los turnos, y porque «la
# huella no coincide» no sirve para nada: lo accionable es «las instrucciones de
# sistema cambiaron».
#
# Y la medida: **`cache_read` es lo que hace correcta la decisión de compactar.**
# Sin saber cuánto se está sirviendo de caché, conservar el historial parece caro
# cuando a menudo es casi gratis, y resumirlo parece barato cuando cuesta una
# inferencia y pierde información.

if __name__ == "__main__":
    asyncio.run(main())
