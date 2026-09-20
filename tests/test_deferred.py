"""SYN-40 · Catálogo diferido.

Lo que se protege aquí no es el ahorro —eso es aritmética— sino las tres cosas
que hacen que diferir sea seguro: que el prefijo no cambie, que el riesgo no se
blanquee, y que el modelo pueda corregir cuando se equivoca de argumentos.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest

from synaptum import (
    Agent,
    Risk,
    Role,
    Session,
    deferred,
    merece_la_pena,
    tool,
)
from synaptum.context import prefix_fingerprint
from synaptum.core.types import Request
from synaptum.testing import FakeGateway, calls, says

RAIZ = __import__('pathlib').Path(__file__).resolve().parents[1]

EJECUTADAS: list[tuple[str, dict]] = []


def _catalogo(n: int = 20, con_destructiva: bool = True):
    herramientas = []
    for i in range(n):
        async def f(consulta: Annotated[str, "Qué buscar"], limite: int = 10) -> str:
            return "resultados"
        f.__name__ = f"consultar_facturas_{i:02d}"
        f.__doc__ = f"Consulta las facturas del periodo {i} en el sistema contable."
        herramientas.append(tool(f))

    @tool
    async def buscar_cliente(nombre: Annotated[str, "Nombre del cliente"]) -> str:
        """Busca un cliente en el padrón por su nombre."""
        EJECUTADAS.append(("buscar_cliente", {"nombre": nombre}))
        return f"cliente {nombre}: activo"

    herramientas.append(buscar_cliente)

    if con_destructiva:
        @tool(risk=Risk.DESTRUCTIVE)
        async def purgar(tabla: Annotated[str, "Tabla"]) -> str:
            """Purga una tabla del almacén."""
            EJECUTADAS.append(("purgar", {"tabla": tabla}))
            return "purgada"

        herramientas.append(purgar)
    return herramientas


def _llamar(herramienta, **kw):
    return asyncio.run(herramienta.fn(**kw))


# ── El prefijo no cambia: es el motivo entero ─────────────────────────────────

def test_the_prefix_never_changes_however_much_is_searched():
    """Medido antes de escribir esto: añadir UNA herramienta a mitad de run
    destruye la caché entera, historial incluido. Por eso el catálogo se mueve
    al historial en vez de cargarse en el prefijo.
    """
    diferido = deferred(_catalogo())
    agente = Agent("a", model="openai-compatible:m", instructions="Ayudas.", tools=diferido)
    antes = prefix_fingerprint(Request(model=agente.model, system=agente.instructions,
                                       tools=agente.tools))

    _llamar(diferido[0], consulta="facturas del periodo")
    _llamar(diferido[0], consulta="clientes")

    despues = prefix_fingerprint(Request(model=agente.model, system=agente.instructions,
                                         tools=agente.tools))
    assert antes == despues, "buscar cambió el prefijo: la caché se habría perdido"


def test_only_two_tools_reach_the_catalogue():
    diferido = deferred(_catalogo(40))
    assert [t.definition.name for t in diferido] == [
        "buscar_herramientas", "usar_herramienta"
    ]


def test_it_says_when_deferring_is_not_worth_it():
    """Por debajo del umbral esto es estrictamente peor, y mejor decirlo que
    dejar que se descubra midiendo una factura."""
    assert merece_la_pena(_catalogo(20))
    assert not merece_la_pena(_catalogo(3, con_destructiva=False))


# ── El riesgo no se blanquea ──────────────────────────────────────────────────

def test_the_dispatcher_inherits_the_worst_risk_in_the_catalogue():
    """Esconder cuarenta herramientas detrás de una las blanquearía a todas:
    el gateway vería lectura y dejaría pasar la que borra el disco."""
    con = deferred(_catalogo(5, con_destructiva=True))
    sin = deferred(_catalogo(5, con_destructiva=False))

    assert con[1].definition.risk is Risk.DESTRUCTIVE
    assert sin[1].definition.risk is Risk.READ
    assert con[1].definition.idempotent is False, "detrás puede haber cualquier cosa"


# ── Buscar ────────────────────────────────────────────────────────────────────

def test_a_search_returns_names_descriptions_and_schemas():
    """El esquema es lo que el modelo necesita para llamar, y llega por el
    historial en vez de por el prefijo."""
    salida = _llamar(deferred(_catalogo())[0], consulta="buscar un cliente en el padrón")

    assert "buscar_cliente" in salida
    assert "padrón" in salida
    assert '"nombre"' in salida, "no llegó el esquema"


def test_a_search_that_matches_nothing_says_so_instead_of_returning_everything():
    salida = _llamar(deferred(_catalogo())[0], consulta="zzz nada que ver")
    assert "Ninguna" in salida


def test_results_are_capped_so_the_model_does_not_choose_worse():
    """Elegir entre cuarenta es medio problema que esto resuelve."""
    salida = _llamar(deferred(_catalogo(30), max_resultados=3)[0], consulta="facturas")
    assert salida.count("argumentos:") <= 3


def test_the_ranking_does_not_depend_on_the_order_of_the_catalogue():
    """Dos catálogos con las mismas herramientas en distinto orden buscan igual.

    Es lo que garantiza el desempate por nombre, y es una garantía real: el
    orden de un diccionario es estable dentro de un proceso, pero el orden en
    que alguien construye su catálogo no lo es —basta con que salga de un `set`,
    de un glob o de un servidor MCP que lista en el orden que le apetezca.

    Si el orden cambiara, el contexto reconstruido al reanudar no sería idéntico
    y se perdería la caché.

    Nota honesta: la primera versión de este test comparaba dos llamadas en el
    mismo proceso. Quité el desempate para comprobar que lo cazaba y **no se
    enteró** — dentro de un proceso el orden ya es estable por inserción. El
    test no probaba lo que decía probar.
    """
    catalogo = _catalogo()
    al_reves = list(reversed(catalogo))

    una = _llamar(deferred(catalogo)[0], consulta="facturas del periodo")
    otra = _llamar(deferred(al_reves)[0], consulta="facturas del periodo")

    assert una == otra, "el resultado depende del orden en que llegó el catálogo"


# ── Usar ──────────────────────────────────────────────────────────────────────

def test_the_dispatcher_runs_the_real_tool():
    EJECUTADAS.clear()
    salida = _llamar(
        deferred(_catalogo())[1], nombre="buscar_cliente", argumentos='{"nombre": "ACME"}'
    )

    assert EJECUTADAS == [("buscar_cliente", {"nombre": "ACME"})]
    assert "activo" in salida


@pytest.mark.parametrize(
    "argumentos,esperado",
    [
        ('{"nombre":', "no son JSON válido"),
        ('["ACME"]', "deben ser un objeto"),
        ("{}", "Faltan argumentos"),
    ],
    ids=["json roto", "no es objeto", "faltan campos"],
)
def test_a_bad_call_comes_back_correctable_instead_of_breaking(argumentos, esperado):
    """El modelo ya no tiene el esquema delante: vive varios turnos atrás.

    Así que un error tiene que traer de vuelta lo que hace falta para corregir,
    no solo decir que algo falló.
    """
    salida = _llamar(deferred(_catalogo())[1], nombre="buscar_cliente", argumentos=argumentos)
    assert esperado in salida


def test_missing_arguments_bring_the_schema_back():
    salida = _llamar(deferred(_catalogo())[1], nombre="buscar_cliente", argumentos="{}")
    assert '"nombre"' in salida, "no devolvió el esquema, y el modelo ya no lo tiene"


def test_an_unknown_tool_points_back_to_the_search():
    salida = _llamar(deferred(_catalogo())[1], nombre="no_existe")
    assert "buscar_herramientas" in salida


# ── En el bucle ───────────────────────────────────────────────────────────────

def test_the_whole_cycle_works_in_the_loop():
    """Buscar, leer el esquema del resultado, y usar."""
    EJECUTADAS.clear()
    diferido = deferred(_catalogo())
    agente = Agent("a", model="openai-compatible:m", instructions="Ayudas.", tools=diferido)

    def responder(peticion):
        respuestas = sum(1 for m in peticion.messages if m.role is Role.TOOL)
        if respuestas == 0:
            return calls("buscar_herramientas", consulta="buscar un cliente")
        if respuestas == 1:
            return calls("usar_herramienta", nombre="buscar_cliente",
                         argumentos='{"nombre": "ACME"}')
        return says("El cliente ACME está activo.")

    async def ir():
        return [
            p async for p in agente.run(
                "¿está activo ACME?", session=Session("r1", FakeGateway(*[responder] * 8, tools=diferido))
            )
        ]

    pasos = asyncio.run(ir())
    assert EJECUTADAS == [("buscar_cliente", {"nombre": "ACME"})]
    assert pasos[-1].output == "El cliente ACME está activo."
