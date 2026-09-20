"""SYN-41 · Delegar a un subagente.

Componer agentes a mano ya funcionaba. Lo que esto cierra son los tres agujeros
que estaban documentados como limitaciones — el coste invisible, el subagente que
no es un paso durable, y el riesgo que no se propaga— y cada uno tiene su test.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest

from synaptum import (
    ALLOW,
    Agent,
    Decision,
    Denied,
    DelegateStep,
    Disposition,
    FinalStep,
    Limits,
    LocalGateway,
    MemoryCheckpointer,
    Phase,
    Risk,
    Role,
    Session,
    tool,
)
from synaptum.agent.delegation import Delegate, delegate_risk, sub_run_id
from synaptum.testing import FakeGateway, calls, says

EJECUCIONES: list[str] = []


@tool(idempotent=True)
async def cotizacion(valor: Annotated[str, "Ticker"]) -> str:
    """Cotización de un valor."""
    EJECUCIONES.append("cotizacion")
    return "187,34 USD"


@tool(risk=Risk.DESTRUCTIVE)
async def liquidar(posicion: Annotated[str, "Posición"]) -> str:
    """Liquida una posición."""
    EJECUCIONES.append("liquidar")
    return "liquidada"


def _analista(tools=(cotizacion,)) -> Agent:
    return Agent(
        "analista", model="openai-compatible:m",
        instructions="Analizas valores con las herramientas. Sé conciso.",
        tools=list(tools),
    )


def _supervisor(sub: Agent) -> Agent:
    return Agent(
        "mesa", model="openai-compatible:m", instructions="Enrutas al especialista.",
        delegates=[sub],
    )


def _responder(peticion):
    """Un doble que contesta según el contexto, como haría un modelo.

    Un guion posicional respondería lo mismo la segunda vez y volvería a delegar
    al reanudar — que es justo lo que la reanudación evita.

    Y para de insistir cuando recibe un error, como haría un modelo razonable.
    Uno que no parase sería correcto también: para eso está `max_steps`.
    """
    if any(m.role is Role.TOOL for m in peticion.messages):
        return says("ACME cotiza a 187,34 USD.")
    if any(t.name == "analista" for t in peticion.tools):
        return calls("analista", brief="¿Cómo va ACME?")
    return calls("cotizacion", valor="ACME")


def _correr(agente, gateway, store=None, run_id="r1"):
    async def ir():
        sesion = Session(run_id, gateway, store) if store else Session(run_id, gateway)
        return [p async for p in agente.run("¿cómo va ACME?", session=sesion)]

    return asyncio.run(ir())


# ── El subagente como herramienta ─────────────────────────────────────────────

def test_a_delegate_is_offered_as_a_single_parameter_tool():
    """El catálogo del padre no crece con el del hijo.

    Es lo que hace barato delegar: el analista puede tener quince herramientas y
    el supervisor ve una firma de una línea.
    """
    supervisor = _supervisor(_analista())

    definicion = next(t for t in supervisor.tools if t.name == "analista")
    assert set(definicion.parameters["properties"]) == {"brief"}
    assert "cotizacion" not in {t.name for t in supervisor.tools}


def test_the_description_says_what_the_specialist_is_for():
    supervisor = _supervisor(_analista())
    definicion = next(t for t in supervisor.tools if t.name == "analista")

    assert "Analizas valores" in definicion.description


# ── 1 · El riesgo se propaga ──────────────────────────────────────────────────

def test_delegating_to_a_destructive_agent_is_destructive():
    """Envolver un agente en una función lo blanqueaba a READ.

    Aquí sí se puede **derivar**: quien delega no sabe qué herramientas tiene el
    otro, pero el framework sí.
    """
    assert delegate_risk(_analista()) is Risk.READ
    assert delegate_risk(_analista(tools=(cotizacion, liquidar))) is Risk.DESTRUCTIVE


def test_risk_survives_two_levels_of_delegation():
    """Un riesgo que se pierde a dos saltos se pierde igual."""
    peligroso = _analista(tools=(liquidar,))
    intermedio = Agent("intermedio", model="openai-compatible:m", delegates=[peligroso])

    assert delegate_risk(intermedio) is Risk.DESTRUCTIVE


def test_the_subagents_destructive_tool_is_still_stopped():
    """Hasta dónde llega hoy la propagación del riesgo, dicho con precisión.

    El riesgo **se declara** —el modelo lo ve, y el arnés en el handshake— pero
    la delegación **no cruza la costura**: el bucle arranca al subagente sin
    preguntar. Lo que sí cruza son las herramientas del subagente, así que el
    efecto destructivo se detiene igual.

    Lo que se pierde es detenerlo *antes* de pagar la inferencia del hijo, y eso
    exigiría un método de la costura que autorice sin ejecutar — un cambio de
    contrato que va por el canal de coordinación.
    """
    vistos: list = []

    def politica(check):
        vistos.append((check.name, check.risk))
        if check.kind == "tool" and check.risk is Risk.DESTRUCTIVE:
            return Decision(disposition=Disposition.DENY_STEP, reason_code="sin_supervision")
        return ALLOW

    def responder(peticion):
        if any(m.role is Role.TOOL for m in peticion.messages):
            return says("no pude hacerlo")
        sub = next((t.name for t in peticion.tools), None)
        return calls(sub, brief="liquida ACME") if sub else says("listo")

    supervisor = _supervisor(_analista(tools=(liquidar,)))
    modelo = FakeGateway(*[responder] * 12)
    gateway = LocalGateway(model=modelo.invoke_model, tools=[liquidar], policy=politica, warn=False)

    EJECUCIONES.clear()
    _correr(supervisor, gateway)

    assert "liquidar" not in EJECUCIONES, "el efecto destructivo ocurrió"
    assert ("liquidar", Risk.DESTRUCTIVE) in vistos, "la herramienta del hijo no cruzó la costura"
    assert ("analista", Risk.DESTRUCTIVE) not in vistos, (
        "si esto empieza a pasar, la limitación se cerró y este test hay que reescribirlo"
    )


# ── 2 · El coste deja de ser invisible ────────────────────────────────────────

def test_the_parents_total_includes_what_the_subagent_spent():
    """`FinalStep.usage` medía solo las llamadas del padre.

    Un sistema que gasta cinco veces más parecía igual de barato.
    """
    pasos = _correr(_supervisor(_analista()), FakeGateway(*[_responder] * 6, tools=[cotizacion]))
    final = pasos[-1]
    delegacion = next(
        p for p in pasos if isinstance(p, DelegateStep) and p.phase is Phase.COMPLETED
    )

    assert delegacion.usage.input, "la delegación no registró el consumo del subagente"
    assert final.usage.input > delegacion.usage.input, (
        "el total del padre no incluye lo que gastó el subagente"
    )


# ── 3 · Un subagente es un paso durable ───────────────────────────────────────

class _Corte(Exception):
    pass


def test_resuming_does_not_re_run_the_subagent():
    """Si el proceso muere a mitad, al reanudar el subagente no se repite.

    Antes el journal veía la delegación como una llamada, no como un run con sus
    propios pasos, así que se reejecutaba entera.
    """
    store = MemoryCheckpointer()
    supervisor = _supervisor(_analista())
    EJECUCIONES.clear()

    async def a_medias():
        gateway = FakeGateway(*[_responder] * 6, tools=[cotizacion])
        try:
            async for paso in supervisor.run("¿cómo va ACME?", session=Session("r1", gateway, store)):
                if isinstance(paso, DelegateStep) and paso.phase is Phase.COMPLETED:
                    raise _Corte
        except _Corte:
            pass

    asyncio.run(a_medias())
    assert EJECUCIONES == ["cotizacion"], "el subagente no llegó a trabajar"

    EJECUCIONES.clear()
    segundo = FakeGateway(*[_responder] * 6, tools=[cotizacion])
    _correr(supervisor, segundo, store=store)

    assert EJECUCIONES == [], "el subagente se reejecutó al reanudar"
    assert segundo.model_calls == 1, (
        f"se pagó inferencia de más: {segundo.model_calls} llamadas donde solo faltaba una"
    )


def test_the_subagent_gets_its_own_journal():
    """Su historial se audita donde está, no en el prompt del padre."""
    store = MemoryCheckpointer()
    _correr(_supervisor(_analista()), FakeGateway(*[_responder] * 6, tools=[cotizacion]), store=store)

    del_padre = asyncio.run(store.load("r1"))
    del_hijo = asyncio.run(store.load(sub_run_id("r1", "000001-delegate")))

    assert del_hijo.next_seq > 0, "el subagente no dejó diario"
    assert not any(e.kind == "tool" for e in del_padre.events), (
        "los pasos del hijo se colaron en el diario del padre"
    )


# ── Contexto aislado ──────────────────────────────────────────────────────────

def test_the_subagent_never_sees_the_parents_conversation():
    """Duplicar el contexto se paga dos veces y hereda los errores del primero."""
    vistos: list = []

    def espia(peticion):
        vistos.append(peticion)
        return _responder(peticion)

    _correr(_supervisor(_analista()), FakeGateway(*[espia] * 6, tools=[cotizacion]))

    # La petición del subagente es la que lleva sus instrucciones.
    del_hijo = next(p for p in vistos if p.system and "Analizas valores" in p.system)
    texto = " ".join(
        t.text for m in del_hijo.messages for p in m.content
        for t in (getattr(p, "content", None) or [p]) if hasattr(t, "text")
    )
    assert "Enrutas al especialista" not in (del_hijo.system or "")
    assert "¿cómo va ACME?" not in texto, "le llegó la tarea original del padre"
    assert "¿Cómo va ACME?" in texto, "no le llegó el brief"


def test_the_parents_stream_shows_a_delegation_not_the_childs_steps():
    """Quien mira el run del padre no debería leer la conversación ajena."""
    pasos = _correr(_supervisor(_analista()), FakeGateway(*[_responder] * 6, tools=[cotizacion]))

    tipos = {p.kind for p in pasos}
    assert "delegate" in tipos
    assert "tool" not in tipos, "los pasos del subagente se reenviaron al padre"


# ── Profundidad ───────────────────────────────────────────────────────────────

def test_a_delegation_cycle_is_stopped():
    """`max_steps` no lo cubre: cada nivel tiene su propio contador.

    Dos agentes que se deleguen mutuamente no terminarían nunca.
    """
    fondo = Agent("fondo", model="openai-compatible:m", instructions="Eres el fondo.")
    nivel = fondo
    for i in range(5):
        nivel = Agent(
            f"nivel{i}", model="openai-compatible:m", instructions="Delegas.",
            delegates=[nivel], limits=Limits(max_delegation_depth=2),
        )

    def siempre_delega(peticion):
        # Deja de insistir al recibir el error del tope, como haría un modelo.
        if any(m.role is Role.TOOL for m in peticion.messages):
            return says("no se puede seguir bajando")
        sub = next((t.name for t in peticion.tools), None)
        return calls(sub, brief="sigue") if sub else says("fondo")

    pasos = _correr(nivel, FakeGateway(*[siempre_delega] * 80))

    assert isinstance(pasos[-1], FinalStep), "el ciclo no terminó"
    profundidad = sum(1 for p in pasos if isinstance(p, DelegateStep) and p.phase is Phase.ATTEMPTED)
    assert profundidad <= 2, f"delegó {profundidad} niveles con un tope de 2"
