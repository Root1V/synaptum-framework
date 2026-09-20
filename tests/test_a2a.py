"""SYN-44 · Delegar en un agente remoto por A2A.

Contra un servidor A2A de verdad, arrancado por el test. Un doble de nuestro
propio cliente comprobaría que llamamos a nuestras funciones, que es lo único
que no hace falta comprobar — y ya nos costó una vez creer que un verde probaba
algo.

Lo que se prueba es lo que decide el diseño: que la reanudación sea una consulta
y no una apuesta, y que un estado que no conocemos no se dé por terminado.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from servidores.a2a_demo import Servidor  # noqa: E402

from synaptum import Agent, MemoryCheckpointer, Risk, Role, Session, Usage  # noqa: E402
from synaptum.a2a import A2AClient, RemoteDelegate, TaskState  # noqa: E402
from synaptum.a2a.wire import deterministic_message_id, task_from_wire  # noqa: E402
from synaptum.testing import FakeGateway, calls, says  # noqa: E402


@pytest.fixture
def servidor():
    creados: list[Servidor] = []

    def montar(**kw):
        s = Servidor(**kw)
        creados.append(s)
        return s

    yield montar
    for s in creados:
        s.cerrar()


# ── Normalización, sin servidor ───────────────────────────────────────────────

def test_an_unknown_state_is_never_treated_as_terminal():
    """La especificación tiene extensiones desde la v1.0.1: habrá estados nuevos.

    Dar por terminada una tarea que no lo está pierde su resultado en silencio,
    que es el peor de los dos errores posibles.
    """
    tarea = task_from_wire({"result": {"id": "t", "status": {"state": "algo_nuevo"}}})

    assert tarea.state is TaskState.WORKING
    assert not tarea.terminal


def test_the_grpc_style_prefix_is_accepted():
    """Un servidor generado desde los `.proto` responde `TASK_STATE_COMPLETED`."""
    tarea = task_from_wire({"result": {"id": "t", "status": {"state": "TASK_STATE_COMPLETED"}}})
    assert tarea.state is TaskState.COMPLETED


def test_artifacts_win_over_the_last_message():
    """Un artefacto es el entregable; un mensaje es conversación."""
    tarea = task_from_wire({"result": {
        "id": "t", "status": {"state": "completed", "message": {"parts": [{"text": "ya está"}]}},
        "artifacts": [{"parts": [{"text": "187,34 USD"}]}],
    }})
    assert tarea.result == "187,34 USD"


def test_the_message_id_is_deterministic_and_carries_no_attempt_counter():
    """Un contador lo haría no determinista, y reanudar generaría trabajo nuevo siempre."""
    uno = deterministic_message_id("run-1/000001-delegate")
    otro = deterministic_message_id("run-1/000001-delegate")

    assert uno == otro
    assert uno != deterministic_message_id("run-1/000002-delegate")


# ── El cliente, contra un servidor ────────────────────────────────────────────

def test_the_agent_card_is_read(servidor):
    s = servidor()
    tarjeta = asyncio.run(A2AClient(s.url).agent_card())

    assert tarjeta.name == "analista"
    assert tarjeta.skills


def test_the_context_id_we_send_is_the_one_that_comes_back(servidor):
    """Es la pieza de la que depende reencontrar la tarea."""
    s = servidor()

    async def ir():
        c = A2AClient(s.url)
        return await c.send_message("t", context_id="r/000001-delegate", message_id="m1")

    assert asyncio.run(ir()).context_id == "r/000001-delegate"
    assert s.enviados[0]["contextId"] == "r/000001-delegate"


# ── Reconciliar: la consulta, no la apuesta ───────────────────────────────────

def _delegado(url, **kw):
    return RemoteDelegate(name="analista", url=url, risk=Risk.READ, poll_every=0.01, **kw)


def test_a_first_run_sends_the_message(servidor):
    s = servidor()
    salida, consumo = asyncio.run(
        _delegado(s.url).execute("¿Cómo va ACME?", None, "r/000001-delegate")
    )

    assert "187,34" in salida
    assert len(s.enviados) == 1
    assert consumo.input == 120, "el consumo que el remoto reportó no subió"


def test_resuming_finds_the_task_instead_of_sending_again(servidor):
    """La propiedad que justifica el diseño.

    El `taskId` lo asigna el servidor y tras una caída no lo tenemos. El
    `contextId` lo ponemos nosotros, así que se pregunta en vez de apostar — y
    esto funciona **aunque el servidor no deduplique** por `messageId`, porque
    ya no dependemos de que lo haga.
    """
    s = servidor()
    contexto = "r/000001-delegate"

    asyncio.run(_delegado(s.url).execute("¿Cómo va ACME?", None, contexto))
    enviados_tras_la_primera = len(s.enviados)

    # Segunda vuelta: el padre reanuda y vuelve a entrar en la delegación.
    salida, _ = asyncio.run(_delegado(s.url).execute("¿Cómo va ACME?", None, contexto))

    assert len(s.enviados) == enviados_tras_la_primera, (
        "se envió una segunda tarea: la delegación se pagó dos veces"
    )
    assert "187,34" in salida


def test_without_tasks_list_it_falls_back_instead_of_breaking(servidor):
    """No todo servidor implementa `tasks/list`. Bajar un peldaño, no reventar."""
    s = servidor(con_list=False)

    salida, _ = asyncio.run(
        _delegado(s.url).execute("¿Cómo va ACME?", None, "r/000001-delegate")
    )

    assert "187,34" in salida
    assert s.enviados[0]["messageId"] == deterministic_message_id("r/000001-delegate"), (
        "sin `tasks/list`, el `messageId` determinista es lo único que queda"
    )


def test_two_live_tasks_for_one_context_do_not_get_guessed(servidor):
    """Si no se puede saber cuál es la nuestra, no se elige por nosotros."""
    from synaptum import UncertainEffect

    s = servidor(turnos_trabajando=99)
    contexto = "r/000001-delegate"
    asyncio.run(A2AClient(s.url).send_message("a", context_id=contexto, message_id="m1"))
    asyncio.run(A2AClient(s.url).send_message("b", context_id=contexto, message_id="m2"))

    with pytest.raises(UncertainEffect):
        asyncio.run(_delegado(s.url).execute("t", None, contexto))


# ── Los ocho estados ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("estado", ["failed", "rejected", "canceled"])
def test_a_failure_comes_back_as_evidence_not_as_a_crash(servidor, estado):
    """Un modelo que no ve el error no puede corregirlo."""
    s = servidor(estado_final=estado)
    salida, _ = asyncio.run(_delegado(s.url).execute("t", None, "r/000001-delegate"))

    assert estado in salida
    assert "no completó" in salida


@pytest.mark.parametrize("estado", ["input_required", "auth_required"])
def test_a_waiting_task_is_reported_as_waiting_not_as_done(servidor, estado):
    """Es nuestro `ApprovalStep` del otro lado de la red.

    Darlo por terminado haría que el padre siguiera con una respuesta que el
    remoto nunca dio.
    """
    s = servidor(estado_final=estado)
    salida, _ = asyncio.run(_delegado(s.url).execute("t", None, "r/000001-delegate"))

    assert "esperando" in salida and estado in salida


def test_usage_that_nobody_reported_stays_unmeasured(servidor):
    """A2A no define un campo de consumo. Cero diría «no costó nada»."""
    from synaptum.a2a.delegate import _consumo

    assert _consumo(task_from_wire({"result": {"id": "t"}})) == Usage()


# ── En el bucle ───────────────────────────────────────────────────────────────

def test_the_loop_cannot_tell_a_remote_delegate_from_a_local_one(servidor):
    """El paso sigue siendo durable, el consumo sigue subiendo, el riesgo viaja.

    Es lo que hace que esto valga la pena: nada del bucle cambia porque el
    subagente viva en otra parte.
    """
    s = servidor()
    remoto = RemoteDelegate(name="analista", url=s.url, risk=Risk.READ, poll_every=0.01)
    supervisor = Agent("mesa", model="openai-compatible:m", instructions="Enrutas.",
                       delegates=[remoto])

    def responder(peticion):
        if any(m.role is Role.TOOL for m in peticion.messages):
            return says("El analista dice 187,34 USD.")
        return calls("analista", brief="¿Cómo va ACME?")

    store = MemoryCheckpointer()

    async def ir():
        return [
            paso
            async for paso in supervisor.run(
                "¿cómo va ACME?", session=Session("r1", FakeGateway(*[responder] * 6), store)
            )
        ]

    pasos = asyncio.run(ir())
    delegacion = next(p for p in pasos if p.kind == "delegate" and p.phase.value == "completed")

    assert delegacion.usage.input == 120, "el consumo del remoto no llegó al paso"
    assert pasos[-1].usage.input > 120, "el total del padre no incluye al remoto"
    assert asyncio.run(store.load("r1")).next_seq > 0, "la delegación remota no dejó diario"
