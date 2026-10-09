"""S-5 · El diario vive en el arnés, y el bucle no se entera.

Lo que de verdad se comprueba aquí no es que el HTTP funcione: es que **dos
sub-runs del mismo padre conserven cada uno su diario**. El primer paso de
cualquier run se llama `000000-model`, así que si el sub-run no entra en la
clave, el segundo se registra como duplicado — y un duplicado es un no-op
silencioso, no un error. Esa es la razón de que el identificador se parta.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import pytest

sys.path.insert(0, str(Path(__file__).parent / "servidores"))

from diario_http import Diario  # noqa: E402

from synaptum import (  # noqa: E402
    Agent,
    ConfigurationError,
    HttpCheckpointer,
    ModelStep,
    Phase,
    ProviderError,
    Risk,
    Role,
    Session,
    tool,
)
from synaptum.run.http import partir  # noqa: E402
from synaptum.testing import FakeGateway, calls, says  # noqa: E402


@pytest.fixture
def diario():
    servidor = Diario()
    yield servidor
    servidor.cerrar()


def test_the_parent_goes_in_the_path_and_the_sub_run_in_the_body():
    """La barra de un sub-run no viaja en la ruta, ni cruda ni escapada.

    Cruda no enruta —el patrón casa un segmento— y `%2F` funciona hoy y es lo
    que normalizan o rechazan los proxies: pasa en el test y falla detrás de un
    gateway, que es la peor forma de fallo disponible.
    """
    assert partir("r1") == ("r1", "")
    assert partir("r1/000001-delegate") == ("r1", "000001-delegate")
    # Un nieto conserva la ruta entera: la delegación anida y esto es una ruta,
    # no un identificador de paso.
    assert partir("r1/000001-delegate/000002-delegate") == (
        "r1", "000001-delegate/000002-delegate",
    )


def test_a_top_level_run_omits_the_key_instead_of_sending_it_empty(diario):
    """«No sé de sub-runs» y «digo que no hay» son cosas distintas.

    La columna tiene su valor por defecto para lo primero; mandar `""` afirma
    lo segundo.
    """
    store = HttpCheckpointer(diario.url)
    asyncio.run(store.append("r1", ModelStep(
        run_id="r1", step_id="000000-model", step_seq=0, phase=Phase.ATTEMPTED,
    )))

    assert "sub_run_id" not in diario.peticiones[0]


def test_two_sub_runs_of_one_run_keep_their_own_journals(diario):
    """El caso por el que pedimos que el sub-run entre en la clave primaria."""
    store = HttpCheckpointer(diario.url)

    for sub in ("r1/000001-delegate", "r1/000003-delegate"):
        asyncio.run(store.append(sub, ModelStep(
            run_id=sub, step_id="000000-model", step_seq=0, phase=Phase.COMPLETED,
        )))

    a = asyncio.run(store.load("r1/000001-delegate"))
    b = asyncio.run(store.load("r1/000003-delegate"))

    assert len(a.events) == 1 and len(b.events) == 1, (
        "un sub-run se tragó al otro: los dos pasos se llaman 000000-model"
    )
    assert a.events[0].run_id == "r1/000001-delegate"
    assert b.events[0].run_id == "r1/000003-delegate"


def test_a_duplicate_is_reported_as_such_and_not_as_an_error(diario):
    store = HttpCheckpointer(diario.url)
    paso = ModelStep(
        run_id="r1", step_id="000000-model", step_seq=0, phase=Phase.ATTEMPTED,
    )

    primera = asyncio.run(store.append("r1", paso))
    segunda = asyncio.run(store.append("r1", paso))

    assert not primera.duplicate
    assert segunda.duplicate, "un append repetido es un no-op, no un error"
    assert segunda.seq == primera.seq
    assert not segunda.payload_diverged


def test_a_run_survives_the_process_through_the_harness_journal(diario):
    """De punta a punta: el diario está al otro lado de una red y el bucle no lo sabe.

    Es la misma propiedad que con SQLite —reanudar no vuelve a pagar la
    inferencia— y por eso el test es el mismo: lo que cambia es el relleno de la
    costura.
    """
    ejecuciones: list[str] = []

    @tool(risk=Risk.HARD_WRITE)
    async def pagar(importe: Annotated[int, "Importe"]) -> str:
        """Paga."""
        ejecuciones.append(importe)
        return "ok"

    def responder(peticion):
        if any(m.role is Role.TOOL for m in peticion.messages):
            return says("Pago confirmado.")
        return calls("pagar", importe=100)

    agente = Agent("a", model="fake:m", instructions="Paga.", tools=[pagar])

    primera = FakeGateway(*[responder] * 6, tools=[pagar])
    asyncio.run(_agotar(agente, Session("r1", primera, HttpCheckpointer(diario.url))))
    assert ejecuciones == [100]
    assert primera.model_calls == 2

    ejecuciones.clear()
    segunda = FakeGateway(*[responder] * 6, tools=[pagar])
    asyncio.run(_agotar(agente, Session("r1", segunda, HttpCheckpointer(diario.url))))

    assert ejecuciones == [], "se repitió un pago que el diario ya registraba"
    assert segunda.model_calls == 0, "se volvió a pagar una inferencia ya pagada"


async def _agotar(agente, sesion):
    return [p async for p in agente.run("paga 100", session=sesion)]


def test_the_token_travels_and_its_absence_is_an_error_you_can_read():
    servidor = Diario(exige_token="s3cr3t")
    try:
        con = HttpCheckpointer(servidor.url, token="s3cr3t")
        asyncio.run(con.append("r1", ModelStep(
            run_id="r1", step_id="000000-model", step_seq=0, phase=Phase.ATTEMPTED,
        )))

        sin = HttpCheckpointer(servidor.url)
        with pytest.raises(ProviderError) as fallo:
            asyncio.run(sin.append("r1", ModelStep(
                run_id="r1", step_id="000001-model", step_seq=1, phase=Phase.ATTEMPTED,
            )))
        assert fallo.value.status == 401
    finally:
        servidor.cerrar()


def test_a_404_says_where_to_look(diario):
    """El 404 más probable no es «ese run no existe»: es apuntar al servicio
    que arranca runs en vez de al que posee el diario. Comparten prefijo."""
    store = HttpCheckpointer(f"{diario.url}/otro-servicio")

    with pytest.raises(ProviderError, match="dueño del diario"):
        asyncio.run(store.append("r1", ModelStep(
            run_id="r1", step_id="000000-model", step_seq=0, phase=Phase.ATTEMPTED,
        )))


def test_an_absurdly_long_sub_run_is_refused_before_the_network():
    with pytest.raises(ConfigurationError, match="512"):
        partir("r1/" + "x" * 600)
