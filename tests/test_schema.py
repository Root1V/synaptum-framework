"""SYN-16 · Salida estructurada sin atarse a una librería de modelos."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Annotated, Literal

import pytest

from synaptum import (
    Agent,
    MemoryCheckpointer,
    ConfigurationError,
    FinalStep,
    NoObjectGeneratedError,
    ProviderError,
    Role,
    Schema,
    Session,
    schema_for,
    tool,
)
from synaptum.schema.protocol import DataclassSchema, RawSchema
from synaptum.testing import FakeGateway, calls, says


@dataclasses.dataclass
class Evaluacion:
    veredicto: Annotated[Literal["aprueba", "rechaza"], "Decisión final"]
    puntuacion: int
    motivos: list[str] = dataclasses.field(default_factory=list)


def drain(agent: Agent, task: str, session: Session) -> list:
    async def go():
        return [e async for e in agent.run(task, session=session)]

    return asyncio.run(go())


# ── Resolución ────────────────────────────────────────────────────────────────

def test_a_dataclass_resolves_without_any_third_party_library():
    resolved = schema_for(Evaluacion)
    assert isinstance(resolved, DataclassSchema)

    schema = resolved.json_schema()
    assert schema["type"] == "object"
    assert schema["properties"]["veredicto"]["enum"] == ["aprueba", "rechaza"]
    assert schema["properties"]["veredicto"]["description"] == "Decisión final"
    assert schema["required"] == ["veredicto", "puntuacion"], "motivos tiene default"


def test_a_raw_json_schema_is_accepted_as_is():
    resolved = schema_for({"type": "object", "properties": {"n": {"type": "integer"}}})
    assert isinstance(resolved, RawSchema)
    assert resolved.json_schema()["properties"]["n"] == {"type": "integer"}


def test_a_raw_schema_does_not_validate_and_says_so():
    """No hay validador en la librería estándar y no se arrastra uno para esto."""
    resolved = schema_for({"type": "object", "required": ["n"]})
    assert resolved.validate({"otra": "cosa"}) == {"otra": "cosa"}


def test_anything_implementing_the_protocol_passes_through():
    class Propio:
        def json_schema(self):
            return {"type": "string"}

        def validate(self, data):
            return str(data).upper()

        def dump(self, obj):
            return obj

    propio = Propio()
    assert isinstance(propio, Schema), "estructural: no hereda nada"
    assert schema_for(propio) is propio


def test_an_incomplete_schema_is_not_accepted_as_one():
    """Sin ``dump`` el journal no puede guardar la salida, así que no vale a medias."""

    class Incompleto:
        def json_schema(self):
            return {"type": "string"}

        def validate(self, data):
            return data

    assert not isinstance(Incompleto(), Schema)
    with pytest.raises(ConfigurationError):
        schema_for(Incompleto())


def test_something_that_is_not_a_schema_fails_clearly():
    with pytest.raises(ConfigurationError, match="No sé usar"):
        schema_for(42)


def test_pydantic_is_detected_without_importing_it():
    """Preguntar por Pydantic no debe convertirlo en dependencia dura."""
    from synaptum.schema.protocol import _is_pydantic_model

    class Impostor:
        @staticmethod
        def model_validate(data):
            return data

        @staticmethod
        def model_json_schema():
            return {"type": "object"}

    assert _is_pydantic_model(Impostor) is True
    assert _is_pydantic_model(Evaluacion) is False


# ── Validación ────────────────────────────────────────────────────────────────

def test_the_journal_keeps_the_serialisable_form_and_the_caller_the_object():
    """La salida tipada se deriva, no se almacena — como el contexto."""
    store = MemoryCheckpointer()
    agent = Agent("evaluador", model="fake:m", output=Evaluacion)
    gateway = FakeGateway('{"veredicto": "aprueba", "puntuacion": 9}')
    events = drain(agent, "evalúa", Session("run-1", gateway, store))
    assert isinstance(events[-1].output, Evaluacion)

    recorded = asyncio.run(store.load("run-1")).events[-1]
    assert isinstance(recorded.output, dict), "en disco va la forma serializable"
    assert recorded.output["puntuacion"] == 9


def test_resuming_a_closed_run_gives_back_the_typed_object():
    """Sin rehidratar, la misma llamada devolvería un dict o un objeto según
    hubiera corrido antes — el tipo dependiendo del historial."""
    store = MemoryCheckpointer()
    agent = Agent("evaluador", model="fake:m", output=Evaluacion)
    drain(agent, "evalúa", Session("run-1", FakeGateway('{"veredicto": "rechaza", "puntuacion": 2}'), store))

    events = drain(agent, "evalúa", Session("run-1", FakeGateway(), store))
    assert isinstance(events[-1].output, Evaluacion)
    assert events[-1].output.veredicto == "rechaza"


def test_a_dataclass_schema_builds_the_typed_object():
    resolved = schema_for(Evaluacion)
    resultado = resolved.validate(
        {"veredicto": "aprueba", "puntuacion": 8, "motivos": ["solvencia", "historial"]}
    )
    assert isinstance(resultado, Evaluacion)
    assert resultado.veredicto == "aprueba"
    assert resultado.motivos == ["solvencia", "historial"]


def test_a_missing_field_is_a_retryable_failure():
    """El muestreo es estocástico: una segunda pasada suele acertar."""
    with pytest.raises(NoObjectGeneratedError) as caught:
        schema_for(Evaluacion).validate({"puntuacion": 8})
    assert caught.value.retryable is True


def test_the_offending_output_is_kept_for_debugging():
    with pytest.raises(NoObjectGeneratedError) as caught:
        schema_for(Evaluacion).validate({"puntuacion": 8})
    assert "puntuacion" in caught.value.raw


# ── En el bucle ───────────────────────────────────────────────────────────────

def test_the_request_carries_the_schema_and_the_final_step_the_object():
    gateway = FakeGateway('{"veredicto": "aprueba", "puntuacion": 9}')
    agent = Agent("evaluador", model="fake:m", output=Evaluacion)
    events = drain(agent, "evalúa", Session("run-1", gateway))

    enviado = gateway.requests[0].response_format
    assert enviado is not None
    assert enviado.kind == "json_schema"
    assert enviado.name == "Evaluacion"
    assert enviado.schema["properties"]["puntuacion"] == {"type": "integer"}

    final = events[-1]
    assert isinstance(final, FinalStep)
    assert isinstance(final.output, Evaluacion), "el cierre entrega el objeto, no el texto"
    assert final.output.puntuacion == 9


def test_an_agent_without_output_still_returns_text():
    gateway = FakeGateway("una respuesta cualquiera")
    events = drain(Agent("a", model="fake:m"), "hola", Session("run-1", gateway))
    assert gateway.requests[0].response_format is None
    assert events[-1].output == "una respuesta cualquiera"


def test_a_malformed_object_is_retried_and_the_second_pass_wins():
    """La taxonomía ya decía que esto es reintentable; el bucle solo la obedece."""
    gateway = FakeGateway(
        "esto no es JSON",
        '{"veredicto": "rechaza", "puntuacion": 3}',
    )
    agent = Agent("evaluador", model="fake:m", output=Evaluacion)
    events = drain(agent, "evalúa", Session("run-1", gateway))

    assert gateway.model_calls == 2
    assert events[-1].output.veredicto == "rechaza"


def test_a_malformed_object_is_asked_again_showing_the_error():
    """Repetir la misma petición no es reintentar.

    Antes, una salida que no validaba hacía repetir el request **byte a byte**
    esperando otra suerte del muestreo, y el modelo nunca llegaba a ver qué
    había fallado. Ahora se le enseña el error, que es lo único que le permite
    corregir.
    """
    gateway = FakeGateway("no es JSON", '{"veredicto": "aprueba", "puntuacion": 7}')
    agent = Agent("evaluador", model="fake:m", output=Evaluacion)

    eventos = drain(agent, "evalúa", Session("run-1", gateway))

    correctivo = [m for m in gateway.requests[-1].messages if m.role is Role.USER][-1]
    assert "no es JSON" in correctivo.text, "el turno correctivo no enseña qué volvió"

    final = next(e for e in eventos if isinstance(e, FinalStep))
    assert (final.output.veredicto, final.output.puntuacion) == ("aprueba", 7)


def test_a_persistently_malformed_object_gives_up_with_the_evidence():
    """Y se rinde diciendo **cuál de los dos fallos** fue.

    «Nunca llegó a entregar» y «entregó y ninguna validó» se arreglan en sitios
    distintos —el prompt y el esquema— así que se distinguen por tipo y no por
    el texto del mensaje.
    """
    from synaptum.agent.agent import Limits

    gateway = FakeGateway(*["no es JSON"] * 10)
    agent = Agent("evaluador", model="fake:m", output=Evaluacion, limits=Limits(max_steps=4))

    with pytest.raises(NoObjectGeneratedError) as caught:
        drain(agent, "evalúa", Session("run-1", gateway))

    assert caught.value.raw == "no es JSON"
    assert "4 vez/veces" in str(caught.value) or "veces" in str(caught.value)


def test_the_schema_is_not_enforced_on_a_turn_that_calls_tools():
    """Un turno que pide herramientas no es la respuesta final: no hay objeto que validar."""

    @tool(idempotent=True)
    def leer(path: str) -> str:
        """Lee."""
        return "contenido"

    gateway = FakeGateway(
        calls("leer", path="/x"),
        says('{"veredicto": "aprueba", "puntuacion": 7}'),
        tools=[leer],
    )
    agent = Agent("evaluador", model="fake:m", tools=[leer], output=Evaluacion)
    events = drain(agent, "lee y evalúa", Session("run-1", gateway))

    assert gateway.tool_calls == 1
    assert events[-1].output.puntuacion == 7


def test_a_provider_error_is_still_classified_by_its_own_rule():
    """Pedir salida estructurada no cambia cómo se clasifican los errores del proveedor."""
    gateway = FakeGateway(ProviderError("mal formada", status=422))
    agent = Agent("evaluador", model="fake:m", output=Evaluacion)

    with pytest.raises(ProviderError):
        drain(agent, "evalúa", Session("run-1", gateway))
    assert gateway.model_calls == 1, "un 4xx no se reintenta"


# ── S-2 · entrega por herramienta ────────────────────────────────────────────
#
# Pedido por Veritium y decidido por ellos: el modo *submit tool* **sustituye**
# a `response_format` en vez de acompañarlo. Dos restricciones que piden lo
# mismo pueden divergir, y con algunos motores una gramática de salida en la
# misma petición que un catálogo de herramientas impide que el modelo emita
# tool calls — la forma de entregar estorbando a la de trabajar.

def test_the_submit_tool_replaces_the_response_format():
    agente = Agent("extractor", model="fake:m", output=Evaluacion, submit_tool=True)

    assert agente._format is None, "no se envían las dos restricciones"
    submit = next(t for t in agente.tools if t.name == "submit")
    assert set(submit.parameters["properties"]) >= {"veredicto", "puntuacion"}
    assert "json" not in (agente.instructions or "").lower(), (
        "el esquema ya viaja como parámetros de la tool; repetirlo en el prompt "
        "sería la segunda restricción que puede divergir"
    )


def test_an_invalid_submit_produces_a_corrective_turn_and_the_second_one_ends_the_run():
    """El criterio de aceptación de Veritium, literal."""
    from synaptum import Phase, ToolStep

    def responder(peticion):
        entregas = sum(1 for m in peticion.messages if m.role is Role.TOOL)
        if entregas == 0:
            return calls("submit", id="s1", veredicto="aprueba")      # falta puntuacion
        return calls("submit", id="s2", veredicto="aprueba", puntuacion=7)

    agente = Agent("evaluador", model="fake:m", output=Evaluacion, submit_tool=True)
    store = MemoryCheckpointer()
    eventos = drain(agente, "evalúa", Session("run-1", FakeGateway(*[responder] * 6), store))

    entregas = [
        e for e in eventos
        if isinstance(e, ToolStep) and e.phase is Phase.COMPLETED and e.call.name == "submit"
    ]
    assert len(entregas) == 2, "una entrega fallida y una buena"
    assert entregas[0].result.is_error
    assert "no valida" in entregas[0].result.content[0].text
    assert not entregas[1].result.is_error

    final = next(e for e in eventos if isinstance(e, FinalStep))
    assert (final.output.veredicto, final.output.puntuacion) == ("aprueba", 7), (
        "el objeto validado también llega al FinalStep, no solo al ToolResult"
    )

    # (3) todo queda en el journal
    import asyncio

    estado = asyncio.run(store.load("run-1"))
    pasos = [e.step_id for e in estado.events]
    assert pasos.count("000001-tool") == 2, "la entrega fallida también se registra"
    assert any(e.step_id.endswith("-final") for e in estado.events)


def test_a_run_that_never_submits_fails_differently_from_one_that_never_validates():
    """Dos diagnósticos distintos, separados **por tipo** y no por el texto.

    Uno se arregla con el prompt y el otro con el esquema, así que quien atrapa
    el error suele querer actuar distinto — y leer un mensaje para decidirlo es
    la clase de contrato que se rompe al reescribirlo.
    """
    from synaptum import LimitExceeded
    from synaptum.agent.agent import Limits

    callado = Agent(
        "evaluador", model="fake:m", output=Evaluacion, submit_tool=True,
        limits=Limits(max_steps=3),
    )
    with pytest.raises(LimitExceeded, match="nunca llegó a entregar"):
        drain(callado, "evalúa", Session("r1", FakeGateway(*["hablo pero no entrego"] * 9)))

    def siempre_mal(_peticion):
        return calls("submit", id="s", veredicto="aprueba")        # nunca valida

    insistente = Agent(
        "evaluador", model="fake:m", output=Evaluacion, submit_tool=True,
        limits=Limits(max_steps=3),
    )
    with pytest.raises(NoObjectGeneratedError) as fallo:
        drain(insistente, "evalúa", Session("r2", FakeGateway(*[siempre_mal] * 9)))
    assert "ninguna validó" in str(fallo.value)
    assert fallo.value.raw, "el error lleva dentro lo último que no validó"
