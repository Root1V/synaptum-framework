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


def test_a_persistently_malformed_object_gives_up_with_the_evidence():
    from synaptum.agent.agent import Limits

    gateway = FakeGateway(*["no es JSON"] * 4)
    agent = Agent("evaluador", model="fake:m", output=Evaluacion, limits=Limits(max_retries=2))

    with pytest.raises(NoObjectGeneratedError) as caught:
        drain(agent, "evalúa", Session("run-1", gateway))
    assert caught.value.raw == "no es JSON"
    assert gateway.model_calls == 3, "el intento inicial más dos reintentos"


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
