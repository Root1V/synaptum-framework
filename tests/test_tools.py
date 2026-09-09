"""SYN-20 · SYN-21 · SYN-65 · El decorador y el doble de desarrollo."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
from typing import Annotated, Any, Literal

import pytest

from synaptum import (
    Agent,
    ConfigurationError,
    Disposition,
    Decision,
    FinalStep,
    InvalidToolCallError,
    ProviderError,
    Risk,
    Session,
    ToolStep,
    ToolExecutionError,
    Usage,
    tool,
)
from synaptum.testing import FakeGateway, calls, says


# ── SYN-20 · El esquema sale de la firma ──────────────────────────────────────

def test_name_and_description_come_from_the_function():
    @tool
    def buscar(consulta: str) -> str:
        """Busca en el índice.

        Este párrafo no entra en la descripción.
        """
        return "ok"

    assert buscar.definition.name == "buscar"
    assert buscar.definition.description == "Busca en el índice."


def test_annotated_metadata_becomes_the_field_description():
    @tool
    def transferir(cuenta: Annotated[str, "IBAN de destino"], importe: float) -> str:
        """Transfiere."""
        return "ok"

    props = transferir.definition.parameters["properties"]
    assert props["cuenta"] == {"type": "string", "description": "IBAN de destino"}
    assert props["importe"] == {"type": "number"}


def test_a_parameter_without_default_is_required():
    @tool
    def leer(path: str, encoding: str = "utf-8") -> str:
        """Lee."""
        return "ok"

    schema = leer.definition.parameters
    assert schema["required"] == ["path"]
    assert schema["properties"]["encoding"]["default"] == "utf-8"
    assert schema["additionalProperties"] is False


def test_containers_literals_and_enums_translate():
    class Modo(str, enum.Enum):
        RAPIDO = "rapido"
        LENTO = "lento"

    @tool
    def procesar(
        ids: list[int],
        etiquetas: dict[str, str],
        prioridad: Literal["alta", "baja"],
        modo: Modo,
        nota: str | None = None,
    ) -> str:
        """Procesa."""
        return "ok"

    props = procesar.definition.parameters["properties"]
    assert props["ids"] == {"type": "array", "items": {"type": "integer"}}
    assert props["etiquetas"] == {"type": "object", "additionalProperties": {"type": "string"}}
    assert props["prioridad"] == {"enum": ["alta", "baja"]}
    assert props["modo"] == {"enum": ["rapido", "lento"]}
    assert props["nota"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_a_dataclass_parameter_becomes_a_nested_object():
    @dataclasses.dataclass
    class Filtro:
        campo: Annotated[str, "Columna"]
        desde: int = 0

    @tool
    def consultar(filtro: Filtro) -> str:
        """Consulta."""
        return "ok"

    nested = consultar.definition.parameters["properties"]["filtro"]
    assert nested["type"] == "object"
    assert nested["properties"]["campo"]["description"] == "Columna"
    assert nested["required"] == ["campo"]


# ── SYN-20 · Fallar al decorar, no al invocar ─────────────────────────────────

def test_an_untranslatable_type_fails_at_decoration_time():
    """Romper el arranque es mejor que romper una llamada ya pagada."""
    class Opaco:
        pass

    with pytest.raises(ConfigurationError, match="No sé traducir"):
        @tool
        def raro(x: Opaco) -> str:
            """Raro."""
            return "ok"


def test_a_parameter_without_an_annotation_fails_at_decoration_time():
    with pytest.raises(ConfigurationError, match="no tiene anotación"):
        @tool
        def sin_tipo(x) -> str:  # type: ignore[no-untyped-def]
            """Sin tipo."""
            return "ok"


def test_varargs_are_rejected():
    """El modelo solo sabe rellenar un esquema, no pasar posicionales."""
    with pytest.raises(ConfigurationError, match=r"\*args"):
        @tool
        def variadica(*args: int) -> str:
            """Variádica."""
            return "ok"


# ── SYN-21 · Riesgo e idempotencia se declaran, no se deducen ─────────────────

def test_risk_is_permissive_by_default_and_durability_conservative():
    """Quien calla obtiene el riesgo más inocuo y la garantía más cara."""
    @tool
    def leer(path: str) -> str:
        """Lee."""
        return "ok"

    assert leer.definition.risk is Risk.READ
    assert leer.definition.idempotent is False


def test_declaring_destructive_is_an_explicit_act():
    @tool(risk=Risk.DESTRUCTIVE)
    def borrar(path: str) -> str:
        """Borra."""
        return "ok"

    assert borrar.definition.risk is Risk.DESTRUCTIVE


# ── Invocación ────────────────────────────────────────────────────────────────

def test_a_tool_is_still_an_ordinary_function():
    @tool
    def sumar(a: int, b: int) -> int:
        """Suma."""
        return a + b

    assert sumar(2, 3) == 5


def test_a_failing_tool_returns_the_evidence_instead_of_raising():
    """Un modelo que no ve el error no puede corregirlo."""
    @tool
    def fallona(x: int) -> str:
        """Falla."""
        raise ToolExecutionError("el servicio está caído", tool="fallona")

    result = asyncio.run(fallona.invoke("c1", {"x": 1}))
    assert result.is_error is True
    assert "caído" in result.content[0].text


def test_bad_arguments_raise_because_the_model_cannot_fix_them():
    """No es algo que se arregle leyendo un mensaje: el esquema y la firma no cuadran."""
    @tool
    def leer(path: str) -> str:
        """Lee."""
        return "ok"

    with pytest.raises(InvalidToolCallError):
        asyncio.run(leer.invoke("c1", {"ruta": "/x"}))


def test_a_non_string_result_is_serialised_deterministically():
    @tool
    def contar(n: int) -> dict[str, int]:
        """Cuenta."""
        return {"b": 2, "a": 1}

    result = asyncio.run(contar.invoke("c1", {"n": 1}))
    assert result.content[0].text == '{"a":1,"b":2}'


def test_an_async_tool_is_awaited():
    @tool
    async def lento(x: int) -> str:
        """Lento."""
        await asyncio.sleep(0)
        return f"listo {x}"

    assert asyncio.run(lento.invoke("c1", {"x": 7})).content[0].text == "listo 7"


# ── SYN-65 · El doble de desarrollo ───────────────────────────────────────────

@tool(idempotent=True)
def leer_fichero(path: Annotated[str, "Ruta"]) -> str:
    """Lee un fichero."""
    return f"contenido de {path}"


def drain(agent: Agent, task: str, session: Session) -> list:
    async def go():
        return [e async for e in agent.run(task, session=session)]

    return asyncio.run(go())


def test_the_fake_runs_the_real_tool():
    gateway = FakeGateway(
        calls("leer_fichero", path="/x"), says("dice hola"), tools=[leer_fichero]
    )
    agent = Agent("a", model="fake:m", tools=[leer_fichero])
    events = drain(agent, "lee /x", Session("run-1", gateway))

    result = next(e for e in events if isinstance(e, ToolStep) and e.result is not None)
    assert result.result.content[0].text == "contenido de /x"
    assert events[-1].output == "dice hola"


def test_the_agent_accepts_a_decorated_tool_directly():
    """Sin que el bucle tenga que importar el decorador."""
    agent = Agent("a", model="fake:m", tools=[leer_fichero])
    assert agent.tools[0].name == "leer_fichero"
    assert agent.tools[0].idempotent is True


def test_an_unknown_tool_comes_back_as_an_error_the_model_can_read():
    gateway = FakeGateway(calls("inexistente"), says("vale, no existe"), tools=[leer_fichero])
    events = drain(Agent("a", model="fake:m"), "usa algo", Session("run-1", gateway))

    result = next(e for e in events if isinstance(e, ToolStep) and e.result is not None)
    assert result.result.is_error is True
    assert "leer_fichero" in result.result.content[0].text


def test_a_string_in_the_script_is_shorthand_for_an_answer():
    events = drain(Agent("a", model="fake:m"), "hola", Session("run-1", FakeGateway("42")))
    assert events[-1].output == "42"


def test_the_script_can_inject_errors_and_denials():
    retried = FakeGateway(ProviderError("saturado", status=429), "a la segunda")
    assert drain(Agent("a", model="fake:m"), "x", Session("run-1", retried))[-1].output == "a la segunda"

    denied = FakeGateway(Decision(Disposition.TERMINATE_RUN, reason_code="budget.exhausted"))
    final = drain(Agent("a", model="fake:m"), "x", Session("run-2", denied))[-1]
    assert isinstance(final, FinalStep)
    assert final.meta["reason_code"] == "budget.exhausted"


def test_a_script_item_can_react_to_what_the_loop_sent():
    gateway = FakeGateway(lambda request: says(f"me llegó: {request.messages[-1].text}"))
    events = drain(Agent("a", model="fake:m"), "ping", Session("run-1", gateway))
    assert events[-1].output == "me llegó: ping"


def test_an_exhausted_script_says_so_clearly():
    gateway = FakeGateway(calls("leer_fichero", path="/x"), tools=[leer_fichero])
    with pytest.raises(AssertionError, match="se agotó"):
        drain(Agent("a", model="fake:m", tools=[leer_fichero]), "lee", Session("run-1", gateway))


def test_the_default_usage_is_measured_zero_not_unmeasured():
    """Un fake que devolviera None en todo haría creer que la economía no funciona."""
    events = drain(Agent("a", model="fake:m"), "hola", Session("run-1", FakeGateway("ok")))
    usage = events[-1].usage
    assert usage.input == 100 and usage.cache_write == 0
    assert usage.total is not None


def test_an_unreported_counter_can_be_simulated():
    """El caso Prometheus: input, output y cache_read, y nada más."""
    prometheus = Usage(input=100, output=20, cache_read=80, estimated=True)
    events = drain(
        Agent("a", model="fake:m"), "hola",
        Session("run-1", FakeGateway(says("ok", usage=prometheus))),
    )
    total = events[-1].usage
    assert total.cache_write is None, "un tramo no lo midió: el total no se sabe"
    assert total.estimated is True


# ── H2 · Streaming y cancelación ──────────────────────────────────────────────

def test_the_stream_emits_the_start_delta_end_cycle_and_finishes_with_usage():
    gateway = FakeGateway(says("hola mundo, esto va troceado"), chunk_size=4)

    async def go():
        from synaptum import CallContext, Request

        ctx = CallContext(run_id="run-1", step_id="000000-model")
        return [e async for e in gateway.stream_model(Request(model="fake:m"), ctx)]

    events = asyncio.run(go())
    kinds = [e.kind for e in events]
    assert kinds[0] == "stream_start" and kinds[-1] == "finish"
    assert "text_start" in kinds and "text_delta" in kinds and "text_end" in kinds
    assert events[-1].response.usage.input == 100


def test_closing_the_iterator_stops_the_generation():
    """Cerrar el iterador *es* la señal: no hace falta un evento de cancelación.

    Es la misma propiedad que el harness mide como «upstream stopped after N of
    M chunks» y que Axonium mide contra Prometheus.  Aquí se comprueba nuestro
    tramo, que es el primero de los tres.
    """
    largo = "x" * 400
    gateway = FakeGateway(says(largo), chunk_size=4)

    async def go():
        from synaptum import CallContext, Request

        ctx = CallContext(run_id="run-1", step_id="000000-model")
        stream = gateway.stream_model(Request(model="fake:m"), ctx)
        seen = 0
        async for event in stream:
            if event.kind == "text_delta":
                seen += 1
                if seen == 2:
                    break
        await stream.aclose()
        return seen

    seen = asyncio.run(go())
    assert seen == 2
    assert gateway.cancelled is True, "el generador supo que lo cerraron"
    assert gateway.chunks_emitted < 100 // 4, "dejó de producir muy por debajo del total"
