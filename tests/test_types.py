"""RM-01 · Las garantías del vocabulario unificado.

No se prueban los dataclasses — se prueban las promesas que otros equipos van a
dar por ciertas al implementar la especificación de normalización.
"""

from __future__ import annotations

import json

import pytest

from synaptum import (
    FinishReason,
    Image,
    Message,
    Request,
    Response,
    ResponseFormat,
    Role,
    Text,
    ToolCall,
    ToolChoice,
    ToolResult,
    Usage,
    dumps,
    to_jsonable,
)


# ── El mensaje de sistema no es un mensaje ────────────────────────────────────

def test_system_message_is_rejected_in_the_message_list():
    """Ningún proveedor trata el system como turno; el tipo lo impide."""
    with pytest.raises(ValueError, match="Request.system"):
        Request(
            model="p:m",
            messages=(Message(Role.SYSTEM, (Text("eres un agente"),)),),
        )


def test_system_travels_in_its_own_field():
    req = Request(model="p:m", system="eres un agente", messages=(Message.user("hola"),))
    assert req.system == "eres un agente"
    assert all(m.role is not Role.SYSTEM for m in req.messages)


# ── Usage — H3 ────────────────────────────────────────────────────────────────

def test_usage_accumulates_all_five_counters():
    a = Usage(input=10, output=5, reasoning=3, cache_read=100, cache_write=20)
    b = Usage(input=1, output=2, reasoning=4, cache_read=8, cache_write=16)
    total = a + b
    assert (total.input, total.output, total.reasoning) == (11, 7, 7)
    assert (total.cache_read, total.cache_write) == (108, 36)


def test_usage_total_counts_reasoning_tokens():
    """El razonamiento se factura; omitirlo subestima el coste del turno."""
    assert Usage(input=10, output=5, reasoning=100).total == 115


def test_cache_hit_ratio_is_the_input_served_from_cache():
    assert Usage(input=25, cache_read=75).cache_hit_ratio == 0.75
    assert Usage(input=10, cache_read=0).cache_hit_ratio == 0.0


# ── Medido, derivado y sin medir son tres cosas distintas ─────────────────────

def test_an_unmeasured_counter_is_none_not_zero():
    """Cero dice «no hubo»; la verdad puede ser «nadie lo midió»."""
    prometheus = Usage(input=100, output=20, cache_read=80)
    assert prometheus.cache_write is None, "llama.cpp no reporta escritura de caché"
    assert prometheus.reasoning is None
    assert Usage(cache_write=0).cache_write == 0, "esto sí es un cero medido"


def test_an_unmeasured_counter_makes_the_total_unknown():
    """Sumar solo lo conocido daría una cota inferior con aspecto de cifra exacta."""
    assert Usage(input=10, output=5, reasoning=2).total == 17
    assert Usage(input=10, output=5).total is None


def test_the_ratio_is_unknown_when_the_cache_was_not_measured():
    assert Usage(input=10).cache_hit_ratio is None
    assert Usage().cache_hit_ratio is None


def test_unknown_propagates_through_accumulation():
    """Si un solo tramo no midió un contador, el total tampoco se sabe."""
    total = Usage.zero() + Usage(input=10, output=5, reasoning=0, cache_read=0, cache_write=0)
    total = total + Usage(input=10, output=5, reasoning=0, cache_read=0)

    assert total.input == 20
    assert total.cache_write is None, "un tramo no lo reportó: el total no se sabe"


def test_zero_is_the_starting_point_and_differs_from_nothing_measured():
    assert Usage.zero().input == 0
    assert Usage().input is None


def test_an_estimate_contaminates_the_sum():
    """Un total que contiene una estimación es una estimación."""
    derived = Usage(input=100, output=20, estimated=True)
    reported = Usage(input=50, output=10)
    assert (reported + derived).estimated is True
    assert (reported + reported).estimated is False


def test_usage_refuses_to_add_foreign_types():
    with pytest.raises(TypeError):
        Usage() + 5  # type: ignore[operator]


# ── Serialización determinista ────────────────────────────────────────────────

def test_serialisation_is_byte_identical_for_equal_objects():
    """Sin esto no hay comparación de replay ni prefijo estable de caché."""
    a = Request(model="p:m", messages=(Message.user("hola"),), temperature=0.2)
    b = Request(model="p:m", messages=(Message.user("hola"),), temperature=0.2)
    assert dumps(a) == dumps(b)


def test_serialisation_does_not_depend_on_mapping_insertion_order():
    one = Request(model="p:m", provider_options={"b": 2, "a": 1})
    two = Request(model="p:m", provider_options={"a": 1, "b": 2})
    assert dumps(one) == dumps(two)


def test_none_fields_are_omitted():
    """Añadir un campo opcional no debe cambiar la serialización de quien no lo usa."""
    payload = json.loads(dumps(Request(model="p:m")))
    assert "temperature" not in payload
    assert "system" not in payload
    assert payload["model"] == "p:m"


def test_enums_serialise_as_their_wire_value():
    payload = to_jsonable(Message.user("hola"))
    assert payload["role"] == "user"
    assert payload["content"][0]["kind"] == "text"


def test_dumps_emits_valid_json():
    response = Response(
        message=Message.assistant("hecho"),
        finish_reason=FinishReason.STOP,
        usage=Usage(input=1, output=2),
        model="p:m",
    )
    assert json.loads(dumps(response))["finish_reason"] == "stop"


# ── Invariantes de las partes de contenido ────────────────────────────────────

def test_image_requires_exactly_one_source():
    with pytest.raises(ValueError):
        Image(media_type="image/png")
    with pytest.raises(ValueError):
        Image(media_type="image/png", data="AAA", url="https://x/y.png")
    assert Image(media_type="image/png", data="AAA").url is None


def test_named_tool_choice_requires_a_name():
    with pytest.raises(ValueError):
        ToolChoice(mode="named")
    with pytest.raises(ValueError):
        ToolChoice(mode="auto", name="buscar")
    assert ToolChoice(mode="named", name="buscar").name == "buscar"


def test_json_schema_format_requires_a_schema():
    with pytest.raises(ValueError):
        ResponseFormat(kind="json_schema")
    assert ResponseFormat(kind="json_schema", schema={"type": "object"}).strict is True


# ── Acceso a contenido ────────────────────────────────────────────────────────

def test_message_text_ignores_non_text_parts():
    msg = Message(
        Role.ASSISTANT,
        (Text("hola "), ToolCall(id="c1", name="buscar"), Text("mundo")),
    )
    assert msg.text == "hola mundo"


def test_tool_calls_are_reachable_from_the_response():
    response = Response(
        message=Message(Role.ASSISTANT, (ToolCall(id="c1", name="buscar", arguments={"q": "x"}),)),
        finish_reason=FinishReason.TOOL_CALLS,
    )
    assert [c.name for c in response.tool_calls] == ["buscar"]
    assert response.tool_calls[0].arguments == {"q": "x"}


def test_tool_result_preserves_error_evidence():
    """Un modelo que no ve el error no puede corregirlo."""
    result = ToolResult.of("c1", "timeout tras 30s", is_error=True)
    assert result.is_error is True
    assert result.content[0].text == "timeout tras 30s"


def test_frozen_values_cannot_be_mutated():
    """El journal referencia estos tipos; si mutasen dejaría de describir lo ocurrido."""
    with pytest.raises(AttributeError):
        Usage().input = 5  # type: ignore[misc]
