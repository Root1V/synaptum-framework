"""SYN-17 · El registro de proveedores."""

from __future__ import annotations

import pytest

from synaptum import ConfigurationError, providers
from synaptum.providers.base import _Broken, _reset_discovery


class _Falso:
    name = "falso"

    def to_wire(self, request):
        return {}

    def from_wire(self, body):
        raise NotImplementedError

    def stream_from_wire(self, chunks):
        raise NotImplementedError


def test_the_base_dialect_comes_registered():
    """No es un plugin que instalar aparte: es el caso base."""
    assert "openai-compatible" in providers.available()


def test_a_model_reference_splits_into_adapter_and_model():
    adapter, model = providers.resolve("openai-compatible:llama3-8b-q4")
    assert adapter.name == "openai-compatible"
    assert model == "llama3-8b-q4", "el nombre viaja tal cual"


def test_a_reference_without_a_prefix_is_rejected():
    with pytest.raises(ConfigurationError, match="no nombra un modelo"):
        providers.resolve("llama3-8b-q4")


def test_an_unknown_adapter_says_which_ones_hay():
    with pytest.raises(ConfigurationError, match="openai-compatible"):
        providers.get("inexistente")


def test_manual_registration_wins_over_discovery():
    """Un test o un adaptador interno pueden sustituir a uno instalado."""
    providers.register(_Falso())
    try:
        assert providers.get("falso").name == "falso"
        assert "falso" in providers.available()
    finally:
        providers.base._MANUAL.pop("falso", None)


def test_a_plain_class_satisfies_the_protocol_without_inheriting():
    assert isinstance(_Falso(), providers.Provider)


def test_a_broken_adapter_fails_when_used_not_when_installed():
    """Uno roto no puede tumbar a los demás: quien no lo pida, ni se entera."""
    roto = _Broken("roto", RuntimeError("falta una dependencia"))
    with pytest.raises(ConfigurationError, match="no carga"):
        roto.from_wire({})


def test_discovery_is_cached_and_resettable():
    _reset_discovery()
    primero = providers.available()
    assert providers.available() == primero


def test_the_reasoning_cycle_closes_when_a_tool_call_starts():
    """La fase de razonamiento termina al empezar *cualquier* otra cosa.

    Cerrábamos el ciclo solo al llegar ``content``, así que un modelo de
    razonamiento que llama a una herramienta —75 deltas de pensamiento y ni un
    token de respuesta, que es la grabación real— dejaba el ``reasoning_end``
    cayendo en mitad de la llamada: el bloque de pensamiento seguía abierto
    mientras los argumentos ya estaban llegando.

    Se comprueba aquí además de en el corpus porque el corpus vive fuera del
    repo, y una propiedad de nuestro adaptador tiene que poder fallar sin él.
    """
    from synaptum.providers.openai_compatible import OpenAICompatible

    adapter = OpenAICompatible()
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "pienso"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "f", "arguments": '{"a":'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                      "finish_reason": "tool_calls"}]},
    ]
    kinds = [event.kind for event in adapter.stream_from_wire(chunks)]

    assert kinds.index("reasoning_end") < kinds.index("tool_call_start"), (
        "el razonamiento tiene que cerrarse antes de que empiece la llamada"
    )
    assert kinds == [
        "stream_start",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_end",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_end",
        "finish",
    ]


def test_a_second_sentinel_cannot_change_the_result():
    """Parar en el primer centinela, vengan uno o vengan dos.

    Es la conducta que dejó sin facturar todo el streaming de Prometheus hasta
    el manifest v8: el cliente correcto paraba en el primero, y el registro de
    la petición vivía pasado ese punto.  Ya viene uno solo — razón de más para
    que esto no dependa de cuántos vengan.
    """
    from synaptum.testing import split_sse

    uno = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\ndata: [DONE]\n'
    dos = uno + 'data: {"choices":[{"delta":{"content":" BASURA"}}]}\ndata: [DONE]\n'

    assert split_sse(uno) == split_sse(dos)


def test_an_image_travels_and_a_document_is_refused():
    """S-7 · Las imágenes viajan; un documento se rechaza **a propósito**.

    Convertir un PDF aquí sería decidir por quien lo manda cómo se ve una
    página, y eso lo decide quien la recortó. Veritium lo pidió así:
    *«para `Document`, preferimos que el adaptador lo rechace»*.
    """
    from synaptum import ConfigurationError, Document, Image, Message, Role, Text
    from synaptum.providers.openai_compatible import _message_to_wire

    con_imagen = _message_to_wire(Message(
        role=Role.USER,
        content=(Text("¿qué dice esta boleta?"), Image(data="QUJD", media_type="image/png")),
    ))[0]
    assert con_imagen["content"] == [
        {"type": "text", "text": "¿qué dice esta boleta?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]

    # Sin partes no textuales, `content` sigue siendo una cadena: cambiarlo
    # siempre haría distinto el cuerpo de todos los runs que hoy funcionan.
    assert _message_to_wire(Message.user("hola"))[0]["content"] == "hola"

    with pytest.raises(ConfigurationError, match="document"):
        _message_to_wire(Message(
            role=Role.USER,
            content=(Document(data="x", media_type="application/pdf"),),
        ))


def _descartado_en_silencio_ya_no_pasa():
    """Lo que no se sabe mandar, no se calla.

    Una imagen puesta en el mensaje desaparecía del cuerpo, el modelo contestaba
    sobre un texto sin ella y **nada fallaba**: la respuesta parecía mala y lo
    que estaba mal era el envío.

    Negarse es peor para quien ya tenía un atajo y mejor para todos los demás.
    Un fallo ruidoso se arregla una vez; uno silencioso se paga en cada
    respuesta sin que nadie sepa por qué.
    """
    # Conservado como nota: la primera mitad de S-7 fue negarse en vez de
    # descartar, y la segunda hacer que las imágenes viajen de verdad.


# ── VRT-SYN-004 · un turno que solo razona ───────────────────────────────────
#
# Reportado por Veritium contra 1.0.0rc4: un `Agent` contra llama-server se
# cortaba con `HTTP 400 · Assistant message must contain either 'content' or
# 'tool_calls'!` en el turno **siguiente** a uno en que el modelo respondió solo
# con `reasoning_content`.
#
# Las dos direcciones del adaptador no son independientes, y ahí estaba el
# hueco: la dirección *response* **produce** la parte (`Thinking`) que la
# dirección *request* no sabe mandar. `contratos/normalizacion/spec.md` dice que
# la dirección request «no se normaliza» y que el documento trata la response —
# así que el corpus dorado que ejecutan las dos implementaciones no podía ver
# esto en ninguno de los dos lenguajes.

def _sin_sobres_vacios(body: dict) -> None:
    """La invariante del dialecto, afirmada sobre el cuerpo entero.

    Se comprueba así y no solo sobre el mensaje del caso porque el 400 no lo
    provoca el turno que razona: lo provoca **cualquier** mensaje del asistente
    que salga sin ninguna de las dos claves, y el turno culpable ya pasó.
    """
    for mensaje in body["messages"]:
        if mensaje["role"] != "assistant":
            continue
        assert mensaje.get("content") or mensaje.get("tool_calls"), (
            f"un mensaje del asistente sale sin `content` ni `tool_calls`: {mensaje}"
        )


def test_an_assistant_turn_with_only_reasoning_is_not_sent_as_an_empty_envelope():
    """VRT-SYN-004 · El razonamiento no vuelve al proveedor; el sobre tampoco.

    Antes el mensaje salía como `{"role": "assistant", "content": null}`, que el
    propio dialecto declara inválido. No se manda `content: ""` en su lugar:
    este servidor lo aceptaría, pero hay dialectos que rechazan un bloque de
    texto vacío, así que la forma «válida» dependería de quién esté al otro
    lado. No mandar nada vale en todos.
    """
    from synaptum import Message, Request, Role, Text, Thinking, ToolCall
    from synaptum.providers.openai_compatible import OpenAICompatible, _message_to_wire

    assert _message_to_wire(Message(Role.ASSISTANT, (Thinking(text="mmm"),))) == []

    cuerpo = OpenAICompatible().to_wire(Request(model="m", messages=(
        Message.user("extrae el total"),
        Message(Role.ASSISTANT, (Thinking(text="el usuario quiere el total..."),)),
        Message.user("sigue"),
    )))
    _sin_sobres_vacios(cuerpo)
    assert [m["role"] for m in cuerpo["messages"]] == ["user", "user"], (
        "el turno que no lleva nada enviable no deja un sobre vacío detrás"
    )

    # Y lo que sí lleva algo no cambia: el razonamiento se sigue omitiendo, pero
    # el mensaje viaja por lo demás.  Un turno de razonar y llamar a una tool es
    # lo normal en un modelo de razonamiento, y ahí el sobre es obligatorio.
    con_tool = _message_to_wire(Message(Role.ASSISTANT, (
        Thinking(text="voy a leerlo"), ToolCall(id="c1", name="leer", arguments={"path": "/x"}),
    )))[0]
    assert con_tool["content"] is None and con_tool["tool_calls"][0]["id"] == "c1"

    con_texto = _message_to_wire(Message(Role.ASSISTANT, (
        Thinking(text="mmm"), Text("el total es 42"),
    )))[0]
    assert con_texto["content"] == "el total es 42"
