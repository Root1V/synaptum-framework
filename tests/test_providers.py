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
