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
