"""SYN-28 · Prompts como configuración versionada."""

from __future__ import annotations

import json

import pytest

from synaptum import (
    Agent,
    ConfigurationError,
    FilePrompts,
    InMemoryPrompts,
    PromptProvider,
    PromptRegistry,
    PromptTemplate,
    fmt_dict,
    fmt_list,
    fmt_records,
)


# ── La plantilla ──────────────────────────────────────────────────────────────

def test_the_version_travels_with_the_content():
    """Cuando una respuesta sale mal, la primera pregunta es con qué prompt se generó."""
    t = PromptTemplate(content="Eres un analista.", version="2.1", description="Analista")
    assert t.version == "2.1"
    assert str(t) == "Eres un analista."


def test_arguments_win_over_the_templates_own_variables():
    t = PromptTemplate(content="Eres {rol}.", variables={"rol": "analista"})
    assert t.render() == "Eres analista."
    assert t.render(rol="auditor") == "Eres auditor."


def test_a_missing_variable_fails_instead_of_leaking_the_placeholder():
    """Una llave sin sustituir no falla: llega al modelo como texto y empeora la respuesta."""
    t = PromptTemplate(content="Evalúa a {cliente} en {sector}.", version="1.3")
    with pytest.raises(ConfigurationError) as caught:
        t.render(cliente="Acme")
    assert "sector" in str(caught.value)
    assert "1.3" in str(caught.value), "la versión ayuda a encontrar el prompt"


def test_the_placeholders_are_discoverable():
    t = PromptTemplate(content="{a} y {b} y otra vez {a}.")
    assert t.placeholders == {"a", "b"}


# ── Proveedores ───────────────────────────────────────────────────────────────

def test_the_short_form_is_just_the_text():
    prompts = InMemoryPrompts({"saludo": "Hola."})
    assert prompts.get("saludo").content == "Hola."
    assert prompts.get("saludo").version == "1.0"


def test_a_missing_prompt_says_which_ones_hay():
    prompts = InMemoryPrompts({"a": "x"})
    with pytest.raises(KeyError, match="'a'"):
        prompts.get("inexistente")


def test_json_needs_no_dependencies(tmp_path):
    fichero = tmp_path / "prompts.json"
    fichero.write_text(json.dumps({
        "analista.system": {
            "content": "Eres un analista de {sector}.",
            "version": "3.0",
            "variables": {"sector": "banca"},
        },
        "corto": "Sé breve.",
    }))

    prompts = FilePrompts(fichero)
    assert prompts.get("analista.system").render() == "Eres un analista de banca."
    assert prompts.get("analista.system").version == "3.0"
    assert prompts.get("corto").content == "Sé breve."


def test_an_unsupported_format_is_rejected(tmp_path):
    fichero = tmp_path / "prompts.txt"
    fichero.write_text("cualquier cosa")
    with pytest.raises(ConfigurationError, match="no soportado"):
        FilePrompts(fichero).exists("x")


def test_a_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigurationError, match="No existe"):
        FilePrompts(tmp_path / "no-esta.json").get("x")


def test_reload_picks_up_an_edit(tmp_path):
    """Editar prompts sin reiniciar es lo que hace utilizable tenerlos fuera."""
    fichero = tmp_path / "prompts.json"
    fichero.write_text(json.dumps({"a": "primera"}))
    prompts = FilePrompts(fichero)
    assert prompts.get("a").content == "primera"

    fichero.write_text(json.dumps({"a": "segunda"}))
    assert prompts.get("a").content == "primera", "cacheado"
    prompts.reload()
    assert prompts.get("a").content == "segunda"


# ── El registro ───────────────────────────────────────────────────────────────

def test_the_first_provider_with_the_prompt_wins():
    registry = PromptRegistry(
        InMemoryPrompts({"saludo": "de producción"}),
        InMemoryPrompts({"saludo": "de reserva", "otro": "solo aquí"}),
    )
    assert registry.get("saludo").content == "de producción"
    assert registry.get("otro").content == "solo aquí"


def test_prepending_overrides_without_editing_the_shared_file():
    registry = PromptRegistry(InMemoryPrompts({"saludo": "compartido"}))
    registry.prepend(InMemoryPrompts({"saludo": "mío en local"}))
    assert registry.get("saludo").content == "mío en local"


def test_a_prompt_in_no_provider_names_the_chain():
    registry = PromptRegistry(InMemoryPrompts({}))
    with pytest.raises(KeyError, match="InMemoryPrompts"):
        registry.get("inexistente")


def test_a_plain_class_satisfies_the_protocol_without_inheriting():
    class Propio:
        def get(self, name):
            return PromptTemplate(content=f"generado para {name}")

        def exists(self, name):
            return True

    assert isinstance(Propio(), PromptProvider)
    assert PromptRegistry(Propio()).get("loquesea").content == "generado para loquesea"


# ── En el agente ──────────────────────────────────────────────────────────────

def test_the_agent_takes_a_template_or_a_string():
    plantilla = PromptTemplate(content="Eres {rol}.", variables={"rol": "auditor"})
    assert Agent("a", model="fake:m", instructions=plantilla).instructions == "Eres auditor."
    assert Agent("b", model="fake:m", instructions="literal").instructions == "literal"
    assert Agent("c", model="fake:m").instructions is None


# ── Serialización de datos ────────────────────────────────────────────────────

def test_the_helpers_produce_data_and_nothing_else():
    """Sin encabezados ni viñetas: ese texto pertenece al fichero de prompts.

    Es lo único que hace que cambiar el encuadre no exija tocar código — el
    motivo entero de que los prompts vivan fuera.
    """
    assert fmt_dict({"nombre": "Acme", "ingresos": 1_000_000}) == "nombre: Acme\ningresos: 1000000"
    assert fmt_list(["riesgo A", "riesgo B"]) == "· riesgo A\n· riesgo B"
    assert fmt_records(
        [{"n": "Ana", "s": 9.2}, {"n": "Beto", "s": 7.8}], "{n}: {s}"
    ) == "Ana: 9.2\nBeto: 7.8"


def test_nested_values_are_serialised_deterministically():
    assert fmt_dict({"datos": {"b": 2, "a": 1}}) == 'datos: {"a":1,"b":2}'
