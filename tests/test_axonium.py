"""SYN-52 · El puente hacia Axonium.

Se prueba contra los **tipos reales del SDK**, construidos a mano. Lo que no se
puede probar aquí es la llamada de red: exige una instancia de Prometheus, y por
P10 no hay ninguna al alcance. Eso queda dicho en vez de simulado.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("axonium", reason="extra 'axonium' no instalado")

from axonium import ChatCompletion, Usage as AxUsage  # noqa: E402

from synaptum import (  # noqa: E402
    FinishReason,
    ProviderError,
    Request,
    Response,
    Thinking,
    ToolCall,
)
from synaptum.providers.axonium import (  # noqa: E402
    AxoniumModel,
    message_from_axonium,
    usage_from_axonium,
)


def _completion(**message) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "llama3-8b-q4",
            "choices": [{"index": 0, "message": {"role": "assistant", **message},
                         "finish_reason": message.pop("_finish", "stop")}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 20,
                      "total_tokens": 35, "cache_read_tokens": 3, "estimated": True},
        }
    )


# ── Lo que motivó que este módulo exista ──────────────────────────────────────

def test_the_cache_and_estimated_fields_survive_the_translation():
    """Pasarlo por un lector de cable los buscaría donde no están y los perdería."""
    unificado = usage_from_axonium(
        AxUsage(prompt_tokens=15, completion_tokens=20, total_tokens=35,
                cache_read_tokens=3, estimated=True)
    )
    assert unificado.input == 15, "inclusivo: contiene lo cacheado"
    assert unificado.cache_read == 3
    assert unificado.estimated is True


def test_what_prometheus_cannot_measure_stays_unmeasured():
    """Sin medir, no cero — cero diría «no hubo»."""
    unificado = usage_from_axonium(
        AxUsage(prompt_tokens=15, completion_tokens=20, total_tokens=35)
    )
    assert unificado.reasoning is None
    assert unificado.cache_write is None
    assert unificado.cache_read is None, "este backend no lo reportó"


def test_no_usage_at_all_is_everything_unmeasured():
    assert usage_from_axonium(None).input is None


# ── Mensajes ──────────────────────────────────────────────────────────────────

def test_reasoning_and_answer_stay_separate():
    completion = _completion(content="la respuesta", reasoning_content="pensando")
    message = message_from_axonium(completion.choices[0].message)

    assert isinstance(message.content[0], Thinking)
    assert message.text == "la respuesta", "el razonamiento no se concatena"


def test_a_null_content_with_tool_calls_produces_no_empty_text_part():
    completion = _completion(
        content=None,
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Lima"}'}}],
    )
    message = message_from_axonium(completion.choices[0].message)

    assert [p.kind for p in message.content] == ["tool_call"]
    assert message.text == ""


def test_tool_call_arguments_arrive_decoded():
    completion = _completion(
        content=None,
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Lima"}'}}],
    )
    call = message_from_axonium(completion.choices[0].message).tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.arguments == {"city": "Lima"}


def test_unreadable_arguments_are_not_turned_into_an_empty_object():
    """Un `{}` silencioso ejecutaría la herramienta sin argumentos."""
    from synaptum import ProviderError

    completion = _completion(
        content=None,
        tool_calls=[{"id": "c", "type": "function",
                     "function": {"name": "t", "arguments": "{roto"}}],
    )
    with pytest.raises(ProviderError, match="ilegibles"):
        message_from_axonium(completion.choices[0].message)


# ── El modelo, contra un cliente falso ────────────────────────────────────────

class _ClienteFalso:
    def __init__(self, completion):
        self.completion = completion
        self.llamadas: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.llamadas.append(kwargs)
        return self.completion


def test_the_request_reaches_axonium_with_the_bare_model_name():
    """Axonium recibe el nombre a secas: el prefijo dice quién normaliza, no qué modelo."""
    cliente = _ClienteFalso(_completion(content="hola"))
    modelo = AxoniumModel(client=cliente)

    respuesta = asyncio.run(
        modelo.complete(Request(model="axonium:llama3-8b-q4", system="sé breve"))
    )

    enviado = cliente.llamadas[0]
    assert enviado["model"] == "llama3-8b-q4"
    assert enviado["messages"][0] == {"role": "system", "content": "sé breve"}
    assert isinstance(respuesta, Response)
    assert respuesta.text == "hola"
    assert respuesta.finish_reason is FinishReason.STOP
    assert respuesta.usage.cache_read == 3, "el desglose llegó hasta el bucle"


def test_a_missing_sdk_says_where_to_get_it():
    from synaptum.core.errors import ConfigurationError
    from synaptum.providers import axonium as puente

    original = puente._build_client
    try:
        def explota(**_):
            raise ConfigurationError("Axonium no está instalado.")

        puente._build_client = explota
        with pytest.raises(ConfigurationError, match="no está instalado"):
            AxoniumModel()
    finally:
        puente._build_client = original


# ── Streaming del puente ──────────────────────────────────────────────────────
#
# El puente tiene su PROPIA implementación de streaming: no reutiliza la del
# adaptador de cable, porque Axonium ya entrega objetos normalizados.  Eso
# significa que un arreglo en el adaptador no lo alcanza — y no lo alcanzó.

class _StreamFalso:
    """Cliente falso que entrega los fragmentos como los entrega el SDK."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.chat = self
        self.completions = self

    def stream(self, **_):
        chunks = self.chunks

        class _Ctx:
            async def __aenter__(self):
                async def gen():
                    for c in chunks:
                        yield c

                return gen()

            async def __aexit__(self, *_):
                return False

        return _Ctx()


def _delta(**campos):
    from types import SimpleNamespace

    completos = {"content": None, "reasoning_content": None, "tool_calls": None}
    completos.update(campos)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(**completos), finish_reason=None)],
        usage=None,
    )


def _cierre(motivo="stop"):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(delta=None, finish_reason=motivo)], usage=None
    )


def _eventos(chunks):
    modelo = AxoniumModel(client=_StreamFalso(chunks))

    async def go():
        return [e async for e in modelo.stream(Request(model="axonium:m"))]

    return asyncio.run(go())


def test_the_bridge_emits_the_reasoning_cycle():
    """Acumularlo sin emitirlo deja a quien consume sin nada que ver.

    Encontrado ejecutando el puente contra la plataforma por primera vez: el
    adaptador de cable ya lo hacía bien —lo fijó el corpus dorado— y el puente
    no, porque el corpus prueba normalización de cuerpos y esto no pasa por
    ningún cuerpo. Un arreglo en un sitio no alcanza al otro.
    """
    eventos = _eventos([
        _delta(reasoning_content="pien"),
        _delta(reasoning_content="so"),
        _delta(content="hola"),
        _cierre(),
    ])
    tipos = [e.kind for e in eventos]

    assert tipos == [
        "stream_start",
        "reasoning_start", "reasoning_delta", "reasoning_delta", "reasoning_end",
        "text_start", "text_delta", "text_end",
        "finish",
    ]
    assert "".join(e.text for e in eventos if e.kind == "reasoning_delta") == "pienso"
    assert eventos[-1].response.text == "hola", "el razonamiento no se cuela en la respuesta"


def test_the_bridge_closes_the_reasoning_cycle_when_a_tool_call_starts():
    """La fase termina al empezar cualquier otra cosa, no solo el texto.

    Con un modelo de razonamiento que llama a una herramienta no llega ni un
    token de respuesta, así que cerrar solo con `content` deja el bloque de
    pensamiento abierto mientras los argumentos ya están llegando.
    """
    eventos = _eventos([
        _delta(reasoning_content="pienso"),
        _delta(tool_calls=[{"index": 0, "id": "c1",
                            "function": {"name": "f", "arguments": '{"a":1}'}}]),
        _cierre("tool_calls"),
    ])
    tipos = [e.kind for e in eventos]

    assert tipos.index("reasoning_end") < tipos.index("tool_call_start")
    assert eventos[-1].response.tool_calls[0].arguments == {"a": 1}


# ── La identidad del paso como clave de idempotencia ──────────────────────────

def test_the_step_identity_travels_as_an_idempotency_key():
    """``(run_id, step_id)`` es determinista, así que reanudar da la misma clave.

    Cierra un agujero que el journal solo no puede tapar: si el proceso muere
    después de mandar la petición y antes de registrar el resultado, el replay
    reintenta —una llamada al modelo es repetible— y sin clave esa repetición
    es una segunda generación facturable.
    """
    from synaptum import CallContext

    cliente = _ClienteFalso(_completion(content="hola"))
    modelo = AxoniumModel(client=cliente)
    ctx = CallContext(run_id="run-7", step_id="000003-model")

    asyncio.run(modelo.complete(Request(model="axonium:m"), ctx))
    asyncio.run(modelo.complete(Request(model="axonium:m"), ctx))

    claves = [l["idempotency_key"] for l in cliente.llamadas]
    assert claves[0] == claves[1], "reanudar produjo una clave distinta"
    assert claves[0].startswith("run-7/000003-model/"), claves[0]


def test_a_different_request_gets_a_different_key():
    """Una clave identifica UNA petición; reutilizarla para otra es un rechazo.

    Sin la huella del cuerpo, dos runs con el mismo `run_id` y distinta tarea
    chocaban — y también el mismo run tras cambiar las instrucciones del agente,
    que es lo que pasa mientras se desarrolla. El síntoma era un error del
    proveedor durante 24 h y ninguna pista de por qué.
    """
    from synaptum import CallContext

    cliente = _ClienteFalso(_completion(content="hola"))
    modelo = AxoniumModel(client=cliente)
    ctx = CallContext(run_id="run-7", step_id="000003-model")

    asyncio.run(modelo.complete(Request(model="axonium:m", system="sé breve"), ctx))
    asyncio.run(modelo.complete(Request(model="axonium:m", system="sé prolijo"), ctx))

    primera, segunda = (l["idempotency_key"] for l in cliente.llamadas)
    assert primera != segunda, "dos peticiones distintas compartieron clave"


def test_without_a_context_no_key_is_invented():
    """Una clave no determinista es peor que ninguna.

    Convertiría cada reintento en una generación nueva llevando la etiqueta de
    que no lo es — se paga dos veces y además queda mal registrado.
    """
    cliente = _ClienteFalso(_completion(content="hola"))
    asyncio.run(AxoniumModel(client=cliente).complete(Request(model="axonium:m")))

    assert cliente.llamadas[0]["idempotency_key"] is None


def _con_meta(**campos):
    """Respuesta con ``meta``, que en el tipo real es de solo lectura.

    El puente lee todo con ``getattr``, así que un objeto con la misma forma
    vale — y evita depender de si el SDK deja escribir sus propios campos.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hola", reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
        model="m",
        meta=SimpleNamespace(**campos),
    )


def test_platform_metadata_reaches_the_response():
    """``meta`` se descartaba entero, y dos campos de ahí no son decorativos."""
    completion = _con_meta(
        request_id="req-1", trace_id="tr-1", instance="#2",
        instance_id="inst-abc", idempotent_replay=True,
    )
    respuesta = asyncio.run(
        AxoniumModel(client=_ClienteFalso(completion)).complete(Request(model="axonium:m"))
    )

    assert respuesta.provider_metadata["idempotent_replay"] is True
    assert respuesta.provider_metadata["instance_id"] == "inst-abc"
    assert respuesta.provider_metadata["request_id"] == "req-1"


def test_a_replay_names_the_generation_that_was_actually_billed():
    """El `request_id` de un replay no lleva a ninguna fila de facturación.

    Nombra la respuesta que se sirvió, no la generación que se cobró. Sin
    `idempotent_replay_of`, una auditoría que parta de ese id no encuentra nada
    y no sabe por qué.
    """
    respuesta = asyncio.run(
        AxoniumModel(client=_ClienteFalso(_con_meta(
            request_id="req-replay", idempotent_replay=True,
            idempotent_replay_of="req-original",
        ))).complete(Request(model="axonium:m"))
    )

    assert respuesta.provider_metadata["idempotent_replay_of"] == "req-original"


def test_a_failure_carries_the_id_that_makes_it_diagnosable():
    """Sin `request_id`, un fallo del otro lado es una descripción.

    Quien opera la plataforma no puede buscar «un 500 de hace tres días»: busca
    un identificador. Lo traía el SDK y nosotros lo dejábamos dentro del texto
    del mensaje, que es donde van las cosas que nadie lee con un programa.
    """
    from axonium.errors import APIError

    from synaptum.providers.axonium import _translate

    traducido = _translate(APIError(
        "boom", status=500, request_id="req-envenenado", trace_id="tr-9",
    ))

    assert isinstance(traducido, ProviderError)
    assert traducido.request_id == "req-envenenado"
    assert traducido.trace_id == "tr-9"
    assert traducido.status == 500


def test_the_wait_the_sdk_did_on_purpose_reaches_the_journal():
    """Hay dos reintentos apilados: el del SDK y el nuestro encima.

    Sin `waited_s` y `attempts`, una llamada que tardó porque el SDK respetó un
    `Retry-After` es indistinguible de una plataforma lenta. Tres equipos la
    reportaron como un cuelgue y ninguno como espera.
    """
    respuesta = asyncio.run(
        AxoniumModel(client=_ClienteFalso(_con_meta(
            request_id="req-1", waited_s=12.5, attempts=3,
        ))).complete(Request(model="axonium:m"))
    )

    assert respuesta.provider_metadata["waited_s"] == 12.5
    assert respuesta.provider_metadata["attempts"] == 3


def test_a_replay_served_through_the_gateway_is_not_counted_twice():
    """De punta a punta: puente → LocalGateway → bucle.

    Las dos mitades se prueban por separado, y esto comprueba que se tocan: el
    puente tiene que exponer el bit y el bucle tiene que mirarlo. Cualquiera de
    las dos sola deja el total inflado sin que nada falle.
    """
    from synaptum import Agent, LocalGateway, Session

    puente = AxoniumModel(client=_ClienteFalso(_con_meta(idempotent_replay=True)))
    gateway = LocalGateway(model=puente.complete, warn=False)

    async def go():
        return [
            paso
            async for paso in Agent("a", model="axonium:m").run(
                "t", session=Session("run-1", gateway)
            )
        ]

    pasos = asyncio.run(go())
    assert pasos[-1].usage.input == 0, "un replay infló el total del run"


def test_a_retry_after_from_the_sdk_survives_the_translation():
    """El otro extremo sabe cuándo estará libre; perderlo es adivinar en su lugar."""
    class _Lenta(Exception):
        status_code = 429
        retry_after = 3.0

    from synaptum.providers.axonium import _translate

    traducido = _translate(_Lenta("frena"))
    assert traducido.status == 429
    assert traducido.retryable is True
    assert traducido.retry_after == 3.0


# ── Canario de forma del SDK ──────────────────────────────────────────────────
#
# `_field()` lee atributo **o** clave, y eso lo hace bueno para sobrevivir a un
# cambio de forma del SDK y **malo para enterarse**. Entre `rc3` y `rc4` las
# tool calls pasaron de dicts a objetos tipados: no nos rompió, y tampoco nos
# avisó. Nos enteramos porque otro equipo lo publicó.
#
# Estos tests no comprueban nuestro código: comprueban **la forma que el SDK nos
# entrega hoy**. Si cambia otra vez, fallan aquí —con el nombre del campo— en
# vez de pasar desapercibidos hasta que algo se pierda en silencio.

sdk = pytest.importorskip("axonium", reason="el puente es un extra opcional")


def _expone(modelo, nombre: str) -> bool:
    """¿El modelo entrega este nombre, sea como sea?

    Mirar solo `model_fields` no basta y el canario lo encontró a la primera:
    `meta` no es un campo de pydantic, es una **propiedad** de `APIObject`. Un
    canario que solo mire una de las dos formas canta cuando no debe, y eso es
    peor que no tenerlo — se acaba silenciando.
    """
    campos = getattr(modelo, "model_fields", None) or {}
    if nombre in campos:
        return True
    return isinstance(getattr(modelo, nombre, None), property)


def _campo_de(modelo, nombre: str):
    campos = getattr(modelo, "model_fields", None) or {}
    return campos.get(nombre)


def test_the_sdk_still_exposes_the_response_fields_the_bridge_reads():
    """Lo que leemos de una respuesta, y de dónde.

    Si el SDK renombra o mueve alguno, aquí se ve el nombre exacto que falta.
    """
    from axonium.models.chat import ChatCompletion

    for campo in ("choices", "usage", "model", "meta"):
        assert _expone(ChatCompletion, campo), (
            f"ChatCompletion ya no expone '{campo}' — el puente lo lee"
        )


def test_the_sdk_still_exposes_the_usage_counters_we_translate():
    """Los cinco contadores y el bit de estimación, por nombre.

    Perder uno en silencio es exactamente el fallo que el `Usage` de tres
    estados existe para evitar.
    """
    from axonium.models.chat import Usage as UsageAxonium

    for campo in ("prompt_tokens", "completion_tokens", "cache_read_tokens", "estimated"):
        assert _expone(UsageAxonium, campo), (
            f"Usage del SDK ya no expone '{campo}' — lo traducimos"
        )


def test_the_shape_of_a_tool_call_is_the_one_the_bridge_assumes():
    """Dicts u objetos: da igual cuál, pero que no cambie sin que lo sepamos.

    Este es el canario. En `rc3` eran `list[dict]` y en `rc4` son `list[ToolCall]`
    con `.function.name` y `.name`. `_field()` absorbe las dos, así que un tercer
    cambio tampoco nos rompería — y tampoco nos enteraríamos.
    """
    from axonium.models.chat import CompletionMessage

    campo = _campo_de(CompletionMessage, "tool_calls")
    assert campo is not None, "CompletionMessage ya no expone 'tool_calls'"

    forma = str(campo.annotation)
    assert "ToolCall" in forma or "dict" in forma, (
        f"la forma de tool_calls cambió a algo que no esperábamos: {forma}. "
        "Comprueba que `_field()` sigue alcanzando el nombre y los argumentos."
    )


def test_the_bridge_reaches_name_and_arguments_whatever_the_shape():
    """Lo único que de verdad importa: que lleguemos al nombre y a los argumentos.

    Se comprueba contra las **dos** formas que el SDK ha tenido, porque un
    consumidor puede estar en cualquiera de las dos y el puente las cubre.
    """
    from types import SimpleNamespace

    from synaptum.providers.axonium import _field

    como_dict = {"index": 0, "id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}}
    como_objeto = SimpleNamespace(
        index=0, id="c1", name="f",
        function=SimpleNamespace(name="f", arguments='{"a":1}'),
    )

    for forma, llamada in (("dict", como_dict), ("objeto", como_objeto)):
        funcion = _field(llamada, "function")
        assert _field(funcion, "name") == "f", f"no se alcanza el nombre en forma {forma}"
        assert _field(funcion, "arguments") == '{"a":1}', f"argumentos, forma {forma}"
        assert _field(llamada, "id") == "c1", f"id, forma {forma}"


def test_no_field_we_read_changed_its_type():
    """Lo que el canario de arriba **no** puede ver.

    Aquel comprueba que los campos que leemos sigan existiendo. Este comprueba
    que no cambien de tipo — que es justo lo que `_field()` se traga sin avisar,
    porque lee atributo **o** clave y las dos formas le valen.

    Entre `rc3` y `rc4` las tool calls pasaron de `list[dict]` a `list[ToolCall]`.
    No nos rompió, no nos avisó, y nos enteramos porque otro equipo lo publicó.
    Con esta comprobación habría salido en una línea, cuatro días antes.

    El método es de Axonium. Si falla, mira si te afecta y anota la nueva forma:

        uv run python scripts/forma_del_sdk.py --write
    """
    import subprocess
    import sys
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    hecho = subprocess.run(
        [sys.executable, "scripts/forma_del_sdk.py", "--check"],
        cwd=raiz, capture_output=True, text=True,
    )
    assert hecho.returncode == 0, hecho.stderr
