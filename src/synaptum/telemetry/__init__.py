"""
SYN-37 · Trazas de la **estructura del bucle**.

Lo que este módulo emite, y sobre todo lo que **no**
-----------------------------------------------------
Emite lo que solo el bucle sabe: dónde empieza y acaba un run, dónde acaba un
turno y empieza el siguiente, cuándo se delega, y cuándo el gobierno paró algo.

**No emite la llamada al modelo ni la ejecución de una herramienta.** Esas las
emite quien las ejecuta —en el camino gobernado, el arnés— y duplicarlas daría
dos spans para un mismo hecho, con dos duraciones que nunca coinciden y un
lector eligiendo la que le parezca. Un dato que existe dos veces es peor que uno
que falta: el que falta se nota.

Cómo se engancha, y por qué así
--------------------------------
Envolviendo el iterador, no tocando el bucle:

    async for paso in traced(agente.run(tarea, session=sesion), run_id="r1"):
        ...

El bucle ya cede el control en cada frontera significativa, así que un
observador que consume esos eventos ve exactamente lo mismo que el bucle sin
que el bucle sepa que existe. Y el núcleo se queda sin dependencia: si el extra
``[otel]`` no está instalado, lo único que falla es esta función, y lo dice.

Los spans llevan las marcas de tiempo **de los eventos**, no las del momento en
que se observan: un paso reproducido del diario cuenta su duración original, no
los microsegundos que tarda en releerse.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from ..core.events import (
    ApprovalStep,
    DelegateStep,
    Disposition,
    FinalStep,
    ModelStep,
    Phase,
    StepEvent,
)

__all__ = ["traced", "ATRIBUTO_GUARDRAIL", "describe_tracing"]

#: El atributo que pone un span en el camino caliente de una plataforma de
#: observabilidad: alerta en segundos en vez de esperar al almacén.
#:
#: **Este nombre es la convención de una plataforma concreta, no nuestra**, y
#: por eso `traced` lo acepta como argumento: quien exporte a otra cosa pone el
#: suyo sin tocar nada más. Está aquí como valor por defecto porque un
#: framework que se niegue a nombrarlo no se puede enrutar, y entonces la
#: decisión la acaba tomando cada despliegue a mano.
#:
#: Lleva **el tipo de lo que paró el run, no un booleano**. «Denegado por
#: política» y «esperando a una persona» necesitan los dos una respuesta rápida
#: y **respuestas distintas**, así que un `true` obligaría a quien recibe la
#: alerta a ir a buscar cuál de las dos es.
ATRIBUTO_GUARDRAIL = "argus.guardrail"

#: Las disposiciones que son gobierno parando algo, y por tanto van al camino
#: caliente.  ``ALLOW`` no está: el sistema funcionando no es una alerta.
_GOBIERNO = {
    Disposition.DENY_STEP,
    Disposition.TERMINATE_RUN,
    Disposition.REQUIRE_APPROVAL,
}


def _tracer() -> Any:
    try:
        from opentelemetry import trace
    except ImportError as falta:  # pragma: no cover - depende del entorno
        raise ImportError(
            "Las trazas necesitan el extra `otel`: pip install synaptum[otel]. "
            "Sin él, `traced` no emite nada — y no hacerlo en silencio sería "
            "observabilidad que parece configurada y no lo está."
        ) from falta
    return trace.get_tracer("synaptum")


def describe_tracing() -> str:
    """Qué proveedor de trazas hay puesto, dicho en una línea.

    Existe por el modo de fallo que **no** produce ningún error: si nadie
    configuró un proveedor, OpenTelemetry deja uno que no hace nada, los spans
    se emiten, no llegan a ningún sitio, y la aplicación funciona
    perfectamente. No hay excepción que mirar — solo la ausencia de trazas, que
    es justo lo que nadie está mirando cuando acaba de montar el tracing.

    Que arrancar diga **a dónde cree que exporta** convierte «no llegan trazas»
    —que pueden ser diez cosas— en una.
    """
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover
        return "sin trazas: falta el extra `otel`"

    proveedor = trace.get_tracer_provider()
    nombre = type(proveedor).__name__
    destino = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "sin declarar")

    # Se pregunta **abriendo un span y mirando si graba**, no por el nombre de
    # la clase. La primera versión de esto miraba el nombre y daba por bueno un
    # `ProxyTracerProvider` —que es exactamente el que hay cuando nadie ha
    # configurado nada— así que la sonda escrita para detectar el silencio
    # informaba de que todo iba bien. Un nombre es una etiqueta; grabar o no
    # grabar es la propiedad.
    sonda = proveedor.get_tracer("synaptum").start_span("synaptum.probe")
    graba = sonda.is_recording()
    sonda.end()

    if not graba:
        return (
            f"trazas EN NINGUNA PARTE: el proveedor ({nombre}) las descarta. Los "
            f"spans se emiten, no llegan a ningún sitio, y nada falla — solo "
            f"faltan trazas, que es lo que nadie mira. "
            f"(OTEL_EXPORTER_OTLP_ENDPOINT={destino})"
        )
    return f"trazas por {nombre} · OTEL_EXPORTER_OTLP_ENDPOINT={destino}"


def _ns(evento: StepEvent) -> int | None:
    """La marca del evento en nanosegundos, que es lo que OTel espera."""
    return None if evento.at is None else int(evento.at * 1_000_000_000)


async def traced(
    pasos: AsyncIterator[Any],
    *,
    run_id: str,
    agent: str = "",
    model: str = "",
    guardrail_attribute: str = ATRIBUTO_GUARDRAIL,
) -> AsyncIterator[Any]:
    """Cede los mismos pasos y, de paso, emite la estructura del run.

    Args:
        pasos: lo que devuelve ``Agent.run`` o ``Agent.stream``.
        run_id: identidad del run, que es por lo que se busca en un incidente.
        agent: nombre del agente, para distinguir runs de distintos agentes.
        model: el modelo pedido.  Se declara aquí porque el span del propio
            `chat` lo emite quien lo ejecuta y no nosotros.
        guardrail_attribute: con qué nombre se marca lo que el gobierno paró.
            El valor por defecto es la convención de una plataforma concreta;
            quien exporte a otra pone el suyo.
    """
    from opentelemetry import trace

    tracer = _tracer()
    raiz = tracer.start_span(
        "agent.run",
        attributes={
            "synaptum.run_id": run_id,
            "synaptum.agent": agent,
            "gen_ai.request.model": model,
            "gen_ai.operation.name": "agent",
        },
    )

    turno: Any = None
    numero = 0
    abiertos: dict[str, Any] = {}

    try:
        with trace.use_span(raiz, end_on_exit=False):
            async for paso in pasos:
                if isinstance(paso, StepEvent):
                    turno, numero = _observar(
                        tracer, paso, raiz, turno, numero, abiertos, run_id,
                        guardrail_attribute,
                    )
                yield paso
    finally:
        # Cerrar lo que quedara abierto: un run cancelado a mitad deja turnos
        # vivos, y un span sin cerrar no aparece en ninguna traza — se perdería
        # justo el run que más interesa mirar.
        for span in (*abiertos.values(), turno):
            if span is not None:
                span.end()
        raiz.end()


def _observar(tracer, paso, raiz, turno, numero, abiertos, run_id, guardrail):
    """Traduce un evento del bucle a spans.  Devuelve `(turno, numero)`."""
    from opentelemetry import trace

    # ── Frontera de turno ────────────────────────────────────────────────────
    if isinstance(paso, ModelStep) and paso.phase is Phase.ATTEMPTED:
        if turno is not None:
            turno.end(end_time=_ns(paso))
        numero += 1
        atributos = {"synaptum.turn": numero, "synaptum.run_id": run_id}
        # Qué prompt produjo este turno. Está en el `meta` del paso porque ahí
        # queda **en el diario**; aquí se copia para que una traza y una
        # auditoría contesten lo mismo sin cruzar dos sistemas.
        atributos.update(
            {k: v for k, v in (paso.meta or {}).items() if k.startswith("prompt.")}
        )
        with trace.use_span(raiz, end_on_exit=False):
            turno = tracer.start_span(
                "agent.turn", start_time=_ns(paso), attributes=atributos,
            )

    padre = turno if turno is not None else raiz

    # ── Delegación y aprobación: estructura que solo conoce el bucle ─────────
    if isinstance(paso, (DelegateStep, ApprovalStep)):
        nombre = "agent.delegate" if isinstance(paso, DelegateStep) else "agent.approval"
        if paso.phase is Phase.ATTEMPTED:
            with trace.use_span(padre, end_on_exit=False):
                span = tracer.start_span(nombre, start_time=_ns(paso))
                if isinstance(paso, DelegateStep):
                    span.set_attribute("synaptum.delegate", paso.agent)
                    span.set_attribute("synaptum.sub_run_id", f"{run_id}/{paso.step_id}")
                else:
                    span.set_attribute("synaptum.subject", paso.subject)
                    # Un run suspendido esperando a una persona es algo que
                    # alguien tiene que atender, así que va al camino caliente.
                    span.set_attribute(guardrail, Disposition.REQUIRE_APPROVAL.value)
                abiertos[paso.step_id] = span
        else:
            span = abiertos.pop(paso.step_id, None)
            if span is not None:
                if isinstance(paso, DelegateStep):
                    _poner_consumo(span, paso.usage)
                span.end(end_time=_ns(paso))

    # ── Lo que el gobierno paró ──────────────────────────────────────────────
    decision = getattr(paso, "decision", None)
    if decision is not None and decision.disposition in _GOBIERNO:
        # **No se marca el span como error**, y es deliberado: una denegación de
        # política es el sistema funcionando. Reportarla como error enterraría
        # los fallos de verdad bajo un flujo de rechazos correctos.
        padre.set_attribute(guardrail, decision.disposition.value)
        padre.add_event(
            "synaptum.denied",
            attributes={
                "synaptum.step_id": paso.step_id,
                "synaptum.reason_code": decision.reason_code,
                "synaptum.message": decision.message,
            },
        )

    # ── Cierre ───────────────────────────────────────────────────────────────
    if isinstance(paso, FinalStep):
        if turno is not None:
            turno.end(end_time=_ns(paso))
            turno = None
        _poner_consumo(raiz, paso.usage)
        raiz.set_attribute("synaptum.finish", paso.reason or "")

    return turno, numero


def _poner_consumo(span, usage) -> None:
    """El consumo, con los nombres de la convención GenAI.

    Un contador **sin medir no se escribe**: un atributo a cero diría «no
    costó nada», y en un panel eso no se distingue de un run gratis.
    """
    for atributo, valor in (
        ("gen_ai.usage.input_tokens", usage.input),
        ("gen_ai.usage.output_tokens", usage.output),
        ("synaptum.usage.cache_read", usage.cache_read),
    ):
        if valor is not None:
            span.set_attribute(atributo, valor)
