"""
SYN-38 · Qué costó un run y por qué.

``Usage`` dice **cuánto** se gastó. Esto dice **por qué**, que es lo único que se
puede accionar. Tres preguntas que solo se ven agregando los turnos:

**¿Qué fracción del contexto se sirvió de caché?** Un run sano sube hacia el 95 %
y se queda ahí. Si cae de golpe en el turno 7, algo rompió el prefijo — y hoy eso
solo se ve mirando números a mano.

**¿Cuántas veces se reescribió el prefijo?** Debería ser **cero**. Reanudar con
otra configuración ya está prohibido (`prefix`), pero dentro de un mismo run el
prefijo todavía puede cambiar solo: unas instrucciones con la fecha dentro lo
reescriben en cada turno, y el síntoma es una caché que nunca arranca.

**¿Cómo crece el coste por turno?** El historial se reenvía entero cada vez, así
que la entrada crece. Saber la pendiente es lo que dice cuándo merece la pena
compactar — y esa decisión **es incorrecta sin ``cache_read``**: sin saber cuánto
se sirve de caché no se puede comparar conservar contra resumir.

Se calcula **del journal**, no en vivo
---------------------------------------
Así funciona sobre un run terminado, sobre uno reanudado y sin tener el agente
delante. Un informe que solo se pudiera obtener mientras el run corre no serviría
para lo único que hace falta: mirar ayer.

Lo que esto **no** hace
------------------------
No exporta nada. Los nombres de atributo, las unidades y el transporte son de la
instrumentación (`SYN-37`), y esa forma la fija la plataforma de observabilidad
que los consuma. Emitir aquí un formato propio sería construir algo que habría
que tirar.

Y no habla de dinero. Los tokens los sabemos; los precios no, y convertirlos con
una tarifa inventada daría una cifra con aspecto de exacta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.events import ModelStep, Phase
from ..core.protocols import RunState
from ..core.types import Usage
from .prefix import describe_prefix_change, prefix_fingerprint

__all__ = ["ContextEconomy", "TurnEconomy", "economy"]


@dataclass(frozen=True, slots=True)
class TurnEconomy:
    """Un turno: una llamada al modelo y lo que costó."""

    step_id: str
    usage: Usage
    prefix_rewritten: bool
    """El prefijo estable cambió respecto al turno anterior.

    En el primer turno es ``False``: no hay nada anterior con lo que compararlo,
    y llamarlo reescritura sería contar el estreno como una regresión."""

    @property
    def cache_hit_ratio(self) -> float | None:
        """Fracción de la entrada servida desde caché, o ``None`` si no se midió.

        ``None`` y ``0.0`` son cosas distintas: una es «nadie lo midió» y la otra
        «se midió y no hubo acierto». Confundirlas hace creer que un run sin
        instrumentar tiene la caché rota."""
        return self.usage.cache_hit_ratio


@dataclass(frozen=True, slots=True)
class ContextEconomy:
    """El informe de un run."""

    run_id: str
    turns: tuple[TurnEconomy, ...] = ()
    prefix_changes: tuple[str, ...] = field(default_factory=tuple)
    """Qué cambió en cada reescritura, en términos de quien lo tiene que arreglar."""

    @property
    def total(self) -> Usage:
        """Suma de todos los turnos.  ``None`` en un contador se propaga."""
        acumulado = Usage.zero()
        for turno in self.turns:
            acumulado += turno.usage
        return acumulado

    @property
    def cache_hit_ratio(self) -> float | None:
        """Acierto de caché del run entero.

        Se calcula sobre los totales y no como media de los turnos: una media
        pesaría igual un turno de 50 tokens que uno de 50.000."""
        entrada, cacheado = self.total.input, self.total.cache_read
        if not entrada or cacheado is None:
            return None
        return cacheado / entrada

    @property
    def prefix_rewrites(self) -> int:
        """Cuántas veces cambió el prefijo estable dentro del run.

        **Debería ser cero.** Cada reescritura tira toda la caché posterior."""
        return sum(1 for t in self.turns if t.prefix_rewritten)

    @property
    def input_growth(self) -> float | None:
        """Cuánto crece la entrada por turno, en tokens.

        Es la pendiente entre el primer turno y el último, no una regresión: con
        cinco puntos una regresión da una cifra más precisa y no más cierta.
        ``None`` si hay menos de dos turnos medidos."""
        medidos = [t.usage.input for t in self.turns if t.usage.input is not None]
        if len(medidos) < 2:
            return None
        return (medidos[-1] - medidos[0]) / (len(medidos) - 1)

    def report(self) -> str:
        """El informe en texto, para leerlo en un terminal o pegarlo en un ticket."""
        lineas = [f"Run {self.run_id} · {len(self.turns)} turnos"]

        acierto = self.cache_hit_ratio
        lineas.append(
            f"  caché      {acierto:.0%} de la entrada servida de caché"
            if acierto is not None
            else "  caché      sin medir — el proveedor no reporta cache_read"
        )

        crecimiento = self.input_growth
        if crecimiento is not None:
            lineas.append(f"  crecimiento {crecimiento:+,.0f} tokens de entrada por turno")

        if self.prefix_rewrites:
            lineas.append(
                f"  ⚠ prefijo  reescrito {self.prefix_rewrites} vez/veces "
                "— cada una tira toda la caché posterior"
            )
            lineas += [f"      · {c}" for c in self.prefix_changes]
        else:
            lineas.append("  prefijo    estable durante todo el run")

        total = self.total
        lineas.append(
            f"  total      entrada={_o(total.input)} salida={_o(total.output)} "
            f"caché={_o(total.cache_read)}"
        )
        return "\n".join(lineas)


def _o(valor: int | None) -> str:
    return "sin medir" if valor is None else f"{valor:,}".replace(",", ".")


def economy(state: RunState) -> ContextEconomy:
    """Calcula el informe de un run a partir de su journal.

    Empareja cada intención de modelo con su resultado: la petición vive en
    ``ATTEMPTED`` —es donde está el prefijo— y el consumo en ``COMPLETED``.
    Un turno sin resultado no se cuenta: se intentó y no se sabe qué costó.
    """
    peticiones: dict[str, object] = {}
    turnos: list[TurnEconomy] = []
    cambios: list[str] = []

    anterior_huella: str | None = None
    anterior_peticion = None

    for evento in state.events:
        if not isinstance(evento, ModelStep):
            continue

        if evento.phase is Phase.ATTEMPTED and evento.request is not None:
            peticiones[evento.step_id] = evento.request
            continue

        if evento.phase is not Phase.COMPLETED:
            continue

        peticion = peticiones.get(evento.step_id)
        reescrito = False
        if peticion is not None:
            huella = prefix_fingerprint(peticion)
            if anterior_huella is not None and huella != anterior_huella:
                reescrito = True
                cambios.append(
                    f"{evento.step_id}: {describe_prefix_change(anterior_peticion, peticion)}"
                )
            anterior_huella, anterior_peticion = huella, peticion

        turnos.append(
            TurnEconomy(step_id=evento.step_id, usage=evento.usage, prefix_rewritten=reescrito)
        )

    return ContextEconomy(
        run_id=state.run_id, turns=tuple(turnos), prefix_changes=tuple(cambios)
    )
