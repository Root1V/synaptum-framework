# 08 · Herramientas que no escribiste tú

> **Esto es un fichero que se ejecuta:** [`examples/agentes/08_herramientas_mcp.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/08_herramientas_mcp.py) ↗
> Esta página lo transcribe y enseña lo que imprime. Si dejan de coincidir, falla un test.

**Dominio: cualquiera de tus repos.** Hasta aquí cada herramienta era una función
con `@tool`. En una aplicación real, buena parte de lo que un agente necesita ya
existe detrás de un servidor MCP — git, ficheros, bases de datos, la API de tu
propia plataforma — y escribirlas otra vez es trabajo tirado.

`MCPTools` las trae con su esquema y las adapta a lo que el bucle ya consume, así
que **el agente no nota la diferencia**: son herramientas como las demás.

Lo que este ejemplo enseña de verdad no es cómo conectarse —son cuatro líneas—
sino **qué significa que la herramienta sea ajena**. Eso está al final y es la
parte que importa.

## Cómo correrlo

```bash
uv sync --extra mcp
uv run python examples/agentes/08_herramientas_mcp.py
```

No hace falta configurar nada: sin modelo, las respuestas van guionizadas y **todo lo
demás es real** — las herramientas se ejecutan, el journal se escribe, el consumo se mide.
Con `AXONIUM_CLIENT_ID` o `SYNAPTUM_BASE_URL` en el entorno, **el mismo fichero sin tocar**
habla con un modelo de verdad; lo que cambia entonces es lo que diga el modelo, no el
código. Ver [Modelos](04-modelos.md).

## Lo que imprime

```text
08 · Herramientas MCP
─────────────────────
sin inferencia · respuestas guionizadas (exporta SYNAPTUM_BASE_URL para usar un modelo real)

  el servidor publica 4 herramientas:

   · repo.listar_ficheros   riesgo=read         idempotente=True   Lista los ficheros de una carpeta del repositorio.
   · repo.leer_fichero      riesgo=read         idempotente=True   Lee las primeras líneas de un fichero del repositorio.
   · repo.buscar            riesgo=read         idempotente=False  Busca un patrón en los ficheros del repositorio.
   ⚠ repo.aplicar_parche    riesgo=destructive  idempotente=False  Aplica un parche a un fichero del repositorio.

  marcadas destructivas: ['repo.aplicar_parche']
  ↳ `repo.aplicar_parche` no anota nada, y eso NO significa inofensiva

  → repo.buscar(patron, extension)
    src/synaptum/a2a/types.py:52: class AgentCard:
  → repo.leer_fichero(ruta, lineas)
    [project]

  El agente vive en `src/synaptum/agent/agent.py` y el paquete se llama synaptum, versión 1.0.0rc2, sin dependencias duras.

  (el servidor se cerró al salir del `async with`)
```

```python
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synaptum import ALLOW, Agent, Decision, Disposition, FinalStep, Phase, Risk, Session, ToolStep
from synaptum.mcp import MCPTools
from synaptum.testing import calls, says

from comun import encabezado, gateway, nombre_del_modelo

SERVIDOR = Path(__file__).parent / "servidor_mcp" / "repo.py"
```

## Una política que decide sobre lo que no escribió

Aquí es donde `risk` deja de ser documentación. El servidor **insinúa**;
Synaptum **declara**; esto **decide**.

```python
def solo_lectura_sin_supervision(check) -> Decision:
    """Deja pasar lo que lee; detiene lo demás para que alguien lo mire."""
    if check.kind != "tool" or check.risk is Risk.READ:
        return ALLOW
    return Decision(
        disposition=Disposition.REQUIRE_APPROVAL,
        reason_code="herramienta_ajena_no_de_solo_lectura",
        message=(
            f"'{check.name}' viene de un servidor MCP y no se declara de solo "
            f"lectura (riesgo {check.risk.value}). Necesita una persona."
        ),
    )

async def main() -> None:
    encabezado("08 · Herramientas MCP")

    # Cuatro líneas. El `async with` no es decoración: el servidor es un proceso
    # hijo, y salir del bloque es lo que lo cierra.
    async with MCPTools.stdio(sys.executable, str(SERVIDOR), prefix="repo.") as repo:
        print(f"  el servidor publica {len(repo)} herramientas:\n")
        for herramienta in repo:
            d = herramienta.definition
            marca = "·" if d.risk is Risk.READ else "⚠"
            print(f"   {marca} {d.name:22} riesgo={d.risk.value:12} "
                  f"idempotente={str(d.idempotent):5}  {d.description}")

        print(f"\n  marcadas destructivas: {[t.name for t in repo.destructivas]}")
        print("  ↳ `repo.aplicar_parche` no anota nada, y eso NO significa inofensiva\n")

        agente = Agent(
            "explorador",
            model=nombre_del_modelo(),
            instructions=(
                "Exploras un repositorio con las herramientas disponibles. "
                "Responde en dos frases."
            ),
            tools=list(repo),
        )

        guion = [
            calls("repo.buscar", patron="class Agent", extension=".py"),
            calls("repo.leer_fichero", ruta="pyproject.toml", lineas=6),
            says("El agente vive en `src/synaptum/agent/agent.py` y el paquete se "
                 "llama synaptum, versión 1.0.0rc2, sin dependencias duras."),
        ]

        puerta = gateway(
            guion,
            tools=list(repo),
            policy=solo_lectura_sin_supervision,
            deny_tools={},
        )

        async for paso in agente.run(
            "¿Dónde está la clase Agent y qué versión tiene el paquete?",
            session=Session("mcp-1", puerta),
        ):
            match paso:
                case ToolStep(phase=Phase.ATTEMPTED, call=llamada) if llamada:
                    print(f"  → {llamada.name}({', '.join(llamada.arguments)})")
                case ToolStep(phase=Phase.COMPLETED, result=resultado) if resultado:
                    texto = " ".join(p.text for p in resultado.content if hasattr(p, "text"))
                    print(f"    {texto.splitlines()[0][:78] if texto else '—'}")
                case FinalStep(output=respuesta):
                    print(f"\n  {respuesta}")

    print("\n  (el servidor se cerró al salir del `async with`)")

if __name__ == "__main__":
    asyncio.run(main())
```

## Lo que esto enseña

Qué cambia cuando la herramienta es ajena
─────────────────────────────────────────

**1 · El riesgo lo insinúa quien no manda.** MCP trae `readOnlyHint`,
`destructiveHint` e `idempotentHint`, y su propia especificación dice que un
cliente **no debe fiarse de ellas** para decidir seguridad: un servidor
equivocado —o malicioso— puede declarar inocua una herramienta que borra.

Por eso la traducción es conservadora y **ausencia no es inofensivo**: MCP
define `destructiveHint` con defecto verdadero, así que una herramienta sin
anotar entra como `Risk.DESTRUCTIVE`. Produce más avisos de los que uno espera, y
es lo correcto — la alternativa es que algo ajeno y sin declarar se ejecute
sin que nadie lo mire.

Nuestro `@tool` usa `Risk.READ` por defecto porque **el autor está delante** y puede
declarar. Aquí no está.

**2 · Los errores llegan sin explicación.** Una tool local que revienta
devuelve su excepción al modelo, que suele corregir. Un servidor MCP no filtra
sus internos: llega «Error executing tool X» y nada más. El modelo sabe *que*
falló y casi nunca *por qué*.

**3 · Los nombres chocan.** Dos servidores que publiquen `search` no son
distinguibles para el modelo, y gana el último registrado, en silencio. Por eso
aquí va `prefix="repo."`.

**4 · El esquema es de otro.** Si el servidor cambia el suyo, tu agente empieza
a mandar argumentos que ya no encajan — sin que nada en tu repositorio haya
cambiado. Es el mismo problema que `@tool` resuelve derivando el esquema de la
firma, y con MCP vuelve, porque la firma vive en otro sitio.

---

**El fichero entero, para clonarlo y tocarlo:** [`examples/agentes/08_herramientas_mcp.py`](https://github.com/Root1V/synaptum-framework/blob/main/examples/agentes/08_herramientas_mcp.py) ↗

Está en [`examples/`](https://github.com/Root1V/synaptum-framework/tree/main/examples) con los otros quince, y todos corren igual.
