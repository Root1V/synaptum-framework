"""
SYN-39 · Herramientas servidas por un servidor MCP.

Un servidor MCP publica herramientas con su esquema. Aquí se adaptan a lo que el
bucle ya sabe consumir — ``name``, ``definition``, ``invoke`` — así que un
servidor de terceros se usa exactamente como una función decorada con ``@tool``,
y ni el agente ni el gateway notan la diferencia.

Lo que decide el diseño de este módulo
---------------------------------------
**Las anotaciones de MCP son pistas, y el servidor no es autoridad sobre el
riesgo.** La propia especificación lo dice: un cliente no debe fiarse de
``readOnlyHint`` para decidir nada de seguridad. Un servidor equivocado —o
malicioso— puede declarar inocua una herramienta que borra.

De ahí salen las dos reglas de este módulo:

1. **Ausencia de anotación no significa inofensivo.** MCP define
   ``destructiveHint`` con valor por defecto *verdadero* y ``readOnlyHint`` con
   *falso*: quien no dice nada está diciendo «puede destruir». Se respeta, aunque
   produzca más herramientas marcadas destructivas de las que uno esperaría. Es
   lo correcto: la alternativa es que una herramienta ajena y sin declarar entre
   como ``READ`` y se ejecute sin que nadie la mire.

2. **La pista se traduce, no se cree.** Lo que sale de aquí es una *declaración*
   que viaja por la costura, y quien gobierna decide. Que Synaptum marque algo
   ``DESTRUCTIVE`` no ejecuta ninguna política; que lo marque ``READ`` tampoco
   autoriza nada.

Uso::

    async with MCPTools.stdio("uvx", "mcp-server-git", "--repository", ".") as git:
        agente = Agent("dev", model=..., tools=git.tools)

El contexto asíncrono no es decoración: un servidor por stdio es un proceso
hijo, y salir del ``async with`` es lo que lo cierra.
"""

from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator, Mapping, Sequence

from ..core.errors import ConfigurationError, ToolExecutionError
from ..core.types import Risk, Text, ToolDefinition, ToolResult

__all__ = ["MCPTools", "MCPTool", "mcp_risk"]


def mcp_risk(annotations: Any) -> Risk:
    """Traduce las pistas de una herramienta MCP a un nivel de riesgo.

    Los valores por defecto son los de la especificación MCP, no los nuestros:
    **sin anotaciones, destructiva**. Nuestro ``@tool`` usa ``READ`` por defecto
    porque quien escribe la función está delante y puede declarar; aquí el autor
    no está, y suponer en su lugar es suponer a favor.
    """
    if annotations is None:
        return Risk.DESTRUCTIVE

    solo_lectura = getattr(annotations, "read_only_hint", None)
    destructiva = getattr(annotations, "destructive_hint", None)

    if solo_lectura is True:
        return Risk.READ
    if destructiva is False:
        # No destruye, pero escribe: no es lo mismo que leer.
        return Risk.SOFT_WRITE
    return Risk.DESTRUCTIVE


def _mcp_idempotent(annotations: Any) -> bool:
    """Solo cuando se declara explícitamente.

    Lo mismo que en ``@tool``: quien calla paga la garantía cara. Un
    ``idempotentHint`` ausente no es una promesa.
    """
    return getattr(annotations, "idempotent_hint", None) is True


class MCPTool:
    """Una herramienta de un servidor MCP, con la forma que el bucle consume.

    Cumple el mismo contrato que ``Tool``: ``name``, ``definition``, ``invoke``.
    Nada más hace falta, y por eso no hereda de nada.
    """

    def __init__(self, session: Any, spec: Any, *, prefix: str = "") -> None:
        self._session = session
        self._remote_name = spec.name
        esquema = getattr(spec, "input_schema", None) or {"type": "object", "properties": {}}
        self.definition = ToolDefinition(
            name=f"{prefix}{spec.name}" if prefix else spec.name,
            description=(spec.description or "").strip(),
            parameters=dict(esquema),
            risk=mcp_risk(getattr(spec, "annotations", None)),
            idempotent=_mcp_idempotent(getattr(spec, "annotations", None)),
        )

    @property
    def name(self) -> str:
        return self.definition.name

    def __repr__(self) -> str:  # pragma: no cover
        return f"MCPTool({self.name!r}, risk={self.definition.risk.value})"

    async def invoke(self, call_id: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Llama al servidor y traduce su respuesta.

        Un error del servidor vuelve como ``ToolResult`` con ``is_error``, igual
        que el de una función local: un modelo que no ve el error no puede
        corregirlo. Lo que no se traga es un fallo de transporte — que el
        servidor se haya caído no es algo que el modelo pueda arreglar.
        """
        try:
            resultado = await self._session.call_tool(self._remote_name, dict(arguments))
        except Exception as fallo:  # noqa: BLE001
            raise ToolExecutionError(
                f"El servidor MCP falló al ejecutar '{self.name}': {fallo}",
                tool=self.name,
                retryable=True,
            ) from fallo

        return ToolResult.of(call_id, _texto(resultado), is_error=bool(
            getattr(resultado, "is_error", False)
        ))


def _texto(resultado: Any) -> str:
    """El contenido de un ``CallToolResult``, como texto.

    MCP admite contenido de varios tipos; hoy se traduce el texto y **se dice**
    lo que no se traduce en vez de descartarlo en silencio. Una imagen que
    desaparece sin rastro es peor que una línea que avisa de que llegó una.
    """
    partes: list[str] = []
    for parte in getattr(resultado, "content", None) or ():
        texto = getattr(parte, "text", None)
        if texto is not None:
            partes.append(str(texto))
        else:
            partes.append(f"[contenido {getattr(parte, 'type', 'desconocido')} no traducido]")

    estructurado = getattr(resultado, "structured_content", None)
    if not partes and estructurado is not None:
        from ..core.types import dumps

        return dumps(estructurado)
    return "\n".join(partes)


class MCPTools:
    """Las herramientas de un servidor MCP, listas para pasárselas a un ``Agent``.

    Args:
        session: sesión MCP ya inicializada.
        prefix: prefijo para los nombres. Con dos servidores que publiquen
            ``search``, el modelo no puede distinguirlos — y el que gana es el
            que se registró último, en silencio.
    """

    def __init__(self, session: Any, *, prefix: str = "") -> None:
        self._session = session
        self._prefix = prefix
        self.tools: list[MCPTool] = []

    async def discover(self) -> list[MCPTool]:
        """Pregunta al servidor qué publica y lo adapta."""
        listado = await self._session.list_tools()
        self.tools = [
            MCPTool(self._session, spec, prefix=self._prefix)
            for spec in getattr(listado, "tools", None) or ()
        ]
        return self.tools

    def __iter__(self):
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    @property
    def destructivas(self) -> list[MCPTool]:
        """Las que entran marcadas como destructivas.

        Útil para mirarlas antes de dárselas a un agente: con un servidor que no
        anota nada, son **todas**, y eso conviene verlo en vez de descubrirlo.
        """
        return [t for t in self.tools if t.definition.risk is Risk.DESTRUCTIVE]

    @classmethod
    @contextlib.asynccontextmanager
    async def stdio(
        cls, command: str, *args: str, prefix: str = "", env: Mapping[str, str] | None = None
    ) -> AsyncIterator["MCPTools"]:
        """Arranca un servidor MCP por stdio y descubre sus herramientas.

        Es un contexto asíncrono porque el servidor es un **proceso hijo**: salir
        del ``async with`` es lo que lo cierra. Dejarlo abierto deja el proceso
        vivo, y un agente que termina no debería dejar procesos detrás.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as falta:
            raise ConfigurationError(
                "El cliente MCP necesita el extra: `pip install synaptum[mcp]`."
            ) from falta

        parametros = StdioServerParameters(command=command, args=list(args), env=dict(env or {}))
        async with stdio_client(parametros) as (lectura, escritura):
            async with ClientSession(lectura, escritura) as sesion:
                await sesion.initialize()
                herramientas = cls(sesion, prefix=prefix)
                await herramientas.discover()
                yield herramientas

    @classmethod
    @contextlib.asynccontextmanager
    async def connected(cls, session: Any, *, prefix: str = "") -> AsyncIterator["MCPTools"]:
        """Sobre una sesión que ya existe — la abre y la cierra quien la creó."""
        herramientas = cls(session, prefix=prefix)
        await herramientas.discover()
        yield herramientas
