from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    """
    Unidad atómica de un prompt.

    Attributes:
        content:     Texto del prompt. Puede contener variables con sintaxis
                     Python estándar: "Eres un agente llamado {agent_id}."
        version:     Versión semántica del template para trazabilidad.
        description: Descripción legible del propósito del prompt.
        variables:   Valores por defecto para las variables del template.
                     Pueden ser sobreescritos al llamar a render().
    """

    content: str
    version: str = "1.0"
    description: str = ""
    variables: dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """
        Interpola las variables del template.

        Los kwargs tienen prioridad sobre self.variables.

        Example:
            tpl = PromptTemplate("Hola {name}, hoy es {date}.")
            tpl.render(name="Emeric", date="2026-02-28")
        """
        context = {**self.variables, **kwargs}
        return self.content.format_map(context) if context else self.content

    def __str__(self) -> str:
        return self.content
