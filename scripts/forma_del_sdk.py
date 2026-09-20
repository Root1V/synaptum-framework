"""Vuelca la forma pública del SDK de plataforma, para poder restarla entre versiones.

El método es de Axonium y resuelve un agujero que yo había dejado abierto por
escrito: nuestro ``_field()`` lee atributo **o** clave, así que absorbe un cambio
de forma sin romperse **y sin avisar**. Entre `rc3` y `rc4` las tool calls
pasaron de `list[dict]` a `list[ToolCall]`, no nos rompió, y nos enteramos porque
otro equipo lo publicó.

Nuestro canario comprueba que los campos que leemos **sigan existiendo**. Esto
comprueba algo que aquel no ve: que **no cambien de tipo**. Es justo lo que un
accesor tolerante se traga en silencio.

    uv run python scripts/forma_del_sdk.py            # imprime la forma de hoy
    uv run python scripts/forma_del_sdk.py --check    # ¿coincide con la anotada?
    uv run python scripts/forma_del_sdk.py --write    # anota la de hoy

Es el mismo patrón que ``tests/superficie-publica.json`` para nuestra propia API:
un fichero anotado, y un test que falla cuando deja de corresponder. Que falle no
significa que el cambio esté mal — significa que hay que mirarlo.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ANOTADA = RAIZ / "tests" / "forma-del-sdk.json"


def forma() -> dict:
    """La superficie pública del SDK: símbolos y el tipo de cada campo."""
    import axonium

    salida: dict = {
        "version": getattr(axonium, "__version__", "?"),
        "all": sorted(axonium.__all__),
        "models": {},
    }
    for nombre in sorted(axonium.__all__):
        objeto = getattr(axonium, nombre)
        campos = getattr(objeto, "model_fields", None)
        if inspect.isclass(objeto) and campos:
            salida["models"][nombre] = {
                campo: str(info.annotation) for campo, info in sorted(campos.items())
            }
    return salida


def diferencias(antes: dict, ahora: dict) -> list[str]:
    """Qué cambió, **separando lo aditivo de lo que rompe**.

    La distinción es el motivo de existir de esto: un campo nuevo no nos afecta,
    y un campo que cambia de tipo es exactamente lo que nuestro accesor se traga
    sin avisar.
    """
    cambios: list[str] = []

    fuera = set(antes["all"]) - set(ahora["all"])
    if fuera:
        cambios.append(f"SÍMBOLOS QUITADOS  {sorted(fuera)}")
    nuevos = set(ahora["all"]) - set(antes["all"])
    if nuevos:
        cambios.append(f"símbolos añadidos  {sorted(nuevos)}")

    for modelo, campos_antes in antes["models"].items():
        campos_ahora = ahora["models"].get(modelo)
        if campos_ahora is None:
            cambios.append(f"MODELO QUITADO     {modelo}")
            continue
        for campo, tipo in campos_antes.items():
            if campo not in campos_ahora:
                cambios.append(f"CAMPO QUITADO      {modelo}.{campo}")
            elif campos_ahora[campo] != tipo:
                cambios.append(
                    f"CAMBIO DE TIPO     {modelo}.{campo}: {tipo} → {campos_ahora[campo]}"
                )
        for campo in set(campos_ahora) - set(campos_antes):
            cambios.append(f"campo añadido      {modelo}.{campo}")

    return cambios


def main() -> int:
    try:
        ahora = forma()
    except ImportError:
        print("el SDK no está instalado; nada que comprobar")
        return 0

    if "--write" in sys.argv:
        ANOTADA.write_text(json.dumps(ahora, indent=1, sort_keys=True) + "\n")
        print(f"anotada la forma de {ahora['version']} en {ANOTADA.relative_to(RAIZ)}")
        return 0

    if "--check" not in sys.argv:
        print(json.dumps(ahora, indent=1, sort_keys=True))
        return 0

    if not ANOTADA.exists():
        print(f"no hay forma anotada. Crea una con --write", file=sys.stderr)
        return 1

    antes = json.loads(ANOTADA.read_text())
    cambios = diferencias(antes, ahora)
    if not cambios:
        print(f"{antes['version']} → {ahora['version']}: aditivo, cero cambios de tipo")
        return 0

    print(f"{antes['version']} → {ahora['version']}:", file=sys.stderr)
    for cambio in cambios:
        print(f"  {cambio}", file=sys.stderr)
    print(
        "\nMira si alguno te afecta y anota la nueva forma con --write.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
