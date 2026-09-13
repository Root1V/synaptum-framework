"""Synaptum no se acopla a ningún proyecto concreto.

Es un framework que puede invocarse desde cualquier sitio. Habrá despliegues
donde el arnés, el SDK y la plataforma de inferencia sean unos determinados, y
habrá otros donde no exista ninguno de los tres — y los dos casos tienen que
funcionar igual.

Esto se degrada por descuido y no por decisión: una ruta absoluta escrita
mientras se depura, un `import` de conveniencia, una docstring que cuenta el
despliegue de quien la escribió como si fuera la razón de ser del diseño. Nada
de eso duele en la máquina donde se escribió, que es exactamente el problema.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PAQUETE = Path(__file__).resolve().parents[1] / "src" / "synaptum"
RAIZ = Path(__file__).resolve().parents[1]

# El puente a un SDK concreto es un adaptador opcional, como lo sería uno de
# Anthropic: puede nombrar a su proveedor porque **es** su proveedor.  Lo que no
# puede es cargarse solo ni ser necesario para nada.
PUENTES = {"axonium.py"}


def _modulos(carpeta: Path):
    return [p for p in carpeta.rglob("*.py") if "__pycache__" not in p.parts]


def test_importing_synaptum_pulls_in_no_third_party_module():
    """Cero dependencias no es una promesa del `pyproject`: es comprobable.

    Se mide en un intérprete limpio, porque en el de la suite ya hay cargado lo
    que arrastran pytest y los extras — y ahí cualquier import de terceros
    pasaría desapercibido.
    """
    codigo = (
        "import sys, json; "
        "import synaptum; "
        "print(json.dumps(sorted("
        "  m for m in sys.modules"
        "  if not m.startswith('synaptum') and '.' not in m"
        "  and m not in sys.stdlib_module_names and not m.startswith('_')"
        ")))"
    )
    salida = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
    )
    import json

    ajenos = json.loads(salida.stdout)
    assert ajenos == [], f"importar synaptum cargó módulos de terceros: {ajenos}"


def test_no_vendor_bridge_is_imported_eagerly():
    """Un adaptador de proveedor concreto no puede cargarse sin pedirlo.

    Si se cargara, su dependencia pasaría de opcional a obligatoria de hecho, y
    el fallo aparecería al instalar en vez de al usar.
    """
    codigo = (
        "import sys, synaptum, synaptum.testing; "
        "print([m for m in sys.modules if 'axonium' in m])"
    )
    salida = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
    )
    assert salida.stdout.strip() == "[]", f"puente cargado sin pedirlo: {salida.stdout}"


def test_the_package_names_no_specific_deployment():
    """Los nombres propios de un despliegue no pertenecen al paquete.

    Un despliegue concreto vale como **evidencia** —«un gateway real llegó a
    mandar dos centinelas»— y ahí el nombre no aporta nada. Vale como
    **identidad** solo en el adaptador de ese proveedor. En cualquier otro sitio
    le cuenta a quien lo lee que necesita algo que no necesita.
    """
    nombres = re.compile(r"\b(Aeon|Axonium|Prometheus)\b")
    culpables = []
    for modulo in _modulos(PAQUETE):
        if modulo.name in PUENTES:
            continue
        for numero, linea in enumerate(modulo.read_text().splitlines(), 1):
            if nombres.search(linea):
                culpables.append(f"{modulo.relative_to(PAQUETE)}:{numero}")
    assert not culpables, (
        "el paquete nombra un despliegue concreto en: " + ", ".join(culpables)
    )


def test_no_source_file_carries_an_absolute_path():
    """Una ruta absoluta funciona en una máquina y en ninguna otra.

    Pasó: tres ficheros apuntaban a la carpeta de contratos de quien los
    escribió, y uno de ellos era un ejemplo — así que el repositorio venía con
    una ruta que solo existía en un portátil.
    """
    absoluta = re.compile(r"[\"'](/Users/|/home/|[A-Z]:\\\\)")
    culpables = []
    for carpeta in (PAQUETE, RAIZ / "tests", RAIZ / "examples"):
        for modulo in _modulos(carpeta):
            for numero, linea in enumerate(modulo.read_text().splitlines(), 1):
                if absoluta.search(linea):
                    culpables.append(f"{modulo.relative_to(RAIZ)}:{numero}")
    assert not culpables, "rutas absolutas en: " + ", ".join(culpables)


def test_the_suite_passes_without_the_shared_contracts():
    """Quien clona esto sin acceso a los contratos de otros equipos no ha hecho nada mal.

    Los corpus dorados son evidencia adicional. Si su ausencia rompiera la
    suite, serían una dependencia — y una que no se puede instalar.
    """
    entorno = {
        **{k: v for k, v in __import__("os").environ.items() if k != "SYNAPTUM_CONTRACTS"},
        "SYNAPTUM_CONTRACTS": "/no/existe/a/proposito",
    }
    resultado = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--ignore", str(Path(__file__).name and RAIZ / "tests" / "test_independencia.py")],
        cwd=RAIZ, env=entorno, capture_output=True, text=True,
    )
    assert resultado.returncode == 0, (
        "la suite falla sin los contratos compartidos:\n" + resultado.stdout[-2000:]
    )
