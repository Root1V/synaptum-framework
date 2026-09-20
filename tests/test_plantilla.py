"""SYN-75 · La plantilla de proyecto tiene que funcionar.

Una plantilla que nadie ejecuta se pudre, y se pudre en silencio: el primero que
la copie descubre que el punto de partida no arranca. Es peor que no tenerla,
porque llega en el peor momento — cuando alguien está decidiendo si adoptar esto.

Se ejecuta contra el **código local**, no contra el synaptum publicado: así un
cambio que rompa la plantilla falla aquí, antes de publicar, y no en el
repositorio de quien la copió.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PLANTILLA = RAIZ / "plantilla"

pytestmark = pytest.mark.skipif(not PLANTILLA.exists(), reason="sin plantilla/")


def _en_entorno_aislado(*orden: str) -> subprocess.CompletedProcess:
    """Corre algo con el synaptum **de este árbol** y la plantilla **del fuente**.

    `--with-editable` no es un detalle: con `--with` a secas, uv sirve un wheel
    ya construido de la plantilla y **no ve las ediciones del fuente**. La
    primera versión de este test pasaba con un `import NoExiste` metido a
    propósito, en 0,21 segundos. Un test que no puede fallar es peor que no
    tenerlo: ocupa el sitio del que sí comprobaría algo.
    """
    return subprocess.run(
        [
            "uv", "run", "--isolated", "--no-project",
            "--with", str(RAIZ),
            "--with-editable", str(PLANTILLA),
            "--with", "pytest", "--with", "pytest-asyncio",
            *orden,
        ],
        cwd=RAIZ, capture_output=True, text=True, timeout=300,
    )


def test_the_template_runs():
    """`python -m mi_agente` arranca y produce una respuesta."""
    hecho = _en_entorno_aislado("python", "-m", "mi_agente")
    assert hecho.returncode == 0, hecho.stderr[-1500:]
    assert "consumo:" in hecho.stdout, hecho.stdout


def test_the_templates_own_tests_pass():
    """Los tests que la plantilla trae de ejemplo también tienen que pasar.

    Son lo primero que alguien copia para probar **su** agente: si vienen rotos,
    aprende a probar mal.
    """
    hecho = _en_entorno_aislado(
        "python", "-m", "pytest", str(PLANTILLA / "tests"),
        "-q", "-c", str(PLANTILLA / "pyproject.toml"), "--rootdir", str(PLANTILLA),
    )
    assert hecho.returncode == 0, hecho.stdout[-2000:]


def test_the_template_pins_a_version_that_exists():
    """No puede depender de una versión que nadie puede instalar."""
    import tomllib

    datos = tomllib.loads((PLANTILLA / "pyproject.toml").read_text())
    dependencias = datos["project"]["dependencies"]
    synaptum = [d for d in dependencias if d.startswith("synaptum")]

    assert synaptum, "la plantilla no depende de synaptum"
    assert ">=" in synaptum[0], f"sin mínimo declarado: {synaptum[0]}"
