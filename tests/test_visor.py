"""Prueba del visor de alineamiento en modo demostración (sin datos de participantes)."""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visor_alineamiento.py"


def cargar_visor():
    spec = importlib.util.spec_from_file_location("visor_alineamiento", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_html_autocontenido_con_camino_y_celdas_no_admisibles():
    visor = cargar_visor()
    caso, referencia = visor.secuencias_demo()
    html = visor.construir_html(caso, referencia, "prueba", "nota")
    datos = json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1))
    assert "<script src" not in html  # sin dependencias externas
    assert datos["camino"][0] == [0, 0] and datos["camino"][-1] == [len(caso) - 1, len(referencia) - 1]
    assert np.any(np.array(datos["costos"]) < 0)  # hay pares sin soporte, marcados en gris
