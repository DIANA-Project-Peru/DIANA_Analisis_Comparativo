"""Pruebas del núcleo de comparación DTW (Sección 2.4)."""
import math

import numpy as np
import pytest

from diana_comparativo.dtw import alineamiento_dtw, articulaciones_requeridas, costo_local, distancia_dtw


def secuencia(n, fase=0.0, semilla=0):
    rng = np.random.default_rng(semilla)
    base = rng.normal(0, 0.5, size=(17, 2))
    t = np.linspace(0, 2 * np.pi, n)[:, None, None]
    movimiento = np.zeros((1, 17, 2))
    movimiento[0, 7:11, 1] = 0.4
    return (base[None] + movimiento * np.sin(t + fase)).reshape(n, -1)


def test_soporte_minimo_es_9_de_17():
    assert articulaciones_requeridas(17, 0.5) == 9


def test_costo_local_reescala_por_articulaciones_comunes():
    a = np.zeros((17, 2))
    b = np.ones((17, 2))
    completo = costo_local(a, b)
    assert completo == pytest.approx(math.sqrt(17 * 2))
    b_parcial = b.copy()
    b_parcial[:5] = np.nan  # 12 articulaciones comunes
    assert costo_local(a, b_parcial) == pytest.approx(completo)  # el factor J/k conserva la escala


def test_par_sin_soporte_es_no_admisible():
    a = np.zeros((17, 2))
    b = np.zeros((17, 2))
    b[:9] = np.nan  # solo 8 comunes
    assert math.isinf(costo_local(a, b))


def test_distancia_cero_para_secuencias_identicas():
    a = secuencia(25)
    assert distancia_dtw(a, a) == pytest.approx(0.0)


def test_distancia_rapida_coincide_con_alineamiento_explicito():
    a, b = secuencia(20), secuencia(31, fase=0.3, semilla=0)
    b[4:9, 9] = np.nan
    assert distancia_dtw(a, b) == pytest.approx(alineamiento_dtw(a, b)["distancia"], rel=1e-12)


def test_alinea_ejecuciones_de_distinta_duracion():
    lenta, rapida = secuencia(40), secuencia(20)
    otra_accion = secuencia(20, semilla=5)
    assert distancia_dtw(lenta, rapida) < distancia_dtw(lenta, otra_accion)


def test_fotograma_sin_soporte_produce_abstencion():
    a, b = secuencia(20), secuencia(22)
    b[10, :20] = np.nan  # un fotograma con 7 de 17 articulaciones
    assert math.isinf(distancia_dtw(a, b))
    assert alineamiento_dtw(a, b)["camino"] == []
