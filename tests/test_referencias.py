"""Pruebas de la selección de referencias y del optimizador de plantillas (Sección 2.5)."""
import itertools

import numpy as np
import pandas as pd

from diana_comparativo.estabilidad import assignment_comparison, build_resampling, set_geometry, solve_greedy_swap
from diana_comparativo.referencias import k_por_numero_de_niños, seleccionar_referencias


def grupo_sintetico(n_niños=12, por_niño=2, semilla=1):
    rng = np.random.default_rng(semilla)
    ejecuciones = pd.DataFrame({
        "segment_id": [f"S{i:03d}" for i in range(n_niños * por_niño)],
        "excel_children": [str(i // por_niño) for i in range(n_niños * por_niño)],
    })
    puntos = rng.normal(size=(len(ejecuciones), 2))
    puntos[0] = 0.0  # la ejecución S000 es la más central
    matriz = pd.DataFrame([{"segment_id_a": a, "segment_id_b": b, "distance": float(np.linalg.norm(puntos[i] - puntos[j]))}
                           for (i, a), (j, b) in itertools.combinations(enumerate(ejecuciones.segment_id), 2)])
    return matriz, ejecuciones


def test_k_depende_del_numero_de_niños():
    assert [k_por_numero_de_niños(n) for n in (4, 5, 9, 10, 19, 20)] == [1, 2, 2, 3, 3, 4]


def test_primaria_es_el_medoide_y_topk_usa_niños_distintos():
    matriz, ejecuciones = grupo_sintetico()
    seleccion = seleccionar_referencias(matriz, ejecuciones)
    medias = pd.concat([matriz.rename(columns={"segment_id_a": "s"}), matriz.rename(columns={"segment_id_b": "s"})]).groupby("s")["distance"].mean()
    assert seleccion.iloc[0]["segment_id"] == medias.idxmin()
    assert len(seleccion) == 3  # 12 niños -> k = 3
    assert seleccion["excel_children"].is_unique


def test_distancias_no_finitas_se_excluyen_de_la_centralidad():
    matriz, ejecuciones = grupo_sintetico()
    matriz.loc[matriz.segment_id_a.eq("S000"), "distance"] = np.inf
    seleccion = seleccionar_referencias(matriz, ejecuciones)
    assert np.isfinite(seleccion["mean_distance_to_group"]).all()


def test_optimizador_respeta_una_plantilla_por_niño():
    rng = np.random.default_rng(0)
    costos = rng.uniform(size=(10, 8))
    niño_de_cada_candidata = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    solucion = solve_greedy_swap(costos, niño_de_cada_candidata, [f"C{i}" for i in range(8)], k=3, rng=rng)
    assert len({int(niño_de_cada_candidata[i]) for i in solucion["selected_positions"]}) == 3


def test_metricas_de_estabilidad_son_maximas_para_conjuntos_identicos():
    distancias = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    geometria = set_geometry(distancias, [0, 2], [0, 2], between_scale=1.0, within_scale=1.0)
    assert geometria["hungarian_mean_between"] == 0.0
    acuerdo = assignment_comparison(np.array([0, 1, 1]), np.array([0, 1, 1]), np.array([0, 1]), k=2)
    assert acuerdo["assignment_agreement"] == 1.0


def test_remuestreo_reserva_niños_y_anida_subconjuntos():
    plan = build_resampling([str(i) for i in range(20)], 3, [3, 5, 8, 10, 12, 15], np.random.default_rng(1))
    assert len(plan["holdout"]) == 3 and plan["maximum_train"] == 17
    assert set(plan["nested"][5]) <= set(plan["nested"][8]) and not set(plan["holdout"]) & set(plan["nested"][17])
