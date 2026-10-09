"""Pruebas de la comparación con referencias y de los perfiles (Secciones 2.4 y 3.4)."""
import json

import numpy as np
import pandas as pd

from diana_comparativo import comparacion, perfiles
from diana_comparativo.dtw import ARTICULACIONES_COCO17, VERSION_ACONDICIONAMIENTO


def guardar(carpeta, sid, n, desplazamiento=0.0, sin_soporte=False):
    secuencia = np.tile(np.linspace(-1, 1, 17)[:, None], (n, 1, 2)).astype(np.float32) + desplazamiento
    valida = np.ones((n, 17), dtype=bool)
    if sin_soporte:
        valida[n // 2, :10] = False
        secuencia[~valida] = np.nan
    np.savez(carpeta / f"{sid}.npz", sequence=secuencia, valid_mask=valida, confidence=np.ones((n, 17)),
             keypoint_names=np.asarray(ARTICULACIONES_COCO17),
             metadata_json=json.dumps({"preprocessing_version": VERSION_ACONDICIONAMIENTO}))


def escenario(tmp_path):
    entrenamiento = pd.DataFrame({
        "segment_id": ["R1", "R2", "R3", "R4"], "excel_children": ["10", "11", "12", "13"],
        "action_code": "A01", "age_bin_12m_label": ["36_47mo", "36_47mo", "48_59mo", "48_59mo"]})
    validacion = pd.DataFrame({"segment_id": ["V1", "V2"], "excel_children": ["20", "21"],
                               "action_code": "A01", "age_bin_12m_label": ["36_47mo", "48_59mo"]})
    topk = entrenamiento.assign(rank=[1, 2, 1, 2])
    primarias = pd.DataFrame({"action_code": "A01", "age_bin_12m_label": ["36_47mo", "48_59mo"], "medoid_segment_id": ["R1", "R3"]})
    for sid, n, d in [("R1", 10, 0), ("R2", 12, .1), ("R3", 11, .2), ("V1", 9, .05), ("V2", 13, .15)]:
        guardar(tmp_path, sid, n, d)
    guardar(tmp_path, "R4", 10, .3, sin_soporte=True)
    catalogo = comparacion.catalogo_de_referencias(topk, primarias, entrenamiento)
    return entrenamiento, validacion, catalogo


def test_solicitudes_y_comparaciones_previstas(tmp_path):
    _, validacion, catalogo = escenario(tmp_path)
    solicitudes, previstas = comparacion.comparaciones_previstas(validacion, catalogo, tmp_path, grupos=("36_47mo", "48_59mo"))
    assert len(solicitudes) == 6  # 2 ejecuciones x 3 desplazamientos
    assert solicitudes.estado.value_counts().to_dict() == {"solicitada": 4, "fuera_de_dominio_etario": 2}
    assert len(previstas) == 8   # cada solicitud con 2 referencias


def test_abstencion_y_controles(tmp_path):
    entrenamiento, validacion, catalogo = escenario(tmp_path)
    _, previstas = comparacion.comparaciones_previstas(validacion, catalogo, tmp_path, grupos=("36_47mo", "48_59mo"))
    resultado = comparacion.calcular(previstas)
    assert (resultado.loc[resultado.id_referencia == "R4", "estado"] == "sin_camino_admisible").all()
    assert resultado.loc[resultado.id_referencia == "R4", "distancia"].isna().all()
    controles = comparacion.controles_de_separacion(entrenamiento, validacion, catalogo, previstas, resultado)
    assert all(c["cumple"] for c in controles)


def test_resumen_topk_exige_todas_las_referencias(tmp_path):
    _, validacion, catalogo = escenario(tmp_path)
    _, previstas = comparacion.comparaciones_previstas(validacion, catalogo, tmp_path, grupos=("36_47mo", "48_59mo"))
    tabla = perfiles.perfiles_por_ejecucion(comparacion.calcular(previstas)).set_index(["id_ejecucion", "resumen"])
    # V2 (48-59): su grupo esperado contiene R4 sin soporte -> la primaria (R3) da distancia, top-k no.
    assert np.isfinite(tabla.loc[("V2", "primaria"), "d_E"])
    assert np.isnan(tabla.loc[("V2", "mediana_topk"), "d_E"])
    # V1 (36-47) no tiene grupo menor: perfil incompleto por construcción.
    assert not tabla.loc[("V1", "primaria"), "perfil_completo"]
