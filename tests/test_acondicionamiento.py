"""Pruebas del acondicionamiento y la normalización (Sección 2.3)."""
import json

import numpy as np
import pytest

from diana_comparativo.acondicionamiento import acondicionar_archivo, interpolar_brechas, mascara_validez, normalizar
from diana_comparativo.dtw import cargar_secuencia, validar_secuencia


def pose(n=12):
    puntos = np.tile(np.arange(17, dtype=float)[:, None] * [1.0, 2.0] + 10, (n, 1, 1))
    confianza = np.full((n, 17), 0.9)
    return puntos, confianza


def test_valor_centinela_y_baja_confianza_son_invalidos():
    puntos, confianza = pose(3)
    puntos[0, 4] = 0.0
    confianza[1, 5] = 0.2
    valida = mascara_validez(puntos, confianza)
    assert not valida[0, 4] and not valida[1, 5] and valida[2].all()


def test_solo_se_interpolan_brechas_interiores_de_hasta_tres_fotogramas():
    puntos, confianza = pose(14)
    confianza[2:5, 0] = 0.0     # brecha de 3: se interpola
    confianza[7:11, 1] = 0.0    # brecha de 4: queda faltante
    confianza[0, 2] = 0.0       # extremo: queda faltante
    _, valida = interpolar_brechas(puntos, confianza)
    assert valida[2:5, 0].all()
    assert not valida[7:11, 1].any()
    assert not valida[0, 2]


def test_normalizacion_centra_en_caderas_y_escala_por_torso():
    puntos, confianza = pose(4)
    coordenadas, valida = interpolar_brechas(puntos, confianza)
    normalizada, diagnostico = normalizar(coordenadas, valida)
    centro = normalizada[:, [11, 12]].mean(axis=1)
    assert np.allclose(centro, 0.0, atol=1e-6)
    assert diagnostico["root_fallback_frames"] == 0 and not diagnostico["scale_fallback_used"]


def test_origen_alternativo_cuando_faltan_ambas_caderas():
    puntos, confianza = pose(4)
    confianza[1, [11, 12]] = 0.0
    coordenadas, valida = interpolar_brechas(puntos, confianza, brecha_maxima=0)
    _, diagnostico = normalizar(coordenadas, valida)
    assert diagnostico["root_fallback_frames"] == 1


def test_archivo_acondicionado_es_compatible_con_la_comparacion(tmp_path):
    puntos, confianza = pose(10)
    origen = tmp_path / "pose.npz"
    np.savez(origen, keypoints=puntos, confidence=confianza, metadata_json=json.dumps({}))
    destino = tmp_path / "secuencia.npz"
    acondicionar_archivo(origen, destino)
    validar_secuencia(destino)
    assert cargar_secuencia(destino).shape == (10, 34)
    with pytest.raises(ValueError):
        validar_secuencia(destino, version="otra_version")
