"""Acondicionamiento y normalización de secuencias esqueléticas (Sección 2.3).

Pasos, en este orden:

1. **Validez.** Una articulación es válida si sus coordenadas y su confianza son
   finitas, la confianza es >= 0,30 y no corresponde al valor (0, 0) que el
   extractor de pose usa para indicar ausencia.
2. **Interpolación.** Solo las brechas interiores de hasta 3 fotogramas se
   completan linealmente (x e y a la vez). Las brechas largas y los extremos
   quedan como faltantes; no se imputa nada más.
3. **Centrado.** Lo inválido se excluye antes de normalizar. El origen de cada
   fotograma es la media de las caderas válidas; si no hay ninguna, la media de
   las articulaciones visibles.
4. **Escala.** Mediana, en la ejecución, de las distancias positivas entre seis
   pares del torso; si no existe ninguna, la extensión de las coordenadas
   visibles. Mínimo de 1 píxel.

El resultado se guarda como NPZ con la secuencia normalizada, la máscara de
validez (que incluye las interpolaciones permitidas), la confianza original y
los metadatos del procedimiento.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dtw import ARTICULACIONES_COCO17, VERSION_ACONDICIONAMIENTO

UMBRAL_CONFIANZA = 0.30
BRECHA_MAXIMA = 3
CADERAS = [11, 12]
PARES_TORSO = [(5, 6), (11, 12), (5, 11), (6, 12), (5, 12), (6, 11)]


def mascara_validez(coordenadas: np.ndarray, confianza: np.ndarray, umbral: float = UMBRAL_CONFIANZA) -> np.ndarray:
    """Máscara (T, J) de observaciones válidas en la pose original."""
    return (np.isfinite(coordenadas).all(axis=2) & np.isfinite(confianza)
            & (confianza >= umbral) & ~np.all(coordenadas == 0, axis=2))


def interpolar_brechas(coordenadas: np.ndarray, confianza: np.ndarray,
                       umbral: float = UMBRAL_CONFIANZA, brecha_maxima: int = BRECHA_MAXIMA) -> tuple[np.ndarray, np.ndarray]:
    """Marca lo inválido como NaN e interpola solo brechas interiores cortas."""
    salida = np.array(coordenadas, dtype=np.float32, copy=True)
    valida = mascara_validez(salida, confianza, umbral)
    salida[~valida] = np.nan
    for articulacion in range(salida.shape[1]):
        indices = np.flatnonzero(valida[:, articulacion])
        for izquierda, derecha in zip(indices[:-1], indices[1:]):
            brecha = derecha - izquierda - 1
            if 0 < brecha <= brecha_maxima:
                tramo = np.linspace(salida[izquierda, articulacion], salida[derecha, articulacion], brecha + 2)[1:-1]
                salida[izquierda + 1:derecha, articulacion] = tramo
                valida[izquierda + 1:derecha, articulacion] = True
    return salida, valida


def _media_finita(valores: np.ndarray, eje: int) -> np.ndarray:
    cuenta = np.isfinite(valores).sum(axis=eje)
    total = np.nansum(valores, axis=eje)
    return np.divide(total, cuenta, out=np.full_like(total, np.nan), where=cuenta > 0)


def escala_torso(coordenadas: np.ndarray) -> tuple[float, bool]:
    """Escala robusta de la ejecución y si se usó la alternativa de respaldo."""
    distancias = np.concatenate([np.linalg.norm(coordenadas[:, a] - coordenadas[:, b], axis=1) for a, b in PARES_TORSO])
    distancias = distancias[np.isfinite(distancias) & (distancias > 1e-6)]
    if distancias.size:
        return float(max(np.nanmedian(distancias), 1.0)), False
    visibles = coordenadas[np.isfinite(coordenadas).all(axis=2)]
    if visibles.size == 0:
        return 1.0, True
    extension = np.nanmax(visibles, axis=0) - np.nanmin(visibles, axis=0)
    return float(max(np.linalg.norm(extension), 1.0)), True


def normalizar(coordenadas: np.ndarray, valida: np.ndarray) -> tuple[np.ndarray, dict]:
    """Centra en las caderas y divide por la escala del torso."""
    coordenadas = np.array(coordenadas, dtype=np.float32, copy=True)
    valida = valida & np.isfinite(coordenadas).all(axis=2)
    coordenadas[~valida] = np.nan
    origen = _media_finita(coordenadas[:, CADERAS, :], eje=1)
    sin_caderas = ~np.isfinite(origen).all(axis=1)
    respaldo = _media_finita(coordenadas, eje=1)
    origen[sin_caderas] = respaldo[sin_caderas]
    escala, escala_respaldo = escala_torso(coordenadas)
    normalizada = (coordenadas - origen[:, None, :]) / escala
    normalizada[~valida] = np.nan
    diagnostico = {
        "scale": escala,
        "root_joint": "mean(available_hips); visible_joint_mean_fallback",
        "root_fallback_frames": int((sin_caderas & np.isfinite(respaldo).all(axis=1)).sum()),
        "root_no_support_frames": int((~np.isfinite(origen).all(axis=1)).sum()),
        "scale_fallback_used": bool(escala_respaldo),
    }
    return normalizada.astype(np.float32), diagnostico


def acondicionar_archivo(ruta_pose: Path, ruta_salida: Path,
                         umbral: float = UMBRAL_CONFIANZA, brecha_maxima: int = BRECHA_MAXIMA) -> dict:
    """Lee una pose 2D (NPZ con ``keypoints`` y ``confidence``) y guarda la secuencia normalizada."""
    with np.load(ruta_pose, allow_pickle=True) as datos:
        coordenadas = datos["keypoints"].astype(np.float32)[..., :2]
        confianza = datos["confidence"].astype(np.float32)
        metadatos = json.loads(str(datos["metadata_json"].item())) if "metadata_json" in datos.files else {}
    interpolada, valida = interpolar_brechas(coordenadas, confianza, umbral, brecha_maxima)
    normalizada, diagnostico = normalizar(interpolada, valida)
    metadatos.update({
        "preprocessing_version": VERSION_ACONDICIONAMIENTO,
        "normalization": "root_scale",
        "confidence_threshold": umbral,
        "max_interp_gap": brecha_maxima,
        **diagnostico,
    })
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta_salida,
        sequence=normalizada,
        confidence=confianza,
        valid_mask=valida.astype(bool),
        keypoint_names=np.asarray(ARTICULACIONES_COCO17),
        metadata_json=json.dumps(metadatos, sort_keys=True),
    )
    return {"fotogramas": int(len(normalizada)), "faltantes_tras_interpolacion": float((~valida).mean()), **diagnostico}
