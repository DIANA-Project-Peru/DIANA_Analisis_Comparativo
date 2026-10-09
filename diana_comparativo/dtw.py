"""Comparación temporal de secuencias esqueléticas mediante DTW.

Implementa la variante descrita en la Sección 2.4 del Informe Técnico N.° 1:

* El costo local entre dos fotogramas se calcula solo con las articulaciones
  válidas en ambos (conjunto V, de tamaño k):

      c(i, j) = sqrt( (J / k) * sum_{a in V} [(x_ia - x_ja)^2 + (y_ia - y_ja)^2] )

  El factor J/k mantiene la escala del costo cuando faltan articulaciones.
* Un par de fotogramas solo es admisible si k >= ceil(f * J); con J = 17 y
  f = 0,5, se exigen 9 articulaciones comunes. Si no, su costo es infinito.
* El costo acumulado usa pasos vertical, horizontal y diagonal, sin ventana:

      D(i, j) = c(i, j) + min{ D(i-1, j), D(i, j-1), D(i-1, j-1) }

  y la distancia final es D(n, m) / (n + m).
* Si no existe un alineamiento completo admisible, la distancia es infinita y
  la comparación se registra como abstención (nunca como cero).

Las secuencias se representan como arreglos (T, 2J) con NaN en las
observaciones inválidas. La función acelerada usa Numba si está instalado.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # Numba es opcional; acelera el cálculo en corridas completas.
    from numba import njit
except ImportError:  # pragma: no cover - depende del entorno
    def njit(*args, **kwargs):
        return lambda function: function

#: Identificador de la métrica, registrado en todos los resultados.
VERSION_DTW = "joint_overlap_v1"
#: Fracción mínima de articulaciones comunes por par de fotogramas.
SOPORTE_MINIMO = 0.5
#: Versión de acondicionamiento con la que deben haberse generado las secuencias.
VERSION_ACONDICIONAMIENTO = "missing_safe_v1"

ARTICULACIONES_COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
CUERPO_COMPLETO = list(range(17))


def articulaciones_requeridas(n_articulaciones: int, soporte_minimo: float = SOPORTE_MINIMO) -> int:
    """Número mínimo de articulaciones comunes: ceil(f * J)."""
    if not 0 < soporte_minimo <= 1:
        raise ValueError("soporte_minimo debe estar en (0, 1]")
    return int(np.ceil(n_articulaciones * soporte_minimo))


def costo_local(a: np.ndarray, b: np.ndarray, soporte_minimo: float = SOPORTE_MINIMO) -> float:
    """Costo entre dos fotogramas (J, 2) o (2J,); infinito si no hay soporte."""
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, 2), np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if a.shape != b.shape:
        raise ValueError("Los fotogramas deben tener las mismas articulaciones")
    validas = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    k = int(validas.sum())
    if k < articulaciones_requeridas(len(a), soporte_minimo):
        return float("inf")
    delta = a[validas] - b[validas]
    return float(np.sqrt(np.sum(delta * delta) * len(a) / k))


@njit(cache=True)
def _distancia_acumulada(a, b, requeridas):
    n, m, dims = len(a), len(b), a.shape[1]
    if n == 0 or m == 0:
        return np.inf
    anterior = np.full(m + 1, np.inf)
    actual = np.full(m + 1, np.inf)
    anterior[0] = 0.0
    for i in range(n):
        actual[0] = np.inf
        for j in range(m):
            suma, k = 0.0, 0
            for c in range(0, dims, 2):
                if (np.isfinite(a[i, c]) and np.isfinite(a[i, c + 1])
                        and np.isfinite(b[j, c]) and np.isfinite(b[j, c + 1])):
                    dx = a[i, c] - b[j, c]
                    dy = a[i, c + 1] - b[j, c + 1]
                    suma += dx * dx + dy * dy
                    k += 1
            costo = np.sqrt(suma * (dims // 2) / k) if k >= requeridas else np.inf
            actual[j + 1] = costo + min(anterior[j + 1], actual[j], anterior[j])
        anterior, actual = actual, anterior
    return anterior[m] / (n + m)


def distancia_dtw(a: np.ndarray, b: np.ndarray, soporte_minimo: float = SOPORTE_MINIMO) -> float:
    """Distancia DTW entre dos secuencias (T, 2J); infinita si no hay alineamiento."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] % 2 or not a.shape[1]:
        raise ValueError("Se esperaban secuencias (T, 2J) con las mismas articulaciones")
    return float(_distancia_acumulada(a, b, articulaciones_requeridas(a.shape[1] // 2, soporte_minimo)))


def alineamiento_dtw(a: np.ndarray, b: np.ndarray, soporte_minimo: float = SOPORTE_MINIMO) -> dict:
    """Versión explícita (más lenta) que devuelve la matriz de costos y el camino.

    Se utiliza en el visor y en las pruebas; produce la misma distancia que
    :func:`distancia_dtw`.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    costos = np.array([[costo_local(a[i], b[j], soporte_minimo) for j in range(m)] for i in range(n)])
    acumulado = np.full((n + 1, m + 1), np.inf)
    acumulado[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            acumulado[i + 1, j + 1] = costos[i, j] + min(acumulado[i, j + 1], acumulado[i + 1, j], acumulado[i, j])
    camino: list[tuple[int, int]] = []
    if np.isfinite(acumulado[n, m]):
        i, j = n, m
        while (i, j) != (0, 0):
            camino.append((i - 1, j - 1))
            pasos = [(acumulado[i - 1, j - 1], i - 1, j - 1), (acumulado[i - 1, j], i - 1, j), (acumulado[i, j - 1], i, j - 1)]
            _, i, j = min(pasos, key=lambda paso: paso[0])
        camino.reverse()
    return {"distancia": float(acumulado[n, m] / (n + m)), "costos": costos, "camino": camino}


# --------------------------------------------------------------- secuencias
def leer_metadatos(ruta: Path) -> dict:
    with np.load(ruta, allow_pickle=False) as datos:
        if "metadata_json" not in datos.files:
            return {}
        return json.loads(str(datos["metadata_json"].item()))


def validar_secuencia(ruta: Path, version: str = VERSION_ACONDICIONAMIENTO) -> dict:
    """Comprueba que una secuencia normalizada sea compatible antes de compararla.

    Una secuencia inexistente es un insumo faltante; una secuencia existente pero
    incompatible es un error de integridad y detiene el cálculo.
    """
    ruta = Path(ruta)
    metadatos = leer_metadatos(ruta)
    if metadatos.get("preprocessing_version") != version:
        raise ValueError(f"Versión de acondicionamiento incompatible en {ruta}: {metadatos.get('preprocessing_version')!r}")
    with np.load(ruta, allow_pickle=False) as datos:
        faltan = {"sequence", "valid_mask", "confidence", "keypoint_names"} - set(datos.files)
        if faltan:
            raise ValueError(f"Faltan arreglos {sorted(faltan)} en {ruta}")
        secuencia, mascara = datos["sequence"], datos["valid_mask"].astype(bool)
        nombres = tuple(str(valor) for valor in datos["keypoint_names"].tolist())
    if secuencia.ndim != 3 or secuencia.shape[1:] != (17, 2) or len(secuencia) == 0:
        raise ValueError(f"Se esperaba una secuencia (T, 17, 2) en {ruta}")
    if nombres != ARTICULACIONES_COCO17:
        raise ValueError(f"Orden de articulaciones distinto de COCO17 en {ruta}")
    if mascara.shape != secuencia.shape[:2] or not np.isfinite(secuencia[mascara]).all():
        raise ValueError(f"Máscara de validez inconsistente en {ruta}")
    return metadatos


def cargar_secuencia(ruta: Path, articulaciones: list[int] = CUERPO_COMPLETO) -> np.ndarray:
    """Carga una secuencia normalizada como arreglo (T, 2J) con NaN en lo inválido."""
    with np.load(ruta, allow_pickle=False) as datos:
        secuencia = datos["sequence"][:, articulaciones, :].astype(np.float64)
        if "valid_mask" in datos.files:
            secuencia[~datos["valid_mask"][:, articulaciones].astype(bool)] = np.nan
    secuencia[~np.isfinite(secuencia)] = np.nan
    return secuencia.reshape(len(secuencia), -1)
