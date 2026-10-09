"""Matrices por grupo y selección de referencias representativas (Sección 2.5).

Para cada acción y grupo etario del conjunto de entrenamiento:

* **Matriz de distancias:** DTW entre todos los pares de ejecuciones del grupo.
* **Referencia primaria (medoide):** ejecución con menor distancia media al
  resto (solo distancias finitas; empates resueltos por identificador).
* **Conjunto top-k:** la primaria más otras ejecuciones centrales de niños
  distintos que difieran de las ya elegidas al menos en el percentil 25 de las
  distancias del grupo (si ninguna lo cumple, se prueba con los percentiles 20
  y 10). El tamaño k depende del número de niños: 1 con menos de 5, 2 con 5 a
  9, 3 con 10 a 19 y 4 con 20 o más.

Las plantillas adaptativas (problema de p-medianas) se calculan en
:mod:`diana_comparativo.estabilidad`.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .dtw import SOPORTE_MINIMO, cargar_secuencia, distancia_dtw

PERCENTILES_DIVERSIDAD = [("p25", 0.25), ("p20", 0.20), ("p10", 0.10)]


# ------------------------------------------------------------- matrices
def _par(tarea: tuple[str, str, str, str], soporte_minimo: float) -> dict:
    id_a, id_b, ruta_a, ruta_b = tarea
    return {"segment_id_a": id_a, "segment_id_b": id_b,
            "distance": distancia_dtw(cargar_secuencia(Path(ruta_a)), cargar_secuencia(Path(ruta_b)), soporte_minimo)}


def matriz_por_grupo(ejecuciones: pd.DataFrame, carpeta_secuencias: Path, n_procesos: int = 1,
                     soporte_minimo: float = SOPORTE_MINIMO) -> pd.DataFrame:
    """Distancias DTW entre todos los pares de ejecuciones de un grupo (formato largo)."""
    filas = ejecuciones.sort_values("segment_id")
    tareas = [(str(a.segment_id), str(b.segment_id),
               str(carpeta_secuencias / f"{a.segment_id}.npz"), str(carpeta_secuencias / f"{b.segment_id}.npz"))
              for a, b in itertools.combinations(filas.itertuples(index=False), 2)]
    calcular = partial(_par, soporte_minimo=soporte_minimo)
    if n_procesos > 1 and len(tareas) > 1:
        with ProcessPoolExecutor(max_workers=n_procesos) as pool:
            return pd.DataFrame(list(pool.map(calcular, tareas, chunksize=64)))
    return pd.DataFrame([calcular(tarea) for tarea in tareas])


# ------------------------------------------------------------- selección
def niño_canonico(valor, segment_id: str) -> str:
    if pd.isna(valor) or str(valor).strip() == "":
        return f"desconocido::{segment_id}"
    texto = str(valor).strip()
    try:
        numero = float(texto)
        if np.isfinite(numero) and numero.is_integer():
            return str(int(numero))
    except ValueError:
        pass
    return texto


def k_por_numero_de_niños(n_niños: int) -> int:
    if n_niños < 5:
        return 1
    if n_niños < 10:
        return 2
    if n_niños < 20:
        return 3
    return 4


def centralidad(matriz: pd.DataFrame) -> tuple[dict[tuple[str, str], float], pd.Series]:
    """Diccionario de distancias finitas y distancia media de cada ejecución al grupo."""
    matriz = matriz[np.isfinite(pd.to_numeric(matriz["distance"], errors="coerce"))]
    distancias: dict[tuple[str, str], float] = {}
    suma: dict[str, float] = {}
    cuenta: dict[str, int] = {}
    for fila in matriz.itertuples(index=False):
        a, b, d = str(fila.segment_id_a).strip(), str(fila.segment_id_b).strip(), float(fila.distance)
        distancias[(a, b)] = distancias[(b, a)] = d
        for sid in (a, b):
            suma[sid] = suma.get(sid, 0.0) + d
            cuenta[sid] = cuenta.get(sid, 0) + 1
    media = pd.Series({sid: suma[sid] / cuenta[sid] for sid in suma}, name="mean_distance_to_group")
    return distancias, media


def seleccionar_referencias(matriz: pd.DataFrame, ejecuciones: pd.DataFrame) -> pd.DataFrame:
    """Referencia primaria y conjunto top-k de un grupo acción × edad.

    ``ejecuciones`` debe contener ``segment_id`` y ``excel_children`` (niño).
    Devuelve una fila por referencia seleccionada, con ``rank`` = 1 para la primaria.
    """
    distancias, media = centralidad(matriz)
    if media.empty:
        raise ValueError("El grupo no tiene distancias finitas")
    candidatos = media.rename_axis("segment_id").reset_index()
    candidatos["segment_id"] = candidatos["segment_id"].astype(str)
    niños = ejecuciones.drop_duplicates("segment_id").set_index(ejecuciones["segment_id"].astype(str))["excel_children"]
    candidatos["niño"] = [niño_canonico(niños.get(sid), sid) for sid in candidatos["segment_id"]]
    candidatos = candidatos.sort_values(["mean_distance_to_group", "segment_id"]).reset_index(drop=True)
    valores = matriz["distance"].to_numpy(dtype=float)
    valores = valores[np.isfinite(valores)]
    umbrales = {etiqueta: float(np.quantile(valores, q)) for etiqueta, q in PERCENTILES_DIVERSIDAD}
    k = k_por_numero_de_niños(int(candidatos["niño"].nunique()))

    def minima_a_elegidas(sid: str, elegidas: list[str]) -> float:
        finitas = [distancias.get((sid, otra), np.nan) for otra in elegidas]
        finitas = [d for d in finitas if np.isfinite(d)]
        return min(finitas) if finitas else np.nan

    elegidas = [candidatos.iloc[0]["segment_id"]]
    niños_elegidos = {candidatos.iloc[0]["niño"]}
    criterio = ["primaria"]
    while len(elegidas) < k:
        elegida = None
        for etiqueta, _ in PERCENTILES_DIVERSIDAD:
            disponibles = candidatos[~candidatos["segment_id"].isin(elegidas) & ~candidatos["niño"].isin(niños_elegidos)].copy()
            if disponibles.empty:
                continue
            disponibles["minima"] = disponibles["segment_id"].map(lambda sid: minima_a_elegidas(sid, elegidas))
            diversas = disponibles[disponibles["minima"] >= umbrales[etiqueta]]
            if not diversas.empty:
                elegida = diversas.sort_values(["mean_distance_to_group", "segment_id"]).iloc[0]
                criterio.append(etiqueta)
                break
        if elegida is None:
            break
        elegidas.append(elegida["segment_id"])
        niños_elegidos.add(elegida["niño"])

    seleccion = candidatos.set_index("segment_id").loc[elegidas].reset_index()
    seleccion["rank"] = np.arange(1, len(seleccion) + 1)
    seleccion["k_requested"] = k
    seleccion["diversity_threshold_used"] = criterio
    seleccion["n_group_clips"] = len(candidatos)
    seleccion["n_group_unique_children"] = int(candidatos["niño"].nunique())
    return seleccion.rename(columns={"niño": "excel_children"})
