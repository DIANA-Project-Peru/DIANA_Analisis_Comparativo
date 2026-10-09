"""Comparación de ejecuciones de validación con referencias (Secciones 2.4 y 3.4).

Cada ejecución de validación se compara con todas las referencias de su misma
acción en su grupo etario (desplazamiento 0) y en los grupos adyacentes
(desplazamientos -1 y +1, es decir, 12 meses menor y mayor).

El conjunto de comparaciones **previstas** se construye antes de calcular
distancias, a partir de los manifiestos y del catálogo de referencias. Así la
completitud se mide contra lo previsto y no contra las filas que sobreviven al
cálculo. Estados posibles:

* ``calculada``: distancia finita.
* ``sin_camino_admisible``: no existe alineamiento completo con el soporte
  articular mínimo (abstención).
* ``fuera_de_dominio_etario``: solicitud de un grupo menor a 36–47 o mayor a
  72–83 meses; no genera comparaciones.
* ``sin_grupo_de_referencia`` / ``sin_secuencia``: insumos ausentes.
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .dtw import SOPORTE_MINIMO, VERSION_ACONDICIONAMIENTO, VERSION_DTW, cargar_secuencia, distancia_dtw, validar_secuencia

DESPLAZAMIENTOS = (-1, 0, 1)
GRUPOS_SOPORTADOS = ("36_47mo", "48_59mo", "60_71mo", "72_83mo")
PASO_MESES = 12


def inicio_grupo(etiqueta: str) -> int | None:
    coincidencia = re.match(r"^\s*(\d+)_(\d+)mo\s*$", str(etiqueta))
    return int(coincidencia.group(1)) if coincidencia else None


def catalogo_de_referencias(topk: pd.DataFrame, primarias: pd.DataFrame, entrenamiento: pd.DataFrame) -> pd.DataFrame:
    """Une primarias y top-k y verifica que todas provengan del entrenamiento."""
    catalogo = topk[["action_code", "age_bin_12m_label", "segment_id", "rank", "excel_children"]].copy()
    catalogo["segment_id"] = catalogo["segment_id"].astype(str)
    primaria = primarias.set_index(["action_code", "age_bin_12m_label"])["medoid_segment_id"].astype(str)
    claves = list(zip(catalogo["action_code"], catalogo["age_bin_12m_label"]))
    catalogo["es_primaria"] = [primaria.get(clave) == sid for clave, sid in zip(claves, catalogo["segment_id"])]
    if not catalogo.groupby(["action_code", "age_bin_12m_label"])["es_primaria"].sum().eq(1).all():
        raise ValueError("Cada grupo debe tener exactamente una referencia primaria")
    fuera = set(catalogo["segment_id"]) - set(entrenamiento["segment_id"].astype(str))
    if fuera:
        raise ValueError(f"Referencias fuera del conjunto de entrenamiento: {sorted(fuera)[:5]}")
    return catalogo


def comparaciones_previstas(validacion: pd.DataFrame, catalogo: pd.DataFrame, carpeta: Path,
                            desplazamientos=DESPLAZAMIENTOS, grupos=GRUPOS_SOPORTADOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solicitudes (ejecución × desplazamiento) y comparaciones previstas (ejecución × referencia)."""
    por_inicio = {inicio_grupo(g): g for g in grupos}
    solicitudes, previstas = [], []
    for fila in validacion.itertuples(index=False):
        sid = str(fila.segment_id)
        ruta_caso = carpeta / f"{sid}.npz"
        for desplazamiento in desplazamientos:
            objetivo = por_inicio.get(inicio_grupo(fila.age_bin_12m_label) + desplazamiento * PASO_MESES)
            grupo = catalogo[(catalogo["action_code"] == fila.action_code) & (catalogo["age_bin_12m_label"] == objetivo)]
            if objetivo is None:
                estado = "fuera_de_dominio_etario"
            elif grupo.empty:
                estado = "sin_grupo_de_referencia"
            elif not ruta_caso.exists():
                estado = "sin_secuencia"
            else:
                estado = "solicitada"
            solicitudes.append({"id_ejecucion": sid, "id_nino": fila.excel_children, "accion": fila.action_code,
                                "grupo_etario": fila.age_bin_12m_label, "desplazamiento_etario": desplazamiento,
                                "grupo_referencia": objetivo or "", "n_referencias": len(grupo), "estado": estado})
            if estado != "solicitada":
                continue
            for ref in grupo.itertuples(index=False):
                previstas.append({"id_ejecucion": sid, "id_nino": fila.excel_children, "accion": fila.action_code,
                                  "grupo_etario": fila.age_bin_12m_label, "desplazamiento_etario": desplazamiento,
                                  "grupo_referencia": objetivo, "id_referencia": ref.segment_id,
                                  "id_nino_referencia": ref.excel_children, "rango_topk": int(ref.rank),
                                  "es_primaria": bool(ref.es_primaria),
                                  "ruta_caso": str(ruta_caso), "ruta_referencia": str(carpeta / f"{ref.segment_id}.npz")})
    return pd.DataFrame(solicitudes), pd.DataFrame(previstas)


def _distancia(tarea: tuple[str, str], soporte_minimo: float) -> float:
    ruta_a, ruta_b = (Path(r) for r in tarea)
    if not ruta_b.exists():
        return np.nan
    validar_secuencia(ruta_a)
    validar_secuencia(ruta_b)
    return distancia_dtw(cargar_secuencia(ruta_a), cargar_secuencia(ruta_b), soporte_minimo)


def calcular(previstas: pd.DataFrame, n_procesos: int = 1, soporte_minimo: float = SOPORTE_MINIMO) -> pd.DataFrame:
    """Calcula la distancia de cada comparación prevista y asigna su estado."""
    tareas = list(zip(previstas["ruta_caso"], previstas["ruta_referencia"]))
    funcion = partial(_distancia, soporte_minimo=soporte_minimo)
    if n_procesos > 1:
        with ProcessPoolExecutor(max_workers=n_procesos) as pool:
            distancias = list(pool.map(funcion, tareas, chunksize=32))
    else:
        distancias = [funcion(tarea) for tarea in tareas]
    resultado = previstas.drop(columns=["ruta_caso", "ruta_referencia"]).copy()
    resultado["distancia"] = distancias
    resultado["estado"] = np.where(np.isfinite(resultado["distancia"]), "calculada",
                                   np.where(resultado["distancia"].isna(), "sin_secuencia", "sin_camino_admisible"))
    resultado.loc[~np.isfinite(resultado["distancia"]), "distancia"] = np.nan
    resultado["version_dtw"] = VERSION_DTW
    resultado["version_acondicionamiento"] = VERSION_ACONDICIONAMIENTO
    resultado["soporte_minimo"] = soporte_minimo
    return resultado


def controles_de_separacion(entrenamiento: pd.DataFrame, validacion: pd.DataFrame,
                            catalogo: pd.DataFrame, previstas: pd.DataFrame, resultado: pd.DataFrame) -> list[dict]:
    """Controles automáticos que respaldan la separación entre conjuntos."""
    niños_ent = set(entrenamiento["excel_children"].astype(str))
    niños_val = set(validacion["excel_children"].astype(str))
    clave = ["id_ejecucion", "id_referencia", "desplazamiento_etario"]
    previstas_claves = set(map(tuple, previstas[clave].astype(str).to_numpy()))
    observadas = set(map(tuple, resultado[clave].astype(str).to_numpy()))
    controles = [
        ("Ningún niño pertenece a ambos conjuntos", len(niños_ent & niños_val) == 0, len(niños_ent & niños_val)),
        ("Todas las referencias provienen del entrenamiento",
         set(catalogo["segment_id"]) <= set(entrenamiento["segment_id"].astype(str)), int(len(catalogo))),
        ("Ningún niño de validación es su propia referencia",
         not (resultado["id_nino"].astype(str) == resultado["id_nino_referencia"].astype(str)).any(), int(len(resultado))),
        ("Las comparaciones observadas coinciden con las previstas", previstas_claves == observadas,
         {"previstas": len(previstas_claves), "observadas": len(observadas)}),
        ("No hay comparaciones duplicadas", not resultado.duplicated(clave).any(), int(resultado.duplicated(clave).sum())),
        ("Una sola versión de métrica y acondicionamiento",
         resultado["version_dtw"].nunique() == 1 and resultado["version_acondicionamiento"].nunique() == 1,
         [VERSION_DTW, VERSION_ACONDICIONAMIENTO]),
    ]
    return [{"control": nombre, "cumple": bool(cumple), "evidencia": evidencia} for nombre, cumple, evidencia in controles]
