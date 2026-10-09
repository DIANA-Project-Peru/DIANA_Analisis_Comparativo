"""Resultados agregados por ejecución y por niño-acción (Sección 3.4).

Para cada ejecución de validación se calculan las distancias a las referencias
de edad menor (Y, desplazamiento -1), esperada (E, 0) y mayor (O, +1) con tres
resúmenes de referencia:

* ``primaria``: distancia a la referencia primaria del grupo.
* ``mediana_topk``: mediana de las distancias a todas las referencias top-k.
* ``minimo_topk``: mínimo de esas distancias.

Un resumen top-k solo se calcula si **todas** las referencias declaradas del
grupo tienen distancia finita; no se resume sobre las que sobreviven. Las
distancias describen similitud geométrica y temporal con referencias concretas;
no son una puntuación TGMD-3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESUMENES = ("primaria", "mediana_topk", "minimo_topk")
NOMBRES = {-1: "Y", 0: "E", 1: "O"}


def _resumir(filas: pd.DataFrame, resumen: str) -> float:
    if resumen == "primaria":
        filas = filas[filas["es_primaria"]]
        if len(filas) != 1:
            return np.nan
    if filas.empty or not filas["estado"].eq("calculada").all():
        return np.nan
    valores = filas["distancia"].astype(float)
    return float(valores.min() if resumen == "minimo_topk" else valores.median())


def perfiles_por_ejecucion(comparaciones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ejecución y resumen: 1.261 ejecuciones × 3 resúmenes en el informe."""
    filas = []
    for (sid, nino, accion, grupo), datos in comparaciones.groupby(["id_ejecucion", "id_nino", "accion", "grupo_etario"], sort=True):
        for resumen in RESUMENES:
            fila = {"id_ejecucion": sid, "id_nino": nino, "accion": accion, "grupo_etario": grupo, "resumen": resumen}
            for desplazamiento, nombre in NOMBRES.items():
                fila[f"d_{nombre}"] = _resumir(datos[datos["desplazamiento_etario"] == desplazamiento], resumen)
            filas.append(fila)
    perfiles = pd.DataFrame(filas)
    distancias = perfiles[["d_Y", "d_E", "d_O"]]
    perfiles["margen_Y_menos_E"] = perfiles["d_Y"] - perfiles["d_E"]
    perfiles["margen_O_menos_E"] = perfiles["d_O"] - perfiles["d_E"]
    perfiles["perfil_completo"] = distancias.notna().all(axis=1)
    # Grupo etario con la menor distancia disponible; sin valor si hay empate o no hay distancias.
    valores = distancias.to_numpy(dtype=float)
    con_datos = np.isfinite(valores).any(axis=1)
    minimo = np.where(con_datos, np.nanmin(np.where(np.isfinite(valores), valores, np.inf), axis=1), np.nan)
    empatado = (valores == minimo[:, None]).sum(axis=1) > 1
    posicion = np.argmin(np.where(np.isfinite(valores), valores, np.inf), axis=1)
    etiquetas = np.array(["Y", "E", "O"], dtype=object)[posicion]
    perfiles["grupo_mas_cercano"] = np.where(con_datos & ~empatado, etiquetas, None)
    return perfiles


def agregados_nino_accion(perfiles: pd.DataFrame) -> pd.DataFrame:
    """Media, mediana, mínimo y máximo de las ejecuciones de cada niño y acción.

    Una estadística solo se calcula si todas las ejecuciones del niño tienen la
    distancia correspondiente (no se resume sobre las disponibles).
    """
    filas = []
    for (nino, accion, resumen), datos in perfiles.groupby(["id_nino", "accion", "resumen"], sort=True):
        base = {"id_nino": nino, "accion": accion, "resumen": resumen, "n_ejecuciones": len(datos)}
        for estadistica in ("mean", "median", "min", "max"):
            fila = {**base, "estadistica": {"mean": "media", "median": "mediana", "min": "minimo", "max": "maximo"}[estadistica]}
            for nombre in ("Y", "E", "O"):
                columna = datos[f"d_{nombre}"]
                fila[f"d_{nombre}"] = float(getattr(columna, estadistica)()) if columna.notna().all() else np.nan
            filas.append(fila)
    return pd.DataFrame(filas)


def cobertura_por_resumen(perfiles: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de la Tabla 7 del informe."""
    filas = []
    comunes = perfiles[perfiles["perfil_completo"]].groupby("id_ejecucion")["resumen"].nunique()
    comunes = set(comunes[comunes == len(RESUMENES)].index)
    referencia = perfiles[(perfiles["resumen"] == "primaria") & perfiles["id_ejecucion"].isin(comunes)].set_index("id_ejecucion")["grupo_mas_cercano"]
    for resumen in RESUMENES:
        datos = perfiles[perfiles["resumen"] == resumen]
        comun = datos[datos["id_ejecucion"].isin(comunes)].set_index("id_ejecucion")
        cercano = comun["grupo_mas_cercano"].value_counts()
        filas.append({
            "resumen": resumen,
            "ejecuciones": len(datos),
            "con_distancia_E": int(datos["d_E"].notna().sum()),
            "perfil_completo": int(datos["perfil_completo"].sum()),
            "sin_ninguna_distancia": int(datos[["d_Y", "d_E", "d_O"]].isna().all(axis=1).sum()),
            "soporte_comun": len(comun),
            "mediana_Y": comun["d_Y"].median(), "mediana_E": comun["d_E"].median(), "mediana_O": comun["d_O"].median(),
            "mas_cercano_Y": int(cercano.get("Y", 0)), "mas_cercano_E": int(cercano.get("E", 0)), "mas_cercano_O": int(cercano.get("O", 0)),
            "cambio_frente_a_primaria": int((comun["grupo_mas_cercano"] != referencia.reindex(comun.index)).sum()),
        })
    return pd.DataFrame(filas)
