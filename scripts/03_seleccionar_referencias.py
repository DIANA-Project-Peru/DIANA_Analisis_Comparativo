#!/usr/bin/env python
"""Paso 3. Selección de referencias primarias y top-k por acción y grupo etario.

Por defecto usa las matrices de la selección inicial (``matrices_historicas``),
con las que se obtuvo el catálogo reportado (52 primarias y 194 top-k). Con
``--matrices actuales`` usa las matrices del paso 2.

Escribe ``<salida>/referencias/catalogo_primarias.csv`` y
``<salida>/referencias/catalogo_topk.csv``. Con ``--comparar-con-catalogo``
verifica que la selección coincida con el catálogo congelado de rutas.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo.referencias import seleccionar_referencias  # noqa: E402
from diana_comparativo.rutas import Rutas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    parser.add_argument("--matrices", choices=["historicas", "actuales"], default="historicas")
    parser.add_argument("--comparar-con-catalogo", action="store_true")
    args = parser.parse_args()
    rutas = Rutas(args.rutas) if args.rutas else Rutas()

    entrenamiento = pd.read_csv(rutas["manifiesto_entrenamiento"], dtype={"segment_id": str})
    if args.matrices == "historicas":
        indice = pd.read_csv(rutas["matrices_historicas"] / "dtw_pairwise_index.csv")
    else:
        indice = pd.read_csv(rutas.salida / "matrices" / "indice_matrices.csv")
    seleccion = []
    for fila in indice.itertuples(index=False):
        grupo = entrenamiento[(entrenamiento["action_code"] == fila.action_code)
                              & (entrenamiento["age_bin_12m_label"] == fila.age_bin_12m_label)]
        matriz = pd.read_csv(fila.pairwise_path, dtype={"segment_id_a": str, "segment_id_b": str})
        referencias = seleccionar_referencias(matriz, grupo)
        referencias.insert(0, "age_bin_12m_label", fila.age_bin_12m_label)
        referencias.insert(0, "action_code", fila.action_code)
        seleccion.append(referencias)
    topk = pd.concat(seleccion, ignore_index=True)
    primarias = topk[topk["rank"] == 1].rename(columns={"segment_id": "medoid_segment_id"})
    destino = rutas.salida_de("referencias")
    topk.to_csv(destino / "catalogo_topk.csv", index=False)
    primarias.to_csv(destino / "catalogo_primarias.csv", index=False)
    print(f"Grupos: {len(primarias)}; referencias primarias: {len(primarias)} "
          f"({primarias['excel_children'].nunique()} niños); top-k: {len(topk)} ({topk['excel_children'].nunique()} niños)")
    print("Referencias por grupo:", topk.groupby(["action_code", "age_bin_12m_label"]).size().value_counts().sort_index().to_dict())

    if args.comparar_con_catalogo:
        congelado = pd.read_csv(rutas["catalogo_topk"], dtype={"segment_id": str})
        clave = ["action_code", "age_bin_12m_label", "rank", "segment_id"]
        iguales = (set(map(tuple, congelado[clave].astype(str).to_numpy()))
                   == set(map(tuple, topk[clave].astype(str).to_numpy())))
        print(f"Coincide con el catálogo congelado: {'sí' if iguales else 'NO'}")
        raise SystemExit(0 if iguales else 1)


if __name__ == "__main__":
    main()
