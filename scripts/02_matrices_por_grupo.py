#!/usr/bin/env python
"""Paso 2. Matrices de distancias DTW por acción y grupo etario (entrenamiento).

Calcula, para cada uno de los 52 grupos, la distancia entre todos los pares de
ejecuciones del conjunto de entrenamiento y escribe un índice compatible con
los pasos 3 y 4 (``<salida>/matrices/indice_matrices.csv``).

Nota: las referencias reportadas en el informe se seleccionaron con las
matrices de la selección inicial, calculadas antes de adecuar el tratamiento de
faltantes (ruta ``matrices_historicas`` en rutas.json). Este paso calcula las
matrices con el procedimiento actual.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo.referencias import matriz_por_grupo  # noqa: E402
from diana_comparativo.rutas import Rutas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    parser.add_argument("--procesos", type=int, default=4)
    parser.add_argument("--grupos", nargs="*", help="limitar a grupos ACCION:EDAD, por ejemplo A01:36_47mo")
    args = parser.parse_args()
    rutas = Rutas(args.rutas) if args.rutas else Rutas()

    entrenamiento = pd.read_csv(rutas["manifiesto_entrenamiento"], usecols=["segment_id", "action_code", "age_bin_12m_label"])
    destino = rutas.salida_de("matrices")
    indice = []
    for (accion, edad), grupo in entrenamiento.groupby(["action_code", "age_bin_12m_label"]):
        if args.grupos and f"{accion}:{edad}" not in args.grupos:
            continue
        matriz = matriz_por_grupo(grupo, rutas["secuencias"], args.procesos)
        archivo = destino / f"{accion}__{edad}.csv"
        matriz.to_csv(archivo, index=False)
        indice.append({"action_code": accion, "age_bin_12m_label": edad, "part": "full", "n_clips": len(grupo),
                       "n_pairs": len(matriz), "pairwise_path": str(archivo)})
        print(f"{accion} {edad}: {len(grupo)} ejecuciones, {len(matriz)} pares")
    pd.DataFrame(indice).to_csv(destino / "indice_matrices.csv", index=False)


if __name__ == "__main__":
    main()
