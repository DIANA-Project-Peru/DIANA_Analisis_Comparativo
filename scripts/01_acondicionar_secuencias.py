#!/usr/bin/env python
"""Paso 1. Acondiciona y normaliza las secuencias esqueléticas (Sección 2.3).

Lee la pose 2D de cada ejecución elegible (columna ``pose2d_npz_path`` del
manifiesto) y guarda la secuencia normalizada en ``<salida>/secuencias``.
También escribe un resumen por ejecución con los indicadores de calidad citados
en el informe (faltantes tras interpolar, uso del origen alternativo).

Con ``--comparar-con`` verifica que las secuencias generadas coincidan con una
carpeta de secuencias existente (por ejemplo, las usadas en el informe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo.acondicionamiento import acondicionar_archivo  # noqa: E402
from diana_comparativo.rutas import Rutas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None, help="archivo de rutas (por defecto configuracion/rutas.json)")
    parser.add_argument("--limite", type=int, help="procesar solo las primeras N ejecuciones (prueba rápida)")
    parser.add_argument("--comparar-con", type=Path, help="carpeta de secuencias de referencia para verificar igualdad")
    args = parser.parse_args()
    rutas = Rutas(args.rutas) if args.rutas else Rutas()

    manifiesto = pd.read_csv(rutas["manifiesto_elegibles"], usecols=["segment_id", "pose2d_npz_path"])
    if args.limite:
        manifiesto = manifiesto.head(args.limite)
    destino = rutas.salida_de("secuencias")
    filas, diferencias = [], 0
    for fila in manifiesto.itertuples(index=False):
        salida = destino / f"{fila.segment_id}.npz"
        resumen = acondicionar_archivo(Path(fila.pose2d_npz_path), salida)
        if args.comparar_con:
            with np.load(salida) as nueva, np.load(args.comparar_con / f"{fila.segment_id}.npz") as original:
                iguales = np.array_equal(nueva["valid_mask"], original["valid_mask"].astype(bool)) and np.allclose(
                    nueva["sequence"], original["sequence"], equal_nan=True, rtol=0, atol=0)
            resumen["identica_a_referencia"] = bool(iguales)
            diferencias += not iguales
        filas.append({"segment_id": fila.segment_id, **resumen})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(rutas.salida_de("resumenes") / "acondicionamiento_por_ejecucion.csv", index=False)
    print(f"Ejecuciones acondicionadas: {len(tabla)}")
    print(f"Faltantes medios tras interpolación: {tabla['faltantes_tras_interpolacion'].mean():.4f}")
    print(f"Ejecuciones con origen alternativo: {(tabla['root_fallback_frames'] > 0).sum()}")
    if args.comparar_con:
        print(f"Secuencias distintas de la referencia: {diferencias}")
        raise SystemExit(1 if diferencias else 0)


if __name__ == "__main__":
    main()
