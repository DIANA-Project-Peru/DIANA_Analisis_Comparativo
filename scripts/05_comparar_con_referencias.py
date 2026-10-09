#!/usr/bin/env python
"""Paso 5. Comparación de las ejecuciones de validación con las referencias.

Construye las comparaciones previstas (ejecución × referencia de la misma acción
en los grupos etarios esperado y adyacentes), calcula las distancias DTW,
registra las abstenciones y genera los resultados agregados. Escribe en
``<salida>/comparacion``:

* ``solicitudes.csv``: una fila por ejecución y desplazamiento etario.
* ``comparaciones.csv``: matriz de comparación con distancia o abstención.
* ``perfiles_por_ejecucion.csv``: distancias Y, E y O por resumen de referencia.
* ``agregados_nino_accion.csv``: media, mediana, mínimo y máximo por niño y acción.
* ``cobertura_por_resumen.csv``: indicadores de la Tabla 7.
* ``controles.json``: controles de separación y completitud.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo import comparacion, perfiles  # noqa: E402
from diana_comparativo.rutas import Rutas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    parser.add_argument("--procesos", type=int, default=4)
    args = parser.parse_args()
    rutas = Rutas(args.rutas) if args.rutas else Rutas()

    entrenamiento = pd.read_csv(rutas["manifiesto_entrenamiento"], dtype={"segment_id": str})
    validacion = pd.read_csv(rutas["manifiesto_validacion"], dtype={"segment_id": str})
    catalogo = comparacion.catalogo_de_referencias(
        pd.read_csv(rutas["catalogo_topk"], dtype={"segment_id": str}),
        pd.read_csv(rutas["catalogo_primarias"], dtype={"medoid_segment_id": str}),
        entrenamiento,
    )
    solicitudes, previstas = comparacion.comparaciones_previstas(validacion, catalogo, rutas["secuencias"])
    print(f"Solicitudes: {len(solicitudes)} ({solicitudes['estado'].value_counts().to_dict()})")
    print(f"Comparaciones previstas: {len(previstas)}; calculando con {args.procesos} procesos...")
    resultado = comparacion.calcular(previstas, args.procesos)
    controles = comparacion.controles_de_separacion(entrenamiento, validacion, catalogo, previstas, resultado)

    por_ejecucion = perfiles.perfiles_por_ejecucion(resultado)
    destino = rutas.salida_de("comparacion")
    solicitudes.to_csv(destino / "solicitudes.csv", index=False)
    resultado.to_csv(destino / "comparaciones.csv", index=False)
    por_ejecucion.to_csv(destino / "perfiles_por_ejecucion.csv", index=False)
    perfiles.agregados_nino_accion(por_ejecucion).to_csv(destino / "agregados_nino_accion.csv", index=False)
    perfiles.cobertura_por_resumen(por_ejecucion).to_csv(destino / "cobertura_por_resumen.csv", index=False)
    (destino / "controles.json").write_text(json.dumps(controles, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"Estados: {resultado['estado'].value_counts().to_dict()}")
    for control in controles:
        print(f"[{'cumple' if control['cumple'] else 'NO CUMPLE'}] {control['control']}")


if __name__ == "__main__":
    main()
