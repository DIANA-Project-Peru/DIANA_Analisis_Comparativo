#!/usr/bin/env python
"""Paso 4. Estudio de estabilidad y soporte muestral de plantillas adaptativas.

Ejecuta el estudio descrito en las Secciones 2.6 y 3.3 del informe sobre las
matrices de distancias del conjunto de entrenamiento. Antes de empezar verifica
los conteos y las huellas SHA-256 de los insumos declarados en
``configuracion/estabilidad.json``.

Resultados en ``<salida>/estabilidad/<identificador>/tables``; los principales
son ``group_status_summary.csv``, ``k_selection_by_group.csv``,
``n_selection_by_group.csv``, ``final_template_catalog.csv``,
``policy_comparison.csv`` y ``sample_size_curves.csv``.

La semilla y el número de repeticiones fijan el resultado. Con 200
repeticiones y cuatro procesos tarda unos 36 minutos.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo import estabilidad  # noqa: E402
from diana_comparativo.rutas import Rutas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    parser.add_argument("--config", type=Path, default=estabilidad.DEFAULT_CONFIG)
    parser.add_argument("--identificador", default="estudio_200_semilla20260715")
    parser.add_argument("--repeticiones", type=int, default=None)
    parser.add_argument("--procesos", type=int, default=4)
    parser.add_argument("--solo-verificar", action="store_true", help="solo ejecutar la verificación previa de insumos")
    parser.add_argument("--reanudar", action="store_true")
    args = parser.parse_args()
    rutas = Rutas(args.rutas) if args.rutas else Rutas()

    estabilidad.establecer_raiz_datos(rutas.datos_raiz)
    config = estabilidad.load_config(args.config)
    config["paths"]["run_root"] = str(rutas.salida_de("estabilidad"))
    if args.solo_verificar:
        reporte = estabilidad.preflight(config)
        print(json.dumps({"estado": reporte["status"], "controles": len(reporte["checks"])}, indent=2))
        return
    repeticiones = args.repeticiones or int(config["resampling"]["repetitions"])
    procesos = min(args.procesos, int(config["execution"]["worker_cap"]))
    resumen = estabilidad.run_analysis(config, args.identificador, repeticiones, procesos, args.reanudar)
    print(json.dumps({clave: resumen[clave] for clave in
                      ["status", "groups", "selected_templates_total", "k_distribution", "sample_size_status_distribution"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
