#!/usr/bin/env python
"""Resumen de la revisión visual de las referencias primarias (Sección 3.2).

Lee el registro de revisiones (un evento JSON por línea, solo de anexión) y
cuenta el estado vigente de cada referencia: la última revisión de cada revisor
sobre cada referencia. Las referencias del catálogo sin revisión se informan
como pendientes; nunca se cuentan como confirmadas.

Equivalencias de estados: PASS = confirmada, REVIEW = a revisar, FAIL = descartada.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo.rutas import Rutas  # noqa: E402

ESTADOS = {"PASS": "confirmada", "REVIEW": "a revisar", "FAIL": "descartada"}


def estados_vigentes(eventos: list[dict]) -> dict[str, dict]:
    vigentes: dict[tuple[str, str], dict] = {}
    for evento in eventos:
        clave = (evento["reviewer_id"], evento["reference_id"])
        if clave not in vigentes or evento["revision"] >= vigentes[clave]["revision"]:
            vigentes[clave] = evento
    return {referencia: evento for (_, referencia), evento in vigentes.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    args = parser.parse_args()
    rutas = Rutas(args.rutas) if args.rutas else Rutas()

    eventos = [json.loads(linea) for linea in rutas["registro_revision_visual"].read_text(encoding="utf-8").splitlines() if linea.strip()]
    vigentes = estados_vigentes(eventos)
    primarias = pd.read_csv(rutas["catalogo_primarias"], dtype={"medoid_segment_id": str})
    conteo = Counter(ESTADOS[vigentes[sid]["answers"]["final_status"]] if sid in vigentes else "pendiente"
                     for sid in primarias["medoid_segment_id"])
    print(f"Referencias primarias: {len(primarias)}; revisores: {len({e['reviewer_id'] for e in eventos})}; eventos: {len(eventos)}")
    for estado in ["confirmada", "a revisar", "descartada", "pendiente"]:
        print(f"  {estado}: {conteo.get(estado, 0)}")
    descartadas = primarias[primarias["medoid_segment_id"].map(lambda sid: sid in vigentes and vigentes[sid]["answers"]["final_status"] == "FAIL")]
    for fila in descartadas.itertuples(index=False):
        respuestas = vigentes[fila.medoid_segment_id]["answers"]
        print(f"  Descartada: {fila.action_code} {fila.age_bin_12m_label} (actor: {respuestas['actor_identity']}; "
              f"motivos: {', '.join(respuestas['failure_reasons'])})")


if __name__ == "__main__":
    main()
