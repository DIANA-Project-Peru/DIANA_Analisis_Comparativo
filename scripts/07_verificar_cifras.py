#!/usr/bin/env python
"""Paso 7. Verifica que las salidas reproduzcan las cifras del Informe Técnico N.° 1.

Cada control compara un valor calculado a partir de los manifiestos y de las
salidas de los pasos 1 a 5 con la cifra publicada en el informe, e indica la
sección o tabla donde aparece. Los controles cuyos insumos aún no se han
generado se informan como "no disponible".

Termina con código 0 solo si todos los controles disponibles coinciden.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo.rutas import Rutas  # noqa: E402

RESULTADOS: list[tuple[str, str, object, object]] = []


def control(seccion: str, descripcion: str, esperado, calcular) -> None:
    try:
        obtenido = calcular()
    except FileNotFoundError:
        obtenido = None
    RESULTADOS.append((seccion, descripcion, esperado, obtenido))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    parser.add_argument("--estabilidad", type=Path)
    args = parser.parse_args()
    rutas = Rutas(args.rutas) if args.rutas else Rutas()
    salida = rutas.salida
    estudio = (args.estabilidad or salida / "estabilidad" / "estudio_200_semilla20260715") / "tables"
    comparacion = salida / "comparacion"

    elegibles = pd.read_csv(rutas["manifiesto_elegibles"], usecols=["segment_id", "excel_children", "action_code"])
    ent = pd.read_csv(rutas["manifiesto_entrenamiento"], usecols=["segment_id", "excel_children"])
    val = pd.read_csv(rutas["manifiesto_validacion"], usecols=["segment_id", "excel_children"])

    # Tabla 2: muestra y partición por niño
    control("Tabla 2", "Ejecuciones / niños / acciones", (4070, 157, 13),
            lambda: (len(elegibles), elegibles.excel_children.nunique(), elegibles.action_code.nunique()))
    control("Tabla 2", "Entrenamiento: ejecuciones / niños", (2809, 109), lambda: (len(ent), ent.excel_children.nunique()))
    control("Tabla 2", "Validación: ejecuciones / niños", (1261, 48), lambda: (len(val), val.excel_children.nunique()))
    control("Tabla 2", "Niños compartidos entre conjuntos", 0,
            lambda: len(set(ent.excel_children) & set(val.excel_children)))

    # Sección 2.3: acondicionamiento
    def acondicionamiento():
        tabla = pd.read_csv(salida / "resumenes" / "acondicionamiento_por_ejecucion.csv")
        return (len(tabla), round(100 * tabla.faltantes_tras_interpolacion.mean(), 1), int((tabla.root_fallback_frames > 0).sum()))
    control("Sección 2.3", "Ejecuciones / % faltantes tras interpolar / con origen alternativo", (4070, 1.0, 326), acondicionamiento)

    # Tabla 4: referencias primarias y top-k
    def referencias():
        topk = pd.read_csv(rutas["catalogo_topk"], dtype={"segment_id": str})
        primarias = topk[topk["rank"] == 1]
        por_grupo = topk.groupby(["action_code", "age_bin_12m_label"]).size().value_counts().to_dict()
        return (len(primarias), primarias.excel_children.nunique(), len(topk), topk.excel_children.nunique(),
                por_grupo.get(4, 0), por_grupo.get(3, 0), por_grupo.get(2, 0))
    control("Tabla 4", "Primarias / niños / top-k / niños / grupos con 4, 3 y 2", (52, 36, 194, 87, 39, 12, 1), referencias)

    # Tablas 4 y 5, Figuras 3 y 4: estudio de estabilidad
    def plantillas():
        grupos = pd.read_csv(estudio / "group_status_summary.csv")
        k = grupos.selected_k.value_counts().to_dict()
        catalogo = pd.read_csv(estudio / "final_template_catalog.csv")
        return (int(grupos.selected_k.sum()), k.get(1, 0), k.get(2, 0), k.get(3, 0),
                int(grupos.selection_strictly_satisfied.sum()), catalogo.template_excel_children.nunique())
    control("Tabla 4 y Figura 3", "Plantillas / grupos k=1, 2, 3 / selección estricta / niños", (85, 23, 25, 4, 30, 59), plantillas)

    def soporte():
        grupos = pd.read_csv(estudio / "group_status_summary.csv")
        estados = grupos.sample_size_status.value_counts().to_dict()
        return (estados.get("not_stabilized_below_available_maximum", 0), estados.get("insufficient_support", 0),
                estados.get("insufficient_children_for_stability_estimation", 0), int(grupos.n_equivalence.notna().sum()),
                int(grupos.n_1se.min()), int(grupos.n_1se.median()), int(grupos.n_1se.max()))
    control("Tabla 5", "Sin estabilizar / soporte insuf. / niños insuf. / con n suficiente / n 1-SE mín, mediana, máx",
            (45, 4, 3, 0, 5, 20, 30), soporte)

    def reduccion():
        politicas = pd.read_csv(estudio / "policy_comparison.csv")
        tabla = politicas.pivot_table(index=["action_code", "age_bin_12m_label"], columns="policy", values="holdout_loss_mean")
        return tuple(round(100 * (1 - (tabla[p] / tabla["fixed_k1"]).mean()), 2) for p in ["adaptive_operational", "cap_k5"])
    control("Sección 3.2", "Reducción media de pérdida frente a una plantilla: adaptativa / hasta cinco (%)", (5.40, 5.55), reduccion)

    # Sección 3.4 y Tablas 6 y 7: comparación con referencias
    def solicitudes():
        tabla = pd.read_csv(comparacion / "solicitudes.csv")
        estados = tabla.estado.value_counts().to_dict()
        return (len(tabla), estados.get("solicitada", 0), estados.get("fuera_de_dominio_etario", 0))
    control("Sección 3.4", "Solicitudes / solicitadas / fuera de dominio etario", (3783, 3364, 419), solicitudes)

    def comparaciones():
        tabla = pd.read_csv(comparacion / "comparaciones.csv", usecols=["estado", "accion"])
        estados = tabla.estado.value_counts().to_dict()
        seis = tabla[tabla.accion.isin(["A01", "A02", "A03", "A04", "A06", "A11"]) & (tabla.estado == "sin_camino_admisible")]
        return (len(tabla), estados.get("calculada", 0), estados.get("sin_camino_admisible", 0), len(seis))
    control("Tabla 6 y Figura 5", "Comparaciones / calculadas / abstenciones / abstenciones en A01–A04, A06, A11",
            (12860, 10716, 2144, 2098), comparaciones)

    def cobertura():
        tabla = pd.read_csv(comparacion / "cobertura_por_resumen.csv").set_index("resumen")
        return tuple(int(tabla.loc[r, c]) for c in ["con_distancia_E", "perfil_completo", "sin_ninguna_distancia"]
                     for r in ["primaria", "mediana_topk", "minimo_topk"])
    control("Tabla 7", "Distancia E / perfil completo / sin distancia (primaria, mediana, mínimo)",
            (1137, 962, 962, 750, 462, 462, 124, 164, 164), cobertura)

    def variabilidad():
        tabla = pd.read_csv(comparacion / "cobertura_por_resumen.csv").set_index("resumen")
        return (int(tabla.loc["primaria", "soporte_comun"]),
                tuple(int(tabla.loc[r, f"mas_cercano_{g}"]) for r in ["primaria", "mediana_topk", "minimo_topk"] for g in "YEO"),
                int(tabla.loc["mediana_topk", "cambio_frente_a_primaria"]), int(tabla.loc["minimo_topk", "cambio_frente_a_primaria"]))
    control("Tabla 7", "Soporte común / grupo más cercano Y, E, O por resumen / cambios frente a primaria",
            (462, (106, 204, 152, 191, 175, 96, 99, 237, 126), 249, 123), variabilidad)

    control("Sección 4.3", "Controles de separación y completitud cumplidos", (6, 6),
            lambda: (lambda c: (sum(x["cumple"] for x in c), len(c)))(json.loads((comparacion / "controles.json").read_text())))

    # Revisión visual (Sección 3.2)
    def revision():
        from resumen_revision_visual import ESTADOS, estados_vigentes
        eventos = [json.loads(linea) for linea in rutas["registro_revision_visual"].read_text().splitlines() if linea.strip()]
        vigentes = estados_vigentes(eventos)
        primarias = pd.read_csv(rutas["catalogo_primarias"], dtype={"medoid_segment_id": str}).medoid_segment_id
        estados = [ESTADOS[vigentes[s]["answers"]["final_status"]] if s in vigentes else "pendiente" for s in primarias]
        return (estados.count("confirmada"), estados.count("descartada"), estados.count("pendiente"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    control("Sección 3.2", "Revisión visual: confirmadas / descartadas / pendientes", (5, 1, 46), revision)

    fallos = 0
    print(f"{'Ubicación':<20} {'Resultado':<14} Control")
    for seccion, descripcion, esperado, obtenido in RESULTADOS:
        if obtenido is None:
            estado = "no disponible"
        elif obtenido == esperado:
            estado = "coincide"
        else:
            estado = "DIFIERE"
            fallos += 1
        print(f"{seccion:<20} {estado:<14} {descripcion}")
        if estado == "DIFIERE":
            print(f"{'':<35} esperado {esperado}; obtenido {obtenido}")
    disponibles = sum(obtenido is not None for *_, obtenido in RESULTADOS)
    print(f"\n{disponibles - fallos} de {disponibles} controles disponibles coinciden con el informe.")
    raise SystemExit(1 if fallos else 0)


if __name__ == "__main__":
    main()
