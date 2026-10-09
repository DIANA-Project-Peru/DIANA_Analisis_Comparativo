#!/usr/bin/env python
#!/usr/bin/env python
"""Paso 6. Genera las Figuras 1 a 5 del Informe Técnico N.° 1.

| Figura | Contenido | Insumo |
| --- | --- | --- |
| 1 | Procedimiento de análisis comparativo | (diagrama) |
| 2 | Soporte del entrenamiento por acción y grupo etario | manifiesto de entrenamiento |
| 3 | Número de plantillas adaptativas por grupo | estudio de estabilidad (paso 4) |
| 4 | Estabilidad del resultado, asignaciones y conjunto | estudio de estabilidad (paso 4) |
| 5 | Cobertura de la comparación por acción | comparación con referencias (paso 5) |

Las figuras contienen solo conteos agregados o resúmenes por grupo. Se escriben
en ``<salida>/figuras`` (o en ``--destino``) junto con ``registro_figuras.json``,
que guarda los valores graficados.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

# Paleta de referencia (modo claro, impresión). Slots categóricos 1-3 validados.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc2ef", "#2a78d6", "#123f78"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

AGE_BINS = ["36_47mo", "48_59mo", "60_71mo", "72_83mo"]
AGE_LABELS = ["36–47", "48–59", "60–71", "72–83"]


def es_num(x, _pos=None):
    """Formato numérico español: coma decimal, punto de miles."""
    if float(x).is_integer():
        return f"{int(x):,}".replace(",", ".")
    return f"{x:.2f}".rstrip("0").replace(".", ",")


def es_axes(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(es_num))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(es_num))


def box(ax, xy, w, h, title, body, fc="#ffffff", ec=MUTED):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.0))
    ax.text(xy[0] + w / 2, xy[1] + h - 0.2, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK)
    ax.text(xy[0] + w / 2, xy[1] + h - 0.5, body, ha="center", va="top", fontsize=7.3, color=INK2, linespacing=1.35)


def arrow(ax, p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))


# ---------------------------------------------------------------- Figura 1
def fig_procedure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    # Columna de procesamiento común
    cw, ch = 2.55, 1.4
    box(ax, (0.1, 3.7), cw, ch, "Secuencias esqueléticas", "Pose 2D, 17 articulaciones\n4.070 ejecuciones\n157 niños · 13 acciones")
    box(ax, (0.1, 1.95), cw, ch, "Acondicionamiento", "Máscara de validez\nInterpolación de brechas cortas\nCentrado y escala del torso")
    box(ax, (0.1, 0.2), cw, ch, "Comparación DTW", "Articulaciones comunes\n(mínimo 9 de 17)\nDistancia normalizada")
    arrow(ax, (1.375, 3.7), (1.375, 3.35))
    arrow(ax, (1.375, 1.95), (1.375, 1.6))
    # Carril de construcción (entrenamiento)
    ax.add_patch(FancyBboxPatch((3.0, 2.75), 6.9, 2.35, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#f4f8fd", ec=BLUE, lw=0.8))
    ax.text(3.15, 4.9, "Construcción de referencias · entrenamiento: 2.809 ejecuciones, 109 niños", fontsize=7.6, color=BLUE, fontweight="bold", va="center")
    bw, bh, by = 2.05, 1.6, 2.95
    box(ax, (3.15, by), bw, bh, "Matrices por grupo", "Distancias entre\nejecuciones de una\nacción y grupo etario")
    box(ax, (5.45, by), bw, bh, "Referencias", "Medoide, top-k y\nplantillas adaptativas\npor acción y edad")
    box(ax, (7.75, by), bw, bh, "Estabilidad", "Submuestras de n niños\nevaluadas en niños\nreservados")
    arrow(ax, (5.2, by + bh / 2), (5.45, by + bh / 2))
    arrow(ax, (7.5, by + bh / 2), (7.75, by + bh / 2))
    # Carril de evaluación (validación)
    ax.add_patch(FancyBboxPatch((3.0, 0.1), 6.9, 2.35, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fdf5f1", ec=ORANGE, lw=0.8))
    ax.text(3.15, 0.27, "Evaluación comparativa · validación: 1.261 ejecuciones, 48 niños distintos", fontsize=7.6, color="#b5481c", fontweight="bold", va="center")
    box(ax, (3.15, 0.5), 3.2, 1.75, "Comparación con referencias", "Misma acción; grupo etario\nesperado y adyacentes\nDistancia o abstención")
    box(ax, (6.6, 0.5), 3.2, 1.75, "Resultados comparativos", "Matrices, catálogos,\nresultados agregados\ny metadatos técnicos")
    arrow(ax, (6.35, 1.37), (6.6, 1.37))
    # DTW alimenta ambos carriles; las referencias alimentan la evaluación
    arrow(ax, (2.65, 1.1), (3.15, 1.1))
    arrow(ax, (2.65, 1.45), (3.15, 3.3))
    arrow(ax, (5.75, by), (5.75, 2.25))
    fig.savefig(out / "Fig1_procedimiento.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figura 2
def fig_support(manifiesto: Path, out: Path) -> dict:
    tr = pd.read_csv(manifiesto,
                     usecols=["segment_id", "excel_children", "action_code", "age_bin_12m_label"])
    g = tr.groupby(["action_code", "age_bin_12m_label"]).agg(ch=("excel_children", "nunique"),
                                                             ex=("segment_id", "size")).reset_index()
    ch = g.pivot(index="action_code", columns="age_bin_12m_label", values="ch")[AGE_BINS]
    ex = g.pivot(index="action_code", columns="age_bin_12m_label", values="ex")[AGE_BINS]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    im = ax.imshow(ch.values, cmap=SEQ, vmin=0, vmax=40, aspect="auto")
    for i in range(ch.shape[0]):
        for j in range(ch.shape[1]):
            v = ch.values[i, j]
            ax.text(j, i, f"{v} ({ex.values[i, j]})", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v >= 20 else INK)
    ax.set_xticks(range(4), AGE_LABELS)
    ax.set_yticks(range(len(ch)), ch.index)
    ax.set_xlabel("Grupo etario (meses)")
    ax.set_ylabel("Acción")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("niños distintos", fontsize=7.5)
    fig.savefig(out / "Fig2_soporte_entrenamiento.png")
    plt.close(fig)
    return {"children_min": int(ch.values.min()), "children_max": int(ch.values.max()),
            "children_median": float(np.median(ch.values)), "exec_total": int(ex.values.sum())}


# ---------------------------------------------------------------- Figura 3
def fig_templates(run: Path, out: Path) -> dict:
    from matplotlib.colors import ListedColormap, BoundaryNorm
    k = pd.read_csv(run / "tables/k_selection_by_group.csv")
    sel = k.pivot(index="action_code", columns="age_bin_12m_label", values="selected_k")[AGE_BINS]
    strict = k.pivot(index="action_code", columns="age_bin_12m_label", values="selection_strictly_satisfied")[AGE_BINS]
    cmap = ListedColormap(["#d6e6f8", "#6fa6e6", "#1f5fb0"])
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.imshow(sel.values, cmap=cmap, norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3), aspect="auto")
    for i in range(sel.shape[0]):
        for j in range(sel.shape[1]):
            v = int(sel.values[i, j])
            mark = "" if bool(strict.values[i, j]) else "*"
            ax.text(j, i, f"{v}{mark}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v == 3 else INK, fontweight="bold" if v > 1 else "normal")
    ax.set_xticks(range(4), AGE_LABELS)
    ax.set_yticks(range(len(sel)), sel.index)
    ax.set_xlabel("Grupo etario (meses)")
    ax.set_ylabel("Acción")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, ["1 plantilla", "2 plantillas", "3 plantillas"])],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5, frameon=False)
    ax.text(1.04, 0.62, "* número de plantillas\n  de respaldo (no cumple\n  todos los criterios)", transform=ax.transAxes,
            fontsize=7, color=INK2, va="top")
    fig.savefig(out / "Fig3_plantillas_por_grupo.png")
    plt.close(fig)
    return {"k_counts": sel.stack().value_counts().to_dict(), "strict": int(strict.values.sum())}


# ---------------------------------------------------------------- Figura 4
def fig_stability(run: Path, out: Path) -> dict:
    cfg = json.loads((run / "analysis_config.json").read_text())
    thr = cfg["selection_thresholds"]["n"]
    s = pd.read_csv(run / "tables/sample_size_curves.csv")
    g = pd.read_csv(run / "tables/group_status_summary.csv")[["action_code", "age_bin_12m_label", "selected_k"]]
    s = s[s.resampling_scheme.eq("primary_h3")].merge(g, on=["action_code", "age_bin_12m_label"])
    s = s[s.k.eq(s.selected_k)]
    grid = cfg["resampling"]["n_grid"]
    # Solo tamaños de la grilla común y estrictamente menores que el máximo del grupo (N−h)
    s = s[s.n_children.isin(grid) & (s.n_children < s.maximum_train_children)]
    s = s[s.age_bin_12m_label.ne("36_47mo")]
    multi = s[s.selected_k >= 2]

    def agg(frame, col):
        r = frame.groupby("n_children")[col].agg(med="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
        r["groups"] = frame.drop_duplicates(["n_children", "action_code", "age_bin_12m_label"]).groupby("n_children").size()
        return r

    exc = agg(s, "normalized_excess_median")
    asg = agg(multi, "assignment_agreement_median")
    hmed = agg(s, "hungarian_between_median")
    hp90 = agg(s, "hungarian_between_p90")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    ax = axes[0]
    ax.fill_between(exc.index, exc.q1, exc.q3, color=BLUE, alpha=0.15, lw=0)
    ax.plot(exc.index, exc.med, color=BLUE, lw=2, marker="o", ms=3.5)
    ax.axhline(thr["maximum_normalized_excess_median"], color=INK2, lw=1, ls="--")
    ax.text(30, thr["maximum_normalized_excess_median"], "criterio 0,05", fontsize=6.8, color=INK2, va="bottom", ha="right")
    ax.set_title("(a) Resultado: exceso de pérdida", fontsize=8.5, loc="left", color=INK)
    ax.set_ylabel("mediana entre grupos (banda: RIC)")
    ax = axes[1]
    ax.fill_between(asg.index, asg.q1, asg.q3, color=AQUA, alpha=0.15, lw=0)
    ax.plot(asg.index, asg.med, color=AQUA, lw=2, marker="o", ms=3.5, label="acuerdo de asignación")
    ax.axhline(thr["minimum_assignment_agreement_median"], color=INK2, lw=1, ls="--")
    ax.text(30, thr["minimum_assignment_agreement_median"], "criterio 0,8", fontsize=6.8, color=INK2, va="bottom", ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Acuerdo de asignaciones (k ≥ 2)", fontsize=8.5, loc="left", color=INK)
    ax = axes[2]
    ax.fill_between(hp90.index, hp90.q1, hp90.q3, color=ORANGE, alpha=0.15, lw=0)
    ax.plot(hp90.index, hp90.med, color=ORANGE, lw=2, marker="o", ms=3.5, label="percentil 90")
    ax.plot(hmed.index, hmed.med, color=BLUE, lw=2, marker="s", ms=3.5, label="mediana")
    ax.axhline(thr["maximum_hungarian_between_p90"], color=ORANGE, lw=1, ls="--")
    ax.axhline(thr["maximum_hungarian_between_median"], color=BLUE, lw=1, ls="--")
    ax.text(30, thr["maximum_hungarian_between_p90"], "criterio p90 0,20", fontsize=6.8, color=INK2, va="bottom", ha="right")
    ax.text(30, thr["maximum_hungarian_between_median"], "criterio mediana 0,10", fontsize=6.8, color=INK2, va="top", ha="right")
    ax.set_title("(c) Variabilidad del conjunto", fontsize=8.5, loc="left", color=INK)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right")
    for ax, ref in zip(axes, (exc, asg, hp90)):
        ax.set_xticks(ref.index)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("niños usados para construir el conjunto (n)", fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        es_axes(ax, x=False)
        for n, row in ref.iterrows():
            ax.annotate(f"{int(row.groups)}", (n, 0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=6, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "Fig4_estabilidad.png")
    plt.close(fig)
    return {name: frame.round(4).reset_index().to_dict(orient="records")
            for name, frame in (("excess", exc), ("assignment_agreement", asg),
                                ("hungarian_median", hmed), ("hungarian_p90", hp90))}


# ---------------------------------------------------------------- Figura 5
def fig_coverage(comparaciones: Path, out: Path) -> dict:
    L = pd.read_csv(comparaciones, usecols=["accion", "estado"])
    t = pd.crosstab(L.accion, L.estado).reindex(columns=["calculada", "sin_camino_admisible"], fill_value=0)
    t.columns = ["success", "no_admissible_path"]
    t = t.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    y = np.arange(len(t))
    ax.barh(y, t["success"], color=BLUE, height=0.7, label="distancia calculada", edgecolor="#ffffff", linewidth=1)
    ax.barh(y, t["no_admissible_path"], left=t["success"], color=ORANGE, height=0.7,
            label="abstención por soporte articular insuficiente", edgecolor="#ffffff", linewidth=1)
    for i, (_, row) in enumerate(t.iterrows()):
        tot = int(row.sum())
        pct = f"{100 * row['no_admissible_path'] / tot:.1f}".replace(".", ",")
        ax.text(tot + 15, i, f"{row['no_admissible_path']}/{es_num(tot)} ({pct} %)", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("comparaciones ejecución de validación × referencia")
    ax.set_xlim(0, 1450)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, frameon=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    es_axes(ax, y=False)
    fig.savefig(out / "Fig5_cobertura_por_accion.png")
    plt.close(fig)
    return {"by_action": t.to_dict(orient="index")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    parser.add_argument("--estabilidad", type=Path, help="carpeta del estudio (por defecto <salida>/estabilidad/estudio_200_semilla20260715)")
    parser.add_argument("--destino", type=Path, help="carpeta de las figuras (por defecto <salida>/figuras)")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diana_comparativo.rutas import Rutas

    rutas = Rutas(args.rutas) if args.rutas else Rutas()
    estudio = args.estabilidad or rutas.salida / "estabilidad" / "estudio_200_semilla20260715"
    destino = args.destino or rutas.salida_de("figuras")
    destino.mkdir(parents=True, exist_ok=True)
    fig_procedure(destino)
    registro = {
        "Figura 2": fig_support(rutas["manifiesto_entrenamiento"], destino),
        "Figura 3": fig_templates(estudio, destino),
        "Figura 4": fig_stability(estudio, destino),
        "Figura 5": fig_coverage(rutas.salida / "comparacion" / "comparaciones.csv", destino),
    }
    (destino / "registro_figuras.json").write_text(json.dumps(registro, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Figuras escritas en {destino}")


if __name__ == "__main__":
    main()
