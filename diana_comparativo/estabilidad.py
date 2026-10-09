"""Estudio de estabilidad y soporte muestral de plantillas reales (Secciones 2.6 y 3.3).

Para cada acción y grupo etario del conjunto de entrenamiento:

1. **Remuestreo.** En cada una de 200 repeticiones se reservan 3 niños que no
   intervienen en la construcción; los demás se ordenan al azar para formar
   subconjuntos anidados de n niños (3, 5, 8, 10, 12, 15, 20, 25, 30 y el
   máximo disponible).
2. **Plantillas adaptativas.** Para cada n y k = 1..5 se resuelve un problema de
   p-medianas con ejecuciones reales (construcción voraz, intercambios y cinco
   reinicios), con a lo sumo una plantilla por niño. La pérdida de un niño es la
   media, sobre sus repeticiones, de la distancia a su plantilla más cercana.
3. **Estabilidad.** Cada conjunto se compara con el construido con el máximo de
   niños de la misma repetición: exceso de pérdida en los niños reservados,
   acuerdo de asignaciones y distancia húngara entre plantillas, ambas en
   unidades de la distancia típica entre niños del grupo.
4. **Selección de k y de n.** k se elige con ganancia incremental, soporte,
   estabilidad geométrica y de asignaciones (techo operativo 3). Un grupo
   alcanza soporte suficiente en n si cumple todos los criterios en ese tamaño
   y en dos tamaños posteriores.

El estudio no calcula DTW: usa las matrices de distancias por grupo ya
almacenadas y no lee el conjunto de validación (solo verifica la disjunción).
Este módulo conserva la lógica del estudio original; se retiraron únicamente
las figuras internas, los videos de ejemplo y la auditoría MILP.

Se ejecuta con python scripts/04_estudio_estabilidad.py.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm


# Raíz de los datos locales autorizados. Las rutas relativas de la configuración
# se resuelven contra ella. La fija el script de entrada (establecer_raiz_datos)
# y se hereda en los procesos de trabajo mediante la variable de entorno.
ROOT = Path(os.environ.get("DIANA_DATOS_RAIZ", ".")).resolve()
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configuracion/estabilidad.json"


def establecer_raiz_datos(ruta: Path) -> None:
    global ROOT
    ROOT = Path(ruta).resolve()
    os.environ["DIANA_DATOS_RAIZ"] = str(ROOT)
PIPE = "|"
AGE_ORDER = ["36_47mo", "48_59mo", "60_71mo", "72_83mo"]


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def canonical_child(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Missing excel_children")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text:
        raise ValueError("Empty excel_children")
    return text


def split_pipe(value: object) -> list[str]:
    if value is None or pd.isna(value) or str(value) == "":
        return []
    return str(value).split(PIPE)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairwise_bundle_hash(pairwise_directory: Path) -> str:
    files = sorted(pairwise_directory.glob("action_code-*.csv"))
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        digest.update(f"{hash_file(path)}  {rel}\n".encode("utf-8"))
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path.resolve())
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else ROOT / path


def atomic_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def input_hashes(config: dict[str, Any]) -> dict[str, str]:
    values = {
        "train_manifest": hash_file(resolve_path(config, "train_manifest")),
        "validation_manifest": hash_file(resolve_path(config, "validation_manifest")),
        "pairwise_index": hash_file(resolve_path(config, "pairwise_index")),
        "pairwise_bundle": pairwise_bundle_hash(resolve_path(config, "pairwise_directory")),
        "official_medoids": hash_file(resolve_path(config, "official_medoids")),
        "official_topk": hash_file(resolve_path(config, "official_topk")),
    }
    return values


@dataclass
class GroupData:
    group_index: int
    action_code: str
    age_bin: str
    manifest: pd.DataFrame
    segment_ids: list[str]
    segment_to_index: dict[str, int]
    segment_to_child: dict[str, str]
    child_indices: dict[str, np.ndarray]
    distances: np.ndarray
    pairwise_path: Path
    official_medoid: str
    official_topk: list[str]
    between_child_scale: float
    within_child_scale: float

    @property
    def children(self) -> list[str]:
        return sorted(self.child_indices)

    @property
    def key(self) -> str:
        return f"{self.action_code}__{self.age_bin}"


def reconstruct_matrix_from_long(
    pairwise: pd.DataFrame, expected_ids: Sequence[str]
) -> tuple[np.ndarray, dict[str, int]]:
    required = {"segment_id_a", "segment_id_b", "distance"}
    missing = required.difference(pairwise.columns)
    if missing:
        raise ValueError(f"Pairwise missing columns: {sorted(missing)}")
    ids = sorted(map(str, expected_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("Expected segment IDs are duplicated")
    index = {segment_id: i for i, segment_id in enumerate(ids)}
    expected_pairs = len(ids) * (len(ids) - 1) // 2
    if len(pairwise) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pair rows, found {len(pairwise)}")
    matrix = np.full((len(ids), len(ids)), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    seen: set[tuple[str, str]] = set()
    for row in pairwise.itertuples(index=False):
        left = str(row.segment_id_a)
        right = str(row.segment_id_b)
        if left not in index or right not in index:
            raise ValueError(f"Unknown pairwise ID: {left}, {right}")
        if left == right:
            raise ValueError(f"Unexpected diagonal pair: {left}")
        key = tuple(sorted((left, right)))
        if key in seen:
            raise ValueError(f"Duplicated unordered pair: {key}")
        seen.add(key)
        distance = float(row.distance)
        if not np.isfinite(distance):
            raise ValueError(f"Non-finite distance for {key}")
        if distance < 0:
            raise ValueError(f"Negative distance for {key}")
        i, j = index[left], index[right]
        matrix[i, j] = distance
        matrix[j, i] = distance
    if len(seen) != expected_pairs or np.isnan(matrix).any():
        raise ValueError("Pairwise matrix is incomplete")
    if not np.allclose(matrix, matrix.T, atol=0.0, rtol=0.0):
        raise ValueError("Pairwise matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=0.0, rtol=0.0):
        raise ValueError("Pairwise matrix diagonal is not zero")
    return matrix, index


def robust_group_scales(
    distances: np.ndarray, child_indices: dict[str, np.ndarray]
) -> tuple[float, float]:
    children = sorted(child_indices)
    between: list[float] = []
    for left_i, left in enumerate(children):
        for right in children[left_i + 1 :]:
            values = distances[np.ix_(child_indices[left], child_indices[right])]
            between.append(float(np.median(values)))
    within: list[float] = []
    for child in children:
        indices = child_indices[child]
        if len(indices) >= 2:
            tri = distances[np.ix_(indices, indices)][np.triu_indices(len(indices), 1)]
            if tri.size:
                within.append(float(np.median(tri)))
    between_scale = float(np.median(between)) if between else float("nan")
    within_scale = float(np.median(within)) if within else float("nan")
    return between_scale, within_scale


def load_group_data(
    config: dict[str, Any],
    group_index: int,
    action_code: str,
    age_bin: str,
    manifest: pd.DataFrame | None = None,
    pairwise_index: pd.DataFrame | None = None,
    medoids: pd.DataFrame | None = None,
    topk: pd.DataFrame | None = None,
) -> GroupData:
    if manifest is None:
        manifest = pd.read_csv(resolve_path(config, "train_manifest"), dtype={"segment_id": str})
        manifest["excel_children"] = manifest["excel_children"].map(canonical_child)
    if pairwise_index is None:
        pairwise_index = pd.read_csv(resolve_path(config, "pairwise_index"))
    if medoids is None:
        medoids = pd.read_csv(resolve_path(config, "official_medoids"), dtype={"medoid_segment_id": str})
    if topk is None:
        topk = pd.read_csv(resolve_path(config, "official_topk"), dtype={"segment_id": str})
    group_manifest = manifest[
        manifest["action_code"].astype(str).eq(action_code)
        & manifest["age_bin_12m_label"].astype(str).eq(age_bin)
    ].copy()
    if group_manifest.empty:
        raise ValueError(f"Empty manifest group {action_code} {age_bin}")
    group_manifest["segment_id"] = group_manifest["segment_id"].astype(str)
    group_manifest["excel_children"] = group_manifest["excel_children"].map(canonical_child)
    segment_ids = sorted(group_manifest["segment_id"].tolist())
    index_rows = pairwise_index[
        pairwise_index["action_code"].astype(str).eq(action_code)
        & pairwise_index["age_bin_12m_label"].astype(str).eq(age_bin)
    ]
    if len(index_rows) != 1:
        raise ValueError(f"Expected one index row for {action_code} {age_bin}, found {len(index_rows)}")
    pairwise_path = Path(str(index_rows.iloc[0]["pairwise_path"]))
    if not pairwise_path.is_absolute():
        pairwise_path = ROOT / pairwise_path
    if not pairwise_path.exists():
        fallback = resolve_path(config, "pairwise_directory") / pairwise_path.name
        pairwise_path = fallback
    pairwise = pd.read_csv(
        pairwise_path,
        dtype={"segment_id_a": str, "segment_id_b": str},
    )
    distances, segment_to_index = reconstruct_matrix_from_long(pairwise, segment_ids)
    segment_to_child = dict(
        zip(group_manifest["segment_id"], group_manifest["excel_children"], strict=True)
    )
    child_indices_list: dict[str, list[int]] = defaultdict(list)
    for segment_id in segment_ids:
        child_indices_list[segment_to_child[segment_id]].append(segment_to_index[segment_id])
    child_indices = {
        child: np.asarray(indices, dtype=np.int64)
        for child, indices in child_indices_list.items()
    }
    medoid_rows = medoids[
        medoids["action_code"].astype(str).eq(action_code)
        & medoids["age_bin_12m_label"].astype(str).eq(age_bin)
    ]
    if len(medoid_rows) != 1:
        raise ValueError(f"Expected one official medoid for {action_code} {age_bin}")
    topk_rows = topk[
        topk["action_code"].astype(str).eq(action_code)
        & topk["age_bin_12m_label"].astype(str).eq(age_bin)
    ].sort_values("rank")
    between, within = robust_group_scales(distances, child_indices)
    return GroupData(
        group_index=group_index,
        action_code=action_code,
        age_bin=age_bin,
        manifest=group_manifest,
        segment_ids=segment_ids,
        segment_to_index=segment_to_index,
        segment_to_child=segment_to_child,
        child_indices=child_indices,
        distances=distances,
        pairwise_path=pairwise_path,
        official_medoid=str(medoid_rows.iloc[0]["medoid_segment_id"]),
        official_topk=topk_rows["segment_id"].astype(str).tolist(),
        between_child_scale=between,
        within_child_scale=within,
    )


def candidate_cost_matrix(
    group: GroupData, client_children: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    client_children = list(map(str, client_children))
    client_set = set(client_children)
    candidate_indices = np.asarray(
        [
            i
            for i, segment_id in enumerate(group.segment_ids)
            if group.segment_to_child[segment_id] in client_set
        ],
        dtype=np.int64,
    )
    candidate_ids = [group.segment_ids[i] for i in candidate_indices]
    costs = np.vstack(
        [
            group.distances[np.ix_(group.child_indices[child], candidate_indices)].mean(axis=0)
            for child in client_children
        ]
    )
    origins = np.asarray(
        [client_children.index(group.segment_to_child[segment_id]) for segment_id in candidate_ids],
        dtype=np.int64,
    )
    return costs, origins, candidate_ids, candidate_indices


def facility_objective(costs: np.ndarray, selected: Sequence[int]) -> float:
    return float(np.min(costs[:, np.asarray(selected, dtype=int)], axis=1).mean())


def refine_swaps(
    costs: np.ndarray,
    facility_child: np.ndarray,
    facility_ids: Sequence[str],
    initial: Sequence[int],
    tolerance: float = 1e-12,
) -> tuple[float, list[int], int]:
    selected = sorted(map(int, initial), key=lambda i: facility_ids[i])
    current = facility_objective(costs, selected)
    swaps = 0
    while True:
        selected_set = set(selected)
        best_key: tuple[float, tuple[str, ...]] | None = None
        best_selected: list[int] | None = None
        for remove_position in range(len(selected)):
            retained = selected[:remove_position] + selected[remove_position + 1 :]
            retained_children = {int(facility_child[index]) for index in retained}
            retained_loss = (
                np.min(costs[:, retained], axis=1)
                if retained
                else np.full(costs.shape[0], np.inf, dtype=float)
            )
            for candidate in range(costs.shape[1]):
                if candidate in selected_set or int(facility_child[candidate]) in retained_children:
                    continue
                objective = float(np.minimum(retained_loss, costs[:, candidate]).mean())
                proposal = sorted(retained + [candidate], key=lambda i: facility_ids[i])
                key = (objective, tuple(facility_ids[i] for i in proposal))
                if best_key is None or key < best_key:
                    best_key = key
                    best_selected = proposal
        if best_key is None or best_key[0] >= current - tolerance:
            return current, selected, swaps
        current = best_key[0]
        selected = best_selected or selected
        swaps += 1


def solve_greedy_swap(
    costs: np.ndarray,
    facility_child: np.ndarray,
    facility_ids: Sequence[str],
    k: int,
    rng: np.random.Generator,
    n_starts: int = 5,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    if k < 1 or k > len(np.unique(facility_child)):
        raise ValueError(f"Invalid k={k} for {len(np.unique(facility_child))} facility children")
    selected: list[int] = []
    selected_children: set[int] = set()
    running = np.full(costs.shape[0], np.inf, dtype=float)
    for _ in range(k):
        proposals: list[tuple[float, str, int]] = []
        for candidate in range(costs.shape[1]):
            if int(facility_child[candidate]) in selected_children:
                continue
            objective = float(np.minimum(running, costs[:, candidate]).mean())
            proposals.append((objective, facility_ids[candidate], candidate))
        _, _, chosen = min(proposals)
        selected.append(chosen)
        selected_children.add(int(facility_child[chosen]))
        running = np.minimum(running, costs[:, chosen])
    solutions = [refine_swaps(costs, facility_child, facility_ids, selected, tolerance)]
    unique_children = np.unique(facility_child)
    for _ in range(max(0, n_starts - 1)):
        sampled_children = rng.choice(unique_children, size=k, replace=False)
        random_selected = [
            int(rng.choice(np.flatnonzero(facility_child == child))) for child in sampled_children
        ]
        solutions.append(
            refine_swaps(costs, facility_child, facility_ids, random_selected, tolerance)
        )
    objective, best, swaps = min(
        solutions,
        key=lambda item: (item[0], tuple(facility_ids[i] for i in item[1])),
    )
    best = sorted(best, key=lambda i: facility_ids[i])
    if len({int(facility_child[i]) for i in best}) != k:
        raise RuntimeError("Solver selected multiple templates from one child")
    return {
        "objective": float(objective),
        "selected_positions": best,
        "selected_ids": [facility_ids[i] for i in best],
        "n_starts": int(n_starts),
        "swaps": int(swaps),
    }


def evaluate_template_set(
    group: GroupData,
    children: Sequence[str],
    template_indices: Sequence[int],
    ambiguity_threshold: float = 0.0,
) -> dict[str, Any]:
    template_indices = np.asarray(template_indices, dtype=np.int64)
    primary_costs: list[float] = []
    secondary_costs: list[float] = []
    assignments: list[int] = []
    margins: list[float] = []
    repeat_consistency: list[float] = []
    for child in children:
        repeat_matrix = group.distances[np.ix_(group.child_indices[str(child)], template_indices)]
        mean_by_template = repeat_matrix.mean(axis=0)
        assignment = int(np.argmin(mean_by_template))
        primary_costs.append(float(mean_by_template[assignment]))
        secondary_costs.append(float(np.min(repeat_matrix, axis=1).mean()))
        assignments.append(assignment)
        if len(template_indices) > 1:
            ordered = np.sort(mean_by_template)
            margins.append(float(ordered[1] - ordered[0]))
            per_repeat = np.argmin(repeat_matrix, axis=1)
            repeat_consistency.append(float(np.mean(per_repeat == assignment)))
        else:
            margins.append(float("inf"))
            repeat_consistency.append(1.0)
    primary = np.asarray(primary_costs, dtype=float)
    secondary = np.asarray(secondary_costs, dtype=float)
    margin_array = np.asarray(margins, dtype=float)
    finite_margins = margin_array[np.isfinite(margin_array)]
    return {
        "child_primary_losses": primary,
        "child_secondary_losses": secondary,
        "assignments": np.asarray(assignments, dtype=np.int64),
        "margins": margin_array,
        "mean_primary": float(np.mean(primary)),
        "median_primary": float(np.median(primary)),
        "p90_primary": float(np.quantile(primary, 0.9)),
        "mean_secondary": float(np.mean(secondary)),
        "median_margin": float(np.median(finite_margins)) if finite_margins.size else float("nan"),
        "ambiguous_fraction": (
            float(np.mean(finite_margins <= ambiguity_threshold))
            if finite_margins.size
            else 0.0
        ),
        "repeat_consistency": float(np.mean(repeat_consistency)),
    }


def minimum_support(n_children: int, absolute: int = 5, fraction: float = 0.15) -> int:
    return max(int(absolute), int(math.ceil(float(fraction) * int(n_children))))


def support_summary(assignments: np.ndarray, k: int, minimum: int) -> dict[str, Any]:
    counts = np.bincount(assignments, minlength=k).astype(int)
    return {
        "counts": counts,
        "minimum_count": int(counts.min()) if counts.size else 0,
        "support_ok": bool(np.all(counts >= minimum)),
        "coverage_fractions": counts / max(1, int(assignments.size)),
    }


def normalized(value: float, scale: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(value / scale)


def set_geometry(
    distance_matrix: np.ndarray,
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    between_scale: float,
    within_scale: float,
) -> dict[str, Any]:
    left = np.asarray(left_indices, dtype=int)
    right = np.asarray(right_indices, dtype=int)
    if len(left) != len(right):
        raise ValueError("Hungarian set comparison requires equal k")
    costs = distance_matrix[np.ix_(left, right)]
    row_ind, col_ind = linear_sum_assignment(costs)
    matched = costs[row_ind, col_ind]
    hausdorff = max(float(np.max(np.min(costs, axis=1))), float(np.max(np.min(costs, axis=0))))
    average_nn = float(
        (np.mean(np.min(costs, axis=1)) + np.mean(np.min(costs, axis=0))) / 2.0
    )
    mean_value = float(np.mean(matched))
    max_value = float(np.max(matched))
    p90_value = float(np.quantile(matched, 0.9))
    mapping = np.full(len(left), -1, dtype=int)
    mapping[row_ind] = col_ind
    return {
        "hungarian_mean": mean_value,
        "hungarian_max": max_value,
        "hungarian_p90": p90_value,
        "hungarian_mean_between": normalized(mean_value, between_scale),
        "hungarian_max_between": normalized(max_value, between_scale),
        "hungarian_p90_between": normalized(p90_value, between_scale),
        "hungarian_mean_within": normalized(mean_value, within_scale),
        "hungarian_max_within": normalized(max_value, within_scale),
        "hungarian_p90_within": normalized(p90_value, within_scale),
        "hausdorff": hausdorff,
        "hausdorff_between": normalized(hausdorff, between_scale),
        "hausdorff_within": normalized(hausdorff, within_scale),
        "symmetric_average_nn": average_nn,
        "symmetric_average_nn_between": normalized(average_nn, between_scale),
        "symmetric_average_nn_within": normalized(average_nn, within_scale),
        "mapping": mapping,
    }


def assignment_comparison(
    left_assignments: np.ndarray,
    right_assignments: np.ndarray,
    left_to_right_mapping: np.ndarray,
    k: int,
) -> dict[str, float]:
    left_assignments = np.asarray(left_assignments, dtype=int)
    right_assignments = np.asarray(right_assignments, dtype=int)
    if left_assignments.shape != right_assignments.shape:
        raise ValueError("Assignment vectors must have equal length")
    mapped = left_to_right_mapping[left_assignments]
    agreement = float(np.mean(mapped == right_assignments))
    ari = float(adjusted_rand_score(left_assignments, right_assignments))
    cluster_jaccards: list[float] = []
    for right_label in range(k):
        left_set = mapped == right_label
        right_set = right_assignments == right_label
        union = int(np.count_nonzero(left_set | right_set))
        cluster_jaccards.append(
            float(np.count_nonzero(left_set & right_set) / union) if union else 1.0
        )
    return {
        "assignment_agreement": agreement,
        "assignment_ari": ari,
        "cluster_jaccard_mean": float(np.mean(cluster_jaccards)),
        "cluster_jaccard_min": float(np.min(cluster_jaccards)),
    }


def derive_rng(*components: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(value) for value in components]))


def seeded_draw(components: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(derive_rng(*components).integers(0, 2**31 - 1, size=5).tolist())


def build_resampling(
    children: Sequence[str],
    holdout_size: int,
    base_grid: Sequence[int],
    rng: np.random.Generator,
) -> dict[str, Any]:
    children = list(map(str, children))
    if holdout_size < 1 or holdout_size >= len(children):
        raise ValueError("Invalid holdout size")
    permutation = rng.permutation(children).tolist()
    holdout = permutation[:holdout_size]
    training_order = permutation[holdout_size:]
    maximum = len(training_order)
    n_values = sorted({int(n) for n in base_grid if 3 <= int(n) <= maximum} | {maximum})
    nested = {n: training_order[:n] for n in n_values}
    return {
        "holdout": holdout,
        "training_order": training_order,
        "n_values": n_values,
        "nested": nested,
        "maximum_train": maximum,
    }


def one_se_n(curve: pd.DataFrame, max_n: int) -> int | None:
    eligible = curve[curve["n_children"] < max_n].sort_values("n_children")
    if eligible.empty:
        return None
    best_row = eligible.loc[eligible["holdout_loss_mean"].idxmin()]
    threshold = float(best_row["holdout_loss_mean"] + best_row["holdout_loss_se"])
    within = eligible[eligible["holdout_loss_mean"] <= threshold]
    return int(within.iloc[0]["n_children"]) if not within.empty else None


def select_persistent_n(
    curve: pd.DataFrame,
    max_n: int,
    thresholds: dict[str, Any],
) -> int | None:
    work = curve[curve["n_children"] < max_n].sort_values("n_children").reset_index(drop=True)
    required_later = int(thresholds["required_later_nontrivial_sizes"])
    if len(work) < required_later + 1:
        return None
    pass_mask = (
        (work["normalized_excess_median"] <= thresholds["maximum_normalized_excess_median"])
        & (work["normalized_excess_p90"] <= thresholds["maximum_normalized_excess_p90"])
        & (work["hungarian_between_median"] <= thresholds["maximum_hungarian_between_median"])
        & (work["hungarian_between_p90"] <= thresholds["maximum_hungarian_between_p90"])
        & (work["assignment_agreement_median"] >= thresholds["minimum_assignment_agreement_median"])
        & (work["assignment_ari_median"] >= thresholds["minimum_assignment_ari_median"])
        & (work["support_rate"] >= thresholds["minimum_support_success_rate"])
        & (
            work["marginal_improvement_abs_normalized"]
            <= thresholds["maximum_marginal_improvement_per_child_normalized"]
        )
    )
    for position in range(len(work) - required_later):
        window = pass_mask.iloc[position : position + required_later + 1]
        if not bool(window.all()):
            continue
        baseline = float(work.iloc[position]["holdout_loss_mean_normalized"])
        later = work.iloc[position + 1 :]["holdout_loss_mean_normalized"]
        if later.empty or float(later.max() - baseline) <= thresholds["maximum_later_deterioration_normalized"]:
            return int(work.iloc[position]["n_children"])
    return None


def ensure_run_isolation(run_dir: Path, analysis_id: str, resume: bool) -> None:
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run already exists and --resume was not supplied: {run_dir}")
    summary = run_dir / "run_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        existing = data.get("analysis_id")
        if existing and existing != analysis_id:
            raise RuntimeError(f"Run isolation violation: {existing} != {analysis_id}")


def partial_is_complete(partial_dir: Path, digest: str, repetitions: int) -> bool:
    marker = partial_dir / "complete.json"
    required = [
        partial_dir / "optimization_results.parquet",
        partial_dir / "external_resampling_plan.csv",
        partial_dir / "nested_training_sets.csv",
        partial_dir / "full_group_solutions.csv",
        partial_dir / "assignment_child_stability.csv",
        partial_dir / "coassignment_matrix.csv",
    ]
    if not marker.exists() or not all(path.exists() for path in required):
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        data.get("status") == "complete"
        and data.get("config_digest") == digest
        and int(data.get("repetitions", -1)) == int(repetitions)
    )


def solve_training_set(
    group: GroupData,
    training_children: Sequence[str],
    k: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> dict[str, Any]:
    costs, origins, candidate_ids, candidate_indices = candidate_cost_matrix(
        group, training_children
    )
    solver_config = config["solver"]
    solution = solve_greedy_swap(
        costs,
        origins,
        candidate_ids,
        int(k),
        rng,
        n_starts=int(solver_config["n_starts"]),
        tolerance=float(solver_config["swap_tolerance"]),
    )
    positions = np.asarray(solution["selected_positions"], dtype=int)
    matrix_indices = candidate_indices[positions]
    ids = [candidate_ids[position] for position in positions]
    order = np.argsort(np.asarray(ids, dtype=object))
    matrix_indices = matrix_indices[order]
    ids = [ids[index] for index in order]
    training_evaluation = evaluate_template_set(
        group,
        training_children,
        matrix_indices,
        ambiguity_threshold=float(config["selection_thresholds"]["ambiguity"]["margin_between_scale_fraction"])
        * group.between_child_scale,
    )
    template_support = config["templates"]
    support_minimum = minimum_support(
        len(training_children),
        int(template_support["minimum_assigned_children_absolute"]),
        float(template_support["minimum_assigned_children_fraction"]),
    )
    support = support_summary(training_evaluation["assignments"], k, support_minimum)
    return {
        "objective": float(solution["objective"]),
        "selected_ids": ids,
        "selected_matrix_indices": matrix_indices.astype(int).tolist(),
        "selected_children": [group.segment_to_child[segment_id] for segment_id in ids],
        "n_starts": int(solution["n_starts"]),
        "swaps": int(solution["swaps"]),
        "train_evaluation": training_evaluation,
        "support": support,
        "support_minimum": support_minimum,
    }


def jaccard_sets(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def functional_set_overlap(
    group: GroupData, left_indices: Sequence[int], right_indices: Sequence[int], threshold: float
) -> float:
    costs = group.distances[np.ix_(np.asarray(left_indices), np.asarray(right_indices))]
    left_covered = float(np.mean(np.min(costs, axis=1) <= threshold))
    right_covered = float(np.mean(np.min(costs, axis=0) <= threshold))
    return (left_covered + right_covered) / 2.0


def _base_solution_record(
    group: GroupData,
    scheme: str,
    holdout_size: int,
    repetition: int,
    n_children: int,
    maximum_train: int,
    k: int,
    holdout: Sequence[str],
    training_children: Sequence[str],
    solution: dict[str, Any],
) -> dict[str, Any]:
    evaluation = solution["train_evaluation"]
    support = solution["support"]
    return {
        "group_index": group.group_index,
        "action_code": group.action_code,
        "age_bin_12m_label": group.age_bin,
        "resampling_scheme": scheme,
        "holdout_size": int(holdout_size),
        "repetition": int(repetition),
        "n_children": int(n_children),
        "maximum_train_children": int(maximum_train),
        "is_max_train_reference": bool(n_children == maximum_train),
        "k": int(k),
        "n_group_children": len(group.children),
        "n_group_clips": len(group.segment_ids),
        "holdout_children": PIPE.join(holdout),
        "training_children": PIPE.join(training_children),
        "selected_template_ids": PIPE.join(solution["selected_ids"]),
        "selected_template_children": PIPE.join(solution["selected_children"]),
        "train_objective_primary": float(solution["objective"]),
        "train_child_loss_mean": float(evaluation["mean_primary"]),
        "train_child_loss_median": float(evaluation["median_primary"]),
        "train_child_loss_p90": float(evaluation["p90_primary"]),
        "train_child_loss_secondary": float(evaluation["mean_secondary"]),
        "support_minimum_required": int(solution["support_minimum"]),
        "assigned_children_counts": PIPE.join(map(str, support["counts"].tolist())),
        "minimum_assigned_children": int(support["minimum_count"]),
        "support_ok": bool(support["support_ok"]),
        "coverage_fractions": PIPE.join(f"{value:.12g}" for value in support["coverage_fractions"]),
        "solver_n_starts": int(solution["n_starts"]),
        "solver_swaps_final_start": int(solution["swaps"]),
        "between_child_distance_scale": float(group.between_child_scale),
        "within_child_repeat_distance_scale": float(group.within_child_scale),
        "_template_indices": solution["selected_matrix_indices"],
    }


def _append_reference_metrics(
    group: GroupData,
    row: dict[str, Any],
    reference: dict[str, Any],
    holdout: Sequence[str],
    config: dict[str, Any],
) -> None:
    current_indices = row["_template_indices"]
    reference_indices = reference["_template_indices"]
    ambiguity_threshold = (
        float(config["selection_thresholds"]["ambiguity"]["margin_between_scale_fraction"])
        * group.between_child_scale
    )
    current_eval = evaluate_template_set(
        group, holdout, current_indices, ambiguity_threshold=ambiguity_threshold
    )
    reference_eval = evaluate_template_set(
        group, holdout, reference_indices, ambiguity_threshold=ambiguity_threshold
    )
    geometry = set_geometry(
        group.distances,
        current_indices,
        reference_indices,
        group.between_child_scale,
        group.within_child_scale,
    )
    comparison = assignment_comparison(
        current_eval["assignments"],
        reference_eval["assignments"],
        geometry["mapping"],
        int(row["k"]),
    )
    current_ids = split_pipe(row["selected_template_ids"])
    reference_ids = split_pipe(reference["selected_template_ids"])
    exact_threshold_config = config["selection_thresholds"]["identity"]
    functional_threshold = (
        group.within_child_scale
        * float(exact_threshold_config["functional_neighborhood_within_scale_multiplier"])
        if np.isfinite(group.within_child_scale) and group.within_child_scale > 0
        else group.between_child_scale
        * float(exact_threshold_config["functional_neighborhood_between_scale_fallback_multiplier"])
    )
    excess = float(current_eval["mean_primary"] - reference_eval["mean_primary"])
    row.update(
        {
            "holdout_child_loss_mean": current_eval["mean_primary"],
            "holdout_child_loss_median": current_eval["median_primary"],
            "holdout_child_loss_p90": current_eval["p90_primary"],
            "holdout_child_loss_secondary": current_eval["mean_secondary"],
            "holdout_repeat_consistency": current_eval["repeat_consistency"],
            "holdout_median_assignment_margin": current_eval["median_margin"],
            "holdout_ambiguous_children_fraction": current_eval["ambiguous_fraction"],
            "max_train_holdout_child_loss_mean": reference_eval["mean_primary"],
            "max_train_holdout_child_loss_median": reference_eval["median_primary"],
            "max_train_holdout_child_loss_p90": reference_eval["p90_primary"],
            "max_train_selected_template_ids": reference["selected_template_ids"],
            "holdout_excess_loss": excess,
            "normalized_holdout_excess_loss": normalized(excess, group.between_child_scale),
            "exact_set_match_max_train": bool(set(current_ids) == set(reference_ids)),
            "jaccard_max_train": jaccard_sets(current_ids, reference_ids),
            "functional_overlap_max_train": functional_set_overlap(
                group, current_indices, reference_indices, functional_threshold
            ),
            "official_topk_overlap_fraction": float(
                len(set(current_ids) & set(group.official_topk)) / max(1, len(current_ids))
            ),
            "hungarian_mean_to_max_train": geometry["hungarian_mean"],
            "hungarian_max_to_max_train": geometry["hungarian_max"],
            "hungarian_p90_to_max_train": geometry["hungarian_p90"],
            "hungarian_mean_to_max_train_between": geometry["hungarian_mean_between"],
            "hungarian_max_to_max_train_between": geometry["hungarian_max_between"],
            "hungarian_p90_to_max_train_between": geometry["hungarian_p90_between"],
            "hungarian_mean_to_max_train_within": geometry["hungarian_mean_within"],
            "hausdorff_to_max_train": geometry["hausdorff"],
            "hausdorff_to_max_train_between": geometry["hausdorff_between"],
            "symmetric_average_nn_to_max_train": geometry["symmetric_average_nn"],
            "symmetric_average_nn_to_max_train_between": geometry[
                "symmetric_average_nn_between"
            ],
            "assignment_agreement_to_max_train": comparison["assignment_agreement"],
            "assignment_ari_to_max_train": comparison["assignment_ari"],
            "cluster_jaccard_to_max_train_mean": comparison["cluster_jaccard_mean"],
            "cluster_jaccard_to_max_train_min": comparison["cluster_jaccard_min"],
        }
    )


def _append_independent_partner_metrics(
    group: GroupData,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["resampling_scheme"]), int(row["n_children"]), int(row["k"]))].append(row)
    ambiguity_threshold = (
        float(config["selection_thresholds"]["ambiguity"]["margin_between_scale_fraction"])
        * group.between_child_scale
    )
    all_children = group.children
    for subset in grouped.values():
        subset.sort(key=lambda item: int(item["repetition"]))
        if len(subset) < 2:
            for row in subset:
                row.update(
                    {
                        "independent_partner_repetition": np.nan,
                        "independent_hungarian_mean": np.nan,
                        "independent_hungarian_mean_between": np.nan,
                        "independent_hungarian_mean_within": np.nan,
                        "independent_hausdorff_between": np.nan,
                        "independent_symmetric_average_nn_between": np.nan,
                        "independent_assignment_agreement": np.nan,
                        "independent_assignment_ari": np.nan,
                        "independent_cluster_jaccard_mean": np.nan,
                    }
                )
            continue
        for position, row in enumerate(subset):
            partner = subset[(position + 1) % len(subset)]
            geometry = set_geometry(
                group.distances,
                row["_template_indices"],
                partner["_template_indices"],
                group.between_child_scale,
                group.within_child_scale,
            )
            left_eval = evaluate_template_set(
                group, all_children, row["_template_indices"], ambiguity_threshold
            )
            right_eval = evaluate_template_set(
                group, all_children, partner["_template_indices"], ambiguity_threshold
            )
            comparison = assignment_comparison(
                left_eval["assignments"],
                right_eval["assignments"],
                geometry["mapping"],
                int(row["k"]),
            )
            row.update(
                {
                    "independent_partner_repetition": int(partner["repetition"]),
                    "independent_hungarian_mean": geometry["hungarian_mean"],
                    "independent_hungarian_mean_between": geometry["hungarian_mean_between"],
                    "independent_hungarian_mean_within": geometry["hungarian_mean_within"],
                    "independent_hausdorff_between": geometry["hausdorff_between"],
                    "independent_symmetric_average_nn_between": geometry[
                        "symmetric_average_nn_between"
                    ],
                    "independent_assignment_agreement": comparison["assignment_agreement"],
                    "independent_assignment_ari": comparison["assignment_ari"],
                    "independent_cluster_jaccard_mean": comparison["cluster_jaccard_mean"],
                }
            )


def _append_paired_improvements(rows: list[dict[str, Any]]) -> None:
    grouped_n: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    grouped_k: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_n[(str(row["resampling_scheme"]), int(row["repetition"]), int(row["k"]))].append(row)
        grouped_k[(str(row["resampling_scheme"]), int(row["repetition"]), int(row["n_children"]))].append(row)
    for subset in grouped_k.values():
        subset.sort(key=lambda item: int(item["k"]))
        k1_loss = float(subset[0]["holdout_child_loss_mean"])
        previous_loss: float | None = None
        for row in subset:
            loss = float(row["holdout_child_loss_mean"])
            row["relative_holdout_loss_reduction_vs_k1"] = (
                float((k1_loss - loss) / k1_loss) if k1_loss > 0 else 0.0
            )
            row["incremental_relative_gain_from_previous_k"] = (
                float((previous_loss - loss) / previous_loss)
                if previous_loss is not None and previous_loss > 0
                else np.nan
            )
            previous_loss = loss
    for subset in grouped_n.values():
        subset.sort(key=lambda item: int(item["n_children"]))
        previous: dict[str, Any] | None = None
        for row in subset:
            if previous is None:
                row["improvement_per_added_child"] = np.nan
                row["improvement_per_added_child_normalized"] = np.nan
            else:
                delta_n = int(row["n_children"]) - int(previous["n_children"])
                improvement = (
                    float(previous["holdout_child_loss_mean"])
                    - float(row["holdout_child_loss_mean"])
                ) / delta_n
                row["improvement_per_added_child"] = improvement
                row["improvement_per_added_child_normalized"] = normalized(
                    improvement, float(row["between_child_distance_scale"])
                )
            previous = row


def _full_solution_rows(
    group: GroupData, config: dict[str, Any], master_seed: int
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    solutions: dict[int, dict[str, Any]] = {}
    children = group.children
    for k in config["templates"]["k_values"]:
        if int(k) > len(children):
            continue
        solution = solve_training_set(
            group,
            children,
            int(k),
            derive_rng(master_seed, group.group_index, 900, int(k)),
            config,
        )
        solutions[int(k)] = solution
        evaluation = solution["train_evaluation"]
        support = solution["support"]
        rows.append(
            {
                "group_index": group.group_index,
                "action_code": group.action_code,
                "age_bin_12m_label": group.age_bin,
                "solution_scope": "full_group_descriptive",
                "n_children": len(children),
                "n_clips": len(group.segment_ids),
                "k": int(k),
                "selected_template_ids": PIPE.join(solution["selected_ids"]),
                "selected_template_children": PIPE.join(solution["selected_children"]),
                "objective_primary": float(solution["objective"]),
                "child_loss_mean": evaluation["mean_primary"],
                "child_loss_median": evaluation["median_primary"],
                "child_loss_p90": evaluation["p90_primary"],
                "child_loss_secondary": evaluation["mean_secondary"],
                "assigned_children_counts": PIPE.join(map(str, support["counts"].tolist())),
                "minimum_support_required": int(solution["support_minimum"]),
                "support_ok": bool(support["support_ok"]),
                "between_child_distance_scale": group.between_child_scale,
                "within_child_repeat_distance_scale": group.within_child_scale,
            }
        )
    return rows, solutions


def _assignment_stability_outputs(
    group: GroupData,
    rows: list[dict[str, Any]],
    full_solutions: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_max_rows = [
        row
        for row in rows
        if row["resampling_scheme"] == "primary_h3" and row["is_max_train_reference"]
    ]
    child_rows: list[dict[str, Any]] = []
    coassignment_rows: list[dict[str, Any]] = []
    all_children = group.children
    ambiguity_threshold = (
        float(config["selection_thresholds"]["ambiguity"]["margin_between_scale_fraction"])
        * group.between_child_scale
    )
    for k, full_solution in full_solutions.items():
        subset = [row for row in primary_max_rows if int(row["k"]) == int(k)]
        if not subset:
            continue
        full_indices = full_solution["selected_matrix_indices"]
        assignment_counts = np.zeros((len(all_children), k), dtype=int)
        ambiguity_counts = np.zeros(len(all_children), dtype=int)
        coassign = np.zeros((len(all_children), len(all_children)), dtype=int)
        for row in subset:
            evaluation = evaluate_template_set(
                group, all_children, row["_template_indices"], ambiguity_threshold
            )
            geometry = set_geometry(
                group.distances,
                row["_template_indices"],
                full_indices,
                group.between_child_scale,
                group.within_child_scale,
            )
            mapped = geometry["mapping"][evaluation["assignments"]]
            for child_position, label in enumerate(mapped):
                assignment_counts[child_position, int(label)] += 1
            finite = np.isfinite(evaluation["margins"])
            ambiguity_counts += (finite & (evaluation["margins"] <= ambiguity_threshold)).astype(int)
            coassign += (mapped[:, None] == mapped[None, :]).astype(int)
        repetitions = len(subset)
        probabilities = assignment_counts / repetitions
        for child_position, child in enumerate(all_children):
            positive = probabilities[child_position][probabilities[child_position] > 0]
            entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
            normalized_entropy = entropy / math.log(k) if k > 1 else 0.0
            child_rows.append(
                {
                    "group_index": group.group_index,
                    "action_code": group.action_code,
                    "age_bin_12m_label": group.age_bin,
                    "k": k,
                    "excel_children": child,
                    "assignment_probabilities_to_full_solution": PIPE.join(
                        f"{value:.12g}" for value in probabilities[child_position]
                    ),
                    "modal_pattern": int(np.argmax(probabilities[child_position]) + 1),
                    "assignment_entropy": entropy,
                    "normalized_assignment_entropy": normalized_entropy,
                    "ambiguous_assignment_frequency": float(
                        ambiguity_counts[child_position] / repetitions
                    ),
                    "n_external_repetitions": repetitions,
                }
            )
        for left in range(len(all_children)):
            for right in range(left, len(all_children)):
                coassignment_rows.append(
                    {
                        "group_index": group.group_index,
                        "action_code": group.action_code,
                        "age_bin_12m_label": group.age_bin,
                        "k": k,
                        "child_a": all_children[left],
                        "child_b": all_children[right],
                        "coassignment_frequency": float(coassign[left, right] / repetitions),
                        "n_external_repetitions": repetitions,
                    }
                )
    return pd.DataFrame(child_rows), pd.DataFrame(coassignment_rows)


def run_group_worker(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    config = task["config"]
    group = load_group_data(
        config,
        int(task["group_index"]),
        str(task["action_code"]),
        str(task["age_bin"]),
    )
    partial_dir = Path(task["partial_dir"])
    partial_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(task["master_seed"])
    repetitions = int(task["repetitions"])
    digest = str(task["config_digest"])
    full_rows, full_solutions = _full_solution_rows(group, config, master_seed)
    plans: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    optimization_rows: list[dict[str, Any]] = []
    base_grid = config["resampling"]["n_grid"]
    n_group = len(group.children)
    primary_holdout = int(config["resampling"]["primary_holdout_size"])
    secondary_holdout = max(
        int(config["resampling"]["secondary_holdout_minimum"]),
        int(math.ceil(float(config["resampling"]["secondary_holdout_fraction"]) * n_group)),
    )
    schemes: list[tuple[str, int, int]] = [("primary_h3", primary_holdout, 0)]
    if n_group - secondary_holdout >= int(
        config["resampling"]["secondary_minimum_train_children"]
    ):
        schemes.append(("secondary_h20", secondary_holdout, 1))
    for scheme, holdout_size, scheme_index in schemes:
        for repetition in range(repetitions):
            sampling = build_resampling(
                group.children,
                holdout_size,
                base_grid,
                derive_rng(master_seed, group.group_index, scheme_index, repetition, 100),
            )
            plans.append(
                {
                    "group_index": group.group_index,
                    "action_code": group.action_code,
                    "age_bin_12m_label": group.age_bin,
                    "resampling_scheme": scheme,
                    "holdout_size": holdout_size,
                    "repetition": repetition,
                    "holdout_children": PIPE.join(sampling["holdout"]),
                    "training_permutation": PIPE.join(sampling["training_order"]),
                    "maximum_train_children": sampling["maximum_train"],
                    "seed_components": PIPE.join(
                        map(str, (master_seed, group.group_index, scheme_index, repetition, 100))
                    ),
                }
            )
            repetition_rows: list[dict[str, Any]] = []
            for n_children in sampling["n_values"]:
                training_children = sampling["nested"][n_children]
                nested_rows.append(
                    {
                        "group_index": group.group_index,
                        "action_code": group.action_code,
                        "age_bin_12m_label": group.age_bin,
                        "resampling_scheme": scheme,
                        "repetition": repetition,
                        "n_children": n_children,
                        "is_max_train_reference": n_children == sampling["maximum_train"],
                        "training_children": PIPE.join(training_children),
                        "holdout_children": PIPE.join(sampling["holdout"]),
                    }
                )
                for k in config["templates"]["k_values"]:
                    if int(k) > n_children:
                        continue
                    solution = solve_training_set(
                        group,
                        training_children,
                        int(k),
                        derive_rng(
                            master_seed,
                            group.group_index,
                            scheme_index,
                            repetition,
                            200,
                            n_children,
                            int(k),
                        ),
                        config,
                    )
                    repetition_rows.append(
                        _base_solution_record(
                            group,
                            scheme,
                            holdout_size,
                            repetition,
                            n_children,
                            sampling["maximum_train"],
                            int(k),
                            sampling["holdout"],
                            training_children,
                            solution,
                        )
                    )
            references = {
                int(row["k"]): row
                for row in repetition_rows
                if bool(row["is_max_train_reference"])
            }
            for row in repetition_rows:
                _append_reference_metrics(
                    group,
                    row,
                    references[int(row["k"])],
                    sampling["holdout"],
                    config,
                )
            optimization_rows.extend(repetition_rows)
    _append_paired_improvements(optimization_rows)
    _append_independent_partner_metrics(group, optimization_rows, config)
    child_stability, coassignment = _assignment_stability_outputs(
        group, optimization_rows, full_solutions, config
    )
    for row in optimization_rows:
        row.pop("_template_indices", None)
    optimization_frame = pd.DataFrame(optimization_rows)
    atomic_parquet(optimization_frame, partial_dir / "optimization_results.parquet")
    atomic_csv(pd.DataFrame(plans), partial_dir / "external_resampling_plan.csv")
    atomic_csv(pd.DataFrame(nested_rows), partial_dir / "nested_training_sets.csv")
    atomic_csv(pd.DataFrame(full_rows), partial_dir / "full_group_solutions.csv")
    atomic_csv(child_stability, partial_dir / "assignment_child_stability.csv")
    atomic_csv(coassignment, partial_dir / "coassignment_matrix.csv")
    elapsed = time.perf_counter() - started
    marker = {
        "status": "complete",
        "completed_at": utc_now(),
        "config_digest": digest,
        "repetitions": repetitions,
        "action_code": group.action_code,
        "age_bin_12m_label": group.age_bin,
        "n_group_children": n_group,
        "n_group_clips": len(group.segment_ids),
        "resampling_schemes": [scheme for scheme, _, _ in schemes],
        "optimization_count": len(optimization_frame),
        "elapsed_seconds": elapsed,
    }
    atomic_json(marker, partial_dir / "complete.json")
    return marker


def group_catalog(config: dict[str, Any]) -> list[tuple[int, str, str]]:
    manifest = pd.read_csv(
        resolve_path(config, "train_manifest"),
        usecols=["action_code", "age_bin_12m_label"],
    )
    groups = manifest.drop_duplicates().copy()
    age_rank = {value: index for index, value in enumerate(AGE_ORDER)}
    rows = sorted(
        [(str(row.action_code), str(row.age_bin_12m_label)) for row in groups.itertuples()],
        key=lambda value: (int(value[0][1:]), age_rank.get(value[1], 99)),
    )
    return [(index, action, age) for index, (action, age) in enumerate(rows)]


def preflight(config: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any = None) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )
        if not passed:
            raise RuntimeError(f"Preflight failed: {name}; observed={observed}; expected={expected}")

    observed_hashes = input_hashes(config)
    expected_hashes = config["expected"]["sha256"]
    for name, expected in expected_hashes.items():
        check(f"sha256_{name}", observed_hashes.get(name) == expected, observed_hashes.get(name), expected)
    manifest = pd.read_csv(
        resolve_path(config, "train_manifest"), dtype={"segment_id": str}
    )
    manifest["excel_children"] = manifest["excel_children"].map(canonical_child)
    validation = pd.read_csv(
        resolve_path(config, "validation_manifest"), dtype={"segment_id": str}
    )
    validation["excel_children"] = validation["excel_children"].map(canonical_child)
    check(
        "train_clip_count",
        len(manifest) == int(config["expected"]["train_clips"]),
        len(manifest),
        int(config["expected"]["train_clips"]),
    )
    check(
        "train_unique_children",
        manifest["excel_children"].nunique() == int(config["expected"]["train_unique_children"]),
        int(manifest["excel_children"].nunique()),
        int(config["expected"]["train_unique_children"]),
    )
    check("train_segment_ids_unique", manifest["segment_id"].is_unique, bool(manifest["segment_id"].is_unique), True)
    check(
        "train_split_label",
        set(manifest["split"].astype(str)) == {"train"},
        sorted(set(manifest["split"].astype(str))),
        ["train"],
    )
    check("validation_split_label", set(validation["split"].astype(str)) == {"validation"}, sorted(set(validation["split"].astype(str))), ["validation"])
    segment_overlap = sorted(set(manifest["segment_id"]) & set(validation["segment_id"]))
    child_overlap = sorted(set(manifest["excel_children"]) & set(validation["excel_children"]))
    check("validation_segment_disjoint", not segment_overlap, segment_overlap[:10], [])
    check("validation_child_disjoint", not child_overlap, child_overlap[:10], [])
    catalog = group_catalog(config)
    check(
        "group_count",
        len(catalog) == int(config["expected"]["groups"]),
        len(catalog),
        int(config["expected"]["groups"]),
    )
    check(
        "action_count",
        manifest["action_code"].nunique() == int(config["expected"]["actions"]),
        int(manifest["action_code"].nunique()),
        int(config["expected"]["actions"]),
    )
    check(
        "age_bins",
        sorted(manifest["age_bin_12m_label"].astype(str).unique()) == sorted(config["expected"]["age_bins"]),
        sorted(manifest["age_bin_12m_label"].astype(str).unique()),
        sorted(config["expected"]["age_bins"]),
    )
    pairwise_index = pd.read_csv(resolve_path(config, "pairwise_index"))
    medoids = pd.read_csv(
        resolve_path(config, "official_medoids"), dtype={"medoid_segment_id": str}
    )
    topk = pd.read_csv(resolve_path(config, "official_topk"), dtype={"segment_id": str})
    check("pairwise_index_group_count", len(pairwise_index) == len(catalog), len(pairwise_index), len(catalog))
    check(
        "pairwise_total_pairs",
        int(pairwise_index["n_pairs"].sum()) == int(config["expected"]["pairwise_pairs"]),
        int(pairwise_index["n_pairs"].sum()),
        int(config["expected"]["pairwise_pairs"]),
    )
    medoid_reproductions: list[dict[str, Any]] = []
    pairwise_ids: set[str] = set()
    for group_index, action_code, age_bin in catalog:
        group = load_group_data(
            config,
            group_index,
            action_code,
            age_bin,
            manifest=manifest,
            pairwise_index=pairwise_index,
            medoids=medoids,
            topk=topk,
        )
        n = len(group.segment_ids)
        means = group.distances.sum(axis=1) / (n - 1)
        computed = group.segment_ids[int(np.argmin(means))]
        index_row = pairwise_index[
            pairwise_index["action_code"].astype(str).eq(action_code)
            & pairwise_index["age_bin_12m_label"].astype(str).eq(age_bin)
        ].iloc[0]
        expected_pairs = n * (n - 1) // 2
        if int(index_row["n_clips"]) != n or int(index_row["n_pairs"]) != expected_pairs:
            raise RuntimeError(f"Index/group mismatch for {group.key}")
        if computed != group.official_medoid:
            raise RuntimeError(
                f"Official medoid mismatch for {group.key}: {computed} != {group.official_medoid}"
            )
        if not np.isfinite(group.distances).all() or np.any(group.distances < 0):
            raise RuntimeError(f"Invalid numeric pairwise matrix for {group.key}")
        pairwise_ids.update(group.segment_ids)
        medoid_reproductions.append(
            {
                "action_code": action_code,
                "age_bin_12m_label": age_bin,
                "n_clips": n,
                "n_children": len(group.children),
                "expected_pairs": expected_pairs,
                "official_medoid": group.official_medoid,
                "computed_medoid": computed,
                "exact_match": True,
                "matrix_symmetric": bool(np.allclose(group.distances, group.distances.T)),
                "matrix_finite": bool(np.isfinite(group.distances).all()),
                "diagonal_zero": bool(np.allclose(np.diag(group.distances), 0.0)),
            }
        )
    check("pairwise_manifest_id_match", pairwise_ids == set(manifest["segment_id"]), len(pairwise_ids), len(manifest))
    check("validation_absent_from_pairwise", not (pairwise_ids & set(validation["segment_id"])), 0, 0)
    check("all_official_medoids_reproduced", len(medoid_reproductions) == len(catalog), len(medoid_reproductions), len(catalog))
    report = {
        "status": "passed",
        "started_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "input_hashes": observed_hashes,
        "checks": checks,
        "medoid_reproductions": medoid_reproductions,
        "validation_used_for_analysis": False,
        "dtw_recomputed": False,
    }
    if output_path is not None:
        atomic_json(report, output_path)
    return report


def summarize_numeric(values: pd.Series, prefix: str) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {f"{prefix}_{suffix}": np.nan for suffix in ["mean", "std", "se", "p05", "p25", "median", "p75", "p90", "p95"]}
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_std": std,
        f"{prefix}_se": std / math.sqrt(len(clean)),
        f"{prefix}_p05": float(clean.quantile(0.05)),
        f"{prefix}_p25": float(clean.quantile(0.25)),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_p75": float(clean.quantile(0.75)),
        f"{prefix}_p90": float(clean.quantile(0.90)),
        f"{prefix}_p95": float(clean.quantile(0.95)),
    }


def build_sample_size_curves(optimization: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "holdout_child_loss_mean": "holdout_loss",
        "holdout_child_loss_median": "holdout_child_median",
        "holdout_child_loss_p90": "holdout_child_p90",
        "holdout_child_loss_secondary": "holdout_secondary",
        "normalized_holdout_excess_loss": "normalized_excess",
        "hungarian_mean_to_max_train_between": "hungarian_between",
        "hungarian_mean_to_max_train_within": "hungarian_within",
        "assignment_agreement_to_max_train": "assignment_agreement",
        "assignment_ari_to_max_train": "assignment_ari",
        "cluster_jaccard_to_max_train_mean": "cluster_jaccard",
        "independent_hungarian_mean_between": "independent_hungarian_between",
        "independent_assignment_ari": "independent_assignment_ari",
        "independent_assignment_agreement": "independent_assignment_agreement",
        "relative_holdout_loss_reduction_vs_k1": "relative_reduction_vs_k1",
        "incremental_relative_gain_from_previous_k": "incremental_relative_gain",
        "improvement_per_added_child_normalized": "marginal_improvement",
        "holdout_ambiguous_children_fraction": "ambiguous_fraction",
        "holdout_repeat_consistency": "repeat_consistency",
    }
    rows: list[dict[str, Any]] = []
    keys = [
        "group_index",
        "action_code",
        "age_bin_12m_label",
        "resampling_scheme",
        "holdout_size",
        "n_children",
        "maximum_train_children",
        "k",
        "n_group_children",
        "n_group_clips",
        "between_child_distance_scale",
        "within_child_repeat_distance_scale",
    ]
    for key, subset in optimization.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key, strict=True))
        row["external_repetitions"] = int(len(subset))
        row["support_rate"] = float(subset["support_ok"].mean())
        row["exact_set_match_max_train_rate"] = float(subset["exact_set_match_max_train"].mean())
        row["jaccard_max_train_mean"] = float(subset["jaccard_max_train"].mean())
        row["functional_overlap_max_train_mean"] = float(subset["functional_overlap_max_train"].mean())
        row["official_topk_overlap_mean"] = float(subset["official_topk_overlap_fraction"].mean())
        selected_sets = subset["selected_template_ids"].map(
            lambda value: PIPE.join(sorted(split_pipe(value)))
        )
        set_counts = selected_sets.value_counts()
        row["exact_set_mode_frequency"] = float(set_counts.iloc[0] / len(subset))
        row["unique_template_set_count"] = int(set_counts.size)
        clip_counts: Counter[str] = Counter(
            segment for value in subset["selected_template_ids"] for segment in split_pipe(value)
        )
        probabilities = np.asarray(list(clip_counts.values()), dtype=float) / (
            len(subset) * int(row["k"])
        )
        row["medoid_selection_entropy"] = float(-np.sum(probabilities * np.log(probabilities)))
        row["unique_selected_template_count"] = int(len(clip_counts))
        for source, prefix in metrics.items():
            row.update(summarize_numeric(subset[source], prefix))
        row["holdout_loss_mean_normalized"] = normalized(
            row["holdout_loss_mean"], float(row["between_child_distance_scale"])
        )
        marginal_values = pd.to_numeric(
            subset["improvement_per_added_child_normalized"], errors="coerce"
        ).dropna()
        row["marginal_improvement_abs_normalized"] = (
            float(marginal_values.abs().median()) if not marginal_values.empty else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["group_index", "resampling_scheme", "k", "n_children"]
    ).reset_index(drop=True)


def build_selection_frequencies(
    config: dict[str, Any], optimization: pd.DataFrame
) -> pd.DataFrame:
    manifest = pd.read_csv(resolve_path(config, "train_manifest"), dtype={"segment_id": str})
    manifest["excel_children"] = manifest["excel_children"].map(canonical_child)
    index = pd.read_csv(resolve_path(config, "pairwise_index"))
    medoids = pd.read_csv(resolve_path(config, "official_medoids"), dtype={"medoid_segment_id": str})
    topk = pd.read_csv(resolve_path(config, "official_topk"), dtype={"segment_id": str})
    output: list[dict[str, Any]] = []
    primary = optimization[optimization["resampling_scheme"].eq("primary_h3")]
    for group_index, action_code, age_bin in group_catalog(config):
        group = load_group_data(config, group_index, action_code, age_bin, manifest, index, medoids, topk)
        subset_group = primary[
            primary["action_code"].eq(action_code)
            & primary["age_bin_12m_label"].eq(age_bin)
        ]
        identity_config = config["selection_thresholds"]["identity"]
        threshold = (
            group.within_child_scale
            * float(identity_config["functional_neighborhood_within_scale_multiplier"])
            if np.isfinite(group.within_child_scale) and group.within_child_scale > 0
            else group.between_child_scale
            * float(identity_config["functional_neighborhood_between_scale_fallback_multiplier"])
        )
        for (n_children, k), subset in subset_group.groupby(["n_children", "k"], sort=False):
            selected_lists = [split_pipe(value) for value in subset["selected_template_ids"]]
            exact_counts = Counter(itertools.chain.from_iterable(selected_lists))
            child_any_counts: Counter[str] = Counter()
            functional_counts = np.zeros(len(group.segment_ids), dtype=int)
            for selected in selected_lists:
                selected_indices = [group.segment_to_index[value] for value in selected]
                distances_to_set = np.min(group.distances[:, selected_indices], axis=1)
                functional_counts += (distances_to_set <= threshold).astype(int)
                child_any_counts.update({group.segment_to_child[value] for value in selected})
            repetitions = len(subset)
            for segment_position, segment_id in enumerate(group.segment_ids):
                child = group.segment_to_child[segment_id]
                output.append(
                    {
                        "group_index": group_index,
                        "action_code": action_code,
                        "age_bin_12m_label": age_bin,
                        "n_children": int(n_children),
                        "k": int(k),
                        "segment_id": segment_id,
                        "excel_children": child,
                        "exact_selection_count": int(exact_counts[segment_id]),
                        "exact_selection_frequency": float(exact_counts[segment_id] / repetitions),
                        "functional_selection_frequency": float(
                            functional_counts[segment_position] / repetitions
                        ),
                        "origin_child_any_template_frequency": float(
                            child_any_counts[child] / repetitions
                        ),
                        "functional_neighborhood_threshold": float(threshold),
                        "external_repetitions": repetitions,
                    }
                )
    return pd.DataFrame(output)


def deterministic_mode(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "<missing>"
    counts = clean.value_counts()
    maximum = counts.max()
    return sorted(counts[counts == maximum].index.astype(str))[0]


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]
    if len(values) < 3 or np.unique(labels).size < 2:
        return 0.0
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 0:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group_values = values[labels == label]
        between += len(group_values) * float((group_values.mean() - values.mean()) ** 2)
    return float(between / total)


def permutation_association(
    values: np.ndarray,
    labels: np.ndarray,
    variable_type: str,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, str]:
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size < 2:
        return 0.0, 1.0, "not_applicable_single_template"
    if variable_type == "categorical":
        encoded = pd.factorize(pd.Series(values).fillna("<missing>").astype(str), sort=True)[0]
        effect = float(normalized_mutual_info_score(labels, encoded))
        null = [
            float(normalized_mutual_info_score(rng.permutation(labels), encoded))
            for _ in range(permutations)
        ]
        metric = "normalized_mutual_information"
    else:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(numeric)
        if int(valid.sum()) < 3:
            return float("nan"), float("nan"), "insufficient_nonmissing_values"
        effect = eta_squared(numeric[valid], labels[valid])
        null = [eta_squared(numeric[valid], rng.permutation(labels[valid])) for _ in range(permutations)]
        metric = "eta_squared"
    p_value = float((1 + np.count_nonzero(np.asarray(null) >= effect)) / (permutations + 1))
    return effect, p_value, metric


def child_covariate_frame(group: GroupData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    categorical = [
        "institution_id",
        "excel_city",
        "camera_id",
        "pose2d_quality_label",
        "excel_sex",
    ]
    continuous = ["duration_seconds", "age_months_final_excel"]
    for child, subset in group.manifest.groupby("excel_children", sort=True):
        row: dict[str, Any] = {"excel_children": str(child), "repeat_count": int(len(subset))}
        for column in categorical:
            row[column] = deterministic_mode(subset[column]) if column in subset else "<unavailable>"
        for column in continuous:
            row[column] = float(pd.to_numeric(subset[column], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("excel_children").reset_index(drop=True)


def build_confounder_analysis(
    config: dict[str, Any], full_solutions: pd.DataFrame
) -> pd.DataFrame:
    manifest = pd.read_csv(resolve_path(config, "train_manifest"), dtype={"segment_id": str})
    manifest["excel_children"] = manifest["excel_children"].map(canonical_child)
    index = pd.read_csv(resolve_path(config, "pairwise_index"))
    medoids = pd.read_csv(resolve_path(config, "official_medoids"), dtype={"medoid_segment_id": str})
    topk = pd.read_csv(resolve_path(config, "official_topk"), dtype={"segment_id": str})
    thresholds = config["selection_thresholds"]["confounders"]
    permutations = int(thresholds["permutations"])
    effect_threshold = float(thresholds["dominant_effect_threshold"])
    significance = float(thresholds["significance_threshold"])
    variables = [
        ("institution_id", "categorical", "capture_or_institution", True),
        ("excel_city", "categorical", "capture_or_institution", True),
        ("camera_id", "categorical", "capture_or_institution", True),
        ("pose2d_quality_label", "categorical", "pose_or_segmentation", True),
        ("duration_seconds", "continuous", "capture_or_institution", True),
        ("repeat_count", "continuous", "sampling_structure", False),
        ("excel_sex", "categorical", "biological_covariate_exploratory", False),
        ("age_months_final_excel", "continuous", "within_window_age_exploratory", False),
    ]
    unavailable = [
        ("frame_count", "continuous", "not_available_in_authorized_inputs"),
        ("missing_pose_percentage", "continuous", "not_available_in_authorized_inputs"),
        ("capture_protocol", "categorical", "not_available_as_distinct_field"),
        ("movement_direction", "categorical", "not_available_in_authorized_inputs"),
        ("laterality", "categorical", "not_available_in_authorized_inputs"),
    ]
    output: list[dict[str, Any]] = []
    for group_index, action_code, age_bin in group_catalog(config):
        group = load_group_data(config, group_index, action_code, age_bin, manifest, index, medoids, topk)
        covariates = child_covariate_frame(group)
        if covariates["excel_children"].tolist() != group.children:
            raise RuntimeError(f"Child covariate order mismatch for {group.key}")
        subset_solutions = full_solutions[
            full_solutions["action_code"].eq(action_code)
            & full_solutions["age_bin_12m_label"].eq(age_bin)
        ]
        for solution in subset_solutions.itertuples(index=False):
            k = int(solution.k)
            template_indices = [
                group.segment_to_index[value] for value in split_pipe(solution.selected_template_ids)
            ]
            assignments = evaluate_template_set(group, group.children, template_indices)["assignments"]
            for variable_index, (variable, variable_type, domain, can_reject) in enumerate(variables):
                effect, p_value, metric = permutation_association(
                    covariates[variable].to_numpy(),
                    assignments,
                    variable_type,
                    permutations,
                    derive_rng(
                        int(config["resampling"]["master_seed"]),
                        group_index,
                        k,
                        variable_index,
                        700,
                    ),
                )
                dominant = bool(
                    can_reject
                    and np.isfinite(effect)
                    and np.isfinite(p_value)
                    and effect >= effect_threshold
                    and p_value <= significance
                )
                output.append(
                    {
                        "group_index": group_index,
                        "action_code": action_code,
                        "age_bin_12m_label": age_bin,
                        "k": k,
                        "variable": variable,
                        "variable_type": variable_type,
                        "technical_domain": domain,
                        "availability_status": "available",
                        "association_metric": metric,
                        "effect_size": effect,
                        "permutation_p_value": p_value,
                        "permutations": permutations,
                        "dominant_technical_confounder": dominant,
                        "eligible_to_reject_solution": can_reject,
                    }
                )
            for variable, variable_type, reason in unavailable:
                output.append(
                    {
                        "group_index": group_index,
                        "action_code": action_code,
                        "age_bin_12m_label": age_bin,
                        "k": k,
                        "variable": variable,
                        "variable_type": variable_type,
                        "technical_domain": "unavailable",
                        "availability_status": reason,
                        "association_metric": "not_computed",
                        "effect_size": np.nan,
                        "permutation_p_value": np.nan,
                        "permutations": 0,
                        "dominant_technical_confounder": False,
                        "eligible_to_reject_solution": False,
                    }
                )
    return pd.DataFrame(output)


def choose_k_for_cap(
    candidates: pd.DataFrame,
    cap: int,
    thresholds: dict[str, Any],
    dominant_k: set[int],
) -> tuple[int, str, bool]:
    work = candidates[candidates["k"] <= cap].sort_values("k").copy()
    work["base_eligible"] = (
        (work["support_rate"] >= thresholds["minimum_support_success_rate"])
        & ~work["k"].astype(int).isin(dominant_k)
    )
    base = work[work["base_eligible"]]
    if base.empty:
        return 1, "fallback_k1_no_support_eligible_solution", False
    best = base.loc[base["holdout_loss_mean"].idxmin()]
    one_se_limit = float(best["holdout_loss_mean"] + best["holdout_loss_se"])
    work["within_one_se"] = work["holdout_loss_mean"] <= one_se_limit
    strict_candidates: list[int] = []
    previous_by_k = {int(row.k): row for row in work.itertuples(index=False)}
    for row in work.itertuples(index=False):
        k = int(row.k)
        gain_threshold = (
            thresholds["minimum_incremental_relative_gain_k4_k5"]
            if k >= 4
            else thresholds["minimum_incremental_relative_gain"]
        )
        gain_ok = k == 1 or (
            np.isfinite(row.incremental_relative_gain_median)
            and row.incremental_relative_gain_median >= gain_threshold
        )
        if k > 1 and k - 1 in previous_by_k:
            previous = previous_by_k[k - 1]
            p90_worsening = (
                (row.holdout_child_p90_mean - previous.holdout_child_p90_mean)
                / max(previous.holdout_child_p90_mean, 1e-12)
            )
        else:
            p90_worsening = 0.0
        p90_ok = p90_worsening <= thresholds["maximum_p90_relative_worsening"]
        geometry_ok = (
            row.independent_hungarian_between_median
            <= thresholds["maximum_independent_hungarian_between_median"]
        )
        assignment_ok = k == 1 or (
            row.independent_assignment_ari_median
            >= thresholds["minimum_independent_assignment_ari_median"]
        )
        if (
            bool(row.base_eligible)
            and bool(row.within_one_se)
            and gain_ok
            and p90_ok
            and geometry_ok
            and assignment_ok
        ):
            strict_candidates.append(k)
    if strict_candidates:
        selected = min(strict_candidates)
        return selected, "smallest_k_within_1se_gain_support_geometry_assignments", True
    fallback = work[work["base_eligible"] & work["within_one_se"]]
    if not fallback.empty:
        return int(fallback.iloc[0]["k"]), "smallest_k_within_1se_support_fallback", False
    return int(base.iloc[0]["k"]), "smallest_support_eligible_k_fallback", False


def select_k_by_group(
    curves: pd.DataFrame,
    confounders: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_max = curves[
        curves["resampling_scheme"].eq("primary_h3")
        & curves["n_children"].eq(curves["maximum_train_children"])
    ].copy()
    thresholds = config["selection_thresholds"]["k"]
    selections: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    for (action_code, age_bin), subset in primary_max.groupby(
        ["action_code", "age_bin_12m_label"], sort=False
    ):
        subset = subset.sort_values("k")
        dominant_k = set(
            confounders[
                confounders["action_code"].eq(action_code)
                & confounders["age_bin_12m_label"].eq(age_bin)
                & confounders["dominant_technical_confounder"].eq(True)
            ]["k"].astype(int)
        )
        cap3, reason3, strict3 = choose_k_for_cap(subset, 3, thresholds, dominant_k)
        cap5, reason5, strict5 = choose_k_for_cap(subset, 5, thresholds, dominant_k)
        if cap5 > 3 and strict5:
            adaptive = cap5
            adaptive_reason = f"research_extension_above_cap3:{reason5}"
            adaptive_strict = True
        else:
            adaptive = cap3
            adaptive_reason = f"operational_cap3:{reason3}"
            adaptive_strict = strict3
        selected_row = subset[subset["k"].eq(adaptive)].iloc[0]
        prototype_labels = {
            1: "single_template_sufficient",
            2: "two_templates_preferred",
            3: "three_templates_preferred",
            4: "four_templates_preferred",
            5: "five_templates_preferred",
        }
        if not bool(selected_row["support_rate"] >= thresholds["minimum_support_success_rate"]):
            prototype_status = "insufficient_support"
        elif not adaptive_strict and adaptive > 1:
            prototype_status = "high_heterogeneity_no_stable_k"
        else:
            prototype_status = prototype_labels[adaptive]
        selections.append(
            {
                "group_index": int(selected_row["group_index"]),
                "action_code": action_code,
                "age_bin_12m_label": age_bin,
                "n_group_children": int(selected_row["n_group_children"]),
                "n_group_clips": int(selected_row["n_group_clips"]),
                "selected_k": adaptive,
                "selected_k_cap3": cap3,
                "selected_k_cap5": cap5,
                "selected_k_fixed1": 1,
                "selection_strictly_satisfied": adaptive_strict,
                "selection_reason": adaptive_reason,
                "prototype_count_status": prototype_status,
                "max_train_holdout_loss_mean": float(selected_row["holdout_loss_mean"]),
                "max_train_holdout_loss_se": float(selected_row["holdout_loss_se"]),
                "max_train_holdout_p90_mean": float(selected_row["holdout_child_p90_mean"]),
                "support_rate": float(selected_row["support_rate"]),
                "independent_hungarian_between_median": float(
                    selected_row["independent_hungarian_between_median"]
                ),
                "independent_assignment_ari_median": float(
                    selected_row["independent_assignment_ari_median"]
                ),
                "dominant_technical_confounder_for_selected_k": adaptive in dominant_k,
            }
        )
        for policy, selected in [
            ("fixed_k1", 1),
            ("cap_k3", cap3),
            ("cap_k5", cap5),
            ("adaptive_operational", adaptive),
        ]:
            policy_row = subset[subset["k"].eq(selected)].iloc[0]
            policies.append(
                {
                    "action_code": action_code,
                    "age_bin_12m_label": age_bin,
                    "policy": policy,
                    "selected_k": int(selected),
                    "holdout_loss_mean": float(policy_row["holdout_loss_mean"]),
                    "holdout_loss_se": float(policy_row["holdout_loss_se"]),
                    "holdout_child_p90_mean": float(policy_row["holdout_child_p90_mean"]),
                    "support_rate": float(policy_row["support_rate"]),
                }
            )
    return pd.DataFrame(selections), pd.DataFrame(policies)


def select_n_by_group(
    curves: pd.DataFrame, k_selection: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    thresholds = config["selection_thresholds"]["n"]
    output: list[dict[str, Any]] = []
    for selection in k_selection.itertuples(index=False):
        subset = curves[
            curves["action_code"].eq(selection.action_code)
            & curves["age_bin_12m_label"].eq(selection.age_bin_12m_label)
            & curves["resampling_scheme"].eq("primary_h3")
            & curves["k"].eq(selection.selected_k)
        ].sort_values("n_children")
        max_n = int(subset["maximum_train_children"].iloc[0])
        n_equivalence = select_persistent_n(subset, max_n, thresholds)
        n_one_se = one_se_n(subset, max_n)
        nontrivial = subset[subset["n_children"] < max_n]
        if len(nontrivial) < int(thresholds["required_later_nontrivial_sizes"]) + 1:
            sample_status = "insufficient_children_for_stability_estimation"
            recommended: int | None = None
        elif float(nontrivial["support_rate"].max()) < thresholds["minimum_support_success_rate"]:
            sample_status = "insufficient_support"
            recommended = None
        elif n_equivalence is None:
            sample_status = "not_stabilized_below_available_maximum"
            recommended = None
        else:
            recommended = max(n_equivalence, n_one_se or n_equivalence)
            sample_status = "stabilized_below_available_maximum"
        if recommended is not None:
            selected_curve = subset[subset["n_children"].eq(recommended)].iloc[0]
            if (
                selected_curve["exact_set_mode_frequency"]
                >= config["selection_thresholds"]["identity"]["exact_set_stable_rate"]
                and selected_curve["exact_set_match_max_train_rate"]
                >= config["selection_thresholds"]["identity"]["exact_set_stable_rate"]
            ):
                stability_status = "identity_and_set_stable"
            elif (
                selected_curve["hungarian_between_median"]
                <= thresholds["maximum_hungarian_between_median"]
                and selected_curve["assignment_agreement_median"]
                >= thresholds["minimum_assignment_agreement_median"]
            ):
                stability_status = "template_set_stable_identity_variable"
            else:
                stability_status = "functionally_stable_assignments_variable"
        elif sample_status == "insufficient_support":
            stability_status = "unstable"
        else:
            stability_status = "not_stabilized_below_available_maximum"
        n_lower = min(
            [value for value in [n_equivalence, n_one_se] if value is not None],
            default=None,
        )
        n_upper = recommended if recommended is not None else (
            int(nontrivial["n_children"].max()) if not nontrivial.empty else None
        )
        output.append(
            {
                "group_index": int(selection.group_index),
                "action_code": selection.action_code,
                "age_bin_12m_label": selection.age_bin_12m_label,
                "n_group_children": int(selection.n_group_children),
                "n_group_clips": int(selection.n_group_clips),
                "selected_k": int(selection.selected_k),
                "maximum_train_children": max_n,
                "n_equivalence": n_equivalence,
                "n_1se": n_one_se,
                "n_recommended": recommended,
                "n_uncertainty_lower": n_lower,
                "n_uncertainty_upper": n_upper,
                "n_adjusted_15pct_loss": (
                    int(math.ceil(recommended / 0.85)) if recommended is not None else np.nan
                ),
                "n_adjusted_20pct_loss": (
                    int(math.ceil(recommended / 0.80)) if recommended is not None else np.nan
                ),
                "sample_size_status": sample_status,
                "template_stability_status": stability_status,
            }
        )
    return pd.DataFrame(output)


def technical_qc_status_for_group(
    confounders: pd.DataFrame, action_code: str, age_bin: str, k: int
) -> str:
    subset = confounders[
        confounders["action_code"].eq(action_code)
        & confounders["age_bin_12m_label"].eq(age_bin)
        & confounders["k"].eq(k)
        & confounders["dominant_technical_confounder"].eq(True)
    ]
    if k == 1:
        return "insufficient_qc_evidence"
    if subset.empty:
        return "qc_clear"
    if subset["technical_domain"].eq("pose_or_segmentation").any():
        return "possible_pose_or_segmentation_problem"
    return "possible_capture_or_institution_confound"


def build_final_template_catalog(
    config: dict[str, Any],
    full_solutions: pd.DataFrame,
    frequencies: pd.DataFrame,
    k_selection: pd.DataFrame,
    n_selection: pd.DataFrame,
) -> pd.DataFrame:
    manifest = pd.read_csv(resolve_path(config, "train_manifest"), dtype={"segment_id": str})
    manifest["excel_children"] = manifest["excel_children"].map(canonical_child)
    index = pd.read_csv(resolve_path(config, "pairwise_index"))
    medoids = pd.read_csv(resolve_path(config, "official_medoids"), dtype={"medoid_segment_id": str})
    topk = pd.read_csv(resolve_path(config, "official_topk"), dtype={"segment_id": str})
    output: list[dict[str, Any]] = []
    selection_frame = k_selection.merge(
        n_selection[["action_code", "age_bin_12m_label", "n_recommended", "maximum_train_children"]],
        on=["action_code", "age_bin_12m_label"],
        how="left",
    )
    for selection in selection_frame.itertuples(index=False):
        group = load_group_data(
            config,
            int(selection.group_index),
            str(selection.action_code),
            str(selection.age_bin_12m_label),
            manifest,
            index,
            medoids,
            topk,
        )
        solution_row = full_solutions[
            full_solutions["action_code"].eq(selection.action_code)
            & full_solutions["age_bin_12m_label"].eq(selection.age_bin_12m_label)
            & full_solutions["k"].eq(selection.selected_k)
        ].iloc[0]
        template_ids = split_pipe(solution_row["selected_template_ids"])
        template_indices = [group.segment_to_index[value] for value in template_ids]
        evaluation = evaluate_template_set(group, group.children, template_indices)
        assignments = evaluation["assignments"]
        covariates = child_covariate_frame(group).set_index("excel_children")
        frequency_n = (
            int(selection.n_recommended)
            if pd.notna(selection.n_recommended)
            else int(selection.maximum_train_children)
        )
        frequency_subset = frequencies[
            frequencies["action_code"].eq(selection.action_code)
            & frequencies["age_bin_12m_label"].eq(selection.age_bin_12m_label)
            & frequencies["n_children"].eq(frequency_n)
            & frequencies["k"].eq(selection.selected_k)
        ].set_index("segment_id")
        preliminary: list[dict[str, Any]] = []
        for template_position, (template_id, template_index) in enumerate(
            zip(template_ids, template_indices, strict=True)
        ):
            assigned_positions = np.flatnonzero(assignments == template_position)
            assigned_children = [group.children[position] for position in assigned_positions]
            child_costs = np.asarray(
                [
                    group.distances[
                        np.ix_(group.child_indices[child], np.asarray([template_index]))
                    ].mean()
                    for child in assigned_children
                ],
                dtype=float,
            )
            institution_counts = Counter(
                covariates.loc[child, "institution_id"] for child in assigned_children
            )
            template_manifest = group.manifest[
                group.manifest["segment_id"].eq(template_id)
            ].iloc[0]
            if template_id in frequency_subset.index:
                frequency_row = frequency_subset.loc[template_id]
                exact_frequency = float(frequency_row["exact_selection_frequency"])
                functional_frequency = float(frequency_row["functional_selection_frequency"])
            else:
                exact_frequency = 0.0
                functional_frequency = 0.0
            other_indices = [value for value in template_indices if value != template_index]
            nearest_other = (
                float(np.min(group.distances[template_index, other_indices]))
                if other_indices
                else np.nan
            )
            pose_flags = {
                "pose2d_quality_label": str(template_manifest.get("pose2d_quality_label", "")),
                "target_quality_label_auto": str(template_manifest.get("target_quality_label_auto", "")),
                "risk_score": (
                    float(template_manifest.get("risk_score"))
                    if pd.notna(template_manifest.get("risk_score"))
                    else None
                ),
            }
            preliminary.append(
                {
                    "action_code": selection.action_code,
                    "age_bin_12m_label": selection.age_bin_12m_label,
                    "selected_k": int(selection.selected_k),
                    "template_segment_id": template_id,
                    "template_excel_children": group.segment_to_child[template_id],
                    "assigned_children_count": len(assigned_children),
                    "assigned_children_fraction": len(assigned_children) / len(group.children),
                    "assigned_children_ids": PIPE.join(assigned_children),
                    "mean_child_loss": float(child_costs.mean()) if child_costs.size else np.nan,
                    "median_child_loss": float(np.median(child_costs)) if child_costs.size else np.nan,
                    "p90_child_loss": float(np.quantile(child_costs, 0.9)) if child_costs.size else np.nan,
                    "bootstrap_exact_selection_frequency": exact_frequency,
                    "bootstrap_functional_selection_frequency": functional_frequency,
                    "cluster_stability": functional_frequency,
                    "centrality_within_cluster": float(child_costs.mean()) if child_costs.size else np.nan,
                    "distance_to_global_medoid": float(
                        group.distances[
                            template_index, group.segment_to_index[group.official_medoid]
                        ]
                    ),
                    "nearest_other_template_distance": nearest_other,
                    "institution_distribution": canonical_json(dict(sorted(institution_counts.items()))),
                    "pose_qc_flags": canonical_json(pose_flags),
                    "selection_reason": str(selection.selection_reason),
                    "clip_path": str(template_manifest.get("clip_path", "")),
                    "video_id": str(template_manifest.get("video_id", "")),
                    "filename": str(template_manifest.get("filename", "")),
                    "frequency_reference_n": frequency_n,
                }
            )
        preliminary.sort(
            key=lambda row: (
                -row["assigned_children_count"],
                np.inf if pd.isna(row["mean_child_loss"]) else row["mean_child_loss"],
                -row["bootstrap_functional_selection_frequency"],
                np.inf if pd.isna(row["centrality_within_cluster"]) else row["centrality_within_cluster"],
                row["distance_to_global_medoid"],
                row["template_segment_id"],
            )
        )
        for rank, row in enumerate(preliminary, start=1):
            row["template_rank"] = rank
            row["template_label"] = f"pattern_{rank}"
            row["rank_interpretation"] = (
                "principal_by_coverage_and_centrality_not_clinical_quality"
                if rank == 1
                else "additional_supported_pattern_not_clinical_quality"
            )
            output.append(row)
    columns = [
        "action_code",
        "age_bin_12m_label",
        "selected_k",
        "template_rank",
        "template_label",
        "template_segment_id",
        "template_excel_children",
        "assigned_children_count",
        "assigned_children_fraction",
        "assigned_children_ids",
        "mean_child_loss",
        "median_child_loss",
        "p90_child_loss",
        "bootstrap_exact_selection_frequency",
        "bootstrap_functional_selection_frequency",
        "cluster_stability",
        "centrality_within_cluster",
        "distance_to_global_medoid",
        "nearest_other_template_distance",
        "institution_distribution",
        "pose_qc_flags",
        "selection_reason",
        "rank_interpretation",
        "clip_path",
        "video_id",
        "filename",
        "frequency_reference_n",
    ]
    return pd.DataFrame(output)[columns]


def build_overall_sample_size_summary(n_selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_summary(scope: str, label: str, subset: pd.DataFrame) -> None:
        stabilized = pd.to_numeric(subset["n_recommended"], errors="coerce").dropna()
        row: dict[str, Any] = {
            "scope": scope,
            "scope_value": label,
            "n_groups": len(subset),
            "n_groups_stabilized": int(subset["n_recommended"].notna().sum()),
            "n_groups_not_stabilized": int(
                subset["sample_size_status"].eq("not_stabilized_below_available_maximum").sum()
            ),
            "n_groups_insufficient_support": int(
                subset["sample_size_status"].eq("insufficient_support").sum()
            ),
            "n_groups_insufficient_children": int(
                subset["sample_size_status"].eq(
                    "insufficient_children_for_stability_estimation"
                ).sum()
            ),
        }
        for name, quantile in [
            ("minimum", 0.0),
            ("median", 0.5),
            ("p75", 0.75),
            ("p80", 0.80),
            ("p90", 0.90),
            ("maximum", 1.0),
        ]:
            row[f"n_recommended_{name}"] = (
                float(stabilized.quantile(quantile)) if not stabilized.empty else np.nan
            )
        for loss_column, prefix in [
            ("n_adjusted_15pct_loss", "n_adjusted_15pct"),
            ("n_adjusted_20pct_loss", "n_adjusted_20pct"),
        ]:
            values = pd.to_numeric(subset[loss_column], errors="coerce").dropna()
            row[f"{prefix}_median"] = float(values.median()) if not values.empty else np.nan
            row[f"{prefix}_p80"] = float(values.quantile(0.8)) if not values.empty else np.nan
            row[f"{prefix}_p90"] = float(values.quantile(0.9)) if not values.empty else np.nan
            row[f"{prefix}_maximum"] = float(values.max()) if not values.empty else np.nan
        rows.append(row)

    append_summary("global", "all_groups", n_selection)
    for action, subset in n_selection.groupby("action_code", sort=True):
        append_summary("action", str(action), subset)
    for age in AGE_ORDER:
        subset = n_selection[n_selection["age_bin_12m_label"].eq(age)]
        append_summary("age", age, subset)
    return pd.DataFrame(rows)


def consolidate_partials(
    config: dict[str, Any], run_dir: Path
) -> dict[str, pd.DataFrame]:
    partial_root = run_dir / "partial"
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    file_map = {
        "optimization": "optimization_results.parquet",
        "plans": "external_resampling_plan.csv",
        "nested": "nested_training_sets.csv",
        "full_solutions": "full_group_solutions.csv",
        "assignment_child": "assignment_child_stability.csv",
        "coassignment": "coassignment_matrix.csv",
    }
    for group_index, action_code, age_bin in group_catalog(config):
        partial_dir = partial_root / f"{action_code}__{age_bin}"
        for key, filename in file_map.items():
            path = partial_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing completed partial: {path}")
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            frames[key].append(frame)
    return {key: pd.concat(values, ignore_index=True) for key, values in frames.items()}


def input_integrity_table(
    config: dict[str, Any], before: dict[str, str], after: dict[str, str]
) -> pd.DataFrame:
    path_keys = {
        "train_manifest": "train_manifest",
        "validation_manifest": "validation_manifest",
        "pairwise_index": "pairwise_index",
        "pairwise_bundle": "pairwise_directory",
        "official_medoids": "official_medoids",
        "official_topk": "official_topk",
    }
    roles = {
        "train_manifest": "authorized_analysis_input",
        "validation_manifest": "preflight_disjointness_only_not_analysis_input",
        "pairwise_index": "authorized_analysis_input",
        "pairwise_bundle": "authorized_stored_dtw_distances",
        "official_medoids": "descriptive_reference_and_preflight",
        "official_topk": "descriptive_baseline",
    }
    return pd.DataFrame(
        [
            {
                "input_name": name,
                "path": str(resolve_path(config, path_keys[name])),
                "role": roles[name],
                "sha256_before": before[name],
                "sha256_after": after[name],
                "hash_unchanged": before[name] == after[name],
                "matches_configured_expected_hash": after[name]
                == config["expected"]["sha256"][name],
            }
            for name in path_keys
        ]
    )


def postprocess_run(
    config: dict[str, Any],
    run_dir: Path,
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
) -> dict[str, Any]:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_now()}] Consolidando parciales", flush=True)
    consolidated = consolidate_partials(config, run_dir)
    optimization = consolidated["optimization"]
    atomic_parquet(optimization, tables_dir / "optimization_results.parquet")
    atomic_csv(consolidated["plans"], tables_dir / "external_resampling_plan.csv")
    atomic_csv(consolidated["nested"], tables_dir / "nested_training_sets.csv")
    atomic_csv(consolidated["full_solutions"], tables_dir / "full_group_solutions.csv")
    atomic_csv(consolidated["assignment_child"], tables_dir / "assignment_child_stability.csv")
    atomic_csv(consolidated["coassignment"], tables_dir / "coassignment_matrix.csv")
    max_train = optimization[optimization["is_max_train_reference"]].copy()
    atomic_csv(max_train, tables_dir / "max_train_solutions.csv")

    print(f"[{utc_now()}] Resumiendo curvas y frecuencias", flush=True)
    curves = build_sample_size_curves(optimization)
    atomic_csv(curves, tables_dir / "sample_size_curves.csv")
    holdout_columns = [
        column
        for column in curves.columns
        if column in [
            "group_index",
            "action_code",
            "age_bin_12m_label",
            "resampling_scheme",
            "holdout_size",
            "n_children",
            "maximum_train_children",
            "k",
            "external_repetitions",
        ]
        or column.startswith("holdout_")
        or column.startswith("normalized_excess")
        or column.startswith("relative_reduction")
        or column.startswith("incremental_relative")
    ]
    atomic_csv(curves[holdout_columns], tables_dir / "holdout_performance.csv")
    geometry_columns = [
        column
        for column in curves.columns
        if column in [
            "group_index",
            "action_code",
            "age_bin_12m_label",
            "resampling_scheme",
            "n_children",
            "maximum_train_children",
            "k",
            "external_repetitions",
        ]
        or "hungarian" in column
        or "jaccard_max_train" in column
        or "functional_overlap" in column
    ]
    atomic_csv(curves[geometry_columns], tables_dir / "template_set_geometry.csv")
    assignment_columns = [
        column
        for column in curves.columns
        if column in [
            "group_index",
            "action_code",
            "age_bin_12m_label",
            "resampling_scheme",
            "n_children",
            "maximum_train_children",
            "k",
            "external_repetitions",
        ]
        or "assignment" in column
        or "cluster_jaccard" in column
        or "ambiguous" in column
        or "repeat_consistency" in column
    ]
    atomic_csv(curves[assignment_columns], tables_dir / "assignment_stability.csv")
    frequencies = build_selection_frequencies(config, optimization)
    atomic_csv(frequencies, tables_dir / "template_selection_frequencies.csv")

    print(f"[{utc_now()}] Analizando confusores y seleccionando k/n", flush=True)
    confounders = build_confounder_analysis(config, consolidated["full_solutions"])
    atomic_csv(confounders, tables_dir / "technical_confounder_analysis.csv")
    k_selection, policy_comparison = select_k_by_group(curves, confounders, config)
    n_selection = select_n_by_group(curves, k_selection, config)
    atomic_csv(k_selection, tables_dir / "k_selection_by_group.csv")
    atomic_csv(n_selection, tables_dir / "n_selection_by_group.csv")
    atomic_csv(policy_comparison, tables_dir / "policy_comparison.csv")
    group_status = k_selection.merge(
        n_selection.drop(columns=["group_index", "n_group_children", "n_group_clips", "selected_k"]),
        on=["action_code", "age_bin_12m_label"],
        how="left",
    )
    group_status["technical_qc_status"] = group_status.apply(
        lambda row: technical_qc_status_for_group(
            confounders,
            str(row["action_code"]),
            str(row["age_bin_12m_label"]),
            int(row["selected_k"]),
        ),
        axis=1,
    )
    atomic_csv(group_status, tables_dir / "group_status_summary.csv")
    not_stabilized = group_status[
        ~group_status["sample_size_status"].eq("stabilized_below_available_maximum")
    ].copy()
    atomic_csv(not_stabilized, tables_dir / "groups_not_stabilized.csv")
    insufficient = group_status[
        group_status["sample_size_status"].isin(
            ["insufficient_support", "insufficient_children_for_stability_estimation"]
        )
    ].copy()
    atomic_csv(insufficient, tables_dir / "groups_insufficient_support.csv")
    overall = build_overall_sample_size_summary(n_selection)
    atomic_csv(overall, tables_dir / "overall_sample_size_summary.csv")

    print(f"[{utc_now()}] Construyendo catálogo final", flush=True)
    catalog = build_final_template_catalog(
        config,
        consolidated["full_solutions"],
        frequencies,
        k_selection,
        n_selection,
    )
    atomic_csv(catalog, tables_dir / "final_template_catalog.csv")
    integrity = input_integrity_table(config, before_hashes, after_hashes)
    atomic_csv(integrity, tables_dir / "input_integrity_and_hashes.csv")

    return {
        "optimization_count": int(len(optimization)),
        "max_train_solution_count": int(len(max_train)),
        "groups": int(len(group_status)),
        "selected_templates_total": int(k_selection["selected_k"].sum()),
        "k_distribution": {
            str(int(key)): int(value)
            for key, value in k_selection["selected_k"].value_counts().sort_index().items()
        },
        "sample_size_status_distribution": {
            str(key): int(value)
            for key, value in group_status["sample_size_status"].value_counts().items()
        },
        "all_input_hashes_unchanged": bool(integrity["hash_unchanged"].all()),
        "all_input_hashes_expected": bool(integrity["matches_configured_expected_hash"].all()),
    }


def run_analysis(
    config: dict[str, Any], analysis_id: str, repetitions: int, n_jobs: int, resume: bool
) -> dict[str, Any]:
    started_wall = utc_now()
    started = time.perf_counter()
    config = json.loads(json.dumps({key: value for key, value in config.items() if key != "_config_path"}))
    config["analysis_id"] = analysis_id
    config["resampling"]["repetitions"] = int(repetitions)
    config["execution"]["n_jobs"] = int(n_jobs)
    config["execution"]["resume"] = bool(resume)
    for name, value in config["execution"]["thread_limits"].items():
        os.environ[str(name)] = str(value)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    run_root = resolve_path(config, "run_root")
    run_dir = run_root / analysis_id
    ensure_run_isolation(run_dir, analysis_id, resume)
    for directory in ["partial", "tables", "logs"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    atomic_json(config, run_dir / "analysis_config.json")
    digest = config_digest(config)
    log_path = run_dir / "logs" / "run.log"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("Starting blocking preflight immediately before full execution")
    preflight_report = preflight(config, run_dir / "logs" / "preflight_report.json")
    before_hashes = preflight_report["input_hashes"]
    atomic_json(before_hashes, run_dir / "logs" / "input_hashes_before.json")
    log(f"Preflight passed ({len(preflight_report['checks'])} checks; 52 medoids reproduced)")
    catalog = group_catalog(config)
    tasks: list[dict[str, Any]] = []
    resumed_groups: list[str] = []
    for group_index, action_code, age_bin in catalog:
        partial_dir = run_dir / "partial" / f"{action_code}__{age_bin}"
        if resume and partial_is_complete(partial_dir, digest, repetitions):
            resumed_groups.append(f"{action_code}__{age_bin}")
            continue
        tasks.append(
            {
                "config": config,
                "config_digest": digest,
                "group_index": group_index,
                "action_code": action_code,
                "age_bin": age_bin,
                "partial_dir": str(partial_dir),
                "master_seed": int(config["resampling"]["master_seed"]),
                "repetitions": repetitions,
            }
        )
    log(
        f"Full run: groups={len(catalog)}, pending={len(tasks)}, resumed={len(resumed_groups)}, "
        f"repetitions={repetitions}, workers={n_jobs}"
    )
    errors: list[dict[str, str]] = []
    completed_markers: list[dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(run_group_worker, task): task for task in tasks}
            progress = tqdm(total=len(catalog), initial=len(resumed_groups), desc="52 groups", unit="group")
            for future in as_completed(futures):
                task = futures[future]
                group_key = f"{task['action_code']}__{task['age_bin']}"
                try:
                    marker = future.result()
                    completed_markers.append(marker)
                    progress.set_postfix_str(
                        f"current={group_key} errors={len(errors)} resumed={len(resumed_groups)}"
                    )
                    log(
                        f"Completed {group_key}: {marker['optimization_count']} optimizations "
                        f"in {marker['elapsed_seconds']:.1f}s"
                    )
                except Exception as error:  # pragma: no cover - exercised only on worker failure
                    errors.append(
                        {
                            "group": group_key,
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    progress.set_postfix_str(
                        f"current={group_key} errors={len(errors)} resumed={len(resumed_groups)}"
                    )
                    log(f"ERROR {group_key}: {error!r}")
                progress.update(1)
            progress.close()
    if errors:
        atomic_json(errors, run_dir / "logs" / "worker_errors.json")
        raise RuntimeError(f"Full run failed for {len(errors)} groups; see worker_errors.json")
    incomplete = [
        f"{action}__{age}"
        for _, action, age in catalog
        if not partial_is_complete(run_dir / "partial" / f"{action}__{age}", digest, repetitions)
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete partials after execution: {incomplete}")
    after_hashes = input_hashes(config)
    atomic_json(after_hashes, run_dir / "logs" / "input_hashes_after.json")
    if before_hashes != after_hashes:
        raise RuntimeError("Official input hashes changed during execution")
    log("All 52 group partials complete; input hashes unchanged; starting consolidation")
    product_summary = postprocess_run(config, run_dir, before_hashes, after_hashes)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "analysis_id": analysis_id,
        "methodological_goal": config["methodological_goal"],
        "started_at": started_wall,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "master_seed": int(config["resampling"]["master_seed"]),
        "repetitions": repetitions,
        "n_jobs": n_jobs,
        "worker_cap": int(config["execution"]["worker_cap"]),
        "groups_expected": len(catalog),
        "groups_completed": len(catalog),
        "groups_resumed": len(resumed_groups),
        "resumed_group_keys": resumed_groups,
        "worker_errors": 0,
        "preflight_status": "passed",
        "dtw_recomputed": False,
        "validation_used_for_selection_or_evaluation": False,
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        **product_summary,
    }
    atomic_json(summary, run_dir / "run_summary.json")
    log(
        f"Run complete in {elapsed:.1f}s; optimizations={summary['optimization_count']}; "
        f"templates={summary['selected_templates_total']}"
    )
    return summary
