from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np


SELECTOR_MODES = (
    "area_smallest",
    "csef_low_evidence",
    "csef_low_evidence_boundary_protected",
    "pareto_area_csef",
    "random_same_count",
)


@dataclass(frozen=True)
class CompactionSignals:
    vertices: np.ndarray
    faces: np.ndarray
    render_contribution: np.ndarray | None = None
    sparse_support: np.ndarray | None = None
    normal_support: np.ndarray | None = None
    positive_surface_evidence: np.ndarray | None = None
    negative_free_space: np.ndarray | None = None
    explanation_debt: np.ndarray | None = None
    topology_cost: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    protected_faces: np.ndarray | None = None
    labels: np.ndarray | None = None


def _as_float_signal(value: np.ndarray | None, count: int, default: float) -> np.ndarray:
    if value is None:
        return np.full((count,), float(default), dtype=np.float64)
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.shape[0] != count:
        raise ValueError(f"signal length {out.shape[0]} does not match face count {count}")
    return out


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(value)
    if not finite.any():
        return np.zeros_like(value, dtype=np.float64)
    lo = float(value[finite].min())
    hi = float(value[finite].max())
    if hi <= lo:
        return np.zeros_like(value, dtype=np.float64)
    out = (value - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norm, 1e-12)


def boundary_face_risk(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edge_counts: dict[tuple[int, int], int] = {}
    face_edges: list[list[tuple[int, int]]] = []
    for face in faces:
        edges = [
            tuple(sorted((int(face[0]), int(face[1])))),
            tuple(sorted((int(face[1]), int(face[2])))),
            tuple(sorted((int(face[2]), int(face[0])))),
        ]
        face_edges.append(edges)
        for edge in edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    risk = np.zeros((faces.shape[0],), dtype=np.float64)
    for idx, edges in enumerate(face_edges):
        risk[idx] = sum(1 for edge in edges if edge_counts[edge] == 1) / 3.0
    return risk


def local_redundancy(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = triangle_areas(vertices, faces)
    normals = face_normals(vertices, faces)
    faces = np.asarray(faces, dtype=np.int64)
    incident: dict[int, list[int]] = {}
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            incident.setdefault(int(vertex_id), []).append(face_id)
    redundancy = np.zeros((faces.shape[0],), dtype=np.float64)
    for face_id, face in enumerate(faces):
        neigh = sorted({j for vertex_id in face for j in incident[int(vertex_id)] if j != face_id})
        if not neigh:
            redundancy[face_id] = 0.0
            continue
        agreement = np.abs(normals[neigh] @ normals[face_id]).mean()
        redundancy[face_id] = float(agreement)
    small_area = 1.0 - _normalize(areas)
    return 0.6 * _normalize(redundancy) + 0.4 * small_area


def build_score_table(signals: CompactionSignals) -> dict[str, np.ndarray]:
    vertices = np.asarray(signals.vertices, dtype=np.float64)
    faces = np.asarray(signals.faces, dtype=np.int64)
    count = faces.shape[0]
    areas = triangle_areas(vertices, faces)
    area_smallness = 1.0 - _normalize(areas)
    if faces.shape[0] > 500_000:
        boundary = np.zeros((faces.shape[0],), dtype=np.float64)
        redundancy = area_smallness.copy()
    else:
        boundary = boundary_face_risk(faces)
        redundancy = local_redundancy(vertices, faces)

    render = _as_float_signal(signals.render_contribution, count, 0.0)
    sparse = _as_float_signal(signals.sparse_support, count, 0.0)
    normal = _as_float_signal(signals.normal_support, count, 0.0)
    positive = _as_float_signal(signals.positive_surface_evidence, count, np.nan)
    if np.isnan(positive).all():
        positive = 0.34 * _normalize(render) + 0.33 * _normalize(sparse) + 0.33 * _normalize(normal)
    negative = _as_float_signal(signals.negative_free_space, count, 0.0)
    debt = _as_float_signal(signals.explanation_debt, count, 0.0)
    topology_cost = _as_float_signal(signals.topology_cost, count, np.nan)
    if np.isnan(topology_cost).all():
        topology_cost = area_smallness
    uncertainty = _as_float_signal(signals.uncertainty, count, 0.0)
    protected = np.asarray(signals.protected_faces, dtype=bool).reshape(-1) if signals.protected_faces is not None else np.zeros((count,), dtype=bool)
    if protected.shape[0] != count:
        raise ValueError("protected_faces length does not match face count")

    csef_low_evidence = (
        1.25 * _normalize(topology_cost)
        + 1.00 * _normalize(redundancy)
        + 0.75 * _normalize(negative)
        + 0.35 * _normalize(uncertainty)
        - 1.25 * _normalize(positive)
        - 1.10 * _normalize(debt)
        - 0.65 * _normalize(boundary)
    )
    csef_boundary_protected = csef_low_evidence.copy()
    csef_boundary_protected[protected | (_normalize(debt) > 0.65)] = -np.inf

    area_rank = np.argsort(np.argsort(-area_smallness)).astype(np.float64)
    csef_rank = np.argsort(np.argsort(-csef_low_evidence)).astype(np.float64)
    pareto = -(area_rank + csef_rank)
    pareto[protected] = -np.inf

    return {
        "face_id": np.arange(count, dtype=np.int64),
        "area": areas,
        "area_smallness": area_smallness,
        "render_contribution": render,
        "sparse_support": sparse,
        "normal_support": normal,
        "local_redundancy": redundancy,
        "boundary_risk": boundary,
        "positive_surface_evidence": positive,
        "negative_free_space": negative,
        "explanation_debt": debt,
        "topology_cost": topology_cost,
        "uncertainty": uncertainty,
        "protected": protected.astype(np.int32),
        "score_area_smallest": area_smallness,
        "score_csef_low_evidence": csef_low_evidence,
        "score_csef_low_evidence_boundary_protected": csef_boundary_protected,
        "score_pareto_area_csef": pareto,
    }


def _target_count(face_count: int, target_prune_fraction: float) -> int:
    if not 0.0 <= target_prune_fraction < 1.0:
        raise ValueError("target_prune_fraction must be in [0, 1)")
    return min(face_count, max(1, int(round(face_count * target_prune_fraction))))


def select_faces(
    signals: CompactionSignals,
    mode: str,
    target_prune_fraction: float,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if mode not in SELECTOR_MODES:
        raise ValueError(f"unknown selector mode {mode!r}; expected one of {SELECTOR_MODES}")
    table = build_score_table(signals)
    count = table["face_id"].shape[0]
    target = _target_count(count, target_prune_fraction)
    if mode == "random_same_count":
        rng = np.random.default_rng(seed)
        protected = table["protected"].astype(bool)
        candidates = np.flatnonzero(~protected)
        if candidates.shape[0] < target:
            candidates = np.arange(count)
        selected = rng.choice(candidates, size=target, replace=False)
        return np.sort(selected.astype(np.int64)), table
    score_key = f"score_{mode}"
    scores = table[score_key]
    finite = np.isfinite(scores)
    candidate_ids = np.flatnonzero(finite)
    if candidate_ids.shape[0] < target:
        raise ValueError(f"mode {mode} has only {candidate_ids.shape[0]} selectable faces for target {target}")
    order = candidate_ids[np.argsort(scores[candidate_ids])[::-1]]
    return np.sort(order[:target].astype(np.int64)), table


def summarize_selection(selected: np.ndarray, table: dict[str, np.ndarray], labels: np.ndarray | None = None) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.int64)
    selected_mask = np.zeros_like(table["face_id"], dtype=bool)
    selected_mask[selected] = True
    label_counts: dict[str, int] = {}
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        for label in sorted(set(str(x) for x in labels.tolist())):
            label_counts[label] = int(np.logical_and(selected_mask, labels == label).sum())
    return {
        "face_count": int(table["face_id"].shape[0]),
        "selected_count": int(selected.shape[0]),
        "selected_fraction": float(selected.shape[0] / max(table["face_id"].shape[0], 1)),
        "selected_area_mean": float(table["area"][selected].mean()) if selected.shape[0] else 0.0,
        "selected_boundary_risk_mean": float(table["boundary_risk"][selected].mean()) if selected.shape[0] else 0.0,
        "selected_positive_evidence_mean": float(table["positive_surface_evidence"][selected].mean()) if selected.shape[0] else 0.0,
        "selected_explanation_debt_mean": float(table["explanation_debt"][selected].mean()) if selected.shape[0] else 0.0,
        "label_counts": label_counts,
    }


def write_selector_outputs(
    out_dir: str | Path,
    selected: np.ndarray,
    table: dict[str, np.ndarray],
    mode: str,
    target_prune_fraction: float,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_selection(selected, table, labels)
    payload = {
        "mode": mode,
        "target_prune_fraction": float(target_prune_fraction),
        "selected_faces": [int(x) for x in np.asarray(selected, dtype=np.int64).tolist()],
        "summary": summary,
    }
    (out / "compaction_candidates.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(out / "compaction_score_table.npz", **table)
    with (out / "compaction_summary.csv").open("w", encoding="utf-8") as fp:
        fp.write("key,value\n")
        for key, value in summary.items():
            fp.write(f"{key},{json.dumps(value)}\n")
    lines = [
        "# Compaction Candidate Report",
        "",
        f"- mode: `{mode}`",
        f"- target prune fraction: `{target_prune_fraction}`",
        f"- selected faces: `{summary['selected_count']}` / `{summary['face_count']}`",
        f"- selected area mean: `{summary['selected_area_mean']:.8f}`",
        f"- selected boundary risk mean: `{summary['selected_boundary_risk_mean']:.8f}`",
        f"- selected positive evidence mean: `{summary['selected_positive_evidence_mean']:.8f}`",
        f"- selected explanation debt mean: `{summary['selected_explanation_debt_mean']:.8f}`",
        "",
        "## Label Counts",
        "",
    ]
    for key, value in summary["label_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "compaction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
