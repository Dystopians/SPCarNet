from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np


SELECTOR_MODES = (
    "area_smallest",
    "csef_low_evidence",
    "csef_low_evidence_boundary_protected",
    "csef_adaptive_policy",
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


@dataclass(frozen=True)
class AdaptiveCompactionPolicyDecision:
    mode: str
    score_mode: str
    target_prune_fraction: float
    selected_count: int
    face_count: int
    confidence: float
    objective: float
    reason: str
    risk: dict[str, float]
    candidates: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float_signal(value: np.ndarray | None, count: int, default: float) -> np.ndarray:
    if value is None:
        return np.full((count,), float(default), dtype=np.float32)
    out = np.asarray(value)
    if not np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float32, copy=False)
    else:
        out = out.astype(np.float32, copy=False)
    out = out.reshape(-1)
    if out.shape[0] != count:
        raise ValueError(f"signal length {out.shape[0]} does not match face count {count}")
    return out


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float32, copy=False)
    finite = np.isfinite(value)
    if not finite.any():
        return np.zeros_like(value, dtype=np.float32)
    lo = float(value[finite].min())
    hi = float(value[finite].max())
    if hi <= lo:
        return np.zeros_like(value, dtype=np.float32)
    out = (value - lo) / (hi - lo)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] > 1_000_000:
        out = np.empty((faces.shape[0],), dtype=np.float32)
        chunk = 250_000
        for start in range(0, faces.shape[0], chunk):
            end = min(start + chunk, faces.shape[0])
            tri = vertices[faces[start:end]]
            out[start:end] = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
        return out.astype(np.float64, copy=False)
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


def build_score_table(signals: CompactionSignals, *, fast_large_mesh: bool = False) -> dict[str, np.ndarray]:
    vertices = np.asarray(signals.vertices, dtype=np.float64)
    faces = np.asarray(signals.faces, dtype=np.int64)
    count = faces.shape[0]
    render_probe = _as_float_signal(signals.render_contribution, count, 0.0)
    if bool(fast_large_mesh) and count > 1_000_000:
        areas = np.ones((count,), dtype=np.float64)
        area_smallness = 1.0 - _normalize(render_probe)
    else:
        areas = triangle_areas(vertices, faces)
        area_smallness = 1.0 - _normalize(areas)
    if faces.shape[0] > 500_000:
        boundary = np.zeros((faces.shape[0],), dtype=np.float64)
        redundancy = area_smallness.copy()
    else:
        boundary = boundary_face_risk(faces)
        redundancy = local_redundancy(vertices, faces)

    render = render_probe
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

    if count > 1_000_000:
        pareto = 0.5 * _normalize(area_smallness) + 0.5 * _normalize(csef_low_evidence)
    else:
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


def build_fast_large_csef_table(signals: CompactionSignals) -> dict[str, np.ndarray]:
    faces = np.asarray(signals.faces)
    count = int(faces.shape[0])
    vertices = np.asarray(signals.vertices)
    render = _as_float_signal(signals.render_contribution, count, 0.0)
    sparse = _as_float_signal(signals.sparse_support, count, 0.0)
    normal = _as_float_signal(signals.normal_support, count, 0.0)
    positive = _as_float_signal(signals.positive_surface_evidence, count, np.nan)
    if np.isnan(positive).all():
        positive = 0.34 * _normalize(render) + 0.33 * _normalize(sparse) + 0.33 * _normalize(normal)
    uncertainty = _as_float_signal(signals.uncertainty, count, 0.0)
    protected = (
        np.asarray(signals.protected_faces, dtype=bool).reshape(-1)
        if signals.protected_faces is not None
        else np.zeros((count,), dtype=bool)
    )
    positive_norm = _normalize(positive)
    low_positive = 1.0 - positive_norm
    if vertices.shape[0] > int(faces.max(initial=0)):
        areas = triangle_areas(vertices, faces)
        area_smallness = 1.0 - _normalize(areas)
    else:
        areas = np.ones((count,), dtype=np.float32)
        area_smallness = low_positive.copy()
    redundancy = area_smallness.copy()
    zeros = np.zeros((count,), dtype=np.float32)
    # Large vehicle meshes have millions of triangles. Empirically the stable
    # deletion order is geometric redundancy; render-only per-face probes are too
    # coarse to be used as a hard local importance score at this scale. Keep the
    # evidence channels in the table for adaptive fraction choice and auditing,
    # but rank faces by redundancy unless explicit protection is provided.
    csef_low_evidence = _normalize(area_smallness)
    csef_low_evidence[protected] = -np.inf
    return {
        "face_id": np.arange(count, dtype=np.int64),
        "area": areas,
        "area_smallness": area_smallness,
        "render_contribution": render,
        "sparse_support": sparse,
        "normal_support": normal,
        "local_redundancy": redundancy,
        "boundary_risk": zeros,
        "positive_surface_evidence": positive,
        "negative_free_space": zeros,
        "explanation_debt": zeros,
        "topology_cost": area_smallness,
        "uncertainty": uncertainty,
        "protected": protected.astype(np.int32),
        "score_area_smallest": area_smallness,
        "score_csef_low_evidence": csef_low_evidence,
        "score_csef_low_evidence_boundary_protected": csef_low_evidence,
        "score_pareto_area_csef": 0.5 * area_smallness + 0.5 * _normalize(csef_low_evidence),
    }


def _sample_indices(count: int, max_count: int = 500_000) -> np.ndarray:
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    stride = int(np.ceil(count / float(max_count)))
    return np.arange(0, count, stride, dtype=np.int64)


def _mean(table: dict[str, np.ndarray], key: str, ids: np.ndarray) -> float:
    if ids.shape[0] == 0:
        return 0.0
    values = np.asarray(table[key][ids], dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    return float(values[finite].mean())


def decide_adaptive_compaction_policy(
    signals: CompactionSignals,
    *,
    min_fraction: float = 0.50,
    max_fraction: float = 0.82,
    seed: int = 0,
) -> tuple[AdaptiveCompactionPolicyDecision, dict[str, np.ndarray]]:
    """Choose a scene-specific compaction fraction from CSEF signal distributions.

    The policy scores prefix candidates on a fixed fraction grid, but the selected
    row is data-driven: it rewards removable low-evidence topology and penalizes
    removal of faces with positive evidence, boundary risk, or explanation debt.
    This makes the compact-recovery entrypoint a single policy instead of a
    hand-selected prune-fraction sweep.
    """
    del seed
    face_count_hint = int(np.asarray(signals.faces).shape[0])
    if face_count_hint > 1_000_000:
        sample = _sample_indices(face_count_hint)
        render = _as_float_signal(signals.render_contribution, face_count_hint, 0.0)[sample]
        sparse = _as_float_signal(signals.sparse_support, face_count_hint, 0.0)[sample]
        normal = _as_float_signal(signals.normal_support, face_count_hint, 0.0)[sample]
        positive = _as_float_signal(signals.positive_surface_evidence, face_count_hint, np.nan)[sample]
        if np.isnan(positive).all():
            positive = 0.34 * _normalize(render) + 0.33 * _normalize(sparse) + 0.33 * _normalize(normal)
        sampled_signals = CompactionSignals(
            vertices=signals.vertices,
            faces=np.asarray(signals.faces)[sample],
            render_contribution=render,
            sparse_support=sparse,
            normal_support=normal,
            positive_surface_evidence=positive,
        )
        table = build_fast_large_csef_table(sampled_signals)
        sample_scale = float(face_count_hint / max(sample.shape[0], 1))
    else:
        table = build_score_table(signals, fast_large_mesh=True)
        sample_scale = 1.0
    face_count = face_count_hint
    eval_face_count = int(table["face_id"].shape[0])
    if face_count <= 0:
        raise ValueError("cannot select from an empty mesh")

    finite_score = np.isfinite(table["score_csef_low_evidence_boundary_protected"])
    score_mode = "csef_low_evidence_boundary_protected"
    if int(finite_score.sum()) < max(1, int(round(eval_face_count * min_fraction))):
        score_mode = "csef_low_evidence"
        finite_score = np.isfinite(table["score_csef_low_evidence"])
    scores = table[f"score_{score_mode}"]
    candidate_ids = np.flatnonzero(finite_score)
    if candidate_ids.shape[0] == 0:
        raise ValueError("adaptive CSEF policy found no finite candidate faces")
    eval_candidate_ids = candidate_ids
    eval_scale = 1.0
    if face_count > 1_000_000 and candidate_ids.shape[0] > 500_000:
        stride = int(np.ceil(candidate_ids.shape[0] / 500_000.0))
        eval_candidate_ids = candidate_ids[::stride]
        eval_scale = float(candidate_ids.shape[0] / max(eval_candidate_ids.shape[0], 1))
    min_fraction = float(np.clip(min_fraction, 0.0, 0.95))
    max_fraction = float(np.clip(max_fraction, min_fraction, 0.95))
    max_feasible_fraction = float(candidate_ids.shape[0] * sample_scale / max(face_count, 1))
    max_fraction = min(max_fraction, max_feasible_fraction)
    if max_fraction < min_fraction:
        min_fraction = max(0.01, max_fraction)

    fractions = np.unique(np.round(np.linspace(min_fraction, max_fraction, 9), 4))
    eval_all_ids = np.arange(eval_face_count, dtype=np.int64)
    global_positive = float(np.maximum(_mean(table, "positive_surface_evidence", eval_all_ids), 1e-6))
    global_debt = float(np.maximum(_mean(table, "explanation_debt", eval_all_ids), 1e-6))
    global_boundary = float(np.maximum(_mean(table, "boundary_risk", eval_all_ids), 1e-6))

    rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    best_ids: np.ndarray | None = None
    for fraction in fractions:
        target = min(int(round(candidate_ids.shape[0] * sample_scale)), _target_count(face_count, float(fraction)))
        eval_target = min(eval_candidate_ids.shape[0], max(1, int(round(target / max(sample_scale * eval_scale, 1e-6)))))
        if face_count > 1_000_000:
            local_scores = scores[eval_candidate_ids]
            part = np.argpartition(local_scores, local_scores.shape[0] - eval_target)[-eval_target:]
            ids = eval_candidate_ids[part]
        else:
            order = candidate_ids[np.argsort(scores[candidate_ids])[::-1]]
            ids = order[:target]
        if ids.shape[0] == 0:
            continue
        positive = _mean(table, "positive_surface_evidence", ids)
        debt = _mean(table, "explanation_debt", ids)
        boundary = _mean(table, "boundary_risk", ids)
        redundancy = _mean(table, "local_redundancy", ids)
        area_small = _mean(table, "area_smallness", ids)
        uncertainty = _mean(table, "uncertainty", ids)
        removable = _mean(table, f"score_{score_mode}", ids)
        positive_risk = positive / global_positive
        debt_risk = debt / global_debt
        boundary_risk_value = boundary / global_boundary
        render_only_evidence = signals.sparse_support is None and signals.normal_support is None
        positive_weight = 0.10 if render_only_evidence else 0.42
        residual_weight = max(0.0, 1.0 - positive_weight)
        risk = (
            positive_weight * positive_risk
            + residual_weight * (0.48 * debt_risk + 0.34 * boundary_risk_value + 0.18 * uncertainty)
        )
        compaction_utility = 0.75 * (1.0 - float(np.exp(-2.1 * float(fraction))))
        benefit = compaction_utility + 0.30 * redundancy + 0.25 * area_small + 0.10 * removable
        high_fraction_prior = 1.8 * max(0.0, float(fraction) - 0.72) ** 2
        low_risk_penalty = 0.22 * risk
        high_risk_penalty = 3.0 * max(0.0, risk - 0.14) ** 2
        objective = benefit - low_risk_penalty - high_risk_penalty - high_fraction_prior
        row = {
            "fraction": float(fraction),
            "target": float(target),
            "objective": float(objective),
            "benefit": float(benefit),
            "risk": float(risk),
            "positive_risk": float(positive_risk),
            "debt_risk": float(debt_risk),
            "boundary_risk": float(boundary_risk_value),
            "uncertainty": float(uncertainty),
            "redundancy": float(redundancy),
            "area_smallness": float(area_small),
            "removable_score": float(removable),
        }
        rows.append(row)
        if best_row is None or row["objective"] > best_row["objective"]:
            best_row = row
            best_ids = ids

    if best_row is None or best_ids is None:
        raise ValueError("adaptive CSEF policy could not evaluate any feasible compaction fraction")
    safe_rows = [row for row in rows if float(row["risk"]) <= 0.13]
    if safe_rows:
        knee_row = max(safe_rows, key=lambda row: (float(row["fraction"]), float(row["objective"])))
        if float(best_row["risk"]) > 0.13:
            best_row = knee_row
    near_optimal_rows = [
        row
        for row in rows
        if float(row["fraction"]) >= 0.66 and float(row["objective"]) >= float(best_row["objective"]) - 0.020
    ]
    if near_optimal_rows:
        best_row = min(near_optimal_rows, key=lambda row: (float(row["fraction"]), -float(row["objective"])))

    risk_value = float(best_row["risk"])
    confidence = float(np.clip(1.0 / (1.0 + np.exp(2.0 * (risk_value - 1.0))), 0.05, 0.99))
    reason = (
        "selected_fraction_by_csef_objective:"
        f"benefit={best_row['benefit']:.4f},risk={best_row['risk']:.4f},"
        f"score_mode={score_mode}"
    )
    decision = AdaptiveCompactionPolicyDecision(
        mode="csef_adaptive_policy",
        score_mode=score_mode,
        target_prune_fraction=float(best_row["fraction"]),
        selected_count=int(best_row["target"]),
        face_count=face_count,
        confidence=confidence,
        objective=float(best_row["objective"]),
        reason=reason,
        risk={
            "policy_risk": float(best_row["risk"]),
            "positive_evidence_risk": float(best_row["positive_risk"]),
            "explanation_debt_risk": float(best_row["debt_risk"]),
            "boundary_risk": float(best_row["boundary_risk"]),
            "uncertainty": float(best_row["uncertainty"]),
            "max_feasible_fraction": float(max_feasible_fraction),
        },
        candidates=rows,
    )
    return decision, table


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
    if mode == "csef_adaptive_policy":
        decision, table = decide_adaptive_compaction_policy(signals, seed=seed)
        if int(np.asarray(signals.faces).shape[0]) > int(table["face_id"].shape[0]):
            table = build_fast_large_csef_table(signals)
        scores = table[f"score_{decision.score_mode}"]
        finite = np.isfinite(scores)
        candidate_ids = np.flatnonzero(finite)
        target = int(decision.selected_count)
        target = min(target, int(candidate_ids.shape[0]))
        order = candidate_ids[np.argsort(scores[candidate_ids])[::-1]]
        selected = np.sort(order[:target].astype(np.int64))
        return selected, table
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
    face_count = int(table["face_id"].shape[0])
    return {
        "face_count": int(face_count),
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
    policy_decision: AdaptiveCompactionPolicyDecision | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_selection(selected, table, labels)
    selected = np.asarray(selected, dtype=np.int64)
    payload = {
        "mode": mode,
        "target_prune_fraction": float(target_prune_fraction),
        "summary": summary,
    }
    large_payload = selected.shape[0] > 1_000_000 or int(table["face_id"].shape[0]) > 1_000_000
    if large_payload:
        selected_path = out / "selected_faces.npy"
        np.save(selected_path, selected)
        payload["selected_faces_path"] = selected_path.name
        payload["selected_faces_count"] = int(selected.shape[0])
        payload["selected_faces_head"] = [int(x) for x in selected[:32].tolist()]
        payload["selected_faces_tail"] = [int(x) for x in selected[-32:].tolist()]
    else:
        payload["selected_faces"] = [int(x) for x in selected.tolist()]
    if policy_decision is not None:
        payload["adaptive_policy_decision"] = policy_decision.to_dict()
    (out / "compaction_candidates.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if large_payload:
        score_summary = {
            key: {
                "mean": float(np.asarray(value, dtype=np.float64).mean()) if np.asarray(value).size else 0.0,
                "min": float(np.asarray(value, dtype=np.float64).min()) if np.asarray(value).size else 0.0,
                "max": float(np.asarray(value, dtype=np.float64).max()) if np.asarray(value).size else 0.0,
            }
            for key, value in table.items()
            if key != "face_id" and np.asarray(value).ndim == 1 and np.issubdtype(np.asarray(value).dtype, np.number)
        }
        (out / "compaction_score_summary.json").write_text(json.dumps(score_summary, indent=2) + "\n", encoding="utf-8")
    else:
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
    if policy_decision is not None:
        lines += [
            "",
            "## Adaptive Policy",
            "",
            f"- score mode: `{policy_decision.score_mode}`",
            f"- confidence: `{policy_decision.confidence:.6f}`",
            f"- objective: `{policy_decision.objective:.6f}`",
            f"- reason: `{policy_decision.reason}`",
            f"- risk: `{json.dumps(policy_decision.risk, sort_keys=True)}`",
        ]
    for key, value in summary["label_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "compaction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
