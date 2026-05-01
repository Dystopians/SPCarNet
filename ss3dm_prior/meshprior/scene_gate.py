"""Scene evidence gates and rollback utilities for MeshPrior proposals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class ProposalGateResult:
    proposal_id: str
    proposal_type: str
    accepted: bool
    scene_evidence_passed: bool
    object_evidence_used: bool
    metrics: dict[str, float]
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "accepted": self.accepted,
            "scene_evidence_passed": self.scene_evidence_passed,
            "object_evidence_used": self.object_evidence_used,
            "metrics": self.metrics,
            "reasons": self.reasons,
        }


def _edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(u), int(v))))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def _boundary_edge_count(faces: np.ndarray) -> int:
    return sum(1 for count in _edge_counts(faces).values() if count == 1)


def _component_count(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0
    parent = list(range(len(faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owners: dict[int, int] = {}
    for fi, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for v in face:
            v = int(v)
            if v in owners:
                union(fi, owners[v])
            else:
                owners[v] = fi
    return len({find(i) for i in range(len(faces))})


def evaluate_proposal_geometry_delta(
    vertices_before: np.ndarray,
    vertices_after: np.ndarray,
) -> dict[str, float]:
    """Return a local geometry movement proxy."""
    before = np.asarray(vertices_before, dtype=np.float32)
    after = np.asarray(vertices_after, dtype=np.float32)
    n = min(len(before), len(after))
    if n == 0:
        return {"mean_matched_vertex_displacement": 0.0, "max_matched_vertex_displacement": 0.0}
    disp = np.linalg.norm(after[:n] - before[:n], axis=1)
    return {
        "mean_matched_vertex_displacement": float(disp.mean()),
        "max_matched_vertex_displacement": float(disp.max()),
    }


def evaluate_proposal_free_space_delta(
    vertices_before: np.ndarray,
    vertices_after: np.ndarray,
    free_space_violation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    if free_space_violation_fn is None:
        return {"free_space_violation_delta": 0.0}
    before = np.asarray(free_space_violation_fn(np.asarray(vertices_before, dtype=np.float32)), dtype=np.float64)
    after = np.asarray(free_space_violation_fn(np.asarray(vertices_after, dtype=np.float32)), dtype=np.float64)
    return {
        "free_space_violation_before_mean": float(before.mean()) if len(before) else 0.0,
        "free_space_violation_after_mean": float(after.mean()) if len(after) else 0.0,
        "free_space_violation_delta": float((after.mean() if len(after) else 0.0) - (before.mean() if len(before) else 0.0)),
    }


def evaluate_proposal_topology_delta(
    faces_before: np.ndarray,
    faces_after: np.ndarray,
) -> dict[str, float]:
    before_faces = np.asarray(faces_before, dtype=np.int64)
    after_faces = np.asarray(faces_after, dtype=np.int64)
    boundary_before = _boundary_edge_count(before_faces)
    boundary_after = _boundary_edge_count(after_faces)
    components_before = _component_count(before_faces)
    components_after = _component_count(after_faces)
    edge_total_after = max(len(_edge_counts(after_faces)), 1)
    return {
        "triangle_count_before": float(len(before_faces)),
        "triangle_count_after": float(len(after_faces)),
        "triangle_count_delta": float(len(after_faces) - len(before_faces)),
        "boundary_edge_count_before": float(boundary_before),
        "boundary_edge_count_after": float(boundary_after),
        "boundary_edge_delta": float(boundary_before - boundary_after),
        "hole_boundary_score_after": float(boundary_after / edge_total_after),
        "component_count_before": float(components_before),
        "component_count_after": float(components_after),
        "component_count_delta": float(components_after - components_before),
        "floater_count_delta": float(max(components_after - components_before, 0)),
    }


def accept_or_reject(
    *,
    proposal_id: str,
    proposal_type: str,
    metrics: dict[str, float],
    object_evidence: dict[str, float] | None = None,
    max_free_space_delta: float = 0.0,
    max_triangle_growth_ratio: float = 0.5,
    max_uncertainty: float = 0.75,
) -> ProposalGateResult:
    reasons: list[str] = []
    object_evidence = object_evidence or {}
    object_evidence_used = bool(object_evidence)
    triangle_before = max(metrics.get("triangle_count_before", 0.0), 1.0)
    triangle_growth_ratio = metrics.get("triangle_count_delta", 0.0) / triangle_before
    scene_improvements = [
        metrics.get("boundary_edge_delta", 0.0) > 0.0,
        metrics.get("free_space_violation_delta", 0.0) < 0.0,
        metrics.get("component_count_delta", 0.0) < 0.0,
    ]
    scene_evidence_passed = any(scene_improvements)
    hard_reject = False

    if metrics.get("free_space_violation_delta", 0.0) > max_free_space_delta:
        hard_reject = True
        reasons.append("free_space_violation_increased")
    if metrics.get("component_count_delta", 0.0) > 0.0:
        hard_reject = True
        reasons.append("component_count_increased")
    if triangle_growth_ratio > max_triangle_growth_ratio:
        hard_reject = True
        reasons.append("triangle_growth_too_large")
    if object_evidence.get("uncertainty", 0.0) > max_uncertainty:
        hard_reject = True
        reasons.append("object_uncertainty_too_high")
    if not scene_evidence_passed:
        hard_reject = True
        reasons.append("no_scene_metric_improved")

    accepted = not hard_reject
    if accepted:
        reasons.append("accepted_by_scene_evidence")
    return ProposalGateResult(
        proposal_id=proposal_id,
        proposal_type=proposal_type,
        accepted=accepted,
        scene_evidence_passed=scene_evidence_passed,
        object_evidence_used=object_evidence_used,
        metrics={k: float(v) for k, v in metrics.items()},
        reasons=reasons,
    )


def save_rollback_snapshot(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    metadata: dict | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
    return out


def restore_rollback_snapshot(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload.files else {}
    return np.asarray(payload["vertices"], dtype=np.float32), np.asarray(payload["faces"], dtype=np.int64), metadata
