from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .csef_builder import edge_ownership, write_ascii_ply
from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState


@dataclass(frozen=True)
class SnapProposal:
    proposal_id: str
    edit: MeshEdit
    target_type: str
    step_size: float
    max_displacement: float
    expected_error_before: float
    expected_error_after: float
    uncertainty: float
    evidence_source: str
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["edit"] = self.edit.to_dict()
        return data


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    d = -float(np.dot(normal, centroid))
    return normal, d


def point_plane_residual(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return points @ normal + d


def boundary_vertices(faces: np.ndarray) -> set[int]:
    owners = edge_ownership(faces)
    out: set[int] = set()
    for edge, face_ids in owners.items():
        if len(face_ids) == 1:
            out.update(edge)
    return out


def vertex_neighbors(faces: np.ndarray, vertex_count: int) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def _local_plane_support(
    vertices: np.ndarray,
    neighbors: list[set[int]],
    vid: int,
    *,
    min_points: int = 3,
    max_points: int = 32,
) -> np.ndarray:
    first_ring = set(neighbors[vid])
    support = set(first_ring)
    if len(support) < min_points:
        for nid in first_ring:
            support.update(neighbors[nid])
    support.discard(int(vid))
    if len(support) < min_points:
        return vertices
    support_ids = sorted(support)
    if len(support_ids) > max_points:
        center = vertices[int(vid)]
        support_ids = sorted(support_ids, key=lambda i: float(np.linalg.norm(vertices[i] - center)))[:max_points]
    return vertices[np.asarray(support_ids, dtype=np.int64)]


def _vertex_evidence_value(
    vertex_evidence: dict[int, dict[str, float]] | None,
    vid: int,
    key: str,
    default: float,
) -> float:
    if not vertex_evidence:
        return default
    return float(vertex_evidence.get(int(vid), {}).get(key, default))


def make_snap_proposals(
    state: MeshState,
    *,
    candidate_vertices: list[int] | None = None,
    supported_vertices: set[int] | None = None,
    vertex_evidence: dict[int, dict[str, float]] | None = None,
    step_sizes: list[float] | None = None,
    max_displacement_fraction: float = 0.05,
    residual_threshold_fraction: float = 0.015,
    free_space_reject_threshold: float = 0.5,
    evidence_source: str = "local_plane_fit",
) -> list[SnapProposal]:
    vertices = np.asarray(state.vertices, dtype=np.float64)
    if len(vertices) < 3:
        return []
    step_sizes = step_sizes or [0.1, 0.25, 0.5]
    supported_vertices = supported_vertices if supported_vertices is not None else set(range(len(vertices)))
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    max_disp = max(bbox_diag * max_displacement_fraction, 1e-6)
    residual_threshold = max(bbox_diag * residual_threshold_fraction, 1e-6)
    boundary = boundary_vertices(state.faces)
    neighbors = vertex_neighbors(state.faces, len(vertices))

    local_models: dict[int, tuple[np.ndarray, float, float]] = {}
    residual_by_vertex: dict[int, float] = {}
    model_vertex_ids = range(len(vertices)) if candidate_vertices is None else [int(v) for v in candidate_vertices]
    for vid in model_vertex_ids:
        support_points = _local_plane_support(vertices, neighbors, vid)
        normal, d = fit_plane(support_points)
        residual = float(point_plane_residual(vertices[int(vid)].reshape(1, 3), normal, d)[0])
        local_models[int(vid)] = (normal, d, residual)
        residual_by_vertex[int(vid)] = abs(residual)

    if candidate_vertices is None:
        residual_abs = np.zeros((len(vertices),), dtype=np.float64)
        for vid, residual in residual_by_vertex.items():
            residual_abs[int(vid)] = residual
        candidate_vertices = [int(i) for i in np.where(residual_abs > residual_threshold)[0]]
    proposals: list[SnapProposal] = []
    for vid in candidate_vertices:
        vid = int(vid)
        positive_evidence = _vertex_evidence_value(vertex_evidence, vid, "positive_surface_evidence", 1.0 if vid in supported_vertices else 0.0)
        negative_free_space = _vertex_evidence_value(vertex_evidence, vid, "negative_free_space_evidence", 0.0)
        uncertainty_evidence = _vertex_evidence_value(vertex_evidence, vid, "uncertainty", 0.35)
        if vid not in supported_vertices:
            proposals.append(
                SnapProposal(
                    proposal_id=f"snap_{len(proposals):04d}",
                    edit=MeshEdit(
                        edit_id=f"snap_edit_{len(proposals):04d}",
                        edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
                        defect_id="unknown",
                        affected_vertices=[int(vid)],
                    ),
                    target_type="local_plane",
                    step_size=0.0,
                    max_displacement=max_disp,
                    expected_error_before=float(residual_by_vertex[vid]),
                    expected_error_after=float(residual_by_vertex[vid]),
                    uncertainty=max(0.95, uncertainty_evidence),
                    evidence_source=evidence_source,
                    rejected_reason="unsupported_vertex_no_snap",
                )
            )
            continue
        if negative_free_space >= free_space_reject_threshold:
            proposals.append(
                SnapProposal(
                    proposal_id=f"snap_{len(proposals):04d}",
                    edit=MeshEdit(
                        edit_id=f"snap_edit_{len(proposals):04d}",
                        edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
                        defect_id="unknown",
                        affected_vertices=[vid],
                        evidence_summary={
                            "selector": "csef_local_plane_snap",
                            "positive_surface_evidence": positive_evidence,
                            "negative_free_space_evidence": negative_free_space,
                            "free_space_reject_threshold": float(free_space_reject_threshold),
                        },
                        risk_summary={"free_space_risk": negative_free_space, "snap_rejected": True},
                    ),
                    target_type="local_plane",
                    step_size=0.0,
                    max_displacement=max_disp,
                    expected_error_before=float(residual_by_vertex[vid]),
                    expected_error_after=float(residual_by_vertex[vid]),
                    uncertainty=max(0.9, uncertainty_evidence),
                    evidence_source=evidence_source,
                    rejected_reason="negative_free_space_evidence",
                )
            )
            continue
        normal, d, residual = local_models[vid]
        disp_to_plane = -residual * normal
        disp_norm = float(np.linalg.norm(disp_to_plane))
        if disp_norm <= 1e-12:
            continue
        if vid in boundary:
            disp_to_plane *= 0.5
        if np.linalg.norm(disp_to_plane) > max_disp:
            disp_to_plane = disp_to_plane / np.linalg.norm(disp_to_plane) * max_disp
        for step in step_sizes:
            target = vertices[vid] + step * disp_to_plane
            after_residual = float(abs(point_plane_residual(target.reshape(1, 3), normal, d)[0]))
            proposals.append(
                SnapProposal(
                    proposal_id=f"snap_{len(proposals):04d}",
                    edit=MeshEdit(
                        edit_id=f"snap_edit_{len(proposals):04d}",
                        edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
                        defect_id="unknown",
                        affected_vertices=[int(vid)],
                        attribute_changes={"target_positions": {str(int(vid)): [float(x) for x in target]}},
                        evidence_summary={
                            "selector": "csef_local_plane_snap",
                            "target_type": "local_plane",
                            "positive_surface_evidence": positive_evidence,
                            "negative_free_space_evidence": negative_free_space,
                            "local_plane_residual_before": float(abs(residual)),
                            "local_plane_residual_after": after_residual,
                        },
                        risk_summary={
                            "free_space_risk": negative_free_space,
                            "boundary_vertex": bool(vid in boundary),
                            "max_displacement": max_disp,
                        },
                    ),
                    target_type="local_plane",
                    step_size=float(step),
                    max_displacement=max_disp,
                    expected_error_before=float(abs(residual)),
                    expected_error_after=after_residual,
                    uncertainty=max(float(uncertainty_evidence), 0.35 if vid not in boundary else 0.55),
                    evidence_source=evidence_source,
                )
            )
    return proposals


def write_snap_outputs(
    state: MeshState,
    proposals: list[SnapProposal],
    output_dir: str | Path,
    *,
    preview_proposal: SnapProposal | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "snap_proposals.json").write_text(json.dumps([p.to_dict() for p in proposals], indent=2), encoding="utf-8")
    with (out / "snap_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["proposal_id", "step_size", "expected_error_before", "expected_error_after", "uncertainty", "rejected_reason"])
        for p in proposals:
            writer.writerow([p.proposal_id, p.step_size, p.expected_error_before, p.expected_error_after, p.uncertainty, p.rejected_reason])
    if preview_proposal is not None and not preview_proposal.rejected_reason:
        preview = state.copy()
        for vid_text, target in preview_proposal.edit.attribute_changes.get("target_positions", {}).items():
            preview.vertices[int(vid_text)] = np.asarray(target, dtype=np.float64)
        write_ascii_ply(out / "snap_debug_before.ply", state.vertices, state.faces)
        write_ascii_ply(out / "snap_debug_before_after.ply", preview.vertices, preview.faces)
