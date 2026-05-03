from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .csef_builder import edge_ownership, triangle_geometry, write_ascii_ply
from .edit_apply import verify_mesh_integrity
from .edit_types import MeshState


@dataclass(frozen=True)
class TopologyBaselineRun:
    method: str
    budget_fraction: float
    target_faces: int
    output_faces: int
    output_vertices: int
    valid: bool
    errors: list[str]
    notes: list[str]
    mesh_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _target_count(face_count: int, budget_fraction: float) -> int:
    return max(1, int(round(face_count * budget_fraction)))


def _delete_faces(state: MeshState, delete_ids: np.ndarray) -> MeshState:
    mask = np.ones((len(state.faces),), dtype=bool)
    delete_ids = delete_ids[(delete_ids >= 0) & (delete_ids < len(state.faces))]
    mask[delete_ids] = False
    return MeshState(vertices=state.vertices.copy(), faces=state.faces[mask].copy(), attributes=dict(state.attributes))


def _boundary_face_mask(faces: np.ndarray) -> np.ndarray:
    owners = edge_ownership(faces)
    mask = np.zeros((len(faces),), dtype=bool)
    for face_ids in owners.values():
        if len(face_ids) == 1:
            mask[int(face_ids[0])] = True
    return mask


def prism_score_topk_delete(state: MeshState, budget_fraction: float, scores: np.ndarray | None = None) -> MeshState:
    target = _target_count(len(state.faces), budget_fraction)
    remove_count = max(0, len(state.faces) - target)
    if remove_count == 0:
        return state.copy()
    if scores is None:
        _, _, areas = triangle_geometry(state.vertices, state.faces)
        scores = areas
    order = np.argsort(scores)
    return _delete_faces(state, order[:remove_count])


def random_same_count_delete(state: MeshState, budget_fraction: float, seed: int = 0) -> MeshState:
    target = _target_count(len(state.faces), budget_fraction)
    remove_count = max(0, len(state.faces) - target)
    rng = np.random.default_rng(seed)
    delete_ids = rng.choice(np.arange(len(state.faces)), size=remove_count, replace=False) if remove_count else np.asarray([], dtype=np.int64)
    return _delete_faces(state, delete_ids)


def low_visibility_delete(state: MeshState, budget_fraction: float) -> MeshState:
    _, _, areas = triangle_geometry(state.vertices, state.faces)
    proxy_visibility = areas
    return prism_score_topk_delete(state, budget_fraction, scores=proxy_visibility)


def boundary_protected_delete(state: MeshState, budget_fraction: float) -> MeshState:
    target = _target_count(len(state.faces), budget_fraction)
    remove_count = max(0, len(state.faces) - target)
    if remove_count == 0:
        return state.copy()
    boundary = _boundary_face_mask(state.faces)
    interior = np.where(~boundary)[0]
    boundary_ids = np.where(boundary)[0]
    order = np.concatenate([interior, boundary_ids])
    return _delete_faces(state, order[:remove_count])


def _remove_degenerate(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return faces.reshape(0, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    return faces[keep]


def qem_style_edge_collapse(state: MeshState, budget_fraction: float) -> MeshState:
    target = _target_count(len(state.faces), budget_fraction)
    vertices = state.vertices.copy()
    faces = state.faces.copy()
    while len(faces) > target:
        owners = edge_ownership(faces)
        if not owners:
            break
        edges = np.asarray(list(owners.keys()), dtype=np.int64)
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        keep, remove = [int(x) for x in edges[int(np.argmin(lengths))]]
        vertices[keep] = 0.5 * (vertices[keep] + vertices[remove])
        faces[faces == remove] = keep
        new_faces = _remove_degenerate(faces)
        if len(new_faces) == len(faces):
            break
        faces = new_faces
    return MeshState(vertices=vertices, faces=faces, attributes=dict(state.attributes))


def planar_face_merge(state: MeshState, budget_fraction: float, normal_dot_threshold: float = 0.995) -> MeshState:
    target = _target_count(len(state.faces), budget_fraction)
    remove_count = max(0, len(state.faces) - target)
    if remove_count == 0:
        return state.copy()
    _, normals, _ = triangle_geometry(state.vertices, state.faces)
    owners = edge_ownership(state.faces)
    removable: list[int] = []
    for face_ids in owners.values():
        if len(face_ids) != 2:
            continue
        a, b = int(face_ids[0]), int(face_ids[1])
        if abs(float(np.dot(normals[a], normals[b]))) >= normal_dot_threshold:
            removable.append(max(a, b))
        if len(set(removable)) >= remove_count:
            break
    if not removable:
        removable = list(range(min(remove_count, len(state.faces))))
    return _delete_faces(state, np.asarray(sorted(set(removable))[:remove_count], dtype=np.int64))


def run_topology_baselines(
    state: MeshState,
    output_dir: str | Path,
    *,
    budgets: list[float] | None = None,
    methods: list[str] | None = None,
) -> list[TopologyBaselineRun]:
    budgets = budgets or [0.90, 0.75, 0.50, 0.25]
    methods = methods or [
        "prism_score_topk_delete",
        "random_same_count_delete",
        "low_visibility_delete",
        "boundary_protected_delete",
        "qem_style_edge_collapse",
        "planar_face_merge",
        "external_simplification",
    ]
    out = Path(output_dir)
    mesh_dir = out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    runners = {
        "prism_score_topk_delete": prism_score_topk_delete,
        "random_same_count_delete": random_same_count_delete,
        "low_visibility_delete": low_visibility_delete,
        "boundary_protected_delete": boundary_protected_delete,
        "qem_style_edge_collapse": qem_style_edge_collapse,
        "planar_face_merge": planar_face_merge,
    }
    runs: list[TopologyBaselineRun] = []
    for method in methods:
        for budget in budgets:
            target = _target_count(len(state.faces), budget)
            notes: list[str] = []
            if method == "external_simplification":
                runs.append(
                    TopologyBaselineRun(
                        method=method,
                        budget_fraction=budget,
                        target_faces=target,
                        output_faces=len(state.faces),
                        output_vertices=len(state.vertices),
                        valid=False,
                        errors=["optional external simplifier not invoked in R6"],
                        notes=["JSON contract retained for trimesh/pymeshlab integration"],
                    )
                )
                continue
            candidate = runners[method](state.copy(), budget)
            integrity = verify_mesh_integrity(candidate)
            mesh_path = mesh_dir / f"{method}_budget{int(budget * 100):03d}.ply"
            if integrity["valid"]:
                write_ascii_ply(mesh_path, candidate.vertices, candidate.faces)
            runs.append(
                TopologyBaselineRun(
                    method=method,
                    budget_fraction=budget,
                    target_faces=target,
                    output_faces=int(len(candidate.faces)),
                    output_vertices=int(len(candidate.vertices)),
                    valid=bool(integrity["valid"]),
                    errors=list(integrity["errors"]),
                    notes=notes,
                    mesh_path=str(mesh_path) if integrity["valid"] else "",
                )
            )
    write_topology_baseline_outputs(runs, out)
    return runs


def write_topology_baseline_outputs(runs: list[TopologyBaselineRun], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "topology_baseline_runs.json").write_text(json.dumps([r.to_dict() for r in runs], indent=2), encoding="utf-8")
    with (out / "topology_baseline_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "budget_fraction", "target_faces", "output_faces", "output_vertices", "valid", "errors", "mesh_path"])
        for r in runs:
            writer.writerow([r.method, r.budget_fraction, r.target_faces, r.output_faces, r.output_vertices, r.valid, " ".join(r.errors), r.mesh_path])
    lines = ["# Topology Baseline Report", "", f"- runs: `{len(runs)}`", "", "## Runs", ""]
    for r in runs:
        lines.append(
            f"- `{r.method}` budget `{r.budget_fraction}` target `{r.target_faces}` "
            f"output `{r.output_faces}` valid `{r.valid}`"
        )
        if r.errors:
            lines.append(f"  - errors: {'; '.join(r.errors)}")
    (out / "topology_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
