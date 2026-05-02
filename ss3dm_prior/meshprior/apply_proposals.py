"""Apply accepted MeshPrior proposal meshes to a safe copy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MeshArrays:
    vertices: np.ndarray
    faces: np.ndarray


def load_mesh_npz(path: str | Path) -> MeshArrays:
    with np.load(path) as data:
        return MeshArrays(vertices=np.asarray(data["vertices"], dtype=np.float32), faces=np.asarray(data["faces"], dtype=np.int64))


def save_mesh_npz(path: str | Path, mesh: MeshArrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vertices=np.asarray(mesh.vertices, dtype=np.float32), faces=np.asarray(mesh.faces, dtype=np.int64))


def meshes_equal(a: MeshArrays, b: MeshArrays, *, atol: float = 1e-6) -> bool:
    return a.vertices.shape == b.vertices.shape and a.faces.shape == b.faces.shape and np.allclose(a.vertices, b.vertices, atol=atol) and np.array_equal(a.faces, b.faces)


def mesh_stats(mesh: MeshArrays) -> dict[str, int]:
    return {"vertex_count": int(len(mesh.vertices)), "face_count": int(len(mesh.faces))}


def accepted_ids_from_gate(gate_report: dict[str, Any]) -> set[str]:
    return {str(row["proposal_id"]) for row in gate_report.get("results", []) if bool(row.get("accepted"))}


def rejected_ids_from_gate(gate_report: dict[str, Any]) -> set[str]:
    return {str(row["proposal_id"]) for row in gate_report.get("results", []) if not bool(row.get("accepted"))}


def apply_accepted_proposals(
    *,
    proposals: list[dict[str, Any]],
    gate_report: dict[str, Any],
    output_dir: str | Path,
    initial_mesh: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    rollback_dir = out / "rollback"
    applied_dir = out / "applied_steps"
    out.mkdir(parents=True, exist_ok=True)
    accepted_ids = accepted_ids_from_gate(gate_report)
    rejected_ids = rejected_ids_from_gate(gate_report)
    applied: list[dict[str, Any]] = []
    warnings: list[str] = []
    current: MeshArrays | None = load_mesh_npz(initial_mesh) if initial_mesh else None
    initial_stats: dict[str, int] | None = mesh_stats(current) if current is not None else None

    for row in proposals:
        proposal_id = str(row.get("proposal_id", ""))
        if proposal_id not in accepted_ids:
            continue
        before = load_mesh_npz(row["before_npz"])
        after = load_mesh_npz(row["after_npz"])
        if current is None:
            current = before
            initial_stats = mesh_stats(current)
        elif not meshes_equal(current, before):
            warnings.append(f"{proposal_id}: before_npz does not match current mesh; applying proposal after-state as authoritative copy")
        rollback_path = rollback_dir / f"{len(applied):04d}_{proposal_id}_rollback.npz"
        save_mesh_npz(rollback_path, current)
        step_path = applied_dir / f"{len(applied):04d}_{proposal_id}_after.npz"
        save_mesh_npz(step_path, after)
        current = after
        applied.append(
            {
                "proposal_id": proposal_id,
                "proposal_type": row.get("proposal_type", "unknown"),
                "rollback_npz": str(rollback_path),
                "applied_npz": str(step_path),
                "before": mesh_stats(before),
                "after": mesh_stats(after),
            }
        )

    if current is None:
        raise ValueError("no initial mesh and no accepted proposals with before_npz were available")
    final_path = out / "applied_mesh.npz"
    save_mesh_npz(final_path, current)
    return {
        "status": "PASS",
        "accepted_count": len(accepted_ids),
        "rejected_count": len(rejected_ids),
        "applied_count": len(applied),
        "initial": initial_stats,
        "final": mesh_stats(current),
        "applied_mesh": str(final_path),
        "applied": applied,
        "warnings": warnings,
    }
