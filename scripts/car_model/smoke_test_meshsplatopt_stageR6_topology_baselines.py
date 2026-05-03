#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.csef_builder import write_ascii_ply
from ss3dm_prior.meshsplatopt.edit_types import MeshState
from ss3dm_prior.meshsplatopt.topology_baselines import run_topology_baselines


def make_plane_object_mesh() -> MeshState:
    vertices = []
    grid_n = 5
    for y in range(grid_n):
        for x in range(grid_n):
            vertices.append((float(x), float(y), 0.0))
    faces = []
    for y in range(grid_n - 1):
        for x in range(grid_n - 1):
            v0 = y * grid_n + x
            v1 = y * grid_n + x + 1
            v2 = (y + 1) * grid_n + x
            v3 = (y + 1) * grid_n + x + 1
            faces.append((v0, v1, v3))
            faces.append((v0, v3, v2))
    base = len(vertices)
    vertices.extend([(1.5, 1.5, 0.0), (2.5, 1.5, 0.0), (2.5, 2.5, 0.0), (1.5, 2.5, 0.0), (2.0, 2.0, 1.0)])
    faces.extend([(base, base + 1, base + 4), (base + 1, base + 2, base + 4), (base + 2, base + 3, base + 4), (base + 3, base, base + 4)])
    return MeshState(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR6_topology_baselines_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = make_plane_object_mesh()
    write_ascii_ply(out / "synthetic_plane_object_before.ply", state.vertices, state.faces)
    methods = ["prism_score_topk_delete", "boundary_protected_delete", "qem_style_edge_collapse", "planar_face_merge"]
    runs = run_topology_baselines(state, out / "baselines", budgets=[0.75, 0.50], methods=methods)
    by_method = {m: [r for r in runs if r.method == m] for m in methods}
    tolerance = 2
    checks = {
        "delete_runs_valid": all(r.valid and abs(r.output_faces - r.target_faces) <= tolerance for r in by_method["prism_score_topk_delete"]),
        "boundary_protected_runs_valid": all(r.valid and abs(r.output_faces - r.target_faces) <= tolerance for r in by_method["boundary_protected_delete"]),
        "collapse_or_merge_runs_valid": any(
            all(r.valid and r.output_faces <= state.faces.shape[0] for r in by_method[m]) for m in ["qem_style_edge_collapse", "planar_face_merge"]
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "input_faces": int(len(state.faces)),
        "runs": [r.to_dict() for r in runs],
        "checks": checks,
    }
    (out / "topology_baseline_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R6 Topology Baselines Smoke", "", f"Status: `{report['status']}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "topology_baseline_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Stage R6 topology baseline smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
