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

from ss3dm_prior.meshsplatopt.csef_builder import build_csef, write_ascii_ply, write_csef_outputs


def make_synthetic_mesh() -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    vertices = []
    grid_n = 6
    dent_vertex = (4, 2)
    for y in range(grid_n):
        for x in range(grid_n):
            z = -0.35 if (x, y) == dent_vertex else 0.0
            vertices.append((float(x), float(y), z))
    faces = []
    labels: dict[str, list[int]] = {"normal": [], "hole_boundary": [], "dent": [], "floater": []}
    missing_cell = (2, 2)
    for y in range(grid_n - 1):
        for x in range(grid_n - 1):
            if (x, y) == missing_cell:
                continue
            v0 = y * grid_n + x
            v1 = y * grid_n + x + 1
            v2 = (y + 1) * grid_n + x
            v3 = (y + 1) * grid_n + x + 1
            tri0 = len(faces)
            faces.append((v0, v1, v3))
            tri1 = len(faces)
            faces.append((v0, v3, v2))
            cell_labels = []
            if abs(x - missing_cell[0]) + abs(y - missing_cell[1]) == 1:
                cell_labels.append("hole_boundary")
            if dent_vertex in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)):
                cell_labels.append("dent")
            if not cell_labels and 0 < x < grid_n - 2 and 0 < y < grid_n - 2:
                cell_labels.append("normal")
            for name in cell_labels:
                labels[name].extend([tri0, tri1])

    floater_base = len(vertices)
    vertices.extend([(5.0, 5.0, 1.0), (5.6, 5.0, 1.1), (5.2, 5.5, 0.9)])
    labels["floater"].append(len(faces))
    faces.append((floater_base, floater_base + 1, floater_base + 2))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), labels


def mean_for(samples, face_ids: list[int], field: str) -> float:
    if not face_ids:
        return 0.0
    vals = [getattr(samples[i], field) for i in sorted(set(face_ids))]
    return float(np.mean(vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="outputs/carnet/meshsplatopt/stageR3_csef_smoke",
        help="Smoke artifact directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vertices, faces, labels = make_synthetic_mesh()
    mesh_path = out / "synthetic_hole_floater_dent.ply"
    write_ascii_ply(mesh_path, vertices, faces)
    result, samples = build_csef(
        vertices,
        faces,
        scene_model="synthetic_hole_floater_dent",
        scene_source="synthetic",
        mesh_path=str(mesh_path),
    )
    write_csef_outputs(result, samples, out / "csef")

    metrics = {
        "normal_debt": mean_for(samples, labels["normal"], "explanation_debt"),
        "hole_boundary_debt": mean_for(samples, labels["hole_boundary"], "explanation_debt"),
        "floater_uncertainty": mean_for(samples, labels["floater"], "uncertainty"),
        "floater_positive_surface_evidence": mean_for(samples, labels["floater"], "positive_surface_evidence"),
        "dent_debt": mean_for(samples, labels["dent"], "explanation_debt"),
        "region_count": len(result.regions),
    }
    checks = {
        "hole_boundary_high_debt": metrics["hole_boundary_debt"] > metrics["normal_debt"] + 0.15,
        "floater_high_uncertainty": metrics["floater_uncertainty"] >= 0.85,
        "floater_low_positive_evidence": metrics["floater_positive_surface_evidence"] <= 0.25,
        "normal_ground_low_debt": metrics["normal_debt"] < 0.55,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {"status": status, "metrics": metrics, "checks": checks}
    (out / "stageR3_csef_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R3 CSEF Smoke", "", f"Status: `{status}`", "", "## Metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "stageR3_csef_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "PASS":
        raise SystemExit(f"Stage R3 CSEF smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
