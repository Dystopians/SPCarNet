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
from ss3dm_prior.meshsplatopt.defect_mining import mine_defects, write_defect_outputs


def make_parking_ground_void() -> tuple[np.ndarray, np.ndarray]:
    grid_n = 9
    vertices = [(float(x), float(y), 0.0) for y in range(grid_n) for x in range(grid_n)]
    faces = []
    for y in range(grid_n - 1):
        for x in range(grid_n - 1):
            if 3 <= x <= 5 and 3 <= y <= 5:
                continue
            v0 = y * grid_n + x
            v1 = y * grid_n + x + 1
            v2 = (y + 1) * grid_n + x
            v3 = (y + 1) * grid_n + x + 1
            faces.append((v0, v1, v3))
            faces.append((v0, v3, v2))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke",
        help="Smoke artifact directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vertices, faces = make_parking_ground_void()
    mesh_path = out / "synthetic_parking_ground_giant_void.ply"
    write_ascii_ply(mesh_path, vertices, faces)
    csef, samples = build_csef(
        vertices,
        faces,
        scene_model="synthetic_parking_ground_void",
        scene_source="synthetic",
        mesh_path=str(mesh_path),
    )
    write_csef_outputs(csef, samples, out / "csef")
    unknown_hints = [
        {
            "hint_id": "out_of_trajectory_void",
            "boundary_loop_support": 0.0,
            "camera_coverage_score": 0.0,
            "prior_support": 0.2,
            "severity": 0.9,
        }
    ]
    defects = mine_defects(csef, giant_area_threshold=12.0, unknown_void_hints=unknown_hints)
    write_defect_outputs(defects, out / "defects")
    types = [d.defect_type for d in defects]
    unknown = [d for d in defects if d.defect_type == "UNKNOWN_UNOBSERVED_VOID"]
    giant = [d for d in defects if d.defect_type == "GIANT_GROUND_VOID"]
    checks = {
        "giant_ground_void_detected": len(giant) >= 1,
        "unknown_unobserved_void_detected": len(unknown) == 1,
        "unknown_void_has_no_repair": bool(unknown and not unknown[0].candidate_edit_types_allowed),
        "unknown_void_has_reason": bool(unknown and unknown[0].no_repair_reason),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "defect_types": types,
        "defect_count": len(defects),
        "checks": checks,
    }
    (out / "stageR4_defect_mining_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R4 Defect Mining Smoke", "", f"Status: `{report['status']}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Defect Types", ""])
    for defect_type in types:
        lines.append(f"- `{defect_type}`")
    (out / "stageR4_defect_mining_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Stage R4 defect mining smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
