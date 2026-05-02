"""Smoke test for parking mesh patch extraction."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    action_plan = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/action_plan.json"
    clusters = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json"
    state = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt"
    for path in (action_plan, clusters, state):
        if not path.is_file():
            raise SystemExit(f"required input missing: {path}")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mesh_patches"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_extract_parking_mesh_patches.py"),
                "--action_plan",
                str(action_plan),
                "--consolidated_regions",
                str(clusters),
                "--triangle_state",
                str(state),
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        summary = json.loads((out / "mesh_patch_summary.json").read_text(encoding="utf-8"))
        assert summary["patch_count"] == 8, summary
        assert summary["nonempty_patch_count"] == 8, summary
        assert summary["total_patch_faces"] > 0, summary
        first = np.load(summary["patches"][0]["patch_path"], allow_pickle=False)
        assert first["vertices"].shape[1] == 3
        assert first["faces"].shape[1] == 3
        assert len(first["original_face_indices"]) == first["faces"].shape[0]
        assert (out / "mesh_patch_summary.csv").is_file()
        assert (out / "mesh_patch_report.md").is_file()
        print("[meshprior-parking-mesh-patch-extraction-smoke] PASS")
        print(
            json.dumps(
                {
                    "patches": summary["patch_count"],
                    "total_faces": summary["total_patch_faces"],
                    "face_range": [summary["min_patch_faces"], summary["max_patch_faces"]],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
