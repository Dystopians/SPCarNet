"""Smoke test for applying parking patch cleanup to a checkpoint copy."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proposal_report = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json"
    patch_summary = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json"
    state = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt"
    for path in (proposal_report, patch_summary, state):
        if not path.is_file():
            raise SystemExit(f"required input missing: {path}")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "checkpoint_copy_cleanup"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_apply_parking_patch_cleanup_to_checkpoint_copy.py"),
                "--patch_proposal_report",
                str(proposal_report),
                "--mesh_patch_summary",
                str(patch_summary),
                "--triangle_state",
                str(state),
                "--output_dir",
                str(out),
                "--max_applications",
                "2",
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "checkpoint_copy_application_report.json").read_text(encoding="utf-8"))
        assert report["accepted_cleanup_applications"] == 2, report
        assert report["source_model_edited"] is False, report
        assert report["checkpoint_copy_edited"] is True, report
        assert report["face_count_after"] < report["face_count_before"], report
        assert report["vertex_count_after"] < report["vertex_count_before"], report
        copied = torch.load(out / "point_cloud_state_dict.pt", map_location="cpu")
        assert copied["_triangle_indices"].shape[0] == report["face_count_after"], report
        assert copied["triangles_points"].shape[0] == report["vertex_count_after"], report
        assert copied["features_dc"].shape[0] == report["vertex_count_after"], report
        assert copied["importance_score"].shape[0] == report["face_count_after"], report
        assert (out / "checkpoint_copy_application_rows.csv").is_file()
        assert (out / "checkpoint_copy_application_report.md").is_file()
        print("[meshprior-parking-checkpoint-copy-cleanup-smoke] PASS")
        print(json.dumps({"faces_removed": report["faces_removed"], "vertices_removed": report["vertices_removed"]}, indent=2))


if __name__ == "__main__":
    main()
