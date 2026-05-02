"""Smoke test for preparing a parking recovery model directory."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_model = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model"
    copied_checkpoint = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt"
    if not source_model.is_dir() or not copied_checkpoint.is_file():
        raise SystemExit("required recovery model inputs are missing")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "recovery_model"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_prepare_parking_recovery_model.py"),
                "--source_model",
                str(source_model),
                "--copied_checkpoint",
                str(copied_checkpoint),
                "--output_model",
                str(out),
                "--iteration",
                "200",
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "meshprior_recovery_model_report.json").read_text(encoding="utf-8"))
        ckpt = out / "point_cloud/iteration_200/point_cloud_state_dict.pt"
        assert ckpt.is_file(), report
        assert (out / "cfg_args").is_file(), report
        assert (out / "cameras.json").is_file(), report
        state = torch.load(ckpt, map_location="cpu")
        assert state["_triangle_indices"].shape[0] == report["triangles"], report
        assert state["triangles_points"].shape[0] == report["vertices"], report
        assert report["source_model_edited"] is False, report
        print("[meshprior-parking-recovery-model-smoke] PASS")
        print(json.dumps({"triangles": report["triangles"], "vertices": report["vertices"]}, indent=2))


if __name__ == "__main__":
    main()
