"""Smoke test for the Stage23.5 integrated topology collector."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def _write_model(root: Path) -> Path:
    model = root / "model"
    ckpt = model / "point_cloud" / "iteration_3"
    ckpt.mkdir(parents=True)
    torch.save(
        {
            "triangles_points": torch.zeros((4, 3), dtype=torch.float32),
            "_triangle_indices": torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32),
        },
        ckpt / "point_cloud_state_dict.pt",
    )
    (model / "results.json").write_text(json.dumps({"ours_3": {"PSNR": 1.0, "SSIM": 0.5, "LPIPS": 0.2}}), encoding="utf-8")
    geom = model / "geometry_eval_colmap"
    geom.mkdir()
    (geom / "iter_3.json").write_text(
        json.dumps({"depth": {"abs_rel": 0.1, "mae": 0.2}, "normal": {"mean_ang_deg": 30.0}}),
        encoding="utf-8",
    )
    debug = model / "prism_debug"
    debug.mkdir()
    (debug / "final_cleanup_summary.json").write_text(json.dumps({"final_cleanup_enabled": False}), encoding="utf-8")
    meta = model / "prism_round_checkpoints"
    meta.mkdir()
    (meta / "iter_000003_candidate_meta.json").write_text(
        json.dumps(
            {
                "iteration": 3,
                "phase": "candidate",
                "prune_mode": "candidate",
                "committed": True,
                "counterfactual_accept": 1,
                "rollback": 0,
                "pre_prune_triangle_count": 2,
                "post_prune_triangle_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return model


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "summary"
        model = _write_model(Path(td))
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_collect_stage23_5_integrated_topology.py"),
                "--model",
                str(model),
                "--iteration",
                "3",
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "stage23_5_integrated_topology_summary.json").read_text(encoding="utf-8"))
        assert report["gate"] == "PASS", report
        assert report["round_count"] == 1, report
        assert report["committed_round_count"] == 1, report
        print("[meshprior-stage23.5-collector-smoke] PASS")
        print(json.dumps({"gate": report["gate"], "rounds": report["round_count"]}, indent=2))


if __name__ == "__main__":
    main()
