"""Smoke test for topology-control checkpoint-copy ablation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def _write_fake_model(root: Path) -> tuple[Path, Path]:
    model = root / "model"
    ckpt = model / "point_cloud" / "iteration_7"
    ckpt.mkdir(parents=True)
    (model / "cfg_args").write_text("Namespace()\n", encoding="utf-8")
    state = {
        "triangles_points": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
            ],
            dtype=torch.float32,
        ),
        "_triangle_indices": torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.int32),
        "vertex_weight": torch.ones((5, 1), dtype=torch.float32),
        "sigma": torch.tensor(0.0001),
        "active_sh_degree": 3,
        "features_dc": torch.zeros((5, 1, 3), dtype=torch.float32),
        "features_rest": torch.zeros((5, 15, 3), dtype=torch.float32),
        "importance_score": torch.zeros((2,), dtype=torch.float32),
        "image_size": torch.zeros((2,), dtype=torch.float32),
        "pixel_count": torch.zeros((2,), dtype=torch.int32),
    }
    torch.save(state, ckpt / "point_cloud_state_dict.pt")
    return model, ckpt / "point_cloud_state_dict.pt"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source_model, source_checkpoint = _write_fake_model(root)
        output_model = root / "ablation"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_apply_topology_control_ablation.py"),
                "--source_model",
                str(source_model),
                "--source_checkpoint",
                str(source_checkpoint),
                "--output_model",
                str(output_model),
                "--iteration",
                "7",
                "--prune_fraction",
                "0.5",
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((output_model / "topology_control_ablation_report.json").read_text(encoding="utf-8"))
        assert report["source_model_edited"] is False, report
        assert report["checkpoint_copy_edited"] is True, report
        assert report["face_count_before"] == 2, report
        assert report["face_count_after"] == 1, report
        assert report["vertex_count_after"] == 3, report
        out_state = torch.load(output_model / "point_cloud" / "iteration_7" / "point_cloud_state_dict.pt", map_location="cpu")
        assert tuple(out_state["_triangle_indices"].shape) == (1, 3)
        assert tuple(out_state["triangles_points"].shape) == (3, 3)
        print("[meshprior-topology-control-ablation-smoke] PASS")
        print(json.dumps({"triangles": report["face_count_after"], "vertices": report["vertex_count_after"]}, indent=2))


if __name__ == "__main__":
    main()
