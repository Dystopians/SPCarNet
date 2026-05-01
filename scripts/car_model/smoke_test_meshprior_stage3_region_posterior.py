"""Smoke test for MeshPrior Stage 3 scene-region posterior inference."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def _write_box_ply(path: Path) -> None:
    vertices = [
        (-1.0, -0.5, -0.25),
        (1.0, -0.5, -0.25),
        (1.0, 0.5, -0.25),
        (-1.0, 0.5, -0.25),
        (-1.0, -0.5, 0.25),
        (1.0, -0.5, 0.25),
        (1.0, 0.5, 0.25),
        (-1.0, 0.5, 0.25),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ckpt = repo_root / "outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mesh_path = tmp / "box_region.ply"
        mining_dir = tmp / "mining"
        posterior_dir = tmp / "posterior"
        _write_box_ply(mesh_path)
        subprocess.run(
            [
                "python",
                str(repo_root / "scripts/car_model/meshprior_mine_regions.py"),
                "--scene_model",
                str(mesh_path),
                "--scene_source",
                str(tmp / "missing_scene"),
                "--output_dir",
                str(mining_dir),
                "--mode",
                "dry_run",
            ],
            cwd=repo_root,
            check=True,
        )
        missing = subprocess.run(
            [
                "python",
                str(repo_root / "scripts/car_model/meshprior_infer_region_posterior.py"),
                "--regions_json",
                str(mining_dir / "regions.json"),
                "--posterior_checkpoint",
                str(tmp / "missing_checkpoint.pt"),
                "--output_dir",
                str(tmp / "missing_out"),
                "--device",
                "cpu",
                "--limit",
                "1",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert missing.returncode != 0
        assert "posterior_checkpoint not found" in missing.stderr
        if not ckpt.is_file():
            print("[meshprior-stage3-smoke] PASS_MISSING_CHECKPOINT_ONLY")
            print(json.dumps({"checkpoint": str(ckpt), "present": False}, indent=2))
            return
        subprocess.run(
            [
                "python",
                str(repo_root / "scripts/car_model/meshprior_infer_region_posterior.py"),
                "--regions_json",
                str(mining_dir / "regions.json"),
                "--posterior_checkpoint",
                str(ckpt),
                "--output_dir",
                str(posterior_dir),
                "--device",
                "cpu",
                "--limit",
                "1",
                "--grid_resolution",
                "16",
                "--n_points",
                "128",
            ],
            cwd=repo_root,
            check=True,
        )
        index = json.loads((posterior_dir / "posterior_index.json").read_text(encoding="utf-8"))
        assert index["ok_regions"] >= 1, index
        region_dir = posterior_dir / index["regions"][0]["region_id"]
        assert (region_dir / "z_mean.npy").is_file()
        assert (region_dir / "canonical_transform.json").is_file()
        assert (region_dir / "posterior_summary.json").is_file()
        print("[meshprior-stage3-smoke] PASS")
        print(json.dumps({"ok_regions": index["ok_regions"], "region_id": index["regions"][0]["region_id"]}, indent=2))


if __name__ == "__main__":
    main()
