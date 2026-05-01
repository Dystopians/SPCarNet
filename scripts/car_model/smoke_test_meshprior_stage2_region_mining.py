"""Smoke test for MeshPrior Stage 2 region mining."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_two_component_ply(path: Path) -> None:
    vertices = [
        (-1.0, -0.5, -0.25),
        (1.0, -0.5, -0.25),
        (1.0, 0.5, -0.25),
        (-1.0, 0.5, -0.25),
        (-1.0, -0.5, 0.25),
        (1.0, -0.5, 0.25),
        (1.0, 0.5, 0.25),
        (-1.0, 0.5, 0.25),
        (4.0, 0.0, 0.0),
        (4.5, 0.0, 0.0),
        (4.0, 0.5, 0.0),
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
        (8, 9, 10),
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
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mesh_path = tmp / "synthetic_two_component.ply"
        out_dir = tmp / "out"
        _write_two_component_ply(mesh_path)
        cmd = [
            sys.executable,
            str(repo_root / "scripts/car_model/meshprior_mine_regions.py"),
            "--scene_model",
            str(mesh_path),
            "--scene_source",
            str(tmp / "missing_scene"),
            "--output_dir",
            str(out_dir),
            "--mode",
            "dry_run",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)
        data = json.loads((out_dir / "regions.json").read_text(encoding="utf-8"))
        regions = data.get("regions", [])
        assert len(regions) >= 2, f"expected at least 2 regions, got {len(regions)}"
        assert (out_dir / "regions_summary.csv").is_file()
        assert (out_dir / "region_mining_report.md").is_file()
        eligible = sum(1 for r in regions if r["evidence"]["eligible_for_posterior"])
        print("[meshprior-stage2-smoke] PASS")
        print(json.dumps({"regions": len(regions), "eligible_for_posterior": eligible}, indent=2))


if __name__ == "__main__":
    main()
