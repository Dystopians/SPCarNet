"""Smoke test for MeshPrior Stage 6 synthetic damage benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "benchmark"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_run_synthetic_damage_benchmark.py"),
                "--output_dir",
                str(out),
                "--damage_types",
                "local_hole",
                "floater",
                "vertex_noise",
                "density_imbalance",
            ],
            cwd=repo_root,
            check=True,
        )
        report = out / "report.md"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_make_synthetic_damage_report.py"),
                "--metrics_json",
                str(out / "metrics.json"),
                "--output",
                str(report),
            ],
            cwd=repo_root,
            check=True,
        )
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        rows = metrics["inference_time_metrics"]
        floater = [r for r in rows if r["damage_type"] == "floater"][0]
        assert floater["floater_prune_recall"] >= 0.99
        assert floater["valid_surface_protect_recall"] >= 0.9
        assert report.is_file()
        print("[meshprior-stage6-smoke] PASS")
        print(json.dumps({"rows": len(rows), "floater_recall": floater["floater_prune_recall"]}, indent=2))


if __name__ == "__main__":
    main()
