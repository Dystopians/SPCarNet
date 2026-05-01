"""Smoke test for MeshPrior Stage 13 evaluation matrix."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "matrix"
        reports = Path(td) / "reports"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_run_experiment_matrix.py"),
                "--dry_run",
                "--smoke",
                "--group",
                "all",
                "--max_objects",
                "6",
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        matrix_path = out / "matrix_results.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        assert matrix["counts"]["total"] >= 1, matrix
        assert matrix["counts"]["missing"] >= 1, matrix
        assert matrix["counts"]["available"] >= 1, matrix
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_make_neurips_report.py"),
                "--matrix_results",
                str(matrix_path),
                "--output_dir",
                str(reports),
                "--report_dir",
                str(reports / "docs"),
            ],
            cwd=repo_root,
            check=True,
        )
        assert (reports / "object_table.csv").is_file()
        assert (reports / "synthetic_damage_table.csv").is_file()
        assert (reports / "scene_table.csv").is_file()
        assert (reports / "ablation_table.csv").is_file()
        assert (reports / "failure_cases.md").is_file()
        assert (reports / "docs/meshprior_neurips_main_report.md").is_file()
        print("[meshprior-stage13-smoke] PASS")
        print(json.dumps(matrix["counts"], indent=2))


if __name__ == "__main__":
    main()
