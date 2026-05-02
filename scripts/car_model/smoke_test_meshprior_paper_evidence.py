"""Smoke test for the MeshPrior Stage22 paper-evidence collector."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td) / "paper_evidence"
        report_path = Path(td) / "paper_report.md"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_collect_paper_evidence.py"),
                "--output_dir",
                str(output_dir),
                "--report_path",
                str(report_path),
            ],
            cwd=repo_root,
            check=True,
        )
        payload = json.loads((output_dir / "paper_evidence.json").read_text(encoding="utf-8"))
        assert payload["gate"] in {"PASS", "SOFT PASS"}, payload
        classes = payload["metric_classes"]
        assert classes["object_prior"], payload
        assert classes["scene_render_geometry_topology"], payload
        assert classes["failure_cases"], payload
        assert classes["missing_rows"], payload
        prune = [row for row in classes["scene_render_geometry_topology"] if row["row_id"] == "current_branch_prune_50_7000"][0]
        clean = [row for row in classes["scene_render_geometry_topology"] if row["row_id"] == "clean_origin_main_7000"][0]
        assert prune["render_psnr"] > clean["render_psnr"], payload
        assert report_path.is_file(), report_path
        print("[meshprior-paper-evidence-smoke] PASS")
        print(json.dumps({"gate": payload["gate"], "scene_rows": len(classes["scene_render_geometry_topology"])}, indent=2))


if __name__ == "__main__":
    main()
