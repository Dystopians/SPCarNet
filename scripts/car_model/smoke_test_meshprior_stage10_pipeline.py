"""Smoke test for MeshPrior Stage 10 dry-run pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "pipeline"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_run_pipeline.py"),
                "--scene_source",
                "synthetic",
                "--scene_model",
                "synthetic",
                "--posterior_checkpoint",
                "",
                "--output_dir",
                str(out),
                "--proposal_types",
                "protect",
                "prune",
                "fill",
                "--mode",
                "dry_run",
                "--require_gate_pass",
            ],
            cwd=repo_root,
            check=True,
        )
        status = json.loads((out / "pipeline_status.json").read_text(encoding="utf-8"))
        gate = json.loads((out / "scene_gate/gate_report.json").read_text(encoding="utf-8"))
        accepted = json.loads((out / "accepted_proposals.json").read_text(encoding="utf-8"))
        assert status["status"] == "PASS", status
        assert gate["accepted_count"] >= 1, gate
        assert accepted["proposals"], accepted
        assert (out / "pipeline_report.md").is_file()
        assert (out / "regions.json").is_file()
        assert (out / "posterior/posterior_summary.json").is_file()
        print("[meshprior-stage10-smoke] PASS")
        print(json.dumps({"accepted_count": gate["accepted_count"], "rejected_count": gate["rejected_count"]}, indent=2))


if __name__ == "__main__":
    main()
