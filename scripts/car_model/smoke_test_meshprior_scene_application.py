"""Smoke test for safe accepted-proposal application."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pipeline = tmp / "pipeline"
        applied = tmp / "applied"
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
                str(pipeline),
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
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_apply_accepted_proposals.py"),
                "--proposals",
                str(pipeline / "proposals/proposals.json"),
                "--gate_report",
                str(pipeline / "scene_gate/gate_report.json"),
                "--output_dir",
                str(applied),
                "--write_recovery_plan",
                "--scene_source",
                "synthetic",
            ],
            cwd=repo_root,
            check=True,
        )
        manifest = json.loads((applied / "application_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "PASS", manifest
        assert manifest["applied_count"] >= 1, manifest
        assert Path(manifest["applied_mesh"]).is_file(), manifest
        assert (applied / "recovery_commands.sh").is_file()
        print("[meshprior-scene-application-smoke] PASS")
        print(json.dumps({"applied_count": manifest["applied_count"], "final": manifest["final"]}, indent=2))


if __name__ == "__main__":
    main()
