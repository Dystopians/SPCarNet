"""Smoke test for copied-patch parking proposal tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    summary = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json"
    if not summary.is_file():
        raise SystemExit("parking mesh patch summary missing")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "patch_proposal_tests"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_test_parking_patch_proposals.py"),
                "--mesh_patch_summary",
                str(summary),
                "--output_dir",
                str(out),
                "--max_patches",
                "2",
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "patch_proposal_test_report.json").read_text(encoding="utf-8"))
        assert report["patches_tested"] == 2, report
        assert report["proposal_tests"] == 6, report
        assert report["counts"]["cleanup_accepted"] == 2, report
        assert report["counts"]["floater_rejected"] == 2, report
        assert report["counts"]["protect_noop_rejected"] == 2, report
        assert report["source_model_edited"] is False, report
        assert (out / "patch_proposal_test_results.csv").is_file()
        assert (out / "patch_proposal_test_report.md").is_file()
        print("[meshprior-parking-patch-proposals-smoke] PASS")
        print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
