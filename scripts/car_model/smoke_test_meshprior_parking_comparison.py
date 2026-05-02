"""Smoke test for parking comparison summary collection."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    baseline = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model"
    recovery = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup"
    if not (baseline / "results.json").is_file() or not (recovery / "results.json").is_file():
        raise SystemExit("render metrics are missing")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "comparison_summary"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_collect_parking_comparison.py"),
                "--engineering_baseline_model",
                str(baseline),
                "--recovery_model",
                str(recovery),
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "parking_comparison_summary.json").read_text(encoding="utf-8"))
        assert len(report["rows"]) == 2, report
        assert report["paper_baseline_status"] == "MISSING", report
        assert report["decision"] == "SOFT_PASS_STABILITY_ONLY", report
        assert report["rows"][1]["triangles"] < report["rows"][0]["triangles"], report
        assert (out / "parking_comparison_summary.csv").is_file()
        assert (out / "parking_comparison_summary.md").is_file()
        print("[meshprior-parking-comparison-smoke] PASS")
        print(json.dumps({"rows": len(report["rows"]), "paper_baseline_status": report["paper_baseline_status"]}, indent=2))


if __name__ == "__main__":
    main()
