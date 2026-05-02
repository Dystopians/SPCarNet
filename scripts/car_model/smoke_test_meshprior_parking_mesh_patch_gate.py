"""Smoke test for parking mesh patch no-op/protect gate."""

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
        out = Path(td) / "patch_gate"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_gate_parking_mesh_patches.py"),
                "--mesh_patch_summary",
                str(summary),
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "patch_gate_report.json").read_text(encoding="utf-8"))
        assert report["patch_count"] == 8, report
        assert report["protect_ready_count"] == 8, report
        assert report["deferred_count"] == 0, report
        assert report["failed_count"] == 0, report
        assert all(Path(row["rollback_snapshot"]).is_file() for row in report["results"])
        assert (out / "patch_gate_results.csv").is_file()
        assert (out / "patch_gate_report.md").is_file()
        print("[meshprior-parking-mesh-patch-gate-smoke] PASS")
        print(json.dumps({"protect_ready": report["protect_ready_count"]}, indent=2))


if __name__ == "__main__":
    main()
