"""Smoke test for parking metadata proposal gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposals.json"
    if not source.is_file():
        raise SystemExit("parking cluster proposals missing")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "metadata_gate"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_gate_parking_metadata_proposals.py"),
                "--proposals",
                str(source),
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "metadata_gate_report.json").read_text(encoding="utf-8"))
        counts = report["decision_counts"]
        assert report["geometry_edited"] is False, report
        assert sum(counts.values()) == 45, counts
        assert counts["candidate_extract"] > 0, counts
        assert counts["deferred"] >= 9, counts
        assert len(report["mesh_extraction_targets"]) > 0, report
        assert all("prune" not in t["proposal_types"] for t in report["mesh_extraction_targets"])
        assert (out / "action_plan.json").is_file()
        assert (out / "metadata_gate_results.csv").is_file()
        assert (out / "metadata_gate_report.md").is_file()
        print("[meshprior-parking-metadata-gate-smoke] PASS")
        print(json.dumps({"counts": counts, "targets": len(report["mesh_extraction_targets"])}, indent=2))


if __name__ == "__main__":
    main()
