"""Smoke test for parking cluster proposal scoring."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json"
    if not source.is_file():
        raise SystemExit("consolidated parking regions missing")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "proposals"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_score_parking_clusters.py"),
                "--consolidated_regions",
                str(source),
                "--output_dir",
                str(out),
                "--max_clusters",
                "3",
            ],
            cwd=repo_root,
            check=True,
        )
        payload = json.loads((out / "proposals.json").read_text(encoding="utf-8"))
        assert payload["cluster_count"] == 3, payload
        assert payload["proposal_count"] == 15, payload
        assert set(payload["proposal_types"]) == {"protect", "prune", "snap_candidate", "fill_candidate", "uncertainty"}
        assert all(p["metadata"]["metadata_only"] for p in payload["proposals"])
        assert (out / "proposal_scores.csv").is_file()
        assert (out / "proposal_report.md").is_file()
        print("[meshprior-parking-cluster-scoring-smoke] PASS")
        print(json.dumps({"clusters": payload["cluster_count"], "proposals": payload["proposal_count"]}, indent=2))


if __name__ == "__main__":
    main()
