"""Smoke test for parking ROI multi-view consolidation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    image_regions = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions.json"
    if not image_regions.is_file():
        raise SystemExit("image region mining output missing; run meshprior_mine_parking_image_regions.py first")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clusters"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_cluster_parking_regions.py"),
                "--image_regions",
                str(image_regions),
                "--output_dir",
                str(out),
                "--max_clusters",
                "20",
            ],
            cwd=repo_root,
            check=True,
        )
        payload = json.loads((out / "consolidated_regions.json").read_text(encoding="utf-8"))
        assert payload["eligible_input_count"] >= 1, payload
        assert payload["cluster_count"] >= 1, payload
        assert (out / "consolidated_regions_summary.csv").is_file()
        assert (out / "consolidation_report.md").is_file()
        print("[meshprior-parking-region-consolidation-smoke] PASS")
        print(json.dumps({"clusters": payload["cluster_count"], "eligible": payload["eligible_cluster_count"]}, indent=2))


if __name__ == "__main__":
    main()
