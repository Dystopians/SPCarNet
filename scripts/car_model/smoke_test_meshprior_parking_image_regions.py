"""Smoke test for parking image/COLMAP ROI mining."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scene_root = repo_root / "outputs/carnet/meshprior/parking_phone_tiny/dataset_view"
    if not (scene_root / "sparse/0/images.bin").is_file():
        raise SystemExit("parking_phone_tiny dataset_view is missing; run meshprior_prepare_parking_scene.py first")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "regions"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_mine_parking_image_regions.py"),
                "--scene_root",
                str(scene_root),
                "--output_dir",
                str(out),
                "--max_images",
                "12",
                "--min_area_px",
                "1000",
            ],
            cwd=repo_root,
            check=True,
        )
        payload = json.loads((out / "image_regions.json").read_text(encoding="utf-8"))
        assert payload["image_count_considered"] == 12, payload
        assert payload["region_count"] >= 1, payload
        assert (out / "image_regions_summary.csv").is_file()
        assert (out / "image_region_mining_report.md").is_file()
        print("[meshprior-parking-image-regions-smoke] PASS")
        print(json.dumps({"regions": payload["region_count"], "eligible": payload["eligible_count"]}, indent=2))


if __name__ == "__main__":
    main()
