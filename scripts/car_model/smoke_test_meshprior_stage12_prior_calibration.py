"""Smoke test for MeshPrior Stage 12 prior calibration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "calibration"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_calibrate_prior.py"),
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        payload = json.loads((out / "calibration_metrics.json").read_text(encoding="utf-8"))
        rows = {row["profile"]: row for row in payload["rows"]}
        assert rows["surface_support_v1"]["accepted_by_profile"], payload
        assert rows["surface_support_v1"]["snapped_valid_surface_protect_recall"] >= rows["surface_support_v1"]["baseline_valid_surface_protect_recall"], payload
        assert rows["surface_support_v1"]["snapped_valid_surface_protect_recall"] > rows["none"]["snapped_valid_surface_protect_recall"], payload
        assert rows["surface_support_v1"]["surface_distance_delta_mean"] > 0.0, payload
        assert payload["free_space_safe"], payload
        print("[meshprior-stage12-smoke] PASS")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
