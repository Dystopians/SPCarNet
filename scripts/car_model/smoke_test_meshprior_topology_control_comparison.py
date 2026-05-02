"""Smoke test for Stage21.5 topology-control comparison collector."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    required = [
        repo_root / "outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/prune_25/model/results.json",
        repo_root / "outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/prune_50/model/results.json",
        repo_root / "outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/prune_66/model/results.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required Stage21.5 artifacts: " + ", ".join(missing))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "comparison"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_collect_topology_control_ablation.py"),
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "topology_control_ablation.json").read_text(encoding="utf-8"))
        assert report["gate"] == "PASS", report
        assert len(report["rows"]) == 5, report
        rows = {row["label"]: row for row in report["rows"]}
        assert rows["prune_50"]["triangles"] < rows["current_branch_7000"]["triangles"] * 0.6, report
        assert rows["prune_50"]["render_psnr"] > rows["clean_origin_main_7000"]["render_psnr"], report
        assert (out / "topology_control_ablation.csv").is_file()
        assert (out / "topology_control_ablation.md").is_file()
        print("[meshprior-topology-control-comparison-smoke] PASS")
        print(json.dumps({"gate": report["gate"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
