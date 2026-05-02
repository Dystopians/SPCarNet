"""Smoke test for the Stage 18 topology-budget comparison collector."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    required = [
        repo_root / "outputs/carnet/meshprior/parking_phone_tiny/origin_main_2000iter/model/results.json",
        repo_root / "outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/results.json",
        repo_root / "outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model/results.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required metric artifacts: " + ", ".join(missing))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "topology_budget"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_collect_topology_budget_comparison.py"),
                "--output_dir",
                str(out),
            ],
            cwd=repo_root,
            check=True,
        )
        report = json.loads((out / "topology_budget_comparison.json").read_text(encoding="utf-8"))
        assert report["gate"] == "PASS", report
        assert report["decision"] == "QUALITY_GAIN_NOT_TOPOLOGY_NORMALIZED", report
        assert len(report["rows"]) == 3, report
        rows = {row["label"]: row for row in report["rows"]}
        assert rows["stage17_meshprior_2000iter"]["triangles"] > rows["origin_main_2000iter"]["triangles"] * 5, report
        assert rows["stage17_meshprior_2000iter"]["render_psnr"] > rows["current_branch_2000iter"]["render_psnr"], report
        assert (out / "topology_budget_comparison.csv").is_file()
        assert (out / "topology_budget_comparison.md").is_file()
        print("[meshprior-topology-budget-smoke] PASS")
        print(json.dumps({"rows": len(report["rows"]), "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
