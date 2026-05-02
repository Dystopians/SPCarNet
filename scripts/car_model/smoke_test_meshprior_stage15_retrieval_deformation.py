"""Smoke test for MeshPrior Stage 15 retrieval-deformation fallback."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bank = tmp / "anchors.npz"
        out = tmp / "eval"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_build_anchor_bank.py"),
                "--output",
                str(bank),
                "--synthetic_smoke",
                "--points_per_anchor",
                "64",
                "--max_anchors",
                "1",
            ],
            cwd=repo_root,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/car_model/meshprior_eval_retrieval_deformation.py"),
                "--anchor_bank",
                str(bank),
                "--output_dir",
                str(out),
                "--damage_types",
                "local_hole",
                "floater",
                "vertex_noise",
            ],
            cwd=repo_root,
            check=True,
        )
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        rows = metrics["inference_time_metrics"]
        methods = {r["method"] for r in rows}
        assert {"stage3_posterior_proxy", "retrieval_only", "retrieval_deform"} <= methods, methods
        retrieval = [r for r in rows if r["method"] == "retrieval_only"][0]
        proposal_types = set(retrieval["proposal_types"].split(","))
        assert {"protect", "prune", "fill_candidate", "uncertainty"} <= proposal_types, proposal_types
        assert metrics["recommendation"] in {
            "PIVOT_TO_RETRIEVAL_DEFORMATION",
            "KEEP_RETRIEVAL_ONLY_KILL_DEFORMATION",
            "KEEP_AS_BASELINE",
        }
        print("[meshprior-stage15-smoke] PASS")
        print(json.dumps({"rows": len(rows), "recommendation": metrics["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
