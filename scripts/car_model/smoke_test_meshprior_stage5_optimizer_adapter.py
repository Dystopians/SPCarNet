"""Smoke test for MeshPrior Stage 5 optimizer adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.optimizer_adapter import (
    combine_scores,
    load_triangle_scores,
    normalize_scores_per_region,
    prism_present,
)


def main() -> None:
    dtype = [
        ("region_id", "U64"),
        ("face_index", "i8"),
        ("protect", "f4"),
        ("prune", "f4"),
        ("support", "f4"),
        ("violation", "f4"),
    ]
    scores = np.asarray(
        [
            ("r0", 0, 0.2, 0.1, 0.8, 0.2),
            ("r0", 1, 0.8, 0.9, 0.2, 0.8),
            ("r1", 2, 0.4, 0.4, 0.5, 0.5),
            ("r1", 3, 0.4, 0.4, 0.5, 0.5),
        ],
        dtype=dtype,
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "triangle_scores.npz"
        out = tmp / "out"
        np.savez(src, scores=scores)
        loaded = load_triangle_scores(src)
        norm = normalize_scores_per_region(loaded)
        assert np.isfinite(norm["protect"]).all()
        assert np.isfinite(norm["prune"]).all()
        base = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        mp = np.asarray([1.0, 0.5, 2.0], dtype=np.float32)
        combined = combine_scores(base, mp, weight=0.25)
        assert np.max(combined - base) <= 0.250001
        subprocess.run(
            [
                "python",
                str(REPO_ROOT / "scripts/car_model/meshprior_export_optimizer_scores.py"),
                "--triangle_scores",
                str(src),
                "--output_dir",
                str(out),
                "--format",
                "both",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        assert (out / "meshprior_scores.npz").is_file()
        assert (out / "meshprior_prism_scores.json").is_file()
        reloaded = load_triangle_scores(out / "meshprior_scores.npz")
        assert len(reloaded) == len(scores)
        summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
        assert summary["prism_present"] is True
        print("[meshprior-stage5-smoke] PASS")
        print(json.dumps({"rows": len(reloaded), "prism_present": prism_present(REPO_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
