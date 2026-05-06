#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.evidence_conflict_graph import build_evidence_conflict_graph, write_ecg_outputs  # noqa: E402


def main() -> int:
    payload = {
        "image_key": np.asarray(["v0", "v0", "v1", "v1"], dtype=object),
        "point3D_id": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "px": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "py": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "gt_depth": np.asarray([10.0, 10.0, 8.0, 5.0], dtype=np.float64),
        "parent_abs_error": np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float64),
        "candidate_abs_error": np.asarray([0.5, 4.0, 3.0, 0.4], dtype=np.float64),
        "parent_abs_rel": np.asarray([0.1, 0.1, 0.0625, 0.1], dtype=np.float64),
        "candidate_abs_rel": np.asarray([0.05, 0.4, 0.375, 0.08], dtype=np.float64),
        "parent_valid": np.asarray([True, True, True, True], dtype=bool),
        "candidate_valid": np.asarray([True, True, True, True], dtype=bool),
        "parent_rgb_residual": np.asarray([0.4, 0.5, 0.2, 0.3], dtype=np.float64),
        "candidate_rgb_residual": np.asarray([0.2, 0.1, 0.4, 0.1], dtype=np.float64),
        "gate_critical": np.asarray([False, True, True, False], dtype=bool),
        "cluster_id": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "depth_bin": np.asarray(["near_q1", "far_q4", "far_q4", "mid_q2"], dtype=object),
    }
    graph = build_evidence_conflict_graph(payload, source="synthetic", split="train")
    top = graph["cluster_summary"][0]
    assert int(top["cluster_id"]) == 1, top
    assert top["suggested_action"] == "ROLLBACK_ONLY", top
    protected = [r for r in graph["cluster_summary"] if int(r["cluster_id"]) == 0][0]
    assert protected["suggested_action"] != "DELETE_OR_COLLAPSE", protected
    with tempfile.TemporaryDirectory() as td:
        write_ecg_outputs(graph, Path(td))
        assert (Path(td) / "evidence_conflict_graph.json").is_file()
        assert (Path(td) / "ecg_cluster_summary.csv").is_file()
    print("SCE12 evidence conflict graph smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

