#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sparse_depth_regression import (
    SparseDepthRegressionConfig,
    build_sparse_depth_regression_table,
    cluster_summary_rows,
    per_view_summary_rows,
    point_summary_rows,
    summarize_sparse_depth_regressions,
    write_sparse_depth_regression_outputs,
)


def main() -> int:
    payload = {
        "image_name": np.array(["a.png", "a.png", "a.png", "b.png", "b.png"], dtype=object),
        "image_key": np.array(["a", "a", "a", "b", "b"], dtype=object),
        "point3D_id": np.array([10, 11, 12, 10, 13], dtype=np.int64),
        "px": np.array([10, 20, 30, 5, 60], dtype=np.int64),
        "py": np.array([10, 20, 30, 5, 60], dtype=np.int64),
        "width": np.array([100, 100, 100, 100, 100], dtype=np.int64),
        "height": np.array([100, 100, 100, 100, 100], dtype=np.int64),
        "gt_depth": np.array([10.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float64),
        "parent_pred_depth": np.array([10.0, 11.0, 9.0, 20.0, 18.0], dtype=np.float64),
        "candidate_pred_depth": np.array([10.0, 12.0, 9.5, 25.0, 0.0], dtype=np.float64),
    }
    cfg = SparseDepthRegressionConfig(margin_abs=0.25, margin_rel=0.01, gate_top_fraction=0.5, cluster_grid_size=32)
    table = build_sparse_depth_regression_table(payload, cfg)

    assert table["delta_abs_error"][0] == 0.0
    assert np.isclose(table["delta_abs_error"][1], 1.0)
    assert np.isclose(table["delta_abs_rel"][1], 0.1)
    assert bool(table["regressed_abs"][1])
    assert bool(table["regressed_rel"][1])
    assert bool(table["parent_valid"][4])
    assert not bool(table["candidate_valid"][4])
    assert bool(table["regressed_abs"][4])
    assert bool(table["gate_critical"][4])

    summary = summarize_sparse_depth_regressions(table)
    assert summary["global"]["count"] == 5
    assert summary["global"]["candidate_invalid_count"] == 1
    assert summary["global"]["gate_critical_count"] >= 2

    per_view = per_view_summary_rows(table)
    assert {row["image_key"] for row in per_view} == {"a", "b"}
    points = point_summary_rows(table)
    assert {int(row["point3D_id"]) for row in points} == {10, 11, 12, 13}
    clusters = cluster_summary_rows(table)
    assert len(clusters) >= 1

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_sparse_depth_regression_outputs(
            output_dir=out,
            table=table,
            cfg=cfg,
            manifest={
                "source_path": "synthetic",
                "split": "test",
                "parent_model_path": "parent",
                "parent_iteration": 1,
                "candidate_model_path": "candidate",
                "candidate_iteration": 2,
            },
        )
        for name in (
            "correspondence_regressions.csv",
            "correspondence_regressions.npz",
            "per_view_regression_summary.csv",
            "point_regression_summary.csv",
            "cluster_regression_summary.csv",
            "sentinel_candidate_mask.npz",
            "regression_report.md",
            "regression_summary.json",
        ):
            assert (out / name).is_file(), name

    print("SCE1 sparse-depth regression smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
