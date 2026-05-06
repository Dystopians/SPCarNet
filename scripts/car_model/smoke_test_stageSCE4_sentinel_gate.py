#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sentinel_parent_pareto_gate import (  # noqa: E402
    SentinelParentParetoGateConfig,
    evaluate_sentinel_parent_pareto_gate,
)
from utils.sparse_depth_regression import build_sparse_depth_regression_table  # noqa: E402


def _table(candidate):
    return build_sparse_depth_regression_table(
        {
            "image_name": np.array(["a", "a", "b"], dtype=object),
            "image_key": np.array(["a", "a", "b"], dtype=object),
            "point3D_id": np.array([1, 2, 3], dtype=np.int64),
            "px": np.array([0, 1, 0], dtype=np.int64),
            "py": np.array([0, 0, 0], dtype=np.int64),
            "width": np.array([10, 10, 10], dtype=np.int64),
            "height": np.array([10, 10, 10], dtype=np.int64),
            "gt_depth": np.array([10.0, 10.0, 20.0], dtype=np.float64),
            "parent_pred_depth": np.array([10.0, 11.0, 19.0], dtype=np.float64),
            "candidate_pred_depth": np.asarray(candidate, dtype=np.float64),
        }
    )


def main() -> int:
    cfg = SentinelParentParetoGateConfig(
        tolerance_absrel=0.0,
        tolerance_mae=0.0,
        worst_view_regression_count_threshold=10,
        cluster_delta_absrel_threshold=999.0,
        cluster_weight_threshold=999,
    )
    passed = evaluate_sentinel_parent_pareto_gate(_table([10.0, 10.5, 19.5]), cfg)
    assert passed["pass"] is True
    failed = evaluate_sentinel_parent_pareto_gate(_table([10.0, 13.0, 17.0]), cfg)
    assert failed["pass"] is False
    assert failed["checks"]["mean_absrel_nonregression"] is False
    print("SCE4 sentinel parent-pareto gate smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
