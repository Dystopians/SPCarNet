#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sparse_depth_parent_rollback import compute_sparse_depth_parent_rollback_loss  # noqa: E402


def _cache() -> dict:
    n = 5
    return {
        "view": {
            "px": np.asarray([0, 1, 2, 3, 1], dtype=np.int64),
            "py": np.asarray([0, 0, 0, 0, 1], dtype=np.int64),
            "width": np.full(n, 4, dtype=np.int64),
            "height": np.full(n, 4, dtype=np.int64),
            "gt_depth": np.ones(n, dtype=np.float64),
            "parent_abs_error": np.zeros(n, dtype=np.float64),
            "parent_abs_rel": np.zeros(n, dtype=np.float64),
            "sentinel_weight": np.ones(n, dtype=np.float64),
            "cluster_id": np.asarray([0, 0, 1, 1, 2], dtype=np.int64),
            "is_regressed_candidate": np.ones(n, dtype=bool),
            "candidate_delta_abs_error": np.asarray([0.1, 0.2, 3.0, 0.2, 0.1], dtype=np.float64),
            "candidate_delta_abs_rel": np.asarray([0.1, 0.2, 3.0, 0.2, 0.1], dtype=np.float64),
        }
    }


def _loss(aggregation: str, *, pixel_radius: int = 0, patch_reduce: str = "center") -> dict:
    depth = torch.ones((1, 4, 4), dtype=torch.float32)
    depth[0, 0, 2] = 4.0
    if pixel_radius > 0:
        depth[0, 1, 2] = 6.0
    return compute_sparse_depth_parent_rollback_loss(
        current_depth=depth,
        cache_by_image_key=_cache(),
        image_key="view",
        lam=1.0,
        margin_abs=0.0,
        margin_rel=0.0,
        huber_delta=0.05,
        loss_space="mae",
        max_points_per_view=0,
        aggregation=aggregation,
        cvar_fraction=0.2,
        cvar_min_points=1,
        pixel_radius=pixel_radius,
        patch_reduce=patch_reduce,
    )


def main() -> int:
    mean = _loss("mean")
    cvar = _loss("cvar")
    cluster = _loss("cluster_cvar")
    patch = _loss("cvar", pixel_radius=1, patch_reduce="max_violation")
    assert float(cvar["loss_pure"]) > float(mean["loss_pure"]), (mean, cvar)
    assert int(cvar["tail_points"]) == 1, cvar
    assert int(cluster["tail_clusters"]) == 1, cluster
    assert float(patch["max_violation_abs"]) >= float(cvar["max_violation_abs"]), (patch, cvar)
    print("SCE21 tail-risk rollback smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
