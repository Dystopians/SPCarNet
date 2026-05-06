#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sparse_depth_parent_rollback import (  # noqa: E402
    compute_sparse_depth_parent_rollback_loss,
    load_sparse_depth_parent_rollback_cache,
)


def _cache_by_key(weight_scale: float = 1.0):
    return {
        "cam": {
            "px": np.array([0, 1, 2], dtype=np.int64),
            "py": np.array([0, 0, 0], dtype=np.int64),
            "gt_depth": np.array([10.0, 10.0, 10.0], dtype=np.float64),
            "parent_abs_error": np.array([0.0, 1.0, 1.0], dtype=np.float64),
            "parent_abs_rel": np.array([0.0, 0.1, 0.1], dtype=np.float64),
            "sentinel_weight": np.array([1.0, weight_scale, 1.0], dtype=np.float64),
            "cluster_id": np.array([0, 1, 2], dtype=np.int64),
            "is_regressed_candidate": np.array([False, True, False], dtype=bool),
            "candidate_delta_abs_error": np.array([0.0, 2.0, 0.1], dtype=np.float64),
            "candidate_delta_abs_rel": np.array([0.0, 0.2, 0.01], dtype=np.float64),
        }
    }


def _loss_value(res) -> float:
    loss = res["loss_pure"]
    return float(loss.detach().item() if torch.is_tensor(loss) else loss)


def main() -> int:
    parent_equal = torch.tensor([[10.0, 11.0, 9.0]], dtype=torch.float32)
    improved = torch.tensor([[10.0, 10.5, 9.5]], dtype=torch.float32)
    worse = torch.tensor([[10.0, 13.0, 7.0]], dtype=torch.float32)

    common = dict(
        cache_by_image_key=_cache_by_key(),
        image_key="cam",
        lam=1.0,
        margin_abs=0.0,
        margin_rel=0.0,
        huber_delta=0.05,
        loss_space="absrel",
        max_points_per_view=0,
    )
    assert _loss_value(compute_sparse_depth_parent_rollback_loss(current_depth=parent_equal, **common)) == 0.0
    assert _loss_value(compute_sparse_depth_parent_rollback_loss(current_depth=improved, **common)) == 0.0
    worse_res = compute_sparse_depth_parent_rollback_loss(current_depth=worse, **common)
    assert _loss_value(worse_res) > 0.0
    assert worse_res["active_points"] == 2

    unweighted = _loss_value(compute_sparse_depth_parent_rollback_loss(current_depth=worse, **common))
    weighted_common = dict(common)
    weighted_common["cache_by_image_key"] = _cache_by_key(weight_scale=10.0)
    weighted = _loss_value(compute_sparse_depth_parent_rollback_loss(current_depth=worse, **weighted_common))
    assert weighted > unweighted

    combined_small_beta = _loss_value(
        compute_sparse_depth_parent_rollback_loss(current_depth=worse, **{**common, "loss_space": "combined", "combined_mae_beta": 0.02})
    )
    combined_large_beta = _loss_value(
        compute_sparse_depth_parent_rollback_loss(current_depth=worse, **{**common, "loss_space": "combined", "combined_mae_beta": 1.0})
    )
    assert combined_large_beta > combined_small_beta

    regressed_only = compute_sparse_depth_parent_rollback_loss(current_depth=worse, **{**common, "regressed_only": True})
    assert regressed_only["total_points"] == 1
    assert regressed_only["active_points"] == 1

    top_cluster = compute_sparse_depth_parent_rollback_loss(current_depth=worse, **{**common, "cluster_top_k": 1})
    assert top_cluster["total_points"] == 1
    assert top_cluster["active_points"] == 1

    missing = compute_sparse_depth_parent_rollback_loss(current_depth=worse, image_key="missing", **{k: v for k, v in common.items() if k != "image_key"})
    assert missing["reason"] == "missing_camera_key"
    assert _loss_value(missing) == 0.0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.npz"
        manifest = {"split": "test"}
        np.savez_compressed(
            path,
            manifest_json=np.asarray(json.dumps(manifest), dtype=object),
            image_key=np.array(["cam"], dtype=object),
            px=np.array([0], dtype=np.int64),
            py=np.array([0], dtype=np.int64),
            gt_depth=np.array([1.0], dtype=np.float64),
            parent_abs_error=np.array([0.0], dtype=np.float64),
            parent_abs_rel=np.array([0.0], dtype=np.float64),
            sentinel_weight=np.array([1.0], dtype=np.float64),
        )
        try:
            load_sparse_depth_parent_rollback_cache(path)
            raise AssertionError("test split cache should be rejected")
        except RuntimeError:
            pass

    print("SCE3 sparse-depth parent rollback loss smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
