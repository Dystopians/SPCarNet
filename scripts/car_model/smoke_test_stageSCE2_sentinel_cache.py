#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sparse_depth_sentinel_cache import (  # noqa: E402
    SentinelCacheConfig,
    build_sparse_depth_sentinel_cache,
    write_sparse_depth_sentinel_cache,
)


def _table():
    return {
        "image_name": np.array(["tr_a.png", "tr_a.png", "tr_b.png", "tr_b.png"], dtype=object),
        "image_key": np.array(["tr_a", "tr_a", "tr_b", "tr_b"], dtype=object),
        "point3D_id": np.array([1, 2, 3, 4], dtype=np.int64),
        "px": np.array([10, 20, 30, 40], dtype=np.int64),
        "py": np.array([10, 20, 30, 40], dtype=np.int64),
        "gt_depth": np.array([10.0, 10.0, 20.0, 20.0], dtype=np.float64),
        "parent_pred_depth": np.array([10.0, 11.0, 20.0, 19.0], dtype=np.float64),
        "candidate_pred_depth": np.array([10.0, 13.0, 21.0, 18.5], dtype=np.float64),
        "cluster_id": np.array([0, 0, 1, 1], dtype=np.int64),
    }


def main() -> int:
    cfg = SentinelCacheConfig(split="train", seed=11, cluster_balance=True, hard_regression_weight=3.0)
    manifest = {"source_path": "synthetic", "parent_model_path": "parent", "parent_iteration": 1}
    cache1 = build_sparse_depth_sentinel_cache(table=_table(), manifest=manifest, cfg=cfg)
    cache2 = build_sparse_depth_sentinel_cache(table=_table(), manifest=manifest, cfg=cfg)
    assert cache1["manifest"]["no_test_leakage"] is True
    assert cache1["manifest"]["split"] == "train"
    assert np.array_equal(cache1["arrays"]["sentinel_weight"], cache2["arrays"]["sentinel_weight"])
    assert int(np.count_nonzero(cache1["arrays"]["is_regressed_candidate"])) == 3
    assert not np.allclose(cache1["arrays"]["sentinel_weight"], np.ones((4,), dtype=np.float64))

    try:
        build_sparse_depth_sentinel_cache(
            table=_table(),
            manifest=manifest,
            cfg=SentinelCacheConfig(split="test"),
        )
        raise AssertionError("test split cache construction should fail")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sentinel_cache.npz"
        write_sparse_depth_sentinel_cache(output=out, cache=cache1, cfg=cfg)
        for name in ("sentinel_cache.npz", "sentinel_manifest.json", "sentinel_view_summary.csv", "sentinel_report.md"):
            assert (out.parent / name).is_file(), name

    print("SCE2 sparse-depth sentinel cache smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
