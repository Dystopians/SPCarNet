#!/usr/bin/env python3
"""Smoke test for SP-CarNet Stage 1.

Builds a tiny object index (limit=8), opens an ``SPCarObjectDataset`` over all
splits, draws a small batch via the provided collate, and asserts shape / dtype /
finiteness / canonical-transform invertibility.

Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.car_model import build_spcarnet_object_index as build_index  # noqa: E402
from ss3dm_prior.data.spcarnet_object_dataset import (  # noqa: E402
    CanonicalTransform,
    SPCarObjectDataset,
    collate_object_batch,
)


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if arr is None:
        return
    if not np.isfinite(arr).all():
        raise AssertionError(f"non-finite values in {name}: {arr.shape} {arr.dtype}")


def _assert_shape(name: str, arr: np.ndarray, expected_trailing: tuple[int, ...]) -> None:
    if arr.shape[1:] != expected_trailing:
        raise AssertionError(
            f"unexpected shape for {name}: got {arr.shape}, expected (*, {expected_trailing})"
        )


def main() -> int:
    cache_dir = REPO_ROOT / "outputs/ss3dm_prior_car/meshfleet_car_cache_v5"
    if not (cache_dir / "patch_index.jsonl").is_file():
        print(f"[smoke] FAIL: cache not found at {cache_dir}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "object_index_smoke.json"
        rc = build_index.main(
            [
                "--patch_cache_dir",
                str(cache_dir),
                "--output",
                str(out_path),
                "--canonicalization",
                "identity",
                "--limit",
                "8",
            ]
        )
        if rc != 0 or not out_path.is_file():
            print("[smoke] FAIL: index builder did not produce output", file=sys.stderr)
            return 1

        with out_path.open() as f:
            doc = json.load(f)
        if doc.get("version") != 1:
            print(f"[smoke] FAIL: unexpected index version {doc.get('version')}", file=sys.stderr)
            return 1
        n = doc["stats"]["n_objects"]
        if n <= 0:
            print(f"[smoke] FAIL: empty object index (stats={doc['stats']})", file=sys.stderr)
            return 1
        print(f"[smoke] index_ok: n_objects={n} stats={doc['stats']}", file=sys.stderr)

        ds = SPCarObjectDataset(
            out_path,
            splits=("train", "val", "test"),
            estimate_symmetry=False,
        )
        if len(ds) == 0:
            print("[smoke] FAIL: dataset empty after filtering by splits", file=sys.stderr)
            return 1
        print(f"[smoke] dataset_ok: len={len(ds)}", file=sys.stderr)

        single = ds[0]
        for key in (
            "object_id",
            "clean_points_object",
            "canonical_transform",
            "occupancy_query_points",
            "occupancy_query_labels",
            "free_space_query_points",
        ):
            if key not in single:
                print(f"[smoke] FAIL: missing key {key!r} in __getitem__ output", file=sys.stderr)
                return 1

        clean = single["clean_points_object"]
        if clean.dtype != np.float32:
            print(f"[smoke] FAIL: clean_points_object dtype={clean.dtype}", file=sys.stderr)
            return 1
        _assert_shape("clean_points_object", clean, (3,))
        if clean.shape[0] != 2048:
            print(f"[smoke] FAIL: clean_points_object shape[0]={clean.shape[0]}", file=sys.stderr)
            return 1
        _assert_finite("clean_points_object", clean)

        if single["partial_observed_points"] is None:
            print("[smoke] FAIL: partial_observed_points is None", file=sys.stderr)
            return 1
        if single["partial_observed_points"].shape != (768, 3):
            print(
                f"[smoke] FAIL: partial_observed_points shape={single['partial_observed_points'].shape}",
                file=sys.stderr,
            )
            return 1
        _assert_finite("partial_observed_points", single["partial_observed_points"])

        if single["occupancy_query_labels"] is not None:
            keep = ~single["occupancy_query_ignore"] if single["occupancy_query_ignore"] is not None else None
            kept = single["occupancy_query_labels"][keep] if keep is not None else single["occupancy_query_labels"]
            unique = set(int(x) for x in np.unique(kept).tolist())
            if not unique.issubset({0, 1}):
                print(f"[smoke] FAIL: occupancy_query_labels post-mask has values {unique}", file=sys.stderr)
                return 1

        # Canonical-transform round-trip on a synthetic record (PCA mode for non-trivial check).
        synth = np.random.RandomState(0).randn(64, 3).astype(np.float32) * 0.5
        identity = CanonicalTransform.from_dict(single["canonical_transform"])
        round_trip = identity.invert(identity.apply(synth))
        max_err = float(np.max(np.abs(round_trip - synth)))
        if max_err > 1e-5:
            print(f"[smoke] FAIL: identity round-trip error {max_err}", file=sys.stderr)
            return 1
        # Build a non-identity transform and verify round-trip too.
        rot = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tx = CanonicalTransform(type="pca", center=np.array([0.1, -0.2, 0.05], dtype=np.float32), scale=0.7, rotation=rot)
        round_trip_pca = tx.invert(tx.apply(synth))
        max_err_pca = float(np.max(np.abs(round_trip_pca - synth)))
        if max_err_pca > 1e-5:
            print(f"[smoke] FAIL: pca-style round-trip error {max_err_pca}", file=sys.stderr)
            return 1
        print(
            f"[smoke] transform_ok: identity_err={max_err:.2e} pca_err={max_err_pca:.2e}",
            file=sys.stderr,
        )

        # Batch via custom collate.
        items = [ds[i] for i in range(min(2, len(ds)))]
        batch = collate_object_batch(items)
        for key, expected_shape in (
            ("clean_points_object", (len(items), 2048, 3)),
            ("partial_observed_points", (len(items), 768, 3)),
            ("occupancy_query_points", (len(items), 1280, 3)),
            ("free_space_query_points", (len(items), 512, 3)),
        ):
            arr = batch.get(key)
            if not isinstance(arr, np.ndarray):
                print(f"[smoke] FAIL: batch[{key!r}] is not ndarray (got {type(arr)})", file=sys.stderr)
                return 1
            if arr.shape != expected_shape:
                print(f"[smoke] FAIL: batch[{key!r}] shape={arr.shape} expected={expected_shape}", file=sys.stderr)
                return 1
            _assert_finite(f"batch[{key}]", arr)
        print(f"[smoke] batch_ok: keys={sorted(batch.keys())}", file=sys.stderr)

    print("[smoke] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
