from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SentinelCacheConfig:
    split: str = "train"
    seed: int = 7
    max_points_per_view: int = 500
    cluster_grid_size: int = 64
    hard_regression_weight: float = 2.0
    cluster_balance: bool = True


def _arr(table: Mapping[str, Any], key: str, dtype=None) -> np.ndarray:
    return np.asarray(table.get(key, []), dtype=dtype).reshape(-1)


def _str_arr(table: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(table.get(key, []), dtype=object).reshape(-1)


def _finite_abs_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = np.full(gt.shape, np.inf, dtype=np.float64)
    valid = np.isfinite(pred) & (pred > 1e-6) & np.isfinite(gt) & (gt > 1e-6)
    out[valid] = np.abs(pred[valid] - gt[valid])
    return out


def _grid_cluster_ids(image_key: np.ndarray, px: np.ndarray, py: np.ndarray, grid_size: int) -> np.ndarray:
    mapping: dict[tuple[str, int, int], int] = {}
    out = np.zeros(px.shape, dtype=np.int64)
    next_id = 0
    size = max(1, int(grid_size))
    for i in range(px.shape[0]):
        key = (str(image_key[i]), int(px[i]) // size, int(py[i]) // size)
        cid = mapping.get(key)
        if cid is None:
            cid = next_id
            mapping[key] = cid
            next_id += 1
        out[i] = cid
    return out


def build_sparse_depth_sentinel_cache(
    *,
    table: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cfg: SentinelCacheConfig,
) -> dict[str, Any]:
    split = str(cfg.split).lower()
    if split == "test":
        raise ValueError("SCE2 sentinel caches must not be built from test split.")

    image_name = _str_arr(table, "image_name")
    image_key = _str_arr(table, "image_key")
    point_id = _arr(table, "point3D_id", np.int64)
    px = _arr(table, "px", np.int64)
    py = _arr(table, "py", np.int64)
    gt = _arr(table, "gt_depth", np.float64)
    parent_pred = _arr(table, "parent_pred_depth", np.float64)
    candidate_pred = _arr(table, "candidate_pred_depth", np.float64)
    if candidate_pred.shape[0] == 0:
        candidate_pred = np.full(gt.shape, np.nan, dtype=np.float64)
    n = int(gt.shape[0])
    for key, value in {
        "image_name": image_name,
        "image_key": image_key,
        "point3D_id": point_id,
        "px": px,
        "py": py,
        "parent_pred_depth": parent_pred,
        "candidate_pred_depth": candidate_pred,
    }.items():
        if int(value.shape[0]) != n:
            raise ValueError(f"sentinel input length mismatch for {key}: {value.shape[0]} vs {n}")

    parent_abs = _finite_abs_error(parent_pred, gt)
    parent_rel = parent_abs / np.clip(gt, 1e-6, None)
    candidate_abs = _finite_abs_error(candidate_pred, gt)
    candidate_rel = candidate_abs / np.clip(gt, 1e-6, None)
    keep_parent_valid = np.isfinite(parent_abs) & np.isfinite(parent_rel)
    image_name = image_name[keep_parent_valid]
    image_key = image_key[keep_parent_valid]
    point_id = point_id[keep_parent_valid]
    px = px[keep_parent_valid]
    py = py[keep_parent_valid]
    gt = gt[keep_parent_valid]
    parent_pred = parent_pred[keep_parent_valid]
    candidate_pred = candidate_pred[keep_parent_valid]
    parent_abs = parent_abs[keep_parent_valid]
    parent_rel = parent_rel[keep_parent_valid]
    candidate_abs = candidate_abs[keep_parent_valid]
    candidate_rel = candidate_rel[keep_parent_valid]
    n = int(gt.shape[0])
    has_candidate = np.any(np.isfinite(candidate_pred))
    delta_abs = np.full(gt.shape, np.inf, dtype=np.float64)
    delta_rel = np.full(gt.shape, np.inf, dtype=np.float64)
    finite_abs_delta = np.isfinite(candidate_abs) & np.isfinite(parent_abs)
    delta_abs[finite_abs_delta] = candidate_abs[finite_abs_delta] - parent_abs[finite_abs_delta]
    finite_rel_delta = np.isfinite(candidate_rel) & np.isfinite(parent_rel)
    delta_rel[finite_rel_delta] = candidate_rel[finite_rel_delta] - parent_rel[finite_rel_delta]
    is_regressed = np.zeros((n,), dtype=bool)
    if has_candidate:
        is_regressed = (delta_abs > 0.0) | (delta_rel > 0.0)

    cluster_id = _arr(table, "cluster_id", np.int64)
    if cluster_id.shape[0] == int(keep_parent_valid.shape[0]):
        cluster_id = cluster_id[keep_parent_valid]
    if cluster_id.shape[0] != n or np.any(cluster_id < 0):
        cluster_id = _grid_cluster_ids(image_key=image_key, px=px, py=py, grid_size=int(cfg.cluster_grid_size))

    weights = np.ones((n,), dtype=np.float64)
    if has_candidate:
        weights[is_regressed] *= float(cfg.hard_regression_weight)
    if bool(cfg.cluster_balance) and n > 0:
        unique, counts = np.unique(cluster_id, return_counts=True)
        count_map = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        weights = np.asarray(
            [weights[i] / np.sqrt(max(1, count_map[int(cluster_id[i])])) for i in range(n)],
            dtype=np.float64,
        )
        finite = np.isfinite(weights) & (weights > 0)
        if np.any(finite):
            weights = weights / float(np.mean(weights[finite]))

    cache_manifest = dict(manifest)
    cache_manifest.update(
        {
            "split": split,
            "no_test_leakage": True,
            "seed": int(cfg.seed),
            "max_points_per_view": int(cfg.max_points_per_view),
            "cluster_grid_size": int(cfg.cluster_grid_size),
            "cluster_balance": bool(cfg.cluster_balance),
            "num_sentinels": int(n),
            "num_views": int(len(set(image_key.tolist()))),
            "has_candidate": bool(has_candidate),
            "num_regressed_candidate": int(np.count_nonzero(is_regressed)),
        }
    )

    return {
        "manifest": cache_manifest,
        "arrays": {
            "image_name": image_name,
            "image_key": image_key,
            "px": px.astype(np.int64),
            "py": py.astype(np.int64),
            "gt_depth": gt.astype(np.float64),
            "point3D_id": point_id.astype(np.int64),
            "parent_pred_depth": parent_pred.astype(np.float64),
            "parent_abs_error": parent_abs.astype(np.float64),
            "parent_abs_rel": parent_rel.astype(np.float64),
            "candidate_pred_depth": candidate_pred.astype(np.float64),
            "candidate_delta_abs_error": delta_abs.astype(np.float64),
            "candidate_delta_abs_rel": delta_rel.astype(np.float64),
            "sentinel_weight": weights.astype(np.float64),
            "cluster_id": cluster_id.astype(np.int64),
            "is_regressed_candidate": is_regressed.astype(bool),
        },
    }


def sentinel_view_summary_rows(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    arrays = dict(cache["arrays"])
    image_key = np.asarray(arrays["image_key"], dtype=object)
    weights = np.asarray(arrays["sentinel_weight"], dtype=np.float64)
    regressed = np.asarray(arrays["is_regressed_candidate"], dtype=bool)
    parent_rel = np.asarray(arrays["parent_abs_rel"], dtype=np.float64)
    rows = []
    for key in sorted(set(image_key.tolist()), key=str):
        mask = image_key == key
        rows.append(
            {
                "image_key": str(key),
                "sentinel_count": int(np.count_nonzero(mask)),
                "weight_sum": float(np.sum(weights[mask])),
                "regressed_candidate_count": int(np.count_nonzero(mask & regressed)),
                "parent_absrel_mean": float(np.mean(parent_rel[mask][np.isfinite(parent_rel[mask])])) if np.any(np.isfinite(parent_rel[mask])) else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sparse_depth_sentinel_cache(*, output: Path, cache: Mapping[str, Any], cfg: SentinelCacheConfig) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(cache["arrays"])
    manifest = dict(cache["manifest"])
    arrays_for_npz = {k: np.asarray(v) for k, v in arrays.items()}
    arrays_for_npz["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True), dtype=object)
    np.savez_compressed(output, **arrays_for_npz)
    manifest_path = output.with_name("sentinel_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output.with_name("sentinel_view_summary.csv"), sentinel_view_summary_rows(cache))
    output.with_name("sentinel_report.md").write_text(format_sentinel_report(cache, cfg), encoding="utf-8")


def format_sentinel_report(cache: Mapping[str, Any], cfg: SentinelCacheConfig) -> str:
    manifest = dict(cache["manifest"])
    arrays = dict(cache["arrays"])
    n = int(np.asarray(arrays["gt_depth"]).shape[0])
    weights = np.asarray(arrays["sentinel_weight"], dtype=np.float64)
    regressed = np.asarray(arrays["is_regressed_candidate"], dtype=bool)
    return (
        "# Sparse Depth Sentinel Cache Report\n\n"
        "Decision: `SENTINEL_CACHE_BUILT_NO_TEST_LEAKAGE`\n\n"
        f"- split: `{manifest.get('split', '')}`\n"
        f"- no_test_leakage: `{manifest.get('no_test_leakage', False)}`\n"
        f"- source: `{manifest.get('source_path', '')}`\n"
        f"- parent: `{manifest.get('parent_model_path', '')}` @ `{manifest.get('parent_iteration', '')}`\n"
        f"- candidate: `{manifest.get('candidate_model_path', '')}` @ `{manifest.get('candidate_iteration', '')}`\n"
        f"- sentinels: `{n}`\n"
        f"- views: `{manifest.get('num_views', 0)}`\n"
        f"- regressed candidate sentinels: `{int(np.count_nonzero(regressed))}`\n"
        f"- mean sentinel weight: `{float(np.mean(weights)) if n else 0.0:.6f}`\n"
        f"- config: `{json.dumps(asdict(cfg), sort_keys=True)}`\n\n"
        "This cache is valid for train/calibration-time targeted geometry preservation only. Test-split cache construction is rejected by the builder.\n"
    )
