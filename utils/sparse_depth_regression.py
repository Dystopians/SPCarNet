from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.geometry_metrics_utils import depth_metrics


@dataclass(frozen=True)
class SparseDepthRegressionConfig:
    margin_abs: float = 0.0
    margin_rel: float = 0.0
    gate_top_fraction: float = 0.10
    boundary_fraction: float = 0.05
    cluster_grid_size: int = 64
    eps_depth: float = 1e-6


def _as_array(payload: Mapping[str, Any], key: str, dtype=None) -> np.ndarray:
    arr = np.asarray(payload.get(key, []), dtype=dtype)
    return arr.reshape(-1)


def _as_str_array(payload: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload.get(key, []), dtype=object).reshape(-1)


def _safe_rel(abs_error: np.ndarray, gt_depth: np.ndarray, eps: float) -> np.ndarray:
    return abs_error / np.clip(gt_depth, float(eps), None)


def _positive_top_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(values.shape, dtype=bool)
    finite_positive = np.isfinite(values) & (values > 0)
    if not np.any(finite_positive):
        return out
    n = int(np.ceil(float(np.count_nonzero(finite_positive)) * max(0.0, min(1.0, float(fraction)))))
    n = max(1, n)
    idx = np.nonzero(finite_positive)[0]
    order = idx[np.argsort(values[idx], kind="stable")]
    out[order[-n:]] = True
    return out


def _boundary_bin(px: np.ndarray, py: np.ndarray, width: np.ndarray, height: np.ndarray, frac: float) -> np.ndarray:
    x_den = np.maximum(width.astype(np.float64) - 1.0, 1.0)
    y_den = np.maximum(height.astype(np.float64) - 1.0, 1.0)
    dist = np.minimum.reduce(
        [
            px.astype(np.float64) / x_den,
            (x_den - px.astype(np.float64)) / x_den,
            py.astype(np.float64) / y_den,
            (y_den - py.astype(np.float64)) / y_den,
        ]
    )
    out = np.full(px.shape, "interior", dtype=object)
    out[dist <= float(frac)] = "near_boundary"
    out[dist <= float(frac) * 0.5] = "edge_band"
    return out


def _depth_bin(gt_depth: np.ndarray) -> np.ndarray:
    out = np.full(gt_depth.shape, "invalid", dtype=object)
    finite = np.isfinite(gt_depth) & (gt_depth > 0)
    if not np.any(finite):
        return out
    qs = np.quantile(gt_depth[finite], [0.25, 0.50, 0.75])
    out[finite & (gt_depth <= qs[0])] = "near_q1"
    out[finite & (gt_depth > qs[0]) & (gt_depth <= qs[1])] = "mid_q2"
    out[finite & (gt_depth > qs[1]) & (gt_depth <= qs[2])] = "far_q3"
    out[finite & (gt_depth > qs[2])] = "far_q4"
    return out


def _error_quantile_bin(delta: np.ndarray) -> np.ndarray:
    out = np.full(delta.shape, "non_positive_or_invalid", dtype=object)
    finite_positive = np.isfinite(delta) & (delta > 0)
    if not np.any(finite_positive):
        return out
    qs = np.quantile(delta[finite_positive], [0.50, 0.90])
    out[finite_positive & (delta <= qs[0])] = "positive_low50"
    out[finite_positive & (delta > qs[0]) & (delta <= qs[1])] = "positive_50_90"
    out[finite_positive & (delta > qs[1])] = "positive_top10"
    return out


def _assign_grid_clusters(
    image_key: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    gate_critical: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    cluster = np.full(px.shape, -1, dtype=np.int64)
    mapping: dict[tuple[str, int, int], int] = {}
    next_id = 0
    size = max(1, int(grid_size))
    for i in np.nonzero(gate_critical)[0].tolist():
        key = (str(image_key[i]), int(px[i]) // size, int(py[i]) // size)
        cid = mapping.get(key)
        if cid is None:
            cid = next_id
            mapping[key] = cid
            next_id += 1
        cluster[i] = cid
    return cluster


def build_sparse_depth_regression_table(
    payload: Mapping[str, Any],
    cfg: SparseDepthRegressionConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or SparseDepthRegressionConfig()
    image_name = _as_str_array(payload, "image_name")
    n = int(image_name.shape[0])
    image_key = _as_str_array(payload, "image_key")
    if image_key.shape[0] == 0:
        image_key = image_name.copy()
    point_id = _as_array(payload, "point3D_id", np.int64)
    px = _as_array(payload, "px", np.int64)
    py = _as_array(payload, "py", np.int64)
    width = _as_array(payload, "width", np.int64)
    height = _as_array(payload, "height", np.int64)
    gt = _as_array(payload, "gt_depth", np.float64)
    parent = _as_array(payload, "parent_pred_depth", np.float64)
    candidate = _as_array(payload, "candidate_pred_depth", np.float64)
    required = {
        "image_key": image_key,
        "point3D_id": point_id,
        "px": px,
        "py": py,
        "width": width,
        "height": height,
        "gt_depth": gt,
        "parent_pred_depth": parent,
        "candidate_pred_depth": candidate,
    }
    bad = [k for k, v in required.items() if int(v.shape[0]) != n]
    if bad:
        raise ValueError(f"correspondence payload has mismatched lengths for {bad}; expected {n}")

    eps = float(cfg.eps_depth)
    parent_valid = np.isfinite(parent) & (parent > eps) & np.isfinite(gt) & (gt > eps)
    candidate_valid = np.isfinite(candidate) & (candidate > eps) & np.isfinite(gt) & (gt > eps)
    both_valid = parent_valid & candidate_valid

    parent_abs = np.full((n,), np.inf, dtype=np.float64)
    candidate_abs = np.full((n,), np.inf, dtype=np.float64)
    parent_abs[parent_valid] = np.abs(parent[parent_valid] - gt[parent_valid])
    candidate_abs[candidate_valid] = np.abs(candidate[candidate_valid] - gt[candidate_valid])
    parent_rel = _safe_rel(parent_abs, gt, eps)
    candidate_rel = _safe_rel(candidate_abs, gt, eps)
    delta_abs = np.full((n,), np.inf, dtype=np.float64)
    delta_rel = np.full((n,), np.inf, dtype=np.float64)
    finite_delta = np.isfinite(candidate_abs) & np.isfinite(parent_abs)
    delta_abs[finite_delta] = candidate_abs[finite_delta] - parent_abs[finite_delta]
    finite_rel_delta = np.isfinite(candidate_rel) & np.isfinite(parent_rel)
    delta_rel[finite_rel_delta] = candidate_rel[finite_rel_delta] - parent_rel[finite_rel_delta]

    regressed_abs = (delta_abs > float(cfg.margin_abs)) | (parent_valid & ~candidate_valid)
    regressed_rel = (delta_rel > float(cfg.margin_rel)) | (parent_valid & ~candidate_valid)
    top_abs = _positive_top_mask(delta_abs[both_valid], float(cfg.gate_top_fraction))
    top_rel = _positive_top_mask(delta_rel[both_valid], float(cfg.gate_top_fraction))
    top_abs_full = np.zeros((n,), dtype=bool)
    top_rel_full = np.zeros((n,), dtype=bool)
    both_idx = np.nonzero(both_valid)[0]
    top_abs_full[both_idx] = top_abs
    top_rel_full[both_idx] = top_rel
    gate_critical = top_abs_full | top_rel_full | (parent_valid & ~candidate_valid)

    depth_bin = _depth_bin(gt)
    boundary_bin = _boundary_bin(px=px, py=py, width=width, height=height, frac=float(cfg.boundary_fraction))
    error_quantile_bin = _error_quantile_bin(delta_rel)
    cluster_id = _assign_grid_clusters(
        image_key=image_key,
        px=px,
        py=py,
        gate_critical=gate_critical,
        grid_size=int(cfg.cluster_grid_size),
    )

    table = {
        "image_name": image_name,
        "image_key": image_key,
        "point3D_id": point_id,
        "px": px,
        "py": py,
        "width": width,
        "height": height,
        "gt_depth": gt,
        "parent_pred_depth": parent,
        "candidate_pred_depth": candidate,
        "parent_valid": parent_valid,
        "candidate_valid": candidate_valid,
        "parent_abs_error": parent_abs,
        "candidate_abs_error": candidate_abs,
        "parent_abs_rel": parent_rel,
        "candidate_abs_rel": candidate_rel,
        "delta_abs_error": delta_abs,
        "delta_abs_rel": delta_rel,
        "regressed_abs": regressed_abs,
        "regressed_rel": regressed_rel,
        "gate_critical": gate_critical,
        "depth_bin": depth_bin,
        "boundary_bin": boundary_bin,
        "error_quantile_bin": error_quantile_bin,
        "cluster_id": cluster_id,
    }
    for key in (
        "parent_normal_cos",
        "candidate_normal_cos",
        "parent_alpha",
        "candidate_alpha",
        "parent_rgb_residual",
        "candidate_rgb_residual",
    ):
        arr = payload.get(key, None)
        if arr is not None:
            table[key] = np.asarray(arr).reshape(-1)
    return table


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if np.any(finite) else float("nan")


def _summarize_mask(table: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    parent_valid = np.asarray(table["parent_valid"], dtype=bool)
    candidate_valid = np.asarray(table["candidate_valid"], dtype=bool)
    both = mask & parent_valid & candidate_valid
    return {
        "count": int(np.count_nonzero(mask)),
        "both_valid_count": int(np.count_nonzero(both)),
        "candidate_invalid_count": int(np.count_nonzero(mask & parent_valid & ~candidate_valid)),
        "parent_mae": _finite_mean(np.asarray(table["parent_abs_error"])[both]),
        "candidate_mae": _finite_mean(np.asarray(table["candidate_abs_error"])[both]),
        "delta_mae": _finite_mean(np.asarray(table["delta_abs_error"])[both]),
        "parent_absrel": _finite_mean(np.asarray(table["parent_abs_rel"])[both]),
        "candidate_absrel": _finite_mean(np.asarray(table["candidate_abs_rel"])[both]),
        "delta_absrel": _finite_mean(np.asarray(table["delta_abs_rel"])[both]),
        "regressed_abs_count": int(np.count_nonzero(mask & np.asarray(table["regressed_abs"], dtype=bool))),
        "regressed_rel_count": int(np.count_nonzero(mask & np.asarray(table["regressed_rel"], dtype=bool))),
        "gate_critical_count": int(np.count_nonzero(mask & np.asarray(table["gate_critical"], dtype=bool))),
    }


def summarize_sparse_depth_regressions(table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = int(np.asarray(table["gt_depth"]).shape[0])
    all_mask = np.ones((n,), dtype=bool)
    both = np.asarray(table["parent_valid"], dtype=bool) & np.asarray(table["candidate_valid"], dtype=bool)
    summary = {
        "global": _summarize_mask(table, all_mask),
        "valid_depth_metrics": None,
        "by_depth_bin": {},
        "by_boundary_bin": {},
        "by_error_quantile_bin": {},
    }
    if np.any(both):
        summary["valid_depth_metrics"] = {
            "parent": depth_metrics(np.asarray(table["parent_pred_depth"], dtype=np.float64)[both], np.asarray(table["gt_depth"], dtype=np.float64)[both]),
            "candidate": depth_metrics(np.asarray(table["candidate_pred_depth"], dtype=np.float64)[both], np.asarray(table["gt_depth"], dtype=np.float64)[both]),
        }
    for key in ("depth_bin", "boundary_bin", "error_quantile_bin"):
        dst = f"by_{key}"
        for value in sorted({str(v) for v in np.asarray(table[key], dtype=object).tolist()}):
            summary[dst][value] = _summarize_mask(table, np.asarray(table[key], dtype=object) == value)
    return summary


def _group_rows(table: Mapping[str, np.ndarray], group_key: str) -> list[dict[str, Any]]:
    values = np.asarray(table[group_key])
    rows = []
    for value in sorted(set(values.tolist()), key=lambda x: str(x)):
        if group_key == "cluster_id" and int(value) < 0:
            continue
        mask = values == value
        row = {group_key: value}
        row.update(_summarize_mask(table, mask))
        rows.append(row)
    return rows


def per_view_summary_rows(table: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return _group_rows(table, "image_key")


def point_summary_rows(table: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return _group_rows(table, "point3D_id")


def cluster_summary_rows(table: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = _group_rows(table, "cluster_id")
    rows.sort(key=lambda r: (float(r.get("delta_absrel", 0.0)), float(r.get("delta_mae", 0.0))), reverse=True)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def table_to_rows(table: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    keys = list(table.keys())
    n = int(np.asarray(table["gt_depth"]).shape[0])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row = {}
        for key in keys:
            value = np.asarray(table[key])[i]
            if isinstance(value, np.generic):
                value = value.item()
            row[key] = value
        rows.append(row)
    return rows


def write_sparse_depth_regression_outputs(
    *,
    output_dir: Path,
    table: Mapping[str, np.ndarray],
    cfg: SparseDepthRegressionConfig,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_sparse_depth_regressions(table)
    payload = {
        "manifest": dict(manifest),
        "config": asdict(cfg),
        "summary": summary,
    }
    write_csv(output_dir / "correspondence_regressions.csv", table_to_rows(table))
    write_csv(output_dir / "per_view_regression_summary.csv", per_view_summary_rows(table))
    write_csv(output_dir / "point_regression_summary.csv", point_summary_rows(table))
    write_csv(output_dir / "cluster_regression_summary.csv", cluster_summary_rows(table))
    np.savez_compressed(output_dir / "correspondence_regressions.npz", **{k: np.asarray(v) for k, v in table.items()})
    np.savez_compressed(
        output_dir / "sentinel_candidate_mask.npz",
        gate_critical=np.asarray(table["gate_critical"], dtype=bool),
        regressed_abs=np.asarray(table["regressed_abs"], dtype=bool),
        regressed_rel=np.asarray(table["regressed_rel"], dtype=bool),
        cluster_id=np.asarray(table["cluster_id"], dtype=np.int64),
        image_key=np.asarray(table["image_key"], dtype=object),
        point3D_id=np.asarray(table["point3D_id"], dtype=np.int64),
        px=np.asarray(table["px"], dtype=np.int64),
        py=np.asarray(table["py"], dtype=np.int64),
    )
    (output_dir / "regression_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "regression_report.md").write_text(format_regression_report(payload), encoding="utf-8")
    return payload


def format_regression_report(payload: Mapping[str, Any]) -> str:
    manifest = dict(payload.get("manifest", {}))
    summary = dict(payload.get("summary", {}))
    global_summary = dict(summary.get("global", {}))
    metrics = summary.get("valid_depth_metrics", None) or {}
    parent_absrel = metrics.get("parent", {}).get("abs_rel", float("nan")) if isinstance(metrics, dict) else float("nan")
    candidate_absrel = metrics.get("candidate", {}).get("abs_rel", float("nan")) if isinstance(metrics, dict) else float("nan")
    parent_mae = metrics.get("parent", {}).get("mae", float("nan")) if isinstance(metrics, dict) else float("nan")
    candidate_mae = metrics.get("candidate", {}).get("mae", float("nan")) if isinstance(metrics, dict) else float("nan")
    status = "SPARSE_DEPTH_CANDIDATE_WORSE" if (candidate_absrel > parent_absrel or candidate_mae > parent_mae) else "SPARSE_DEPTH_NONREGRESSION"
    return (
        "# Sparse Depth Regression Report\n\n"
        f"Decision: `{status}`\n\n"
        f"- scene/source: `{manifest.get('source_path', '')}`\n"
        f"- split: `{manifest.get('split', '')}`\n"
        f"- parent: `{manifest.get('parent_model_path', '')}` @ `{manifest.get('parent_iteration', '')}`\n"
        f"- candidate: `{manifest.get('candidate_model_path', '')}` @ `{manifest.get('candidate_iteration', '')}`\n"
        f"- correspondences: `{global_summary.get('count', 0)}`\n"
        f"- both valid: `{global_summary.get('both_valid_count', 0)}`\n"
        f"- candidate invalid count: `{global_summary.get('candidate_invalid_count', 0)}`\n"
        f"- gate-critical correspondences: `{global_summary.get('gate_critical_count', 0)}`\n\n"
        "## Aggregate Sparse Depth\n\n"
        "| metric | parent | candidate | candidate - parent |\n"
        "|---|---:|---:|---:|\n"
        f"| AbsRel | {parent_absrel:.9f} | {candidate_absrel:.9f} | {candidate_absrel - parent_absrel:+.9f} |\n"
        f"| Depth MAE | {parent_mae:.9f} | {candidate_mae:.9f} | {candidate_mae - parent_mae:+.9f} |\n\n"
        "Positive deltas mean the candidate is worse than the parent. Test-split outputs are diagnostic only and must not be used as a training sentinel cache.\n"
    )
