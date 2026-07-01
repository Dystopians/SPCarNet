#!/usr/bin/env python3
"""Audit per-view oracle headroom in support-transport apply reports.

This is a read-only diagnostic: it only consumes metrics already serialized in
support_transport_apply_report.json files.  It does not render targets, load GT,
or recompute candidate images.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


REPORT_NAME = "support_transport_apply_report.json"
EPS = 1.0e-12

PREFERRED_SCENE_ORDER = (
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
)

PREFERRED_METRIC_ORDER = (
    "psnr_gain",
    "candidate_psnr",
    "ssim_gain",
    "candidate_ssim",
    "mse_reduction",
    "candidate_mse",
    "candidate_lpips",
    "lpips",
    "lpips_gain",
    "candidate_dists",
    "dists",
)

IGNORED_METRIC_KEYS = {
    "view",
    "base_mse",
    "base_psnr",
    "base_ssim",
    "base_lpips",
    "base_dists",
    "changed_fraction",
    "mean_changed_fraction",
    "mean_abs_delta",
    "positive_view_fraction",
    "ssim_positive_view_fraction",
}


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def _mean(values: list[float]) -> float | None:
    return float(fmean(values)) if values else None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _add_counter(target: Counter[str], source: dict[str, int] | Counter[str]) -> None:
    for key, value in source.items():
        target[str(key)] += int(value)


def _format_float(value: Any, digits: int = 9) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _format_signed(value: Any, digits: int = 9) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:+.{digits}f}"


def _format_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _parse_named_path(spec: str, flag: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"{flag} expects NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name:
        raise ValueError(f"{flag} has an empty NAME in {spec!r}")
    if not raw_path:
        raise ValueError(f"{flag} has an empty PATH in {spec!r}")
    return name, Path(raw_path).expanduser()


def _scene_sort_key(scene: str) -> tuple[int, str]:
    if scene in PREFERRED_SCENE_ORDER:
        return PREFERRED_SCENE_ORDER.index(scene), scene
    return len(PREFERRED_SCENE_ORDER), scene


def _metric_sort_key(metric: str) -> tuple[int, str]:
    if metric in PREFERRED_METRIC_ORDER:
        return PREFERRED_METRIC_ORDER.index(metric), metric
    return len(PREFERRED_METRIC_ORDER), metric


def _infer_metric_direction(metric: str) -> str | None:
    key = metric.lower()
    if key in IGNORED_METRIC_KEYS or key.startswith("base_"):
        return None
    if key.endswith("_gain") or "reduction" in key or "improvement" in key:
        return "max"
    if "psnr" in key or "ssim" in key:
        return "max"
    if any(token in key for token in ("mse", "rmse", "mae", "lpips", "dists", "loss", "error")):
        return "min"
    return None


def _metric_headroom(selected_value: float, best_value: float, direction: str) -> float:
    if direction == "max":
        return float(best_value - selected_value)
    return float(selected_value - best_value)


def _quality_metrics(candidates: dict[str, dict[str, Any]], selected: dict[str, Any]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    keys: set[str] = set(selected)
    for candidate in candidates.values():
        keys.update(candidate)
    for key in sorted(keys, key=_metric_sort_key):
        direction = _infer_metric_direction(key)
        if direction is not None:
            metrics[key] = direction
    return metrics


def _choose_primary_metric(metric_directions: dict[str, str], requested: str | None) -> str | None:
    if requested:
        return requested if requested in metric_directions else None
    for metric in PREFERRED_METRIC_ORDER:
        if metric in metric_directions:
            return metric
    return next(iter(metric_directions), None)


def _best_candidate(
    candidates: dict[str, dict[str, Any]],
    metric: str,
    direction: str,
    selected_variant: str | None,
) -> tuple[str | None, float | None]:
    values: list[tuple[str, float]] = []
    for variant, row in candidates.items():
        value = _as_float(row.get(metric))
        if value is not None:
            values.append((variant, value))
    if not values:
        return None, None

    best_score = max(value for _, value in values) if direction == "max" else min(value for _, value in values)
    tied = [(variant, value) for variant, value in values if abs(value - best_score) <= EPS]
    if selected_variant is not None:
        for variant, value in tied:
            if variant == selected_variant:
                return variant, value
    variant, value = sorted(tied, key=lambda item: item[0])[0]
    return variant, value


def _candidate_metrics(view: dict[str, Any], warnings: Counter[str]) -> dict[str, dict[str, Any]]:
    raw = view.get("candidate_metrics")
    if isinstance(raw, dict):
        candidates: dict[str, dict[str, Any]] = {}
        for variant, metrics in raw.items():
            if isinstance(variant, str) and isinstance(metrics, dict):
                candidates[variant] = metrics
            else:
                warnings["malformed_candidate_metric_entry"] += 1
        if candidates:
            return candidates
        warnings["empty_candidate_metrics"] += 1
        return {}

    fallback: dict[str, dict[str, Any]] = {}
    for variant in ("fixed", "learned", "hybrid"):
        metrics = view.get(variant)
        if isinstance(metrics, dict):
            fallback[variant] = metrics
    if fallback:
        warnings["missing_candidate_metrics_used_legacy_variant_fields"] += 1
        return fallback
    warnings["missing_candidate_metrics"] += 1
    return {}


def _selected_metrics(
    view: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    warnings: Counter[str],
) -> tuple[str | None, dict[str, Any] | None, str]:
    output_variant = view.get("output_variant")
    scene_selected_variant = view.get("selected_variant")
    output_variant = output_variant if isinstance(output_variant, str) else None
    scene_selected_variant = scene_selected_variant if isinstance(scene_selected_variant, str) else None

    selected = view.get("selected")
    if isinstance(selected, dict):
        return output_variant or scene_selected_variant, selected, "selected_field"

    if output_variant is not None and output_variant in candidates:
        warnings["missing_selected_used_output_variant"] += 1
        return output_variant, candidates[output_variant], "output_variant_candidate"

    if scene_selected_variant is not None and scene_selected_variant in candidates:
        warnings["missing_selected_used_scene_selected_variant"] += 1
        return scene_selected_variant, candidates[scene_selected_variant], "selected_variant_candidate"

    warnings["missing_selected_metrics"] += 1
    return output_variant or scene_selected_variant, None, "missing"


def _discover_method_reports(name: str, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[Path] = []
    errors: list[dict[str, Any]] = []
    direct = root / REPORT_NAME
    if direct.is_file():
        reports.append(direct)
    for path in sorted(root.glob(f"*/{REPORT_NAME}"), key=lambda p: _scene_sort_key(p.parent.name)):
        if path not in reports:
            reports.append(path)

    if not reports:
        errors.append(
            {
                "method": name,
                "root": str(root),
                "error": f"no {REPORT_NAME} found at root or one scene directory below root",
            }
        )
    inputs = [
        {
            "method": name,
            "path": path,
            "scene_hint": path.parent.name,
            "source": "method",
        }
        for path in reports
    ]
    return inputs, errors


def _normalize_report_path(path: Path) -> Path:
    if path.is_dir():
        return path / REPORT_NAME
    return path


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "file_not_found"
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc}"
    except OSError as exc:
        return None, f"os_error:{exc}"
    if not isinstance(payload, dict):
        return None, "top_level_json_is_not_object"
    return payload, None


def _metric_summary(acc: dict[str, list[float]]) -> dict[str, Any]:
    selected = acc.get("selected", [])
    oracle = acc.get("oracle", [])
    headroom = acc.get("headroom", [])
    positive = [value for value in headroom if value > EPS]
    return {
        "view_count": int(len(headroom)),
        "selected_mean": _mean(selected),
        "oracle_mean": _mean(oracle),
        "mean_headroom": _mean(headroom),
        "max_headroom": max(headroom) if headroom else None,
        "positive_headroom_count": int(len(positive)),
        "positive_headroom_fraction": (len(positive) / len(headroom)) if headroom else None,
    }


def _empty_scene_summary(method: str, path: Path, scene: str, error: str) -> dict[str, Any]:
    return {
        "method": method,
        "scene": scene,
        "report_path": str(path),
        "status": "load_error",
        "load_error": error,
        "view_count": 0,
        "valid_view_count": 0,
        "schema_warnings": {},
        "metric_directions": {},
        "primary_metric": None,
        "metrics": {},
        "selected_variant_counts": {},
        "scene_selected_variant_counts": {},
        "output_variant_counts": {},
        "best_variant_counts_by_metric": {},
        "decision_source_counts": {},
        "decision_source_buckets": {},
        "largest_misses": [],
        "per_view": [],
    }


def _audit_scene(
    *,
    method: str,
    path: Path,
    payload: dict[str, Any],
    top_k: int,
    requested_primary_metric: str | None,
) -> dict[str, Any]:
    scene = path.parent.name
    per_view = payload.get("per_view")
    warnings: Counter[str] = Counter()
    if not isinstance(per_view, list):
        warnings["missing_per_view_list"] += 1
        return {
            **_empty_scene_summary(method, path, scene, "missing_per_view_list"),
            "status": "insufficient_schema",
            "schema_warnings": _sorted_counter(warnings),
        }

    selected_variant_counts: Counter[str] = Counter()
    scene_selected_variant_counts: Counter[str] = Counter()
    output_variant_counts: Counter[str] = Counter()
    best_variant_counts_by_metric: dict[str, Counter[str]] = defaultdict(Counter)
    decision_source_counts: Counter[str] = Counter()
    decision_source_bucket_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "view_count": 0,
            "primary_headroom": [],
            "primary_positive_headroom_count": 0,
            "best_variant_counts": Counter(),
            "output_variant_counts": Counter(),
        }
    )
    metric_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_directions: dict[str, str] = {}
    per_view_rows: list[dict[str, Any]] = []
    largest_misses: list[dict[str, Any]] = []
    primary_metric = requested_primary_metric

    for index, raw_view in enumerate(per_view):
        if not isinstance(raw_view, dict):
            warnings["per_view_entry_not_object"] += 1
            continue

        view_name = str(raw_view.get("view", index))
        candidates = _candidate_metrics(raw_view, warnings)
        selected_variant, selected, selected_source = _selected_metrics(raw_view, candidates, warnings)
        scene_selected_variant = raw_view.get("selected_variant")
        scene_selected_variant = scene_selected_variant if isinstance(scene_selected_variant, str) else None
        output_variant = raw_view.get("output_variant")
        output_variant = output_variant if isinstance(output_variant, str) else None
        decision_source = raw_view.get("output_decision_source") or raw_view.get("decision_source") or "__missing__"
        decision_source = str(decision_source)

        if scene_selected_variant is not None:
            scene_selected_variant_counts[scene_selected_variant] += 1
        else:
            scene_selected_variant_counts["__missing__"] += 1
        if output_variant is not None:
            output_variant_counts[output_variant] += 1
        else:
            output_variant_counts["__missing__"] += 1
        selected_variant_counts[selected_variant or "__missing__"] += 1
        decision_source_counts[decision_source] += 1

        if selected is None or not candidates:
            continue

        view_metric_directions = _quality_metrics(candidates, selected)
        metric_directions.update(view_metric_directions)
        if primary_metric is None:
            primary_metric = _choose_primary_metric(metric_directions, requested_primary_metric)
        if primary_metric is not None and primary_metric not in view_metric_directions:
            warnings["primary_metric_missing_on_view"] += 1

        metric_oracles: dict[str, dict[str, Any]] = {}
        for metric, direction in view_metric_directions.items():
            selected_value = _as_float(selected.get(metric))
            if selected_value is None:
                warnings[f"missing_selected_{metric}"] += 1
                continue
            best_variant, best_value = _best_candidate(candidates, metric, direction, selected_variant)
            if best_variant is None or best_value is None:
                warnings[f"missing_candidate_{metric}"] += 1
                continue
            headroom = _metric_headroom(selected_value, best_value, direction)
            metric_acc[metric]["selected"].append(selected_value)
            metric_acc[metric]["oracle"].append(best_value)
            metric_acc[metric]["headroom"].append(headroom)
            best_variant_counts_by_metric[metric][best_variant] += 1
            metric_oracles[metric] = {
                "direction": direction,
                "selected_value": selected_value,
                "best_variant": best_variant,
                "best_value": best_value,
                "headroom": headroom,
            }

        if primary_metric is None or primary_metric not in metric_oracles:
            continue

        primary = metric_oracles[primary_metric]
        primary_headroom = float(primary["headroom"])
        primary_best_variant = str(primary["best_variant"])
        bucket = decision_source_bucket_acc[decision_source]
        bucket["view_count"] += 1
        bucket["primary_headroom"].append(primary_headroom)
        if primary_headroom > EPS:
            bucket["primary_positive_headroom_count"] += 1
        bucket["best_variant_counts"][primary_best_variant] += 1
        bucket["output_variant_counts"][selected_variant or "__missing__"] += 1

        selected_metric_values = {
            metric: _as_float(selected.get(metric))
            for metric in sorted(metric_oracles, key=_metric_sort_key)
            if _as_float(selected.get(metric)) is not None
        }
        view_row = {
            "view": view_name,
            "selected_variant": selected_variant,
            "scene_selected_variant": scene_selected_variant,
            "output_variant": output_variant,
            "selected_metric_source": selected_source,
            "decision_source": decision_source,
            "primary_metric": primary_metric,
            "primary_best_variant": primary_best_variant,
            "primary_selected_value": primary["selected_value"],
            "primary_best_value": primary["best_value"],
            "primary_headroom": primary_headroom,
            "selected_metrics": selected_metric_values,
            "metric_oracles": metric_oracles,
        }
        per_view_rows.append(view_row)
        if primary_headroom > EPS:
            largest_misses.append(
                {
                    "scene": scene,
                    "view": view_name,
                    "decision_source": decision_source,
                    "selected_variant": selected_variant,
                    "scene_selected_variant": scene_selected_variant,
                    "output_variant": output_variant,
                    "best_variant": primary_best_variant,
                    "metric": primary_metric,
                    "selected_value": primary["selected_value"],
                    "best_value": primary["best_value"],
                    "headroom": primary_headroom,
                }
            )

    metric_summaries = {
        metric: {
            "direction": metric_directions.get(metric),
            **_metric_summary(acc),
        }
        for metric, acc in sorted(metric_acc.items(), key=lambda item: _metric_sort_key(item[0]))
    }
    primary_metric = _choose_primary_metric(metric_directions, requested_primary_metric)
    if requested_primary_metric and primary_metric is None:
        warnings["requested_primary_metric_not_available"] += 1

    largest_misses = sorted(largest_misses, key=lambda row: (-float(row["headroom"]), row["scene"], row["view"]))[
        : max(int(top_k), 0)
    ]
    decision_source_buckets: dict[str, Any] = {}
    for source, bucket in sorted(decision_source_bucket_acc.items()):
        headroom = bucket["primary_headroom"]
        count = int(bucket["view_count"])
        decision_source_buckets[source] = {
            "view_count": count,
            "primary_mean_headroom": _mean(headroom),
            "primary_max_headroom": max(headroom) if headroom else None,
            "primary_positive_headroom_count": int(bucket["primary_positive_headroom_count"]),
            "primary_positive_headroom_fraction": (
                int(bucket["primary_positive_headroom_count"]) / count if count else None
            ),
            "best_variant_counts": _sorted_counter(bucket["best_variant_counts"]),
            "output_variant_counts": _sorted_counter(bucket["output_variant_counts"]),
        }

    return {
        "method": method,
        "scene": scene,
        "report_path": str(path),
        "status": "ok" if per_view_rows else "insufficient_schema",
        "view_count": int(len(per_view)),
        "valid_view_count": int(len(per_view_rows)),
        "schema_warnings": _sorted_counter(warnings),
        "metric_directions": {
            metric: metric_directions[metric] for metric in sorted(metric_directions, key=_metric_sort_key)
        },
        "primary_metric": primary_metric,
        "metrics": metric_summaries,
        "selected_variant_counts": _sorted_counter(selected_variant_counts),
        "scene_selected_variant_counts": _sorted_counter(scene_selected_variant_counts),
        "output_variant_counts": _sorted_counter(output_variant_counts),
        "best_variant_counts_by_metric": {
            metric: _sorted_counter(counts)
            for metric, counts in sorted(best_variant_counts_by_metric.items(), key=lambda item: _metric_sort_key(item[0]))
        },
        "decision_source_counts": _sorted_counter(decision_source_counts),
        "decision_source_buckets": decision_source_buckets,
        "largest_misses": largest_misses,
        "per_view": per_view_rows,
    }


def _aggregate_metric_summaries(scene_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = sorted(
        {
            metric
            for scene in scene_summaries
            for metric, summary in scene.get("metrics", {}).items()
            if int(summary.get("view_count") or 0) > 0
        },
        key=_metric_sort_key,
    )
    out: dict[str, Any] = {}
    for metric in metrics:
        scene_rows = [
            scene["metrics"][metric]
            for scene in scene_summaries
            if metric in scene.get("metrics", {}) and int(scene["metrics"][metric].get("view_count") or 0) > 0
        ]
        total_views = sum(int(row.get("view_count") or 0) for row in scene_rows)
        if not scene_rows or total_views <= 0:
            continue
        direction = next((row.get("direction") for row in scene_rows if row.get("direction")), None)
        out[metric] = {
            "direction": direction,
            "scene_count": int(len(scene_rows)),
            "view_count": int(total_views),
            "macro_selected_mean": _mean(
                [float(row["selected_mean"]) for row in scene_rows if row.get("selected_mean") is not None]
            ),
            "macro_oracle_mean": _mean(
                [float(row["oracle_mean"]) for row in scene_rows if row.get("oracle_mean") is not None]
            ),
            "macro_mean_headroom": _mean(
                [float(row["mean_headroom"]) for row in scene_rows if row.get("mean_headroom") is not None]
            ),
            "micro_selected_mean": (
                sum(float(row["selected_mean"]) * int(row["view_count"]) for row in scene_rows) / total_views
            ),
            "micro_oracle_mean": (
                sum(float(row["oracle_mean"]) * int(row["view_count"]) for row in scene_rows) / total_views
            ),
            "micro_mean_headroom": (
                sum(float(row["mean_headroom"]) * int(row["view_count"]) for row in scene_rows) / total_views
            ),
            "positive_headroom_count": int(sum(int(row.get("positive_headroom_count") or 0) for row in scene_rows)),
            "positive_headroom_fraction": (
                sum(int(row.get("positive_headroom_count") or 0) for row in scene_rows) / total_views
            ),
            "max_headroom": max(
                float(row["max_headroom"]) for row in scene_rows if row.get("max_headroom") is not None
            ),
        }
    return out


def _aggregate_decision_buckets(scene_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "view_count": 0,
            "primary_headroom_weighted_sum": 0.0,
            "primary_max_headroom": None,
            "primary_positive_headroom_count": 0,
            "best_variant_counts": Counter(),
            "output_variant_counts": Counter(),
        }
    )
    for scene in scene_summaries:
        for source, row in scene.get("decision_source_buckets", {}).items():
            count = int(row.get("view_count") or 0)
            bucket = buckets[source]
            bucket["view_count"] += count
            mean_headroom = _as_float(row.get("primary_mean_headroom"))
            if mean_headroom is not None:
                bucket["primary_headroom_weighted_sum"] += mean_headroom * count
            max_headroom = _as_float(row.get("primary_max_headroom"))
            if max_headroom is not None:
                prev = bucket["primary_max_headroom"]
                bucket["primary_max_headroom"] = max_headroom if prev is None else max(float(prev), max_headroom)
            bucket["primary_positive_headroom_count"] += int(row.get("primary_positive_headroom_count") or 0)
            _add_counter(bucket["best_variant_counts"], row.get("best_variant_counts", {}))
            _add_counter(bucket["output_variant_counts"], row.get("output_variant_counts", {}))

    out: dict[str, Any] = {}
    for source, bucket in sorted(buckets.items()):
        count = int(bucket["view_count"])
        positive = int(bucket["primary_positive_headroom_count"])
        out[source] = {
            "view_count": count,
            "primary_mean_headroom": (
                float(bucket["primary_headroom_weighted_sum"]) / count if count else None
            ),
            "primary_max_headroom": bucket["primary_max_headroom"],
            "primary_positive_headroom_count": positive,
            "primary_positive_headroom_fraction": (positive / count if count else None),
            "best_variant_counts": _sorted_counter(bucket["best_variant_counts"]),
            "output_variant_counts": _sorted_counter(bucket["output_variant_counts"]),
        }
    return out


def _summarize_method(
    *,
    name: str,
    report_inputs: list[dict[str, Any]],
    input_errors: list[dict[str, Any]],
    top_k: int,
    requested_primary_metric: str | None,
) -> dict[str, Any]:
    scene_summaries: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    for item in report_inputs:
        path = _normalize_report_path(Path(item["path"]))
        payload, error = _load_json(path)
        scene = str(item.get("scene_hint") or path.parent.name)
        if error is not None or payload is None:
            load_errors.append({"path": str(path), "scene": scene, "error": error})
            scene_summaries.append(_empty_scene_summary(name, path, scene, str(error)))
            continue
        scene_summaries.append(
            _audit_scene(
                method=name,
                path=path,
                payload=payload,
                top_k=top_k,
                requested_primary_metric=requested_primary_metric,
            )
        )

    scene_summaries = sorted(scene_summaries, key=lambda row: _scene_sort_key(str(row["scene"])))
    metric_summary = _aggregate_metric_summaries(scene_summaries)
    primary_metric = _choose_primary_metric(
        {metric: str(row.get("direction")) for metric, row in metric_summary.items()},
        requested_primary_metric,
    )
    selected_counts: Counter[str] = Counter()
    scene_selected_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    best_counts_by_metric: dict[str, Counter[str]] = defaultdict(Counter)
    warning_counts: Counter[str] = Counter()
    largest_misses: list[dict[str, Any]] = []
    for scene in scene_summaries:
        _add_counter(selected_counts, scene.get("selected_variant_counts", {}))
        _add_counter(scene_selected_counts, scene.get("scene_selected_variant_counts", {}))
        _add_counter(output_counts, scene.get("output_variant_counts", {}))
        _add_counter(decision_counts, scene.get("decision_source_counts", {}))
        _add_counter(warning_counts, scene.get("schema_warnings", {}))
        for metric, counts in scene.get("best_variant_counts_by_metric", {}).items():
            _add_counter(best_counts_by_metric[metric], counts)
        largest_misses.extend(scene.get("largest_misses", []))

    largest_misses = sorted(largest_misses, key=lambda row: (-float(row["headroom"]), row["scene"], row["view"]))[
        : max(int(top_k), 0)
    ]
    valid_scene_count = sum(1 for scene in scene_summaries if scene.get("status") == "ok")
    valid_view_count = sum(int(scene.get("valid_view_count") or 0) for scene in scene_summaries)
    primary_summary = metric_summary.get(primary_metric, {}) if primary_metric is not None else {}
    return {
        "name": name,
        "report_count": int(len(report_inputs)),
        "scene_count": int(len(scene_summaries)),
        "valid_scene_count": int(valid_scene_count),
        "valid_view_count": int(valid_view_count),
        "primary_metric": primary_metric,
        "macro": {
            "primary_metric": primary_metric,
            "scene_count": int(valid_scene_count),
            "view_count": int(valid_view_count),
            "primary": primary_summary,
            "metrics": metric_summary,
        },
        "selected_variant_counts": _sorted_counter(selected_counts),
        "scene_selected_variant_counts": _sorted_counter(scene_selected_counts),
        "output_variant_counts": _sorted_counter(output_counts),
        "best_variant_counts_by_metric": {
            metric: _sorted_counter(counts)
            for metric, counts in sorted(best_counts_by_metric.items(), key=lambda item: _metric_sort_key(item[0]))
        },
        "decision_source_counts": _sorted_counter(decision_counts),
        "decision_source_buckets": _aggregate_decision_buckets(scene_summaries),
        "schema_warning_counts": _sorted_counter(warning_counts),
        "input_errors": input_errors,
        "load_errors": load_errors,
        "largest_misses": largest_misses,
        "scenes": scene_summaries,
    }


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    method_inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_specs: list[dict[str, str]] = []

    for spec in args.method or []:
        name, root = _parse_named_path(spec, "--method")
        reports, errors = _discover_method_reports(name, root)
        method_inputs[name].extend(reports)
        method_errors[name].extend(errors)
        input_specs.append({"kind": "method", "name": name, "path": str(root)})

    for spec in args.report or []:
        name, path = _parse_named_path(spec, "--report")
        report_path = _normalize_report_path(path)
        method_inputs[name].append(
            {
                "method": name,
                "path": report_path,
                "scene_hint": report_path.parent.name,
                "source": "report",
            }
        )
        input_specs.append({"kind": "report", "name": name, "path": str(report_path)})

    if not method_inputs and not method_errors:
        raise ValueError("at least one --method NAME=ROOT or --report NAME=PATH is required")

    methods: list[dict[str, Any]] = []
    for name in sorted(set(method_inputs) | set(method_errors)):
        methods.append(
            _summarize_method(
                name=name,
                report_inputs=method_inputs.get(name, []),
                input_errors=method_errors.get(name, []),
                top_k=int(args.top_k),
                requested_primary_metric=args.primary_metric,
            )
        )

    return {
        "audit": "support_transport_oracle_gap",
        "schema_version": 1,
        "read_only": True,
        "target_gt_or_render_requirement": "none beyond metrics already recorded in report candidate_metrics",
        "oracle_rule": {
            "per_metric": "best candidate over numeric quality metrics in candidate_metrics",
            "headroom": "positive means oracle better than selected; max metrics use oracle-selected, min metrics use selected-oracle",
            "tie_break": "prefer selected variant on exact metric ties, otherwise lexical variant name",
            "primary_metric": args.primary_metric or "auto: psnr_gain, candidate_psnr, then preferred metric order",
            "eps": EPS,
        },
        "args": {
            "method": list(args.method or []),
            "report": list(args.report or []),
            "output_json": str(args.output_json) if args.output_json else None,
            "output_md": str(args.output_md) if args.output_md else None,
            "top_k": int(args.top_k),
            "primary_metric": args.primary_metric,
        },
        "inputs": input_specs,
        "methods": methods,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Support-Transport Oracle Gap Audit",
        "",
        "Read-only diagnostic over metrics already serialized in support_transport_apply_report.json.",
        "",
        "## Macro",
        "",
        "| method | scenes | views | primary | selected mean | oracle mean | mean headroom | positive views | selected counts | best counts |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for method in payload.get("methods", []):
        primary_metric = method.get("primary_metric")
        primary = method.get("macro", {}).get("primary", {}) if isinstance(method.get("macro"), dict) else {}
        best_counts = (
            method.get("best_variant_counts_by_metric", {}).get(primary_metric, {})
            if isinstance(method.get("best_variant_counts_by_metric"), dict)
            else {}
        )
        lines.append(
            "| {name} | {scenes} | {views} | {primary} | {selected} | {oracle} | {headroom} | {positive} | {selected_counts} | {best_counts} |".format(
                name=method.get("name", ""),
                scenes=method.get("valid_scene_count", 0),
                views=method.get("valid_view_count", 0),
                primary=primary_metric or "",
                selected=_format_float(primary.get("macro_selected_mean")),
                oracle=_format_float(primary.get("macro_oracle_mean")),
                headroom=_format_signed(primary.get("macro_mean_headroom")),
                positive=primary.get("positive_headroom_count", ""),
                selected_counts=_format_counts(method.get("output_variant_counts", {})),
                best_counts=_format_counts(best_counts),
            )
        )

    for method in payload.get("methods", []):
        method_name = str(method.get("name", ""))
        lines += [
            "",
            f"## {method_name}",
            "",
            "### Metric Headroom",
            "",
            "| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for metric, row in method.get("macro", {}).get("metrics", {}).items():
            lines.append(
                "| {metric} | {direction} | {scenes} | {views} | {selected} | {oracle} | {macro_headroom} | {micro_headroom} | {positive} |".format(
                    metric=metric,
                    direction=row.get("direction", ""),
                    scenes=row.get("scene_count", ""),
                    views=row.get("view_count", ""),
                    selected=_format_float(row.get("macro_selected_mean")),
                    oracle=_format_float(row.get("macro_oracle_mean")),
                    macro_headroom=_format_signed(row.get("macro_mean_headroom")),
                    micro_headroom=_format_signed(row.get("micro_mean_headroom")),
                    positive=row.get("positive_headroom_count", ""),
                )
            )

        lines += [
            "",
            "### Decision Sources",
            "",
            "| source | views | mean headroom | max headroom | positive views | output counts | best counts |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for source, row in method.get("decision_source_buckets", {}).items():
            lines.append(
                "| {source} | {views} | {mean_headroom} | {max_headroom} | {positive} | {outputs} | {best} |".format(
                    source=source,
                    views=row.get("view_count", ""),
                    mean_headroom=_format_signed(row.get("primary_mean_headroom")),
                    max_headroom=_format_signed(row.get("primary_max_headroom")),
                    positive=row.get("primary_positive_headroom_count", ""),
                    outputs=_format_counts(row.get("output_variant_counts", {})),
                    best=_format_counts(row.get("best_variant_counts", {})),
                )
            )

        lines += [
            "",
            "### Per Scene",
            "",
            "| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |",
            "|---|---|---:|---|---:|---:|---:|---:|---|---|---|",
        ]
        for scene in method.get("scenes", []):
            primary_metric = scene.get("primary_metric")
            metric = scene.get("metrics", {}).get(primary_metric, {}) if primary_metric else {}
            best_counts = scene.get("best_variant_counts_by_metric", {}).get(primary_metric, {}) if primary_metric else {}
            lines.append(
                "| {scene} | {status} | {views} | {primary} | {selected} | {oracle} | {headroom} | {positive} | {outputs} | {best} | {warnings} |".format(
                    scene=scene.get("scene", ""),
                    status=scene.get("status", ""),
                    views=scene.get("valid_view_count", 0),
                    primary=primary_metric or "",
                    selected=_format_float(metric.get("selected_mean")),
                    oracle=_format_float(metric.get("oracle_mean")),
                    headroom=_format_signed(metric.get("mean_headroom")),
                    positive=metric.get("positive_headroom_count", ""),
                    outputs=_format_counts(scene.get("output_variant_counts", {})),
                    best=_format_counts(best_counts),
                    warnings=_format_counts(scene.get("schema_warnings", {})),
                )
            )

        lines += [
            "",
            "### Largest Misses",
            "",
            "| scene | view | source | selected | best | metric | selected value | best value | headroom |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ]
        for row in method.get("largest_misses", []):
            lines.append(
                "| {scene} | {view} | {source} | {selected} | {best} | {metric} | {selected_value} | {best_value} | {headroom} |".format(
                    scene=row.get("scene", ""),
                    view=row.get("view", ""),
                    source=row.get("decision_source", ""),
                    selected=row.get("selected_variant", ""),
                    best=row.get("best_variant", ""),
                    metric=row.get("metric", ""),
                    selected_value=_format_float(row.get("selected_value")),
                    best_value=_format_float(row.get("best_value")),
                    headroom=_format_signed(row.get("headroom")),
                )
            )

        if method.get("input_errors") or method.get("load_errors") or method.get("schema_warning_counts"):
            lines += ["", "### Diagnostics", ""]
            if method.get("schema_warning_counts"):
                lines.append(f"- schema warnings: `{_format_counts(method.get('schema_warning_counts', {}))}`")
            for error in method.get("input_errors", []):
                lines.append(f"- input error: `{error}`")
            for error in method.get("load_errors", []):
                lines.append(f"- load error: `{error}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help=f"Method root as NAME=ROOT. ROOT may contain scene/{REPORT_NAME} files. Repeatable.",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help=f"Single report as NAME=PATH. PATH may be a {REPORT_NAME} file or a directory containing it. Repeatable.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--primary_metric",
        default=None,
        help="Optional metric key for largest-miss sorting. Default: auto, preferring psnr_gain.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = _build_payload(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output_json}")
    if args.output_md is not None:
        _write_markdown(args.output_md, payload)
        print(f"wrote {args.output_md}")
    if args.output_json is None and args.output_md is None:
        print(json.dumps(payload, indent=2, sort_keys=True))

    for method in payload.get("methods", []):
        primary_metric = method.get("primary_metric")
        primary = method.get("macro", {}).get("primary", {})
        print(
            "method={name} scenes={scenes} views={views} primary={primary} "
            "macro_headroom={headroom}".format(
                name=method.get("name"),
                scenes=method.get("valid_scene_count"),
                views=method.get("valid_view_count"),
                primary=primary_metric,
                headroom=_format_signed(primary.get("macro_mean_headroom")),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
