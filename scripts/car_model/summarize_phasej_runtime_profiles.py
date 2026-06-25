#!/usr/bin/env python3
"""Summarize raw Phase-J integrated runtime JSON profiles.

This is intentionally CPU-only and stdlib-only. It aggregates profiler JSON
files produced per scene, normalizes the common runtime fields, and optionally
joins render-only summary rows for comparison columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


KNOWN_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
SKIP_JSON_NAMES = {"summary.json", "driver_summary.json"}
GENERIC_PATH_PARTS = {
    "compact_model",
    "model",
    "point_cloud",
    "outputs",
    "carnet",
    "meshsplatopt",
    "spcarnet",
    "raw",
    "test",
    "train",
}

CSV_FIELDS = [
    "scene",
    "label",
    "benchmark",
    "scope",
    "source_json",
    "model_path",
    "split",
    "loaded_iteration",
    "target_views",
    "available_target_views",
    "view_stride",
    "warmup_views",
    "support_frames",
    "repeats",
    "elapsed_sec_mean",
    "elapsed_sec_stdev",
    "ms_per_view_mean",
    "ms_per_view_stdev",
    "fps_mean",
    "fps_stdev",
    "render_ms_per_view_mean",
    "adapter_ms_per_view_mean",
    "adapter_over_render_ratio_mean",
    "peak_allocated_mib_mean",
    "peak_allocated_mib_max",
    "peak_reserved_mib_mean",
    "peak_reserved_mib_max",
    "triangles",
    "vertices",
    "mean_covered_fraction",
    "mean_confidence",
    "alpha",
    "k",
    "mode",
    "depth_rel_tol",
    "residual_clip",
    "direction_weight",
    "alpha_calibrator_loaded",
    "benefit_calibrator_loaded",
    "render_only_clean_ms_per_view",
    "render_only_compact_ms_per_view",
    "render_only_delta_ms_per_view",
    "render_only_clean_fps",
    "render_only_compact_fps",
    "integrated_minus_render_only_compact_ms_per_view",
    "integrated_over_render_only_compact_ms_ratio",
    "adapter_over_render_only_compact_ms_ratio",
]


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_finite(*values: Any) -> float | None:
    for value in values:
        out = _finite(value)
        if out is not None:
            return out
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [_finite(value) for value in values]
    finite = [value for value in finite if value is not None]
    return float(statistics.mean(finite)) if finite else None


def _stdev(values: Iterable[Any]) -> float:
    finite = [_finite(value) for value in values]
    finite = [value for value in finite if value is not None]
    return float(statistics.stdev(finite)) if len(finite) > 1 else 0.0


def _max(values: Iterable[Any]) -> float | None:
    finite = [_finite(value) for value in values]
    finite = [value for value in finite if value is not None]
    return max(finite) if finite else None


def _weighted_mean(rows: Sequence[dict[str, Any]], value_key: str, weight_key: str = "target_views") -> float | None:
    total_weight = 0.0
    total = 0.0
    for row in rows:
        value = _finite(row.get(value_key))
        weight = _finite(row.get(weight_key))
        if value is None or weight is None or weight <= 0.0:
            continue
        total += value * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return total / total_weight


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _finite(numerator)
    bottom = _finite(denominator)
    if top is None or bottom is None or abs(bottom) <= 1e-12:
        return None
    return top / bottom


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}
    return payload if isinstance(payload, dict) else {"_read_error": "top-level JSON is not an object"}


def _iter_json_files(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("*.json") if path.name not in SKIP_JSON_NAMES)


def _scene_from_parts(parts: Sequence[str]) -> str | None:
    lowered = [part.lower() for part in parts]
    for scene in KNOWN_SCENES:
        if scene in lowered:
            return scene
    for idx, part in enumerate(lowered):
        if part.startswith("ratio_") and idx > 0:
            return parts[idx - 1]
    for part in reversed(parts):
        if not part or part.lower() in GENERIC_PATH_PARTS or part.startswith("iteration_"):
            continue
        if part.startswith("ratio_") or part.startswith("ours_"):
            continue
        return part
    return None


def _scene_from_text(text: str) -> str | None:
    normalized = text.replace("-", "_").replace(".", "_").lower()
    tokens = [token for token in normalized.split("_") if token]
    for scene in KNOWN_SCENES:
        if scene in tokens:
            return scene
    return None


def _infer_scene(payload: dict[str, Any], path: Path) -> str:
    scene = payload.get("scene")
    if scene:
        return str(scene)

    for key in ("model_path", "base_model_path", "point_cloud_dir"):
        value = payload.get(key)
        if value:
            inferred = _scene_from_parts(Path(str(value)).parts)
            if inferred:
                return str(inferred)

    for key in ("label", "benchmark"):
        value = payload.get(key)
        if value:
            inferred = _scene_from_text(str(value))
            if inferred:
                return inferred

    inferred = _scene_from_text(path.stem)
    if inferred:
        return inferred
    return path.stem


def _repeat_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("repeat_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _frame_values(repeat_rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for repeat in repeat_rows:
        frames = repeat.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            value = _finite(frame.get(key))
            if value is not None:
                values.append(value)
    return values


def _repeat_field_values(repeat_rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in repeat_rows:
        value = _finite(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _row_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    repeats = _repeat_rows(payload)
    ms_from_repeats = _repeat_field_values(repeats, "ms_per_view")
    if not ms_from_repeats:
        ms_from_repeats = _repeat_field_values(repeats, "ms_per_target_frame")
    fps_from_repeats = _repeat_field_values(repeats, "fps")
    if not fps_from_repeats:
        fps_from_repeats = _repeat_field_values(repeats, "target_frames_per_sec")

    target_views = _int_or_none(
        _first_present(payload.get("num_views"), payload.get("target_frame_count"), payload.get("views"))
    )
    available_target_views = _int_or_none(
        _first_present(payload.get("available_target_frame_count"), payload.get("available_target_views"))
    )

    mean_covered = _first_finite(
        payload.get("mean_covered_fraction"),
        _mean(repeat.get("mean_covered_fraction") for repeat in repeats),
        _mean(_frame_values(repeats, "covered_fraction")),
    )
    mean_confidence = _first_finite(
        payload.get("mean_confidence"),
        _mean(repeat.get("mean_confidence") for repeat in repeats),
        _mean(_frame_values(repeats, "mean_confidence")),
    )

    elapsed_mean = _first_finite(
        payload.get("elapsed_sec_mean"),
        payload.get("cpu_wall_time_sec_mean"),
        _mean(repeat.get("elapsed_sec") for repeat in repeats),
        _mean(repeat.get("cpu_wall_time_sec") for repeat in repeats),
    )
    ms_mean = _first_finite(
        payload.get("ms_per_view_mean"),
        payload.get("ms_per_target_frame_mean"),
        _mean(ms_from_repeats),
    )
    fps_mean = _first_finite(
        payload.get("fps_mean"),
        payload.get("target_frames_per_sec_mean"),
        _mean(fps_from_repeats),
        1000.0 / ms_mean if ms_mean and ms_mean > 0.0 else None,
    )

    alloc_values = _repeat_field_values(repeats, "peak_allocated_mib") + _repeat_field_values(
        repeats, "cuda_peak_allocated_mib"
    )
    reserved_values = _repeat_field_values(repeats, "peak_reserved_mib") + _repeat_field_values(
        repeats, "cuda_peak_reserved_mib"
    )

    return {
        "scene": _infer_scene(payload, path),
        "label": payload.get("label"),
        "benchmark": payload.get("benchmark"),
        "scope": payload.get("scope"),
        "source_json": str(path),
        "model_path": _first_present(payload.get("model_path"), payload.get("base_model_path")),
        "split": _first_present(payload.get("split"), payload.get("target_split")),
        "loaded_iteration": _int_or_none(payload.get("loaded_iteration")),
        "target_views": target_views,
        "available_target_views": available_target_views,
        "view_stride": _int_or_none(payload.get("view_stride")),
        "warmup_views": _int_or_none(payload.get("warmup_views")),
        "support_frames": _int_or_none(payload.get("support_frame_count")),
        "repeats": _int_or_none(payload.get("repeats")) or len(repeats) or None,
        "elapsed_sec_mean": elapsed_mean,
        "elapsed_sec_stdev": _first_finite(
            payload.get("elapsed_sec_stdev"),
            payload.get("cpu_wall_time_sec_stdev"),
            _stdev(repeat.get("elapsed_sec") for repeat in repeats),
            _stdev(repeat.get("cpu_wall_time_sec") for repeat in repeats),
        ),
        "ms_per_view_mean": ms_mean,
        "ms_per_view_stdev": _first_finite(
            payload.get("ms_per_view_stdev"),
            payload.get("ms_per_target_frame_stdev"),
            _stdev(ms_from_repeats),
        ),
        "fps_mean": fps_mean,
        "fps_stdev": _first_finite(
            payload.get("fps_stdev"),
            payload.get("target_frames_per_sec_stdev"),
            _stdev(fps_from_repeats),
        ),
        "render_ms_per_view_mean": _first_finite(
            payload.get("render_ms_per_view_mean"),
            _mean(repeat.get("render_ms_per_view") for repeat in repeats),
        ),
        "adapter_ms_per_view_mean": _first_finite(
            payload.get("adapter_ms_per_view_mean"),
            payload.get("ms_per_target_frame_mean"),
            _mean(repeat.get("adapter_ms_per_view") for repeat in repeats),
        ),
        "adapter_over_render_ratio_mean": _first_finite(
            payload.get("adapter_over_render_ratio_mean"),
            _mean(repeat.get("adapter_over_render_ratio") for repeat in repeats),
        ),
        "peak_allocated_mib_mean": _first_finite(
            payload.get("peak_allocated_mib_mean"),
            payload.get("cuda_peak_allocated_mib_mean"),
            _mean(alloc_values),
        ),
        "peak_allocated_mib_max": _first_finite(
            payload.get("peak_allocated_mib_max"),
            payload.get("cuda_peak_allocated_mib_max"),
            _max(alloc_values),
        ),
        "peak_reserved_mib_mean": _first_finite(
            payload.get("peak_reserved_mib_mean"),
            payload.get("cuda_peak_reserved_mib_mean"),
            _mean(reserved_values),
        ),
        "peak_reserved_mib_max": _first_finite(
            payload.get("peak_reserved_mib_max"),
            payload.get("cuda_peak_reserved_mib_max"),
            _max(reserved_values),
        ),
        "triangles": _int_or_none(payload.get("triangles")),
        "vertices": _int_or_none(payload.get("vertices")),
        "mean_covered_fraction": mean_covered,
        "mean_confidence": mean_confidence,
        "alpha": _finite(payload.get("alpha")),
        "k": _int_or_none(payload.get("k")),
        "mode": payload.get("mode"),
        "depth_rel_tol": _finite(payload.get("depth_rel_tol")),
        "residual_clip": _finite(payload.get("residual_clip")),
        "direction_weight": _finite(payload.get("direction_weight")),
        "alpha_calibrator_loaded": payload.get("alpha_calibrator_loaded"),
        "benefit_calibrator_loaded": payload.get("benefit_calibrator_loaded"),
    }


def _load_render_only_rows(path_text: str | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    if not path_text:
        return {}, {}, ""
    path = Path(path_text)
    payload = _read_json(path)
    if not payload or payload.get("_read_error"):
        return {}, {"read_error": payload.get("_read_error") if payload else "empty"}, str(path)

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}, payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}, str(path)

    by_scene: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        scene = row.get("scene")
        if scene:
            by_scene[str(scene)] = row
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return by_scene, summary, str(path)


def _join_render_only(rows: list[dict[str, Any]], render_rows: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        render_row = render_rows.get(str(row.get("scene")))
        if not render_row:
            continue
        clean_ms = _finite(render_row.get("clean_ms_per_view"))
        compact_ms = _finite(render_row.get("compact_ms_per_view"))
        row["render_only_clean_ms_per_view"] = clean_ms
        row["render_only_compact_ms_per_view"] = compact_ms
        row["render_only_delta_ms_per_view"] = _finite(render_row.get("delta_ms_per_view"))
        row["render_only_clean_fps"] = _finite(render_row.get("clean_fps"))
        row["render_only_compact_fps"] = _finite(render_row.get("compact_fps"))
        integrated_ms = _finite(row.get("ms_per_view_mean"))
        adapter_ms = _finite(row.get("adapter_ms_per_view_mean"))
        if integrated_ms is not None and compact_ms is not None:
            row["integrated_minus_render_only_compact_ms_per_view"] = integrated_ms - compact_ms
            row["integrated_over_render_only_compact_ms_ratio"] = _ratio(integrated_ms, compact_ms)
        if adapter_ms is not None and compact_ms is not None:
            row["adapter_over_render_only_compact_ms_ratio"] = _ratio(adapter_ms, compact_ms)


def _summary(rows: list[dict[str, Any]], label: str, raw_dir: Path, render_summary: dict[str, Any], render_path: str) -> dict[str, Any]:
    weighted_ms = _weighted_mean(rows, "ms_per_view_mean")
    weighted_render_ms = _weighted_mean(rows, "render_ms_per_view_mean")
    weighted_adapter_ms = _weighted_mean(rows, "adapter_ms_per_view_mean")
    weighted_compact_render_ms = _weighted_mean(rows, "render_only_compact_ms_per_view")
    weighted_clean_render_ms = _weighted_mean(rows, "render_only_clean_ms_per_view")
    target_views = sum(int(row["target_views"]) for row in rows if _int_or_none(row.get("target_views")) is not None)
    repeat_values = sorted({int(row["repeats"]) for row in rows if _int_or_none(row.get("repeats")) is not None})
    render_matches = sum(1 for row in rows if _finite(row.get("render_only_compact_ms_per_view")) is not None)

    out: dict[str, Any] = {
        "label": label,
        "raw_dir": str(raw_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenes": len(rows),
        "target_views": target_views,
        "raw_json_count": len(rows),
        "repeats_per_scene": repeat_values,
        "mean_scene_ms_per_view": _mean(row.get("ms_per_view_mean") for row in rows),
        "weighted_ms_per_view": weighted_ms,
        "weighted_fps": 1000.0 / weighted_ms if weighted_ms and weighted_ms > 0.0 else None,
        "mean_scene_fps": _mean(row.get("fps_mean") for row in rows),
        "weighted_render_ms_per_view": weighted_render_ms,
        "weighted_adapter_ms_per_view": weighted_adapter_ms,
        "weighted_adapter_over_render_ratio": _ratio(weighted_adapter_ms, weighted_render_ms),
        "mean_scene_render_ms_per_view": _mean(row.get("render_ms_per_view_mean") for row in rows),
        "mean_scene_adapter_ms_per_view": _mean(row.get("adapter_ms_per_view_mean") for row in rows),
        "max_peak_allocated_mib": _max(row.get("peak_allocated_mib_max") for row in rows),
        "mean_peak_allocated_mib": _mean(row.get("peak_allocated_mib_max") for row in rows),
        "max_peak_reserved_mib": _max(row.get("peak_reserved_mib_max") for row in rows),
        "mean_peak_reserved_mib": _mean(row.get("peak_reserved_mib_max") for row in rows),
        "mean_covered_fraction": _mean(row.get("mean_covered_fraction") for row in rows),
        "mean_confidence": _mean(row.get("mean_confidence") for row in rows),
        "scope": "phasej_integrated_runtime_raw_aggregation_no_cuda_imports",
    }

    if render_path:
        out.update(
            {
                "render_only_summary_json": render_path,
                "render_only_scene_matches": render_matches,
                "render_only_compact_mean_ms_per_view": render_summary.get("mean_compact_ms_per_view"),
                "render_only_clean_mean_ms_per_view": render_summary.get("mean_clean_ms_per_view"),
                "weighted_render_only_compact_ms_per_view": weighted_compact_render_ms,
                "weighted_render_only_clean_ms_per_view": weighted_clean_render_ms,
                "integrated_over_render_only_compact_ms_ratio": _ratio(weighted_ms, weighted_compact_render_ms),
                "adapter_over_render_only_compact_ms_ratio": _ratio(weighted_adapter_ms, weighted_compact_render_ms),
            }
        )
    return out


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    number = _finite(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}f}"


def _write_md(path: Path, label: str, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    has_render = "render_only_summary_json" in summary
    lines = [
        "# Phase-J Integrated Runtime Profile",
        "",
        "This summary aggregates raw per-scene runtime JSON files using only CPU-side JSON parsing.",
        "",
        "## Summary",
        "",
        f"- label: `{label}`",
        f"- scenes: `{summary['scenes']}`",
        f"- target views: `{summary['target_views']}`",
        f"- repeats per scene: `{summary['repeats_per_scene']}`",
        f"- weighted ms/view: `{_fmt(summary.get('weighted_ms_per_view'))}`",
        f"- weighted FPS: `{_fmt(summary.get('weighted_fps'))}`",
        f"- weighted render ms/view: `{_fmt(summary.get('weighted_render_ms_per_view'))}`",
        f"- weighted adapter ms/view: `{_fmt(summary.get('weighted_adapter_ms_per_view'))}`",
        f"- weighted adapter/render ratio: `{_fmt(summary.get('weighted_adapter_over_render_ratio'))}`",
        f"- max peak allocated MiB: `{_fmt(summary.get('max_peak_allocated_mib'), 3)}`",
        f"- max peak reserved MiB: `{_fmt(summary.get('max_peak_reserved_mib'), 3)}`",
    ]
    if has_render:
        lines.extend(
            [
                f"- render-only summary: `{summary.get('render_only_summary_json')}`",
                f"- render-only matched scenes: `{summary.get('render_only_scene_matches')}`",
                f"- weighted render-only compact ms/view: `{_fmt(summary.get('weighted_render_only_compact_ms_per_view'))}`",
                f"- integrated/render-only compact ms ratio: `{_fmt(summary.get('integrated_over_render_only_compact_ms_ratio'))}`",
                f"- adapter/render-only compact ms ratio: `{_fmt(summary.get('adapter_over_render_only_compact_ms_ratio'))}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-Scene",
            "",
            "| scene | views | repeats | ms/view | FPS | render ms | adapter ms | adapter/render | peak alloc MiB | covered | confidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('scene')} | {row.get('target_views') or 'n/a'} | {row.get('repeats') or 'n/a'} | "
            f"{_fmt(row.get('ms_per_view_mean'))} | {_fmt(row.get('fps_mean'))} | "
            f"{_fmt(row.get('render_ms_per_view_mean'))} | {_fmt(row.get('adapter_ms_per_view_mean'))} | "
            f"{_fmt(row.get('adapter_over_render_ratio_mean'))} | {_fmt(row.get('peak_allocated_mib_max'), 3)} | "
            f"{_fmt(row.get('mean_covered_fraction'))} | {_fmt(row.get('mean_confidence'))} |"
        )

    if has_render:
        lines.extend(
            [
                "",
                "## Render-Only Comparison",
                "",
                "| scene | integrated ms/view | render-only compact ms/view | delta ms/view | integrated/render-only ratio | adapter/render-only ratio |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row.get('scene')} | {_fmt(row.get('ms_per_view_mean'))} | "
                f"{_fmt(row.get('render_only_compact_ms_per_view'))} | "
                f"{_fmt(row.get('integrated_minus_render_only_compact_ms_per_view'))} | "
                f"{_fmt(row.get('integrated_over_render_only_compact_ms_ratio'))} | "
                f"{_fmt(row.get('adapter_over_render_only_compact_ms_ratio'))} |"
            )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Aggregates existing profiler JSON files only; it does not render, call CUDA, compute image metrics, or import project GPU modules.",
            "- Missing optional fields are emitted as `null` in JSON and blank cells in CSV.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_driver_summary(raw_dir: Path) -> dict[str, Any] | None:
    for candidate in (raw_dir / "driver_summary.json", raw_dir.parent / "driver_summary.json"):
        if not candidate.is_file():
            continue
        payload = _read_json(candidate)
        if payload and not payload.get("_read_error"):
            payload["_source_driver_summary_json"] = str(candidate)
            return payload
    return None


def _write_driver_summary(path: Path, rows: Sequence[dict[str, Any]], raw_dir: Path, summary: dict[str, Any]) -> None:
    if not rows:
        return

    source_driver = _load_driver_summary(raw_dir)
    if source_driver is not None and isinstance(source_driver.get("results"), list):
        payload = dict(source_driver)
        by_json = {str(row.get("source_json")): row for row in rows}
        by_scene = {str(row.get("scene")): row for row in rows}
        merged_results = []
        for item in source_driver["results"]:
            if not isinstance(item, dict):
                merged_results.append(item)
                continue
            row = by_json.get(str(item.get("out_json"))) or by_scene.get(str(item.get("scene")))
            merged = dict(item)
            if row:
                merged.update(
                    {
                        "scene": row.get("scene"),
                        "target_views": row.get("target_views"),
                        "repeats": row.get("repeats"),
                        "ms_per_view_mean": row.get("ms_per_view_mean"),
                        "fps_mean": row.get("fps_mean"),
                        "source_json": row.get("source_json"),
                    }
                )
            merged_results.append(merged)
        payload["results"] = merged_results
        payload["aggregate_summary"] = summary
    else:
        payload = {
            "source": "aggregated_raw_profiles",
            "raw_dir": str(raw_dir),
            "generated_at_utc": summary.get("generated_at_utc"),
            "aggregate_summary": summary,
            "results": [
                {
                    "scene": row.get("scene"),
                    "out_json": row.get("source_json"),
                    "target_views": row.get("target_views"),
                    "repeats": row.get("repeats"),
                    "ms_per_view_mean": row.get("ms_per_view_mean"),
                    "fps_mean": row.get("fps_mean"),
                }
                for row in rows
            ],
        }

    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def build_summary(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"--raw_dir is not a directory: {raw_dir}")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in _iter_json_files(raw_dir):
        payload = _read_json(path)
        if not payload or payload.get("_read_error"):
            skipped.append({"path": str(path), "reason": str(payload.get("_read_error") if payload else "empty")})
            continue
        rows.append(_row_from_payload(path, payload))

    rows.sort(key=lambda row: (str(row.get("scene")), str(row.get("source_json"))))
    render_rows, render_summary, render_path = _load_render_only_rows(args.render_only_summary_json)
    _join_render_only(rows, render_rows)
    summary = _summary(rows, str(args.label), raw_dir, render_summary, render_path)
    summary["skipped_json_count"] = len(skipped)
    if skipped:
        summary["skipped_json"] = skipped
    return summary, rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate raw Phase-J integrated runtime JSON profiles into summary.json, per_scene.csv, and summary.md."
    )
    parser.add_argument("--raw_dir", required=True, help="Directory containing raw per-scene integrated runtime JSON files.")
    parser.add_argument("--out_dir", required=True, help="Directory where summary artifacts will be written.")
    parser.add_argument("--label", required=True, help="Human-readable label for this aggregated profile.")
    parser.add_argument(
        "--render_only_summary_json",
        default="",
        help="Optional render-only summary.json with rows keyed by scene for comparison columns.",
    )
    args = parser.parse_args()

    summary, rows, _skipped = build_summary(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"summary": summary, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "per_scene.csv", rows)
    _write_md(out_dir / "summary.md", str(args.label), rows, summary)
    _write_driver_summary(out_dir / "driver_summary.json", rows, Path(args.raw_dir), summary)

    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
