#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
METRIC_OUT = {"PSNR": "psnr", "SSIM": "ssim", "LPIPS": "lpips"}
REFERENCE_PREFIXES = {"clean", "endpoint", "reference", "v101", "v102", "baseline"}
VERSION_PRIORITY = ("v107", "v106", "v105b", "v105", "v104c", "v104", "v103", "v102", "v101")
POD_STAT_KEYS = (
    "valid_triangles",
    "total_accumulated_pixels",
    "mixture_triangles",
    "fallback_only_triangles",
    "view_affine_triangles",
    "fallback_triangles",
    "gate_mean",
    "gain_score_mean",
    "crossfit_gain_mean",
    "crossfit_gain_supported_triangles",
    "stability_score_mean",
    "debt_guard_mean",
    "detail_triangles",
    "boundary_triangles",
    "detail_reliability_mean",
    "boundary_reliability_mean",
    "detail_gain_mean",
    "boundary_gain_mean",
    "detail_full_gain_mean",
    "boundary_full_gain_mean",
    "detail_mse_scale_mean",
    "boundary_mse_scale_mean",
    "detail_debt_guard_mean",
    "boundary_debt_guard_mean",
    "detail_weighted_pixels",
    "boundary_weighted_pixels",
    "detail_crossfit_supported_triangles",
    "boundary_crossfit_supported_triangles",
    "detail_crossfit_gain_mean",
    "boundary_crossfit_gain_mean",
    "detail_crossfit_mse_scale_mean",
    "boundary_crossfit_mse_scale_mean",
    "detail_crossfit_even_fit_triangles",
    "detail_crossfit_odd_fit_triangles",
    "boundary_crossfit_even_fit_triangles",
    "boundary_crossfit_odd_fit_triangles",
    "base_observed_triangles",
    "base_mixture_triangles",
    "base_gate_mean",
    "base_gain_score_mean",
    "base_debt_guard_mean",
    "shrink_alpha_mean",
    "elapsed_sec",
    "field_elapsed_sec",
    "render_elapsed_sec",
    "mean_abs_delta",
    "mean_surface_valid_fraction",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(section: dict[str, Any], metric: str) -> float | None:
    for key in (metric, metric.lower(), metric.upper()):
        if key in section:
            return _float_or_none(section.get(key))
    return None


def _metrics_from_section(section: Any) -> dict[str, float | None]:
    data = _as_dict(section)
    return {METRIC_OUT[metric]: _metric_value(data, metric) for metric in METRICS}


def _has_any_metric(metrics: dict[str, float | None]) -> bool:
    return any(value is not None for value in metrics.values())


def _metrics_from_flat(row: dict[str, Any], prefix: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in METRICS:
        candidates = (
            f"{prefix}_{metric}",
            f"{prefix}_{metric.lower()}",
            f"{prefix.lower()}_{metric}",
            f"{prefix.lower()}_{metric.lower()}",
        )
        result[METRIC_OUT[metric]] = _first_float_from_keys(row, candidates)
    return result


def _first_float_from_keys(row: dict[str, Any], keys: tuple[str, ...] | list[str]) -> float | None:
    for key in keys:
        if key in row:
            value = _float_or_none(row.get(key))
            if value is not None:
                return value
    return None


def _delta(lhs: dict[str, float | None], rhs: dict[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("psnr", "ssim", "lpips"):
        left = lhs.get(key)
        right = rhs.get(key)
        out[key] = None if left is None or right is None else left - right
    return out


def _is_zero_metrics(metrics: dict[str, float | None]) -> bool:
    return all(value == 0.0 for value in metrics.values() if value is not None) and any(
        value is not None for value in metrics.values()
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"json decode failed for {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"json root is not an object: {path}")
    return payload


def _compact_method(raw: Any, scene: Any = None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = re.sub(r"^ours_\d+_", "", text)
    if scene:
        suffix = f"_{scene}"
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text or None


def _infer_version(parts: list[Any]) -> str | None:
    haystack = " ".join(str(part) for part in parts if part not in (None, ""))
    haystack_lower = haystack.lower()
    for version in VERSION_PRIORITY:
        if re.search(rf"(^|[^a-z0-9]){re.escape(version)}([^a-z0-9]|$)", haystack_lower):
            return version
    if "pod_moe" in haystack_lower or "pod-moe" in haystack_lower:
        return "v106"
    return None


def _method_label(row: dict[str, Any], source: Path, metric_prefix: str | None = None) -> tuple[str | None, str | None]:
    scene = row.get("scene")
    output_method = _first_present(row.get("output_method"), row.get("result_key"), row.get("method_key"))
    raw_method = _first_present(row.get("method"), output_method, metric_prefix, row.get("report_label"))
    compact = _compact_method(raw_method, scene)
    version = _infer_version(
        [
            compact,
            output_method,
            row.get("report_label"),
            row.get("claim_boundary"),
            row.get("field"),
            row.get("field_manifest"),
            row.get("support_source"),
            str(source),
        ]
    )
    if compact and version and version not in compact.lower():
        compact = f"{version}_{compact}"
    return compact, version


def _flat_metric_prefixes(row: dict[str, Any]) -> list[str]:
    prefixes: list[str] = []
    for key in row:
        for metric in METRICS:
            suffix = f"_{metric}"
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                if prefix and prefix not in prefixes:
                    prefixes.append(prefix)
    return prefixes


def _choose_flat_prefix(row: dict[str, Any], source: Path) -> str | None:
    prefixes = _flat_metric_prefixes(row)
    if not prefixes:
        return None
    version = _infer_version(
        [
            row.get("method"),
            row.get("output_method"),
            row.get("report_label"),
            row.get("claim_boundary"),
            row.get("field"),
            row.get("field_manifest"),
            str(source),
        ]
    )
    if version:
        for candidate in (version, "v105" if version == "v106" else ""):
            if candidate and candidate in prefixes:
                return candidate
    non_reference = [prefix for prefix in prefixes if prefix.lower() not in REFERENCE_PREFIXES]
    if len(non_reference) == 1:
        return non_reference[0]
    for preferred in VERSION_PRIORITY:
        if preferred in prefixes:
            return preferred
    return non_reference[0] if non_reference else prefixes[0]


def _nested_metric_key(metrics: dict[str, Any], method_hint: Any, source: Path) -> str | None:
    if not metrics:
        return None
    hint = str(method_hint or "").lower()
    if hint:
        for key in metrics:
            key_lower = str(key).lower()
            if key_lower == hint or key_lower in hint or hint in key_lower:
                return str(key)
    source_version = _infer_version([method_hint, str(source)])
    if source_version:
        for key in metrics:
            if source_version in str(key).lower():
                return str(key)
        if source_version == "v106":
            for key in metrics:
                if "v105" in str(key).lower():
                    return str(key)
    for key in metrics:
        key_lower = str(key).lower()
        if key_lower not in REFERENCE_PREFIXES and "endpoint" not in key_lower and "ceiling" not in key_lower:
            return str(key)
    return next(iter(metrics), None)


def _nested_v104c_key(metrics: dict[str, Any]) -> str | None:
    for key in metrics:
        if "v104c" in str(key).lower():
            return str(key)
    return None


def _extract_identity(row: dict[str, Any], key: str) -> Any:
    field_stats = _as_dict(row.get("field_stats"))
    field_identity = _as_dict(row.get("field_identity"))
    manifest_identity = _as_dict(field_identity.get("manifest"))
    render_stats = _as_dict(row.get("render_stats"))
    render_field = _as_dict(render_stats.get("surface_residual_field"))
    return _first_present(row.get(key), field_stats.get(key), manifest_identity.get(key), render_field.get(key))


def _extract_pod_stats(row: dict[str, Any]) -> dict[str, Any]:
    field_stats = _as_dict(row.get("field_stats"))
    render_stats = _as_dict(row.get("render_stats"))
    pod_stats: dict[str, Any] = {}
    for key in POD_STAT_KEYS:
        value = _first_present(row.get(key), field_stats.get(key), render_stats.get(key))
        if value is not None:
            pod_stats[key] = value
    return pod_stats


def _direct_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    metrics = _metrics_from_section(row.get("metrics"))
    if _has_any_metric(metrics):
        return metrics
    return {key: None for key in ("psnr", "ssim", "lpips")}


def _metrics_for_row(row: dict[str, Any], source: Path) -> tuple[dict[str, float | None], str | None]:
    metrics = _direct_metrics(row)
    if _has_any_metric(metrics):
        return metrics, None
    prefix = _choose_flat_prefix(row, source)
    if prefix is None:
        return {key: None for key in ("psnr", "ssim", "lpips")}, None
    return _metrics_from_flat(row, prefix), prefix


def _reference_metrics(row: dict[str, Any], prefix: str, section_key: str) -> dict[str, float | None]:
    nested = _metrics_from_section(row.get(section_key))
    if _has_any_metric(nested):
        return nested
    return _metrics_from_flat(row, prefix)


def _delta_from_row(row: dict[str, Any], label: str) -> dict[str, float | None]:
    deltas = _as_dict(row.get("deltas"))
    nested = _metrics_from_section(deltas.get(f"vs_{label}"))
    if _has_any_metric(nested):
        return nested
    prefix = f"d{{metric}}_vs_{label}"
    out: dict[str, float | None] = {}
    for metric in METRICS:
        out[METRIC_OUT[metric]] = _first_float_from_keys(
            row,
            (
                prefix.format(metric=metric),
                prefix.format(metric=metric.lower()),
                f"diff_{metric.lower()}_vs_{label}",
                f"delta_{metric.lower()}_vs_{label}",
            ),
        )
    return out


def _normalize_flat_row(row: dict[str, Any], source: Path, source_kind: str, default_scene: str | None = None) -> dict[str, Any]:
    row = dict(row)
    if default_scene and not row.get("scene"):
        row["scene"] = default_scene
    metrics, metric_prefix = _metrics_for_row(row, source)
    method, version = _method_label(row, source, metric_prefix)
    clean = _reference_metrics(row, "clean", "clean_metrics")
    v104c = _reference_metrics(row, "v104c", "v104c_metrics")

    diff_vs_clean = _delta_from_row(row, "clean")
    if not _has_any_metric(diff_vs_clean):
        diff_vs_clean = _delta(metrics, clean)

    diff_vs_v104c = _delta_from_row(row, "v104c")
    if not _has_any_metric(diff_vs_v104c):
        if _has_any_metric(v104c):
            diff_vs_v104c = _delta(metrics, v104c)
        elif str(metric_prefix or "").lower() == "v104c" or (method and "v104c" in method.lower()):
            diff_vs_v104c = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

    output_method = _first_present(row.get("output_method"), row.get("result_key"))
    return {
        "source": str(source),
        "source_kind": source_kind,
        "scene": row.get("scene"),
        "method": method,
        "version": version,
        "output_method": output_method,
        "status": row.get("status"),
        "passed": row.get("passed"),
        "metrics": metrics,
        "clean_metrics": clean if _has_any_metric(clean) else None,
        "v104c_metrics": v104c if _has_any_metric(v104c) else None,
        "diff_vs_clean": diff_vs_clean if _has_any_metric(diff_vs_clean) else {key: None for key in ("psnr", "ssim", "lpips")},
        "diff_vs_v104c": diff_vs_v104c if _has_any_metric(diff_vs_v104c) else {key: None for key in ("psnr", "ssim", "lpips")},
        "field_variant": _extract_identity(row, "field_variant"),
        "pod_base_keep_mode": _extract_identity(row, "pod_base_keep_mode"),
        "pod_view_gate_mode": _extract_identity(row, "pod_view_gate_mode"),
        "view_gate_temperature": _extract_identity(row, "view_gate_temperature"),
        "expert_mse_certificate": _extract_identity(row, "expert_mse_certificate"),
        "pod_stats": _extract_pod_stats(row),
    }


def _normalize_scene_dict(
    scene: str,
    scene_payload: dict[str, Any],
    source: Path,
    top_payload: dict[str, Any],
) -> dict[str, Any] | None:
    metrics_by_method = _as_dict(scene_payload.get("metrics"))
    if not metrics_by_method:
        row = dict(scene_payload)
        row.setdefault("scene", scene)
        return _normalize_flat_row(row, source, "summary_scene")

    method_key = _nested_metric_key(metrics_by_method, top_payload.get("method"), source)
    if method_key is None:
        return None
    row = dict(scene_payload)
    row["scene"] = scene
    row["method_key"] = method_key
    row["method"] = _first_present(top_payload.get("method"), method_key)
    row["output_method"] = _first_present(scene_payload.get("output_method"), scene_payload.get("result_key"))
    row["metrics"] = metrics_by_method.get(method_key)
    clean_metrics = _as_dict(metrics_by_method.get("clean"))
    if clean_metrics:
        row["clean_metrics"] = clean_metrics
    v104c_key = _nested_v104c_key(metrics_by_method)
    if v104c_key is not None:
        row["v104c_metrics"] = metrics_by_method.get(v104c_key)
    row["field_stats"] = _first_present(scene_payload.get("field_stats"), scene_payload.get("pod_stats"))
    return _normalize_flat_row(row, source, "summary_scene")


def _normalize_top_mean(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mean_payload = _first_present(payload.get("mean"), payload.get("aggregate"))
    if isinstance(mean_payload, dict):
        row = dict(mean_payload)
        row["scene"] = "__aggregate__"
        row["method"] = payload.get("method")
        rows.append(_normalize_flat_row(row, source, "summary_aggregate"))

    means = _as_dict(payload.get("means"))
    if means:
        method_key = _nested_metric_key(means, payload.get("method"), source)
        if method_key is not None:
            row = {
                "scene": "__aggregate__",
                "method": _first_present(payload.get("method"), method_key),
                "metrics": means.get(method_key),
            }
            if "clean" in means:
                row["clean_metrics"] = means.get("clean")
            v104c_key = _nested_v104c_key(means)
            if v104c_key is not None:
                row["v104c_metrics"] = means.get(v104c_key)
            rows.append(_normalize_flat_row(row, source, "summary_aggregate"))
    return [row for row in rows if _has_any_metric(row.get("metrics", {}))]


def extract_rows(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _as_list(payload.get("rows")):
        if isinstance(row, dict):
            rows.append(_normalize_flat_row(row, source, "summary_row"))

    scenes = payload.get("scenes")
    if isinstance(scenes, dict):
        for scene, scene_payload in sorted(scenes.items()):
            if isinstance(scene_payload, dict):
                normalized = _normalize_scene_dict(str(scene), scene_payload, source, payload)
                if normalized is not None:
                    rows.append(normalized)

    if not rows and ("scene" in payload or "metrics" in payload):
        rows.append(_normalize_flat_row(payload, source, "report"))

    if not rows:
        rows.extend(_normalize_top_mean(payload, source))

    return rows


def _numeric_values(rows: list[dict[str, Any]], getter: Any) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_or_none(getter(row))
        if value is not None:
            values.append(value)
    return values


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _mean_metric(rows: list[dict[str, Any]], section: str, metric: str) -> float | None:
    return _mean_or_none(_numeric_values(rows, lambda row: _as_dict(row.get(section)).get(metric)))


def build_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(_first_present(row.get("method"), row.get("version"), "unknown"))
        by_method.setdefault(method, []).append(row)

    method_summaries: list[dict[str, Any]] = []
    for method in sorted(by_method):
        method_rows = by_method[method]
        method_summaries.append(
            {
                "method": method,
                "rows": len(method_rows),
                "scenes": sorted(str(row.get("scene")) for row in method_rows if row.get("scene") not in (None, "__aggregate__")),
                "metrics_mean": {metric: _mean_metric(method_rows, "metrics", metric) for metric in ("psnr", "ssim", "lpips")},
                "diff_vs_clean_mean": {
                    metric: _mean_metric(method_rows, "diff_vs_clean", metric) for metric in ("psnr", "ssim", "lpips")
                },
                "diff_vs_v104c_mean": {
                    metric: _mean_metric(method_rows, "diff_vs_v104c", metric) for metric in ("psnr", "ssim", "lpips")
                },
            }
        )

    rows_by_version: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        version = str(row.get("version") or "").lower()
        if not version:
            version = _infer_version([row.get("method"), row.get("output_method")]) or "unknown"
        rows_by_version.setdefault(version, []).append(row)
    return {
        "row_count": len(rows),
        "methods": method_summaries,
        "version_vs_v104c": {
            version: {
                "rows": len(version_rows),
                "scenes": sorted(
                    str(row.get("scene"))
                    for row in version_rows
                    if row.get("scene") not in (None, "__aggregate__")
                ),
                "diff_mean": {
                    metric: _mean_metric(version_rows, "diff_vs_v104c", metric)
                    for metric in ("psnr", "ssim", "lpips")
                },
            }
            for version, version_rows in sorted(rows_by_version.items())
        },
    }


def build_output(inputs: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        rows.extend(extract_rows(_read_json(path), path))
    return {
        "schema_version": 1,
        "inputs": [str(path) for path in inputs],
        "row_count": len(rows),
        "rows": rows,
        "aggregate": build_aggregate(rows),
        "notes": [
            "Metric deltas are candidate minus reference; for LPIPS, negative is better.",
            "diff_vs_v104c is null when the input did not carry v104c metrics and the gap cannot be derived.",
        ],
    }


def _fmt(value: Any, signed: bool = False) -> str:
    value_float = _float_or_none(value)
    if value_float is None:
        return ""
    return f"{value_float:+.6f}" if signed else f"{value_float:.6f}"


def _md_delta(value: Any, highlight: bool) -> str:
    text = _fmt(value, signed=True)
    if not text:
        return ""
    return f"**{text}**" if highlight else text


def markdown_table(payload: dict[str, Any]) -> str:
    rows = _as_list(payload.get("rows"))
    aggregate = _as_dict(payload.get("aggregate"))
    lines = [
        "# Candidate Report Comparison",
        "",
        f"- inputs: `{len(_as_list(payload.get('inputs')))}`",
        f"- rows: `{payload.get('row_count')}`",
        "- deltas: candidate minus reference; lower LPIPS is better.",
        "",
        "## Candidate vs v104c",
        "",
        "| scene | method | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c | field variant | POD base | POD view gate | vgt | expert certificate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|",
    ]
    for row in rows:
        metrics = _as_dict(row.get("metrics"))
        gap = _as_dict(row.get("diff_vs_v104c"))
        is_current_candidate = str(row.get("version") or "").lower() in {"v106", "v107"} or any(
            marker in str(row.get("method") or "").lower() for marker in ("v106", "v107")
        )
        lines.append(
            "| {scene} | {method} | {psnr} | {ssim} | {lpips} | {dpsnr} | {dssim} | {dlpips} | {variant} | {pod_base} | {pod_gate} | {vgt} | {cert} |".format(
                scene=row.get("scene", ""),
                method=row.get("method", ""),
                psnr=_fmt(metrics.get("psnr")),
                ssim=_fmt(metrics.get("ssim")),
                lpips=_fmt(metrics.get("lpips")),
                dpsnr=_md_delta(gap.get("psnr"), is_current_candidate),
                dssim=_md_delta(gap.get("ssim"), is_current_candidate),
                dlpips=_md_delta(gap.get("lpips"), is_current_candidate),
                variant=row.get("field_variant") or "",
                pod_base=row.get("pod_base_keep_mode") or "",
                pod_gate=row.get("pod_view_gate_mode") or "",
                vgt=_fmt(row.get("view_gate_temperature")),
                cert=row.get("expert_mse_certificate") or "",
            )
        )

    lines.extend(
        [
            "",
            "## Mean Gap by Method",
            "",
            "| method | rows | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _as_list(aggregate.get("methods")):
        clean_gap = _as_dict(row.get("diff_vs_clean_mean"))
        v104c_gap = _as_dict(row.get("diff_vs_v104c_mean"))
        is_current_candidate = any(marker in str(row.get("method") or "").lower() for marker in ("v106", "v107"))
        lines.append(
            "| {method} | {rows} | {dc_psnr} | {dc_ssim} | {dc_lpips} | {dv_psnr} | {dv_ssim} | {dv_lpips} |".format(
                method=row.get("method", ""),
                rows=row.get("rows", ""),
                dc_psnr=_fmt(clean_gap.get("psnr"), True),
                dc_ssim=_fmt(clean_gap.get("ssim"), True),
                dc_lpips=_fmt(clean_gap.get("lpips"), True),
                dv_psnr=_md_delta(v104c_gap.get("psnr"), is_current_candidate),
                dv_ssim=_md_delta(v104c_gap.get("ssim"), is_current_candidate),
                dv_lpips=_md_delta(v104c_gap.get("lpips"), is_current_candidate),
            )
        )

    lines.extend(
        [
            "",
            "## Key POD Stats",
            "",
            "| scene | method | valid triangles | mixture triangles | fallback only | detail triangles | boundary triangles | gate mean | debt guard | mean abs delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        stats = _as_dict(row.get("pod_stats"))
        lines.append(
            "| {scene} | {method} | {valid} | {mixture} | {fallback} | {detail} | {boundary} | {gate} | {debt} | {mad} |".format(
                scene=row.get("scene", ""),
                method=row.get("method", ""),
                valid=_fmt(stats.get("valid_triangles")),
                mixture=_fmt(stats.get("mixture_triangles")),
                fallback=_fmt(stats.get("fallback_only_triangles")),
                detail=_fmt(stats.get("detail_triangles")),
                boundary=_fmt(stats.get("boundary_triangles")),
                gate=_fmt(stats.get("gate_mean")),
                debt=_fmt(stats.get("debt_guard_mean")),
                mad=_fmt(stats.get("mean_abs_delta")),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline/v104c/candidate report or summary JSON files."
    )
    parser.add_argument("inputs", nargs="+", help="Report JSON or summary JSON paths.")
    parser.add_argument("--out_json", default="", help="Optional path for compact normalized JSON output.")
    parser.add_argument("--out_md", default="", help="Optional path for a Markdown comparison table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    inputs = [Path(path) for path in args.inputs]
    payload = build_output(inputs)
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json_text, encoding="utf-8")
    else:
        sys.stdout.write(json_text)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown_table(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
