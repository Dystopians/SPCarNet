#!/usr/bin/env python3
"""Collect read-only diagnostics for v107 CrossFit POD-MoE field runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_REPORT_ROOT = Path("/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports")
DEFAULT_FIELD_ROOT = Path("/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_field")
DEFAULT_V106_FULL9_JSON = Path("/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_assembled_20260625.json")
DEFAULT_SCENES = ("counter", "flowers", "garden", "bonsai")

REFERENCE_KEYS = ("baseline", "reference", "v102", "clean", "v104", "v106")
COMMON_IDENTITY_KEYS = (
    "manifest_present",
    "method_version",
    "field_type",
    "field_variant",
    "basis_type",
    "builder_variant",
    "gate_source",
    "pod_view_gate_mode",
    "field_sha256",
    "renderer_scaling",
    "residual_dtype",
    "residual_clip",
    "ridge",
    "rank_rtol",
    "min_count",
    "min_views",
    "view_std_floor",
    "view_gate_temperature",
    "gate_boost",
    "expert_mse_certificate",
    "expert_reliability_variant",
    "pod_crossfit_split",
    "no_test_gt_used_for_policy",
)
V107_STAT_KEYS = (
    "valid_triangles",
    "base_observed_triangles",
    "total_accumulated_pixels",
    "base_view_affine_triangles",
    "base_shrink_alpha_mean",
    "detail_crossfit_supported_triangles",
    "detail_crossfit_even_to_odd_supported_triangles",
    "detail_crossfit_odd_to_even_supported_triangles",
    "detail_crossfit_gain_mean",
    "detail_crossfit_mse_scale_mean",
    "detail_full_gain_mean",
    "detail_gain_mean",
    "detail_mse_scale_mean",
    "detail_reliability_mean",
    "detail_debt_guard_mean",
    "detail_solved_triangles",
    "detail_triangles",
    "detail_weighted_pixels",
    "boundary_crossfit_supported_triangles",
    "boundary_crossfit_even_to_odd_supported_triangles",
    "boundary_crossfit_odd_to_even_supported_triangles",
    "boundary_crossfit_gain_mean",
    "boundary_crossfit_mse_scale_mean",
    "boundary_full_gain_mean",
    "boundary_gain_mean",
    "boundary_mse_scale_mean",
    "boundary_reliability_mean",
    "boundary_debt_guard_mean",
    "boundary_solved_triangles",
    "boundary_triangles",
    "boundary_weighted_pixels",
    "crossfit_gain_supported_triangles",
    "crossfit_gain_mean",
    "crossfit_mse_scale_mean",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _status_from_exists(path: Path | None) -> str:
    if path is None:
        return "missing"
    return "present" if path.is_file() else "missing"


def _read_json(path: Path | None, warnings: list[str], label: str) -> dict[str, Any]:
    if path is None:
        warnings.append(f"missing_{label}_path")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.append(f"missing_{label}:{path}")
        return {}
    except json.JSONDecodeError as exc:
        warnings.append(f"json_decode_error_{label}:{path}:{exc}")
        return {}
    except OSError as exc:
        warnings.append(f"read_error_{label}:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"json_root_not_object_{label}:{path}")
        return {}
    return payload


def _metric_value(section: Any, metric: str) -> float | None:
    data = _as_dict(section)
    for key in (metric, metric.lower(), metric.upper(), metric.capitalize()):
        if key in data:
            return _float_or_none(data.get(key))
    return None


def _metrics_from_section(section: Any) -> dict[str, float | None]:
    return {metric: _metric_value(section, metric) for metric in METRICS}


def _has_any_metric(metrics: dict[str, float | None]) -> bool:
    return any(value is not None for value in metrics.values())


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _fmt(value: Any, digits: int = 6, signed: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if signed:
            return f"{value:+.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def _jsonish(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _find_report(report_root: Path, scene: str) -> Path | None:
    scene_dir = report_root / scene
    preferred = scene_dir / f"{scene}_v107_crossfit_pod_moe_report.json"
    if preferred.is_file():
        return preferred
    if scene_dir.is_dir():
        patterns = (
            f"{scene}_v107*_report.json",
            f"{scene}_*crossfit*_report.json",
            f"{scene}_*report.json",
            "*v107*_report.json",
            "*report.json",
        )
        for pattern in patterns:
            candidates = sorted(path for path in scene_dir.glob(pattern) if path.is_file())
            if candidates:
                return candidates[0]
    direct = report_root / f"{scene}_v107_crossfit_pod_moe_report.json"
    if direct.is_file():
        return direct
    return None


def _path_from_report(value: Any) -> Path | None:
    if isinstance(value, str) and value:
        return Path(value)
    return None


def _find_manifest(field_root: Path, scene: str, report: dict[str, Any]) -> Path | None:
    report_manifest = _path_from_report(report.get("field_manifest"))
    if report_manifest is not None and report_manifest.is_file():
        return report_manifest

    report_field = _path_from_report(report.get("field"))
    if report_field is not None:
        candidates = (
            report_field.with_suffix(report_field.suffix + ".manifest.json"),
            report_field.with_suffix(".manifest.json"),
            Path(str(report_field).replace("_field.pt", "_field.manifest.json")),
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    scene_dir = field_root / scene
    if scene_dir.is_dir():
        patterns = ("*_field.manifest.json", "*.manifest.json", "*manifest*.json")
        for pattern in patterns:
            candidates = sorted(path for path in scene_dir.glob(pattern) if path.is_file())
            if candidates:
                return candidates[0]
    return report_manifest


def _field_path_from_sources(report: dict[str, Any], manifest: dict[str, Any], manifest_path: Path | None) -> Path | None:
    for value in (report.get("field"), manifest.get("field_path"), _as_dict(manifest.get("args")).get("output_field")):
        path = _path_from_report(value)
        if path is not None:
            return path
    if manifest_path is not None:
        guessed = Path(str(manifest_path).replace("_field.manifest.json", "_field.pt"))
        if guessed != manifest_path:
            return guessed
    return None


def _load_v106_rows(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    payload = _read_json(path, warnings, "v106_full9_json")
    rows = _as_list(payload.get("rows"))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        scene = row.get("scene")
        if isinstance(scene, str) and scene:
            result[scene] = row
    if path.is_file() and not result:
        warnings.append(f"no_v106_rows:{path}")
    return result, warnings


def _v106_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    direct = _metrics_from_section(row.get("metrics"))
    if _has_any_metric(direct):
        return direct
    return {metric: _metric_value(row, metric) for metric in METRICS}


def _v104_metrics_from_row(row: dict[str, Any]) -> dict[str, float | None]:
    direct = _metrics_from_section(row.get("v104_metrics"))
    if _has_any_metric(direct):
        return direct
    direct = _metrics_from_section(row.get("v104c_metrics"))
    if _has_any_metric(direct):
        return direct
    return {metric: _metric_value(row, f"v104c_{metric}") for metric in METRICS}


def _select_metric_section(report: dict[str, Any], names: tuple[str, ...]) -> tuple[str, dict[str, float | None]]:
    for name in names:
        candidates = (f"{name}_metrics", f"{name}c_metrics", name)
        for candidate in candidates:
            metrics = _metrics_from_section(report.get(candidate))
            if _has_any_metric(metrics):
                return candidate, metrics
    return "", {metric: None for metric in METRICS}


def _collect_reference_metrics(
    report: dict[str, Any],
    v106_row: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}

    baseline_source, baseline_metrics = _select_metric_section(report, ("baseline", "clean", "reference"))
    refs["baseline"] = {"source": baseline_source, "metrics": baseline_metrics}

    clean_source, clean_metrics = _select_metric_section(report, ("clean",))
    refs["clean"] = {"source": clean_source, "metrics": clean_metrics}

    reference_source, reference_metrics = _select_metric_section(report, ("reference", "v102", "v101"))
    refs["reference"] = {"source": reference_source, "metrics": reference_metrics}

    v102_source, v102_metrics = _select_metric_section(report, ("v102", "v101"))
    refs["v102"] = {"source": v102_source, "metrics": v102_metrics}

    v104_source, v104_metrics = _select_metric_section(report, ("v104", "v104c"))
    refs["v104"] = {"source": v104_source, "metrics": v104_metrics}

    v106_source, v106_metrics = _select_metric_section(report, ("v106",))
    if not _has_any_metric(v106_metrics) and v106_row:
        v106_source = "v106_full9_json"
        v106_metrics = _v106_metrics(v106_row)
    refs["v106"] = {"source": v106_source, "metrics": v106_metrics}

    if v106_row:
        v106_v104 = _v104_metrics_from_row(v106_row)
        if _has_any_metric(v106_v104):
            refs["v106_v104"] = {"source": "v106_full9_json:v104c", "metrics": v106_v104}

    return refs


def _collect_deltas(
    method_metrics: dict[str, float | None],
    refs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for name, info in refs.items():
        metrics = info.get("metrics")
        if not isinstance(metrics, dict):
            continue
        deltas[f"vs_{name}"] = {metric: _delta(method_metrics.get(metric), metrics.get(metric)) for metric in METRICS}
    return deltas


def _flatten_checks(checks: dict[str, Any]) -> dict[str, bool | None]:
    return {str(key): _bool_or_none(value) for key, value in checks.items()}


def _ok_from_checks(checks: dict[str, bool | None]) -> bool | None:
    values = [value for value in checks.values() if value is not None]
    if not values:
        return None
    return all(values)


def _failed_checks(prefix: str, checks: dict[str, bool | None]) -> list[str]:
    return [f"{prefix}:{key}" for key, value in sorted(checks.items()) if value is False]


def _identity_from_manifest(manifest: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool | None] = {}
    expected = _as_dict(_as_dict(report.get("field_identity")).get("expected"))
    for key, expected_value in expected.items():
        if key in manifest:
            checks[str(key)] = manifest.get(key) == expected_value

    manifest_from_report = _as_dict(_as_dict(report.get("field_identity")).get("manifest"))
    for key, report_value in manifest_from_report.items():
        if key in manifest and key not in checks:
            checks[str(key)] = manifest.get(key) == report_value

    return {
        "checks": checks,
        "ok": _ok_from_checks(checks),
        "failed": _failed_checks("manifest", checks),
    }


def _collect_identity(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    field_checks = _flatten_checks(_as_dict(_as_dict(report.get("field_identity")).get("checks")))
    render_checks = _flatten_checks(_as_dict(_as_dict(report.get("render_stats")).get("identity_checks")))
    manifest_identity = _identity_from_manifest(manifest, report) if manifest else {"checks": {}, "ok": None, "failed": []}

    failed = _failed_checks("field", field_checks)
    failed.extend(_failed_checks("render", render_checks))
    failed.extend(_as_list(manifest_identity.get("failed")))

    ok_values = [
        value
        for value in (
            _ok_from_checks(field_checks),
            _ok_from_checks(render_checks),
            manifest_identity.get("ok"),
        )
        if value is not None
    ]
    return {
        "field_checks": field_checks,
        "render_checks": render_checks,
        "manifest_checks": manifest_identity.get("checks", {}),
        "field_ok": _ok_from_checks(field_checks),
        "render_ok": _ok_from_checks(render_checks),
        "manifest_ok": manifest_identity.get("ok"),
        "all_ok": all(ok_values) if ok_values else None,
        "failed": failed,
    }


def _collect_solve_stats(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    stats = dict(_as_dict(report.get("field_stats")))
    manifest_stats = _as_dict(manifest.get("solve_stats"))
    for key, value in manifest_stats.items():
        stats[key] = value
    for key in (
        "valid_triangles",
        "total_accumulated_pixels",
        "total_valid_pixels",
        "triangle_count",
        "elapsed_sec",
        "method_version",
        "field_variant",
        "builder_variant",
        "gate_source",
        "pod_view_gate_mode",
        "pod_expert_reliability_variant",
        "pod_base_keep_mode",
        "pod_crossfit_split",
    ):
        if key in manifest and key not in stats:
            stats[key] = manifest.get(key)
    return stats


def _collect_crossfit_experts(stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prefixes: set[str] = set()
    for key in stats:
        if "_crossfit_" in key:
            prefixes.add(str(key).split("_crossfit_", 1)[0])
    for name in ("detail", "boundary"):
        if any(str(key).startswith(f"{name}_") for key in stats):
            prefixes.add(name)

    experts: dict[str, dict[str, Any]] = {}
    for prefix in sorted(prefixes):
        expert: dict[str, Any] = {}
        prefix_text = f"{prefix}_"
        for key, value in stats.items():
            if str(key).startswith(prefix_text):
                expert[str(key)[len(prefix_text) :]] = value
        experts[prefix] = expert
    return experts


def _collect_scene(scene: str, report_root: Path, field_root: Path, v106_row: dict[str, Any] | None) -> dict[str, Any]:
    warnings: list[str] = []
    report_path = _find_report(report_root, scene)
    report = _read_json(report_path, warnings, "report") if report_path is not None else {}
    if report_path is None:
        warnings.append("missing_report")

    manifest_path = _find_manifest(field_root, scene, report)
    manifest = _read_json(manifest_path, warnings, "field_manifest") if manifest_path is not None and manifest_path.is_file() else {}
    if manifest_path is None or not manifest_path.is_file():
        warnings.append("missing_field_manifest")

    field_path = _field_path_from_sources(report, manifest, manifest_path)
    if field_path is None or not field_path.is_file():
        warnings.append("missing_field_pt")

    method_metrics = _metrics_from_section(report.get("metrics"))
    refs = _collect_reference_metrics(report, v106_row)
    if not _has_any_metric(_as_dict(_as_dict(refs.get("v106")).get("metrics"))):
        warnings.append("missing_v106_metrics")
    deltas = _collect_deltas(method_metrics, refs)
    identity = _collect_identity(report, manifest)
    solve_stats = _collect_solve_stats(report, manifest)

    status = "ok"
    if report_path is None:
        status = "missing_report"
    elif manifest_path is None or not manifest_path.is_file():
        status = "missing_field_manifest"
    elif not _has_any_metric(method_metrics):
        status = "missing_metrics"
    elif report.get("passed") is False:
        status = "failed_report"

    return {
        "scene": scene,
        "status": status,
        "warnings": sorted(set(warnings)),
        "report": {
            "exists": report_path is not None and report_path.is_file(),
            "path": str(report_path) if report_path is not None else "",
            "status": report.get("status", ""),
            "passed": _bool_or_none(report.get("passed")),
            "schema_version": report.get("schema_version"),
            "report_label": report.get("report_label", ""),
            "output_method": report.get("output_method", ""),
            "return_codes": _as_dict(report.get("return_codes")),
            "logs": _as_dict(report.get("logs")),
        },
        "field": {
            "manifest_exists": manifest_path is not None and manifest_path.is_file(),
            "manifest_path": str(manifest_path) if manifest_path is not None else "",
            "field_exists": field_path is not None and field_path.is_file(),
            "field_path": str(field_path) if field_path is not None else "",
            "manifest_identity": {
                "method_version": manifest.get("method_version"),
                "field_type": manifest.get("field_type"),
                "field_variant": manifest.get("field_variant"),
                "basis_type": manifest.get("basis_type"),
                "builder_variant": manifest.get("builder_variant"),
                "gate_source": manifest.get("gate_source"),
                "pod_view_gate_mode": manifest.get("pod_view_gate_mode"),
                "field_sha256": manifest.get("field_sha256"),
                "renderer_scaling": manifest.get("renderer_scaling"),
                "residual_dtype": manifest.get("residual_dtype"),
            },
        },
        "metrics": {
            "v107": method_metrics,
            "references": refs,
            "deltas": deltas,
            "embedded_report_deltas": _as_dict(report.get("deltas")),
        },
        "identity": identity,
        "solve_stats": {key: solve_stats.get(key) for key in sorted(solve_stats)},
        "crossfit_experts": _collect_crossfit_experts(solve_stats),
        "claim_boundary": report.get("claim_boundary", ""),
    }


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    report = _as_dict(row.get("report"))
    field = _as_dict(row.get("field"))
    manifest_identity = _as_dict(field.get("manifest_identity"))
    metrics = _as_dict(row.get("metrics"))
    v107_metrics = _as_dict(metrics.get("v107"))
    refs = _as_dict(metrics.get("references"))
    deltas = _as_dict(metrics.get("deltas"))
    identity = _as_dict(row.get("identity"))
    solve_stats = _as_dict(row.get("solve_stats"))

    flat: dict[str, Any] = {
        "scene": row.get("scene"),
        "status": row.get("status"),
        "warnings": ";".join(str(item) for item in _as_list(row.get("warnings"))),
        "report_exists": report.get("exists"),
        "report_path": report.get("path"),
        "report_status": report.get("status"),
        "passed": report.get("passed"),
        "field_manifest_exists": field.get("manifest_exists"),
        "field_manifest_path": field.get("manifest_path"),
        "field_exists": field.get("field_exists"),
        "field_path": field.get("field_path"),
        "identity_all_ok": identity.get("all_ok"),
        "field_identity_ok": identity.get("field_ok"),
        "render_identity_ok": identity.get("render_ok"),
        "manifest_identity_ok": identity.get("manifest_ok"),
        "identity_failed_checks": ";".join(str(item) for item in _as_list(identity.get("failed"))),
        "method_version": manifest_identity.get("method_version") or solve_stats.get("method_version"),
        "field_variant": manifest_identity.get("field_variant") or solve_stats.get("field_variant"),
        "builder_variant": manifest_identity.get("builder_variant") or solve_stats.get("builder_variant"),
        "gate_source": manifest_identity.get("gate_source") or solve_stats.get("gate_source"),
        "pod_view_gate_mode": manifest_identity.get("pod_view_gate_mode") or solve_stats.get("pod_view_gate_mode"),
        "field_sha256": manifest_identity.get("field_sha256"),
    }

    for metric in METRICS:
        flat[f"v107_{metric}"] = v107_metrics.get(metric)
    for ref_name in REFERENCE_KEYS:
        ref = _as_dict(refs.get(ref_name))
        ref_metrics = _as_dict(ref.get("metrics"))
        flat[f"{ref_name}_metric_source"] = ref.get("source", "")
        for metric in METRICS:
            flat[f"{ref_name}_{metric}"] = ref_metrics.get(metric)
            flat[f"d{metric}_v107_vs_{ref_name}"] = _as_dict(deltas.get(f"vs_{ref_name}")).get(metric)

    field_checks = _as_dict(identity.get("field_checks"))
    render_checks = _as_dict(identity.get("render_checks"))
    manifest_checks = _as_dict(identity.get("manifest_checks"))
    for key in COMMON_IDENTITY_KEYS:
        flat[f"id_field_{key}"] = field_checks.get(key)
        flat[f"id_render_{key}"] = render_checks.get(key)
        flat[f"id_manifest_{key}"] = manifest_checks.get(key)

    for key in V107_STAT_KEYS:
        flat[key] = solve_stats.get(key)
    for key in (
        "expert_mse_certificate",
        "expert_reliability_variant",
        "expert_reliability_combine",
        "pod_base_keep_mode",
        "pod_crossfit_split",
        "elapsed_sec",
        "triangle_count",
        "total_valid_pixels",
    ):
        flat[key] = solve_stats.get(key)

    return flat


def _mean_or_none(values: list[Any]) -> float | None:
    numeric = [_float_or_none(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    return mean(numeric) if numeric else None


def _summary_row(flat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_with_reports = [row for row in flat_rows if row.get("report_exists") is True]
    rows_with_manifests = [row for row in flat_rows if row.get("field_manifest_exists") is True]
    passed = [row for row in flat_rows if row.get("passed") is True]
    identity_ok = [row for row in flat_rows if row.get("identity_all_ok") is True]
    summary: dict[str, Any] = {
        "scene": "mean_present",
        "status": f"{len(rows_with_reports)}/{len(flat_rows)} reports, {len(rows_with_manifests)}/{len(flat_rows)} manifests",
        "warnings": "",
        "report_exists": len(rows_with_reports),
        "report_path": "",
        "report_status": "",
        "passed": f"{len(passed)}/{len(flat_rows)}",
        "field_manifest_exists": len(rows_with_manifests),
        "field_manifest_path": "",
        "field_exists": sum(1 for row in flat_rows if row.get("field_exists") is True),
        "field_path": "",
        "identity_all_ok": f"{len(identity_ok)}/{len(flat_rows)}",
        "field_identity_ok": "",
        "render_identity_ok": "",
        "manifest_identity_ok": "",
        "identity_failed_checks": "",
        "method_version": "",
        "field_variant": "",
        "builder_variant": "",
        "gate_source": "",
        "pod_view_gate_mode": "",
        "field_sha256": "",
    }
    metric_prefixes = ["v107"] + list(REFERENCE_KEYS)
    for prefix in metric_prefixes:
        if prefix != "v107":
            summary[f"{prefix}_metric_source"] = ""
        for metric in METRICS:
            summary[f"{prefix}_{metric}"] = _mean_or_none([row.get(f"{prefix}_{metric}") for row in rows_with_reports])
            if prefix != "v107":
                summary[f"d{metric}_v107_vs_{prefix}"] = _mean_or_none(
                    [row.get(f"d{metric}_v107_vs_{prefix}") for row in rows_with_reports]
                )
    for key in COMMON_IDENTITY_KEYS:
        summary[f"id_field_{key}"] = ""
        summary[f"id_render_{key}"] = ""
        summary[f"id_manifest_{key}"] = ""
    for key in V107_STAT_KEYS:
        summary[key] = _mean_or_none([row.get(key) for row in rows_with_manifests])
    for key in (
        "expert_mse_certificate",
        "expert_reliability_variant",
        "expert_reliability_combine",
        "pod_base_keep_mode",
        "pod_crossfit_split",
        "elapsed_sec",
        "triangle_count",
        "total_valid_pixels",
    ):
        summary[key] = _mean_or_none([row.get(key) for row in rows_with_manifests])
    return summary


def _csv_columns(flat_rows: list[dict[str, Any]]) -> list[str]:
    fixed = [
        "scene",
        "status",
        "warnings",
        "report_exists",
        "passed",
        "field_manifest_exists",
        "field_exists",
        "identity_all_ok",
        "field_identity_ok",
        "render_identity_ok",
        "manifest_identity_ok",
        "identity_failed_checks",
        "method_version",
        "field_variant",
        "builder_variant",
        "gate_source",
        "pod_view_gate_mode",
        "v107_PSNR",
        "baseline_PSNR",
        "dPSNR_v107_vs_baseline",
        "v104_PSNR",
        "dPSNR_v107_vs_v104",
        "v106_PSNR",
        "dPSNR_v107_vs_v106",
        "v107_SSIM",
        "baseline_SSIM",
        "dSSIM_v107_vs_baseline",
        "v104_SSIM",
        "dSSIM_v107_vs_v104",
        "v106_SSIM",
        "dSSIM_v107_vs_v106",
        "v107_LPIPS",
        "baseline_LPIPS",
        "dLPIPS_v107_vs_baseline",
        "v104_LPIPS",
        "dLPIPS_v107_vs_v104",
        "v106_LPIPS",
        "dLPIPS_v107_vs_v106",
        "valid_triangles",
        "base_observed_triangles",
        "detail_crossfit_supported_triangles",
        "detail_crossfit_gain_mean",
        "detail_crossfit_mse_scale_mean",
        "boundary_crossfit_supported_triangles",
        "boundary_crossfit_gain_mean",
        "boundary_crossfit_mse_scale_mean",
        "field_sha256",
        "report_path",
        "field_manifest_path",
        "field_path",
    ]
    all_keys = set().union(*(row.keys() for row in flat_rows)) if flat_rows else set()
    return fixed + sorted(key for key in all_keys if key not in fixed)


def _write_csv(path: Path, flat_rows: list[dict[str, Any]]) -> None:
    columns = _csv_columns(flat_rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({key: _jsonish(row.get(key)) for key in columns})


def _write_md(path: Path, flat_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# v107 Field Diagnostics",
        "",
        f"- report_root: `{args.report_root}`",
        f"- field_root: `{args.field_root}`",
        f"- v106_full9_json: `{args.v106_full9_json}`",
        "",
        "Deltas are v107 minus the reference. For LPIPS, negative deltas are better.",
        "",
        "| scene | status | report | passed | manifest | identity | v107 PSNR | dPSNR vs v104 | dPSNR vs v106 | v107 SSIM | dSSIM vs v106 | v107 LPIPS | dLPIPS vs v106 | detail support | detail gain | detail mse scale | boundary support | boundary gain | boundary mse scale | warnings |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in flat_rows:
        lines.append(
            "| {scene} | {status} | {report} | {passed} | {manifest} | {identity} | {psnr} | {dpsnr104} | {dpsnr106} | {ssim} | {dssim106} | {lpips} | {dlpips106} | {detail_support} | {detail_gain} | {detail_mse} | {boundary_support} | {boundary_gain} | {boundary_mse} | {warnings} |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                report="yes" if row.get("report_exists") else "no",
                passed=_jsonish(row.get("passed")),
                manifest="yes" if row.get("field_manifest_exists") else "no",
                identity=_jsonish(row.get("identity_all_ok")),
                psnr=_fmt(row.get("v107_PSNR")),
                dpsnr104=_fmt(row.get("dPSNR_v107_vs_v104"), signed=True),
                dpsnr106=_fmt(row.get("dPSNR_v107_vs_v106"), signed=True),
                ssim=_fmt(row.get("v107_SSIM")),
                dssim106=_fmt(row.get("dSSIM_v107_vs_v106"), signed=True),
                lpips=_fmt(row.get("v107_LPIPS")),
                dlpips106=_fmt(row.get("dLPIPS_v107_vs_v106"), signed=True),
                detail_support=_fmt(row.get("detail_crossfit_supported_triangles"), digits=0),
                detail_gain=_fmt(row.get("detail_crossfit_gain_mean")),
                detail_mse=_fmt(row.get("detail_crossfit_mse_scale_mean")),
                boundary_support=_fmt(row.get("boundary_crossfit_supported_triangles"), digits=0),
                boundary_gain=_fmt(row.get("boundary_crossfit_gain_mean")),
                boundary_mse=_fmt(row.get("boundary_crossfit_mse_scale_mean")),
                warnings=str(row.get("warnings", "")).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--field_root", type=Path, default=DEFAULT_FIELD_ROOT)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v107_field_diagnostics")
    parser.add_argument("--v106_full9_json", type=Path, default=DEFAULT_V106_FULL9_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v106_rows, v106_warnings = _load_v106_rows(args.v106_full9_json)
    rows = [_collect_scene(scene, args.report_root, args.field_root, v106_rows.get(scene)) for scene in args.scenes]

    for warning in v106_warnings:
        for row in rows:
            row.setdefault("warnings", []).append(warning)

    flat_rows = [_flatten_row(row) for row in rows]
    flat_with_summary = flat_rows + [_summary_row(flat_rows)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.prefix}.json"
    csv_path = args.out_dir / f"{args.prefix}.csv"
    md_path = args.out_dir / f"{args.prefix}.md"

    payload = {
        "schema_version": 1,
        "report_root": str(args.report_root),
        "field_root": str(args.field_root),
        "v106_full9_json": str(args.v106_full9_json),
        "scenes": list(args.scenes),
        "rows": rows,
        "flat_rows": flat_rows,
        "summary": flat_with_summary[-1],
        "warnings": sorted(set(v106_warnings)),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, flat_with_summary)
    _write_md(md_path, flat_with_summary, args)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
