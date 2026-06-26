#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_SCENE_ORDER = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
DEFAULT_SOURCE_PRIORITY = ("counter", "hardtriad", "full9")
REPORT_GLOB = "*_v105_evidence_gated_mixture_report.json"
SUMMARY_GLOB = "*_summary.json"
REFERENCE_PREFIXES = {
    "clean",
    "endpoint",
    "reference",
    "baseline",
    "v101",
    "v102",
    "v104",
    "v104c",
}
VERSION_PRIORITY = ("v106", "v105b", "v105", "v104c", "v104", "v103", "v102", "v101")


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


def _read_json(path: Path, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.append({"code": "missing_json", "path": str(path)})
        return {}
    except json.JSONDecodeError as exc:
        warnings.append({"code": "json_decode_error", "path": str(path), "message": str(exc)})
        return {}
    if not isinstance(payload, dict):
        warnings.append({"code": "json_root_not_object", "path": str(path)})
        return {}
    return payload


def _display_path(path: Path | str) -> str:
    return str(path)


def _normalize_source_path(value: Any, container_path: Path) -> str:
    raw = _first_present(value, container_path)
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _metric_value(section: dict[str, Any], metric: str) -> float | None:
    for key in (metric, metric.lower(), metric.upper(), metric.capitalize()):
        if key in section:
            return _float_or_none(section.get(key))
    return None


def _metrics_from_section(section: Any) -> dict[str, float | None]:
    data = _as_dict(section)
    return {metric: _metric_value(data, metric) for metric in METRICS}


def _has_any_metric(metrics: dict[str, float | None]) -> bool:
    return any(value is not None for value in metrics.values())


def _has_all_metrics(metrics: dict[str, float | None]) -> bool:
    return all(metrics.get(metric) is not None for metric in METRICS)


def _metrics_from_flat(row: dict[str, Any], prefix: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    prefix_lower = prefix.lower()
    for metric in METRICS:
        candidates = (
            f"{prefix}_{metric}",
            f"{prefix}_{metric.lower()}",
            f"{prefix_lower}_{metric}",
            f"{prefix_lower}_{metric.lower()}",
        )
        result[metric] = _first_float_from_keys(row, candidates)
    return result


def _first_float_from_keys(row: dict[str, Any], keys: tuple[str, ...] | list[str]) -> float | None:
    for key in keys:
        if key in row:
            value = _float_or_none(row.get(key))
            if value is not None:
                return value
    return None


def _flat_metric_prefixes(row: dict[str, Any]) -> list[str]:
    prefixes: list[str] = []
    for key in row:
        for metric in METRICS:
            for suffix in (f"_{metric}", f"_{metric.lower()}"):
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    if prefix and prefix not in prefixes:
                        prefixes.append(prefix)
    return prefixes


def _infer_version(parts: list[Any]) -> str | None:
    haystack = " ".join(str(part) for part in parts if part not in (None, ""))
    haystack_lower = haystack.lower()
    for version in VERSION_PRIORITY:
        if re.search(rf"(^|[^a-z0-9]){re.escape(version)}([^a-z0-9]|$)", haystack_lower):
            return version
    if "pod_moe" in haystack_lower or "pod-moe" in haystack_lower or "podmoe" in haystack_lower:
        return "v106"
    return None


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


def _nested_metric_key(metrics_by_method: dict[str, Any], method_hint: Any, source: Path) -> str | None:
    if not metrics_by_method:
        return None
    hint = str(method_hint or "").lower()
    if hint:
        for key in metrics_by_method:
            key_lower = str(key).lower()
            if key_lower == hint or key_lower in hint or hint in key_lower:
                return str(key)
    version = _infer_version([method_hint, source])
    if version:
        for key in metrics_by_method:
            if version in str(key).lower():
                return str(key)
        if version == "v106":
            for key in metrics_by_method:
                if "v105" in str(key).lower():
                    return str(key)
    for key in metrics_by_method:
        key_lower = str(key).lower()
        if key_lower not in REFERENCE_PREFIXES and "endpoint" not in key_lower and "ceiling" not in key_lower:
            return str(key)
    return next(iter(metrics_by_method), None)


def _nested_v104c_key(metrics_by_method: dict[str, Any]) -> str | None:
    for key in metrics_by_method:
        if "v104c" in str(key).lower():
            return str(key)
    return None


def _choose_flat_prefix(row: dict[str, Any], source: Path) -> str | None:
    prefixes = _flat_metric_prefixes(row)
    if not prefixes:
        return None
    version = _infer_version(
        [
            row.get("method"),
            row.get("output_method"),
            row.get("result_key"),
            row.get("method_key"),
            row.get("report_label"),
            row.get("claim_boundary"),
            row.get("field_variant"),
            row.get("builder_variant"),
            row.get("field"),
            row.get("field_manifest"),
            source,
        ]
    )
    if version:
        for candidate in (version, "v105" if version == "v106" else ""):
            if candidate and candidate in prefixes:
                return candidate
    for preferred in ("v106", "v105b", "v105"):
        if preferred in prefixes:
            return preferred
    non_reference = [prefix for prefix in prefixes if prefix.lower() not in REFERENCE_PREFIXES]
    return non_reference[0] if non_reference else prefixes[0]


def _candidate_metrics(row: dict[str, Any], source: Path) -> tuple[dict[str, float | None], str | None]:
    direct = _metrics_from_section(row.get("metrics"))
    if _has_any_metric(direct):
        return direct, "metrics"

    metrics_by_method = _as_dict(row.get("metrics"))
    if metrics_by_method:
        method_key = _nested_metric_key(metrics_by_method, row.get("method"), source)
        nested = _metrics_from_section(metrics_by_method.get(method_key)) if method_key is not None else {}
        if _has_any_metric(nested):
            return nested, str(method_key)

    prefix = _choose_flat_prefix(row, source)
    if prefix is not None:
        flat = _metrics_from_flat(row, prefix)
        if _has_any_metric(flat):
            return flat, prefix

    direct_flat = {metric: _first_float_from_keys(row, (metric, metric.lower())) for metric in METRICS}
    if _has_any_metric(direct_flat):
        return direct_flat, "direct"

    return {metric: None for metric in METRICS}, None


def _extract_identity(row: dict[str, Any], key: str) -> Any:
    field_stats = _as_dict(row.get("field_stats"))
    field_identity = _as_dict(row.get("field_identity"))
    manifest_identity = _as_dict(field_identity.get("manifest"))
    render_stats = _as_dict(row.get("render_stats"))
    render_field = _as_dict(render_stats.get("surface_residual_field"))
    return _first_present(row.get(key), field_stats.get(key), manifest_identity.get(key), render_field.get(key))


def _is_base_preserve(row: dict[str, Any], method: str | None, source_path: str, container_path: Path) -> bool:
    parts = [
        method,
        source_path,
        container_path,
        row.get("claim_boundary"),
        row.get("report_label"),
        row.get("field"),
        row.get("field_manifest"),
        _extract_identity(row, "pod_base_keep_mode"),
        _extract_identity(row, "base_variant"),
    ]
    text = " ".join(str(part) for part in parts if part not in (None, "")).lower()
    return (
        "basepreserve" in text
        or "base-preserve" in text
        or "base_preserve" in text
        or "base_preserving" in text
        or "v104c_like_shrink_view_affine" in text
    )


def _infer_scene(row: dict[str, Any], source: Path, default_scene: str | None = None) -> str | None:
    scene = _first_present(row.get("scene"), default_scene)
    if scene:
        return str(scene)
    stem = source.stem
    match = re.match(r"([A-Za-z0-9_-]+)_v105_evidence_gated_mixture_report$", stem)
    if match:
        return match.group(1)
    parent = source.parent.name
    if parent and parent not in (".", "/"):
        return parent
    return None


def _candidate_from_row(
    row: dict[str, Any],
    container_path: Path,
    source_bucket: str,
    source_kind: str,
    default_scene: str | None = None,
) -> dict[str, Any] | None:
    source_path = _normalize_source_path(row.get("report_path"), container_path)
    metrics, metric_source = _candidate_metrics(row, container_path)
    if not _has_any_metric(metrics):
        return None
    scene = _infer_scene(row, Path(source_path), default_scene)
    if not scene or scene == "__aggregate__":
        return None
    raw_method = _first_present(
        row.get("output_method"),
        row.get("method"),
        row.get("method_key"),
        row.get("result_key"),
        row.get("report_label"),
        metric_source,
    )
    method = _compact_method(raw_method, scene) or str(raw_method or "unknown")
    version = _infer_version(
        [
            method,
            row.get("output_method"),
            row.get("claim_boundary"),
            _extract_identity(row, "builder_variant"),
            _extract_identity(row, "field_variant"),
            _extract_identity(row, "pod_base_keep_mode"),
            source_path,
            container_path,
        ]
    )
    if version and version not in method.lower():
        method = f"{version}_{method}"

    candidate = {
        "scene": scene,
        "method": method,
        "version": version,
        "source_bucket": source_bucket,
        "source_kind": source_kind,
        "source_path": source_path,
        "container_path": str(container_path),
        "metric_source": metric_source,
        "status": row.get("status"),
        "passed": row.get("passed"),
        "metrics": metrics,
        "complete_metrics": _has_all_metrics(metrics),
        "is_v106": version == "v106" or "v106" in method.lower(),
        "is_base_preserve": False,
        "output_method": row.get("output_method"),
        "field_variant": _extract_identity(row, "field_variant"),
        "builder_variant": _extract_identity(row, "builder_variant"),
        "pod_base_keep_mode": _extract_identity(row, "pod_base_keep_mode"),
        "expert_mse_certificate": _extract_identity(row, "expert_mse_certificate"),
    }
    candidate["is_base_preserve"] = _is_base_preserve(row, method, source_path, container_path)
    return candidate


def _extract_candidates_from_summary(payload: dict[str, Any], path: Path, source_bucket: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in _as_list(payload.get("rows")):
        if isinstance(row, dict):
            candidate = _candidate_from_row(row, path, source_bucket, "summary_row")
            if candidate is not None:
                candidates.append(candidate)

    scenes = payload.get("scenes")
    if isinstance(scenes, dict):
        for scene, scene_payload in sorted(scenes.items()):
            if not isinstance(scene_payload, dict):
                continue
            metrics_by_method = _as_dict(scene_payload.get("metrics"))
            if metrics_by_method and not _has_any_metric(_metrics_from_section(scene_payload.get("metrics"))):
                method_key = _nested_metric_key(metrics_by_method, payload.get("method"), path)
                if method_key is None:
                    continue
                row = dict(scene_payload)
                row["scene"] = str(scene)
                row["method"] = _first_present(payload.get("method"), method_key)
                row["method_key"] = method_key
                row["metrics"] = metrics_by_method.get(method_key)
                candidate = _candidate_from_row(row, path, source_bucket, "summary_scene", str(scene))
            else:
                row = dict(scene_payload)
                row.setdefault("scene", str(scene))
                candidate = _candidate_from_row(row, path, source_bucket, "summary_scene", str(scene))
            if candidate is not None:
                candidates.append(candidate)

    if not candidates and ("scene" in payload or "metrics" in payload):
        candidate = _candidate_from_row(payload, path, source_bucket, "report")
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _scan_root(root: Path, source_bucket: str, warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not root.exists():
        warnings.append({"code": "missing_report_root", "source_bucket": source_bucket, "path": str(root)})
        return []
    candidates: list[dict[str, Any]] = []
    report_paths = sorted(path for path in root.rglob(REPORT_GLOB) if path.is_file())
    summary_paths = sorted(path for path in root.rglob(SUMMARY_GLOB) if path.is_file())

    for path in report_paths:
        payload = _read_json(path, warnings)
        if not payload:
            continue
        candidate = _candidate_from_row(payload, path, source_bucket, "report", path.parent.name)
        if candidate is not None:
            candidates.append(candidate)
        else:
            warnings.append({"code": "report_missing_candidate_metrics", "source_bucket": source_bucket, "path": str(path)})

    for path in summary_paths:
        payload = _read_json(path, warnings)
        if not payload:
            continue
        extracted = _extract_candidates_from_summary(payload, path, source_bucket)
        if extracted:
            candidates.extend(extracted)
        else:
            warnings.append({"code": "summary_missing_scene_candidates", "source_bucket": source_bucket, "path": str(path)})
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank(candidate: dict[str, Any]) -> tuple[int, int, int]:
        kind_rank = 0 if candidate.get("source_kind") == "report" else 1
        complete_rank = 0 if candidate.get("complete_metrics") else 1
        passed = candidate.get("passed")
        status = str(candidate.get("status") or "").lower()
        ok_rank = 0 if passed is True or status in ("", "ok") or passed is None else 1
        return kind_rank, complete_rank, ok_rank

    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            str(candidate.get("scene")),
            str(candidate.get("method")),
            str(candidate.get("source_path")),
        )
        previous = by_key.get(key)
        if previous is None or rank(candidate) < rank(previous):
            by_key[key] = candidate
    return list(by_key.values())


def _anchor_metrics_from_row(row: dict[str, Any], source: Path) -> tuple[dict[str, float | None], str | None]:
    flat = _metrics_from_flat(row, "v104c")
    if _has_any_metric(flat):
        return flat, "v104c_flat"

    nested = _metrics_from_section(row.get("v104c_metrics"))
    if _has_any_metric(nested):
        return nested, "v104c_metrics"

    metrics_by_method = _as_dict(row.get("metrics"))
    if metrics_by_method:
        method_key = _nested_v104c_key(metrics_by_method)
        nested_by_method = _metrics_from_section(metrics_by_method.get(method_key)) if method_key is not None else {}
        if _has_any_metric(nested_by_method):
            return nested_by_method, str(method_key)

    if "v104c" in " ".join(str(row.get(key, "")) for key in ("method", "output_method", "result_key")).lower():
        direct = _metrics_from_section(row.get("metrics"))
        if _has_any_metric(direct):
            return direct, "metrics"
        direct_flat = {metric: _first_float_from_keys(row, (metric, metric.lower())) for metric in METRICS}
        if _has_any_metric(direct_flat):
            return direct_flat, "direct"

    prefix = _choose_flat_prefix(row, source)
    if prefix and "v104c" in prefix.lower():
        prefixed = _metrics_from_flat(row, prefix)
        if _has_any_metric(prefixed):
            return prefixed, prefix

    return {metric: None for metric in METRICS}, None


def load_v104c_anchors(path: Path, warnings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, warnings)
    anchors: dict[str, dict[str, Any]] = {}

    def add_anchor(row: dict[str, Any], source_kind: str, default_scene: str | None = None) -> None:
        scene = _infer_scene(row, path, default_scene)
        if not scene or scene == "__aggregate__":
            return
        metrics, metric_source = _anchor_metrics_from_row(row, path)
        if not _has_any_metric(metrics):
            warnings.append({"code": "v104c_anchor_metrics_missing", "scene": scene, "source_kind": source_kind, "path": str(path)})
            return
        complete = _has_all_metrics(metrics)
        if not complete:
            warnings.append(
                {
                    "code": "v104c_anchor_metrics_partial",
                    "scene": scene,
                    "metric_source": metric_source,
                    "missing_metrics": [metric for metric in METRICS if metrics.get(metric) is None],
                    "path": str(path),
                }
            )
        if scene in anchors:
            warnings.append(
                {
                    "code": "duplicate_v104c_anchor",
                    "scene": scene,
                    "kept_metric_source": anchors[scene]["metric_source"],
                    "dropped_metric_source": metric_source,
                    "path": str(path),
                }
            )
            return
        anchors[scene] = {
            "scene": scene,
            "metrics": metrics,
            "metric_source": metric_source,
            "source_kind": source_kind,
            "source_path": str(path),
            "method": _compact_method(_first_present(row.get("output_method"), row.get("method"), "v104c"), scene),
        }

    for row in _as_list(payload.get("rows")):
        if isinstance(row, dict):
            add_anchor(row, "summary_row")

    scenes = payload.get("scenes")
    if isinstance(scenes, dict):
        for scene, scene_payload in sorted(scenes.items()):
            if isinstance(scene_payload, dict):
                row = dict(scene_payload)
                row.setdefault("scene", str(scene))
                add_anchor(row, "summary_scene", str(scene))

    if not anchors and ("scene" in payload or "metrics" in payload):
        add_anchor(payload, "report")

    if not anchors:
        warnings.append({"code": "no_v104c_anchors_found", "path": str(path)})
    return anchors


def _scene_sort_key(scene: str) -> tuple[int, str]:
    if scene in DEFAULT_SCENE_ORDER:
        return DEFAULT_SCENE_ORDER.index(scene), scene
    return len(DEFAULT_SCENE_ORDER), scene


def _candidate_ok_rank(candidate: dict[str, Any]) -> int:
    passed = candidate.get("passed")
    status = str(candidate.get("status") or "").lower()
    if passed is False or status in ("failed", "blocker", "missing"):
        return 1
    return 0


def _select_candidate(
    candidates: list[dict[str, Any]],
    source_priority: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    source_rank = {name: idx for idx, name in enumerate(source_priority)}

    def rank(candidate: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
        return (
            _candidate_ok_rank(candidate),
            0 if candidate.get("complete_metrics") else 1,
            0 if candidate.get("is_v106") and candidate.get("is_base_preserve") else 1,
            0 if candidate.get("is_v106") else 1,
            source_rank.get(str(candidate.get("source_bucket")), len(source_rank)),
            str(candidate.get("source_path")),
        )

    ordered = sorted(candidates, key=rank)
    selected = ordered[0]
    reason = (
        "selected by passed/complete metrics, then v106 base-preserve, then v106, "
        f"then source priority {source_priority}"
    )
    return selected, ordered[1:], reason


def _delta(candidate: dict[str, float | None], anchor: dict[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in METRICS:
        lhs = candidate.get(metric)
        rhs = anchor.get(metric)
        out[metric] = None if lhs is None or rhs is None else lhs - rhs
    return out


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _fmt(value: Any, signed: bool = False) -> str:
    value_float = _float_or_none(value)
    if value_float is None:
        return ""
    return f"{value_float:+.6f}" if signed else f"{value_float:.6f}"


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    warnings: list[dict[str, Any]] = []
    source_priority = list(args.source_priority or DEFAULT_SOURCE_PRIORITY)
    unknown_priorities = [name for name in source_priority if name not in DEFAULT_SOURCE_PRIORITY]
    if unknown_priorities:
        raise SystemExit(f"--source_priority contains unknown buckets: {unknown_priorities}")
    for bucket in DEFAULT_SOURCE_PRIORITY:
        if bucket not in source_priority:
            source_priority.append(bucket)

    anchors = load_v104c_anchors(Path(args.v104c_summary_json), warnings)
    root_by_bucket = {
        "counter": Path(args.counter_report_root),
        "hardtriad": Path(args.hardtriad_report_root),
        "full9": Path(args.full9_report_root),
    }
    candidates: list[dict[str, Any]] = []
    for bucket in DEFAULT_SOURCE_PRIORITY:
        candidates.extend(_scan_root(root_by_bucket[bucket], bucket, warnings))

    candidates_by_scene: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidates_by_scene.setdefault(str(candidate["scene"]), []).append(candidate)

    required_scenes = [str(scene) for scene in _as_list(args.require_scenes)]
    if required_scenes:
        scene_names = required_scenes
    else:
        scene_names = sorted(candidates_by_scene, key=_scene_sort_key)

    rows: list[dict[str, Any]] = []
    missing_result_scenes: list[str] = []
    missing_anchor_scenes: list[str] = []

    for scene in scene_names:
        scene_candidates = candidates_by_scene.get(scene, [])
        anchor = anchors.get(scene)
        if not scene_candidates:
            missing_result_scenes.append(scene)
            rows.append(
                {
                    "scene": scene,
                    "status": "missing_result",
                    "method": "",
                    "source_bucket": "",
                    "source_kind": "",
                    "source_path": "",
                    "candidate_count": 0,
                    "candidate_sources": "",
                    "warnings": "missing_result",
                    **{metric: None for metric in METRICS},
                    **{f"v104c_{metric}": _as_dict(anchor).get("metrics", {}).get(metric) if anchor else None for metric in METRICS},
                    **{f"d{metric}_vs_v104c": None for metric in METRICS},
                }
            )
            continue

        selected, dropped, reason = _select_candidate(scene_candidates, source_priority)
        selected_metrics = _as_dict(selected.get("metrics"))
        anchor_metrics = _as_dict(anchor.get("metrics")) if anchor else {}
        deltas = _delta(selected_metrics, anchor_metrics) if anchor else {metric: None for metric in METRICS}

        row_warnings: list[str] = []
        if not selected.get("complete_metrics"):
            row_warnings.append("partial_candidate_metrics")
        if not anchor:
            row_warnings.append("missing_v104c_anchor")
            missing_anchor_scenes.append(scene)
        elif not _has_all_metrics(anchor_metrics):
            row_warnings.append("partial_v104c_anchor")
            missing_anchor_scenes.append(scene)

        candidate_sources = [
            f"{candidate.get('source_bucket')}:{candidate.get('source_kind')}:{candidate.get('source_path')}"
            for candidate in scene_candidates
        ]
        rows.append(
            {
                "scene": scene,
                "status": "ok" if not row_warnings else "warning",
                "method": selected.get("method"),
                "version": selected.get("version"),
                "is_v106": selected.get("is_v106"),
                "is_base_preserve": selected.get("is_base_preserve"),
                "source_bucket": selected.get("source_bucket"),
                "source_kind": selected.get("source_kind"),
                "source_path": selected.get("source_path"),
                "container_path": selected.get("container_path"),
                "metric_source": selected.get("metric_source"),
                "passed": selected.get("passed"),
                "candidate_count": len(scene_candidates),
                "candidate_sources": ";".join(candidate_sources),
                "dropped_candidate_sources": ";".join(str(candidate.get("source_path")) for candidate in dropped),
                "selection_reason": reason,
                "warnings": ";".join(row_warnings),
                "field_variant": selected.get("field_variant"),
                "builder_variant": selected.get("builder_variant"),
                "pod_base_keep_mode": selected.get("pod_base_keep_mode"),
                "expert_mse_certificate": selected.get("expert_mse_certificate"),
                "v104c_anchor_source_path": anchor.get("source_path") if anchor else "",
                "v104c_anchor_metric_source": anchor.get("metric_source") if anchor else "",
                "PSNR": selected_metrics.get("PSNR"),
                "SSIM": selected_metrics.get("SSIM"),
                "LPIPS": selected_metrics.get("LPIPS"),
                "v104c_PSNR": anchor_metrics.get("PSNR"),
                "v104c_SSIM": anchor_metrics.get("SSIM"),
                "v104c_LPIPS": anchor_metrics.get("LPIPS"),
                "dPSNR_vs_v104c": deltas.get("PSNR"),
                "dSSIM_vs_v104c": deltas.get("SSIM"),
                "dLPIPS_vs_v104c": deltas.get("LPIPS"),
            }
        )

    if required_scenes:
        for scene in missing_result_scenes:
            warnings.append({"code": "required_scene_missing_result", "scene": scene})
        for scene in sorted(set(missing_anchor_scenes), key=_scene_sort_key):
            warnings.append({"code": "required_scene_missing_or_partial_v104c_anchor", "scene": scene})

    selected_rows = [row for row in rows if row.get("source_path")]
    mean_values: dict[str, Any] = {
        "available_scenes": len(selected_rows),
        "required_scenes": len(required_scenes) if required_scenes else None,
    }
    for metric in METRICS:
        mean_values[metric] = _mean_metric(selected_rows, metric)
        mean_values[f"v104c_{metric}"] = _mean_metric(selected_rows, f"v104c_{metric}")
        mean_values[f"d{metric}_vs_v104c"] = _mean_metric(selected_rows, f"d{metric}_vs_v104c")

    missing_requirements = bool(required_scenes and (missing_result_scenes or missing_anchor_scenes))
    exit_code = 0 if args.allow_missing or not missing_requirements else 2
    if missing_requirements and args.allow_missing:
        warnings.append(
            {
                "code": "missing_requirements_allowed",
                "missing_result_scenes": missing_result_scenes,
                "missing_anchor_scenes": sorted(set(missing_anchor_scenes), key=_scene_sort_key),
            }
        )

    payload = {
        "schema_version": 1,
        "inputs": {
            "v104c_summary_json": str(args.v104c_summary_json),
            "counter_report_root": str(args.counter_report_root),
            "hardtriad_report_root": str(args.hardtriad_report_root),
            "full9_report_root": str(args.full9_report_root),
        },
        "selection_policy": {
            "source_priority": source_priority,
            "rank_order": [
                "passed_or_not_failed",
                "complete_candidate_metrics",
                "v106_base_preserve",
                "v106",
                "source_priority",
                "source_path",
            ],
        },
        "require_scenes": required_scenes,
        "allow_missing": bool(args.allow_missing),
        "scene_count": len(rows),
        "available_scenes": len(selected_rows),
        "missing_result_scenes": missing_result_scenes,
        "missing_anchor_scenes": sorted(set(missing_anchor_scenes), key=_scene_sort_key),
        "mean": mean_values,
        "rows": rows,
        "warnings": warnings,
        "candidate_count": len(candidates),
        "candidate_sources_by_scene": {
            scene: [
                {
                    "method": candidate.get("method"),
                    "source_bucket": candidate.get("source_bucket"),
                    "source_kind": candidate.get("source_kind"),
                    "source_path": candidate.get("source_path"),
                    "is_v106": candidate.get("is_v106"),
                    "is_base_preserve": candidate.get("is_base_preserve"),
                    "metric_source": candidate.get("metric_source"),
                }
                for candidate in scene_candidates
            ]
            for scene, scene_candidates in sorted(candidates_by_scene.items(), key=lambda item: _scene_sort_key(item[0]))
        },
        "notes": [
            "Deltas are selected candidate minus v104c summary anchor; for LPIPS, negative is better.",
            "Selection prefers passed complete v106 base-preserve reports, then v106 reports, then the configured source priority.",
        ],
    }
    return payload, exit_code


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _as_list(payload.get("rows"))
    fieldnames = [
        "scene",
        "status",
        "method",
        "version",
        "is_v106",
        "is_base_preserve",
        "source_bucket",
        "source_kind",
        "source_path",
        "PSNR",
        "SSIM",
        "LPIPS",
        "v104c_PSNR",
        "v104c_SSIM",
        "v104c_LPIPS",
        "dPSNR_vs_v104c",
        "dSSIM_vs_v104c",
        "dLPIPS_vs_v104c",
        "candidate_count",
        "selection_reason",
        "warnings",
        "candidate_sources",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(payload: dict[str, Any]) -> str:
    rows = _as_list(payload.get("rows"))
    mean_values = _as_dict(payload.get("mean"))
    warnings = _as_list(payload.get("warnings"))
    inputs = _as_dict(payload.get("inputs"))
    policy = _as_dict(payload.get("selection_policy"))
    lines = [
        "# v106 Base-Preserve Full9 Assembly",
        "",
        f"- v104c anchors: `{inputs.get('v104c_summary_json', '')}`",
        f"- source priority: `{', '.join(_as_list(policy.get('source_priority')))}`",
        f"- available scenes: `{payload.get('available_scenes')}` / `{payload.get('scene_count')}`",
        f"- warnings: `{len(warnings)}`",
        "",
        "| scene | source | method | PSNR | SSIM | LPIPS | v104c PSNR | dPSNR | v104c SSIM | dSSIM | v104c LPIPS | dLPIPS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        source = row.get("source_bucket") or row.get("status") or ""
        lines.append(
            "| {scene} | {source} | {method} | {psnr} | {ssim} | {lpips} | {vpsnr} | {dpsnr} | {vssim} | {dssim} | {vlpips} | {dlpips} |".format(
                scene=row.get("scene", ""),
                source=source,
                method=row.get("method", ""),
                psnr=_fmt(row.get("PSNR")),
                ssim=_fmt(row.get("SSIM")),
                lpips=_fmt(row.get("LPIPS")),
                vpsnr=_fmt(row.get("v104c_PSNR")),
                dpsnr=_fmt(row.get("dPSNR_vs_v104c"), True),
                vssim=_fmt(row.get("v104c_SSIM")),
                dssim=_fmt(row.get("dSSIM_vs_v104c"), True),
                vlpips=_fmt(row.get("v104c_LPIPS")),
                dlpips=_fmt(row.get("dLPIPS_vs_v104c"), True),
            )
        )
    lines.append(
        "| mean | selected |  | {psnr} | {ssim} | {lpips} | {vpsnr} | {dpsnr} | {vssim} | {dssim} | {vlpips} | {dlpips} |".format(
            psnr=_fmt(mean_values.get("PSNR")),
            ssim=_fmt(mean_values.get("SSIM")),
            lpips=_fmt(mean_values.get("LPIPS")),
            vpsnr=_fmt(mean_values.get("v104c_PSNR")),
            dpsnr=_fmt(mean_values.get("dPSNR_vs_v104c"), True),
            vssim=_fmt(mean_values.get("v104c_SSIM")),
            dssim=_fmt(mean_values.get("dSSIM_vs_v104c"), True),
            vlpips=_fmt(mean_values.get("v104c_LPIPS")),
            dlpips=_fmt(mean_values.get("dLPIPS_vs_v104c"), True),
        )
    )
    selected_source_rows = [row for row in rows if row.get("source_path")]
    if selected_source_rows:
        lines.extend(["", "## Selected Sources", "", "| scene | source path |", "|---|---|"])
        for row in selected_source_rows:
            lines.append(f"| {row.get('scene', '')} | `{row.get('source_path', '')}` |")
    missing = _as_list(payload.get("missing_result_scenes"))
    if missing:
        lines.extend(["", f"Missing result scenes: `{', '.join(str(scene) for scene in missing)}`"])
    missing_anchor = _as_list(payload.get("missing_anchor_scenes"))
    if missing_anchor:
        lines.extend(["", f"Missing/partial v104c anchors: `{', '.join(str(scene) for scene in missing_anchor)}`"])
    if warnings:
        lines.extend(["", "Warnings are recorded in the JSON output."])
    return "\n".join(lines) + "\n"


def write_md(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(payload), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble v106 base-preserve counter/hardtriad/full9 partial reports against v104c anchors."
    )
    parser.add_argument("--v104c_summary_json", required=True, help="v104c summary JSON containing scene baseline anchors.")
    parser.add_argument("--counter_report_root", required=True, help="Root containing counter report JSON/summary files.")
    parser.add_argument("--hardtriad_report_root", required=True, help="Root containing hard-triad report JSON/summary files.")
    parser.add_argument("--full9_report_root", required=True, help="Root containing full9 partial/complete report JSON/summary files.")
    parser.add_argument("--out_json", required=True, help="Output assembled JSON path.")
    parser.add_argument("--out_csv", required=True, help="Output per-scene CSV path.")
    parser.add_argument("--out_md", required=True, help="Output compact Markdown table path.")
    parser.add_argument(
        "--source_priority",
        nargs="+",
        default=list(DEFAULT_SOURCE_PRIORITY),
        help="Tie-break order for duplicate scene candidates. Valid buckets: counter hardtriad full9.",
    )
    parser.add_argument("--require_scenes", nargs="*", default=[], help="Scene names that must be present.")
    parser.add_argument("--allow_missing", action="store_true", help="Warn instead of exiting non-zero for missing required scenes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    payload, exit_code = build_report(args)
    write_json(payload, Path(args.out_json))
    write_csv(payload, Path(args.out_csv))
    write_md(payload, Path(args.out_md))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "out_csv": args.out_csv,
                "out_md": args.out_md,
                "available_scenes": payload.get("available_scenes"),
                "missing_result_scenes": payload.get("missing_result_scenes"),
                "missing_anchor_scenes": payload.get("missing_anchor_scenes"),
                "warning_count": len(_as_list(payload.get("warnings"))),
                "exit_code": exit_code,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
