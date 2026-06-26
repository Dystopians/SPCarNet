#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable


SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_OUTPUT_ROOT = Path("/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625")
DEFAULT_DETACHED_ROOT = Path("/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
DEFAULT_CLEAN_BASELINE_ROOT = Path("outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing_file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(payload, dict):
        return None, "json_root_not_object"
    return payload, None


def _metric_from_section(section: dict[str, Any], metric: str) -> float | None:
    for key in (metric, metric.lower(), metric.upper()):
        if key in section:
            return _float_or_none(section.get(key))
    return None


def _extract_metrics(section: Any) -> dict[str, float | None]:
    data = _as_dict(section)
    return {metric: _metric_from_section(data, metric) for metric in METRICS}


def _empty_metrics() -> dict[str, float | None]:
    return {metric: None for metric in METRICS}


def _has_all_metrics(metrics: dict[str, float | None]) -> bool:
    return all(metrics.get(metric) is not None for metric in METRICS)


def _delta(lhs: dict[str, float | None], rhs: dict[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in METRICS:
        left = lhs.get(metric)
        right = rhs.get(metric)
        out[metric] = None if left is None or right is None else left - right
    return out


def _normalize_scenes(values: list[str] | tuple[str, ...]) -> list[str]:
    scenes: list[str] = []
    for value in values:
        for part in str(value).split(","):
            scene = part.strip()
            if scene and scene not in scenes:
                scenes.append(scene)
    return scenes or list(SCENES)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        return value
    return None


def _find_exact_key(mapping: dict[str, Any], candidates: list[str]) -> str | None:
    lower_to_key = {str(key).lower(): str(key) for key in mapping}
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in mapping:
            return candidate
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return key
    return None


def _find_matching_key(mapping: dict[str, Any], predicate: Callable[[str], bool]) -> str | None:
    for key in sorted(str(key) for key in mapping):
        if predicate(key.lower()):
            return key
    return None


def _clean_method_key(results: dict[str, Any], scene: str) -> tuple[str | None, list[str]]:
    candidates = [
        "ours_26000",
        f"ours_26000_{scene}",
        f"ours_26000_v101_detached_package_full9_{scene}",
        f"ours_26000_v101_detached_package_full9_{scene}_base",
    ]
    exact = _find_exact_key(results, candidates)
    if exact is not None:
        return exact, candidates
    matched = _find_matching_key(
        results,
        lambda key: (
            scene.lower() in key
            and (
                "clean" in key
                or "baseline" in key
                or f"v101_detached_package_full9_{scene.lower()}" in key
            )
        ),
    )
    return matched, candidates


def _v106_method_key(results: dict[str, Any], scene: str, report: dict[str, Any]) -> tuple[str | None, list[str]]:
    report_parent = _first_present(report.get("parent_method_name"))
    candidates = [
        str(report_parent or ""),
        f"ours_26000_v106_podmoe_basepreserve_{scene}",
    ]
    exact = _find_exact_key(results, candidates)
    if exact is not None:
        return exact, candidates
    matched = _find_matching_key(
        results,
        lambda key: scene.lower() in key and "v106" in key and "basepreserve" in key,
    )
    if matched is None:
        matched = _find_matching_key(results, lambda key: scene.lower() in key and "v106" in key)
    return matched, candidates


def _v110_method_key(results: dict[str, Any], scene: str, report: dict[str, Any]) -> tuple[str | None, list[str]]:
    report_gate = _first_present(report.get("gate_method_name"))
    metrics = _as_dict(report.get("metrics"))
    all_methods = _as_dict(metrics.get("all_methods_in_eval_output"))
    candidates = [
        str(report_gate or ""),
        f"ours_26000_v110_strict_train_even_odd_parent_gate_{scene}",
    ]
    for source in (results, all_methods):
        exact = _find_exact_key(source, candidates)
        if exact is not None and exact in results:
            return exact, candidates
    matched = _find_matching_key(
        results,
        lambda key: scene.lower() in key and "v110" in key and ("parent_gate" in key or "gate" in key),
    )
    if matched is None:
        matched = _find_matching_key(results, lambda key: scene.lower() in key and "v110" in key)
    return matched, candidates


def _metrics_for_method(
    results: dict[str, Any],
    method_key: str | None,
    label: str,
    path: Path,
    missing: list[dict[str, Any]],
    expected: list[str],
) -> dict[str, float | None]:
    if method_key is None:
        missing.append(
            {
                "code": "missing_method",
                "label": label,
                "path": str(path),
                "expected": [item for item in expected if item],
            }
        )
        return _empty_metrics()
    metrics = _extract_metrics(results.get(method_key))
    for metric in METRICS:
        if metrics.get(metric) is None:
            missing.append(
                {
                    "code": "missing_metric",
                    "label": label,
                    "method": method_key,
                    "metric": metric,
                    "path": str(path),
                }
            )
    return metrics


def _report_path(output_root: Path, scene: str) -> Path:
    return output_root / scene / "reports" / f"{scene}_v110_strict_split_parent_gate_report.json"


def _results_path(detached_root: Path, scene: str) -> Path:
    return detached_root / scene / "detached_model" / "results.json"


def _clean_results_path(clean_baseline_root: Path, scene: str) -> Path:
    return clean_baseline_root / scene / "results.json"


def collect_scene(output_root: Path, detached_root: Path, clean_baseline_root: Path, scene: str) -> dict[str, Any]:
    report_path = _report_path(output_root, scene)
    results_path = _results_path(detached_root, scene)
    clean_results_path = _clean_results_path(clean_baseline_root, scene)
    missing: list[dict[str, Any]] = []

    report, report_error = _read_json_object(report_path)
    if report_error is not None:
        missing.append({"code": "missing_report_json" if report_error == "missing_file" else "invalid_report_json", "path": str(report_path), "detail": report_error})
        report = {}

    results, results_error = _read_json_object(results_path)
    if results_error is not None:
        missing.append({"code": "missing_results_json" if results_error == "missing_file" else "invalid_results_json", "path": str(results_path), "detail": results_error})
        results = {}

    clean_results, clean_results_error = _read_json_object(clean_results_path)
    if clean_results_error is not None:
        missing.append(
            {
                "code": "missing_clean_results_json" if clean_results_error == "missing_file" else "invalid_clean_results_json",
                "path": str(clean_results_path),
                "detail": clean_results_error,
            }
        )
        clean_results = {}

    clean_source = clean_results if clean_results else results
    clean_source_path = clean_results_path if clean_results else results_path

    clean_key, clean_expected = _clean_method_key(clean_source, scene)
    v106_key, v106_expected = _v106_method_key(results, scene, report)
    v110_key, v110_expected = _v110_method_key(results, scene, report)

    clean = _metrics_for_method(clean_source, clean_key, "clean_baseline", clean_source_path, missing, clean_expected)
    v106 = _metrics_for_method(results, v106_key, "v106_parent", results_path, missing, v106_expected)
    v110 = _metrics_for_method(results, v110_key, "v110_gated", results_path, missing, v110_expected)

    return {
        "scene": scene,
        "report_path": str(report_path),
        "report_present": report_path.is_file() and bool(report),
        "report_status": report.get("status"),
        "report_dry_run": report.get("dry_run"),
        "results_path": str(results_path),
        "results_present": results_path.is_file() and bool(results),
        "clean_results_path": str(clean_results_path),
        "clean_results_present": clean_results_path.is_file() and bool(clean_results),
        "methods": {
            "clean_baseline": clean_key,
            "v106_parent": v106_key,
            "v110_gated": v110_key,
        },
        "metrics": {
            "clean_baseline": clean,
            "v106_parent": v106,
            "v110_gated": v110,
        },
        "deltas": {
            "v110_gated_minus_clean_baseline": _delta(v110, clean),
            "v110_gated_minus_v106_parent": _delta(v110, v106),
        },
        "missing": missing,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_metrics(rows: list[dict[str, Any]], section: str, label: str | None = None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in METRICS:
        values: list[float] = []
        for row in rows:
            source = _as_dict(row.get(section))
            if label is not None:
                source = _as_dict(source.get(label))
            value = _float_or_none(source.get(metric))
            if value is not None:
                values.append(value)
        out[metric] = _mean(values)
    return out


def build_summary(output_root: Path, detached_root: Path, clean_baseline_root: Path, scenes: list[str]) -> dict[str, Any]:
    rows = [collect_scene(output_root, detached_root, clean_baseline_root, scene) for scene in scenes]
    missing = [dict(item, scene=row["scene"]) for row in rows for item in row["missing"]]
    required_missing_codes = {
        "missing_results_json",
        "invalid_results_json",
        "missing_clean_results_json",
        "invalid_clean_results_json",
        "missing_method",
        "missing_metric",
    }
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "detached_root": str(detached_root),
        "clean_baseline_root": str(clean_baseline_root),
        "scenes": scenes,
        "row_count": len(rows),
        "missing_count": len(missing),
        "all_reports_present": all(bool(row.get("report_present")) for row in rows),
        "all_results_present": all(bool(row.get("results_present")) for row in rows),
        "all_clean_results_present": all(bool(row.get("clean_results_present")) for row in rows),
        "all_required_metrics_present": not any(item.get("code") in required_missing_codes for item in missing),
        "means": {
            "clean_baseline": _mean_metrics(rows, "metrics", "clean_baseline"),
            "v106_parent": _mean_metrics(rows, "metrics", "v106_parent"),
            "v110_gated": _mean_metrics(rows, "metrics", "v110_gated"),
            "v110_gated_minus_clean_baseline": _mean_metrics(rows, "deltas", "v110_gated_minus_clean_baseline"),
            "v110_gated_minus_v106_parent": _mean_metrics(rows, "deltas", "v110_gated_minus_v106_parent"),
        },
        "missing": missing,
        "rows": rows,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    number = _float_or_none(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def _fmt_metric_triplet(metrics: dict[str, Any]) -> str:
    return " / ".join(_fmt(metrics.get(metric)) for metric in METRICS)


def _fmt_delta_triplet(metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    for metric in METRICS:
        value = _float_or_none(metrics.get(metric))
        parts.append("NA" if value is None else f"{value:+.6f}")
    return " / ".join(parts)


def _fmt_method(value: Any) -> str:
    return str(value) if value not in (None, "") else "NA"


def _fmt_present(value: Any) -> str:
    return "yes" if value else "no"


def _missing_text(row: dict[str, Any]) -> str:
    missing = row.get("missing")
    if not isinstance(missing, list) or not missing:
        return "none"
    parts: list[str] = []
    for item in missing:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "missing"))
        label = item.get("label")
        metric = item.get("metric")
        if label and metric:
            parts.append(f"{code}:{label}.{metric}")
        elif label:
            parts.append(f"{code}:{label}")
        else:
            parts.append(code)
    return "; ".join(parts) if parts else "none"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# v110 Strict Split Parent Gate Summary",
        "",
        f"- output_root: `{summary['output_root']}`",
        f"- detached_root: `{summary['detached_root']}`",
        f"- clean_baseline_root: `{summary['clean_baseline_root']}`",
        f"- scenes: `{', '.join(summary['scenes'])}`",
        f"- all_reports_present: `{summary['all_reports_present']}`",
        f"- all_results_present: `{summary['all_results_present']}`",
        f"- all_clean_results_present: `{summary['all_clean_results_present']}`",
        f"- all_required_metrics_present: `{summary['all_required_metrics_present']}`",
        f"- missing_count: `{summary['missing_count']}`",
        "",
        "Metric triplets are `PSNR / SSIM / LPIPS`. Deltas are v110 minus the reference triplet.",
        "",
        "| scene | report | results | clean method | clean | v106 parent method | v106 parent | v110 gated method | v110 gated | delta vs clean | delta vs v106 | missing |",
        "|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        metrics = _as_dict(row.get("metrics"))
        deltas = _as_dict(row.get("deltas"))
        methods = _as_dict(row.get("methods"))
        lines.append(
            "| {scene} | {report} | {results} | `{clean_method}` | {clean} | `{v106_method}` | {v106} | `{v110_method}` | {v110} | {dclean} | {dv106} | {missing} |".format(
                scene=row.get("scene"),
                report=_fmt_present(row.get("report_present")),
                results=_fmt_present(row.get("results_present")),
                clean_method=_fmt_method(methods.get("clean_baseline")),
                clean=_fmt_metric_triplet(_as_dict(metrics.get("clean_baseline"))),
                v106_method=_fmt_method(methods.get("v106_parent")),
                v106=_fmt_metric_triplet(_as_dict(metrics.get("v106_parent"))),
                v110_method=_fmt_method(methods.get("v110_gated")),
                v110=_fmt_metric_triplet(_as_dict(metrics.get("v110_gated"))),
                dclean=_fmt_delta_triplet(_as_dict(deltas.get("v110_gated_minus_clean_baseline"))),
                dv106=_fmt_delta_triplet(_as_dict(deltas.get("v110_gated_minus_v106_parent"))),
                missing=_missing_text(row),
            )
        )

    means = _as_dict(summary.get("means"))
    lines.extend(
        [
            "",
            "## Means",
            "",
            "| label | PSNR | SSIM | LPIPS |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in (
        "clean_baseline",
        "v106_parent",
        "v110_gated",
        "v110_gated_minus_clean_baseline",
        "v110_gated_minus_v106_parent",
    ):
        row = _as_dict(means.get(label))
        lines.append(f"| {label} | {_fmt(row.get('PSNR'))} | {_fmt(row.get('SSIM'))} | {_fmt(row.get('LPIPS'))} |")
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect v110 strict-split parent-gate metrics from detached SPCarNet results.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--detached_root", type=Path, default=DEFAULT_DETACHED_ROOT)
    parser.add_argument("--clean_baseline_root", type=Path, default=DEFAULT_CLEAN_BASELINE_ROOT)
    parser.add_argument("--scenes", nargs="*", default=list(SCENES), help="Scene list; comma-separated tokens are also accepted.")
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser()
    detached_root = args.detached_root.expanduser()
    clean_baseline_root = args.clean_baseline_root.expanduser()
    scenes = _normalize_scenes(args.scenes)
    out_json = (args.out_json or (output_root / "v110_strict_split_parent_gate_summary.json")).expanduser()
    out_md = (args.out_md or (output_root / "v110_strict_split_parent_gate_summary.md")).expanduser()

    summary = build_summary(output_root, detached_root, clean_baseline_root, scenes)
    summary["out_json"] = str(out_json)
    summary["out_md"] = str(out_md)
    write_outputs(summary, out_json, out_md)
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "missing_count": summary["missing_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
