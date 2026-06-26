#!/usr/bin/env python3
"""Collect the current v110/v111/v114 strict-branch package.

The collector is intentionally read-only. It records both completed metrics and
missing artifacts so long jobs can be summarized without hand-written tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_DETACHED_ROOT = Path("/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
DEFAULT_CLEAN_ROOT = Path("outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
DEFAULT_V106_ASSEMBLED = Path("docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.json")
DEFAULT_V110_ROOT = Path("/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625")
DEFAULT_V111_ROOT = Path("/dev/shm/peilincai_spcarnet_v111_end_to_end_strict_parent_gate_20260625")
DEFAULT_V114_ROOT = Path("/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625")


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, "missing_file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"json_decode_error:{exc}"
    except OSError as exc:
        return {}, f"read_error:{exc}"
    if not isinstance(payload, dict):
        return {}, "json_root_not_object"
    return payload, None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(section: Any, metric: str) -> float | None:
    data = _as_dict(section)
    for key in (metric, metric.lower(), metric.upper(), metric.capitalize()):
        value = _float_or_none(data.get(key))
        if value is not None:
            return value
    return None


def _metrics(section: Any) -> dict[str, float | None]:
    return {metric: _metric(section, metric) for metric in METRICS}


def _has_all_metrics(metrics: dict[str, float | None]) -> bool:
    return all(metrics.get(metric) is not None for metric in METRICS)


def _delta(lhs: dict[str, float | None], rhs: dict[str, float | None]) -> dict[str, float | None]:
    return {
        metric: None if lhs.get(metric) is None or rhs.get(metric) is None else float(lhs[metric]) - float(rhs[metric])
        for metric in METRICS
    }


def _fmt(value: Any, signed: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}" if signed else f"{value:.6f}"
    return str(value)


def _triplet(values: dict[str, Any], signed: bool = False) -> str:
    return " / ".join(_fmt(values.get(metric), signed=signed) for metric in METRICS)


def _clean_method_key(results: dict[str, Any]) -> str | None:
    for key in ("ours_26000", "ours_30000"):
        if key in results:
            return key
    keys = sorted(str(key) for key in results)
    return keys[0] if keys else None


def _load_clean(clean_root: Path, scene: str, missing: list[str]) -> tuple[str | None, dict[str, float | None], str]:
    path = clean_root / scene / "results.json"
    payload, error = _read_json(path)
    if error:
        missing.append(f"clean_results:{path}:{error}")
        return None, _metrics({}), str(path)
    key = _clean_method_key(payload)
    if key is None:
        missing.append(f"clean_method_missing:{path}")
        return None, _metrics({}), str(path)
    values = _metrics(payload.get(key))
    if not _has_all_metrics(values):
        missing.append(f"clean_metrics_incomplete:{path}:{key}")
    return key, values, str(path)


def _load_v106(v106_rows: dict[str, dict[str, Any]], scene: str, missing: list[str]) -> dict[str, float | None]:
    row = v106_rows.get(scene, {})
    values = _metrics(row)
    if not _has_all_metrics(values):
        missing.append(f"v106_metrics_missing:{scene}")
    return values


def _v106_rows(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    missing: list[str] = []
    payload, error = _read_json(path)
    if error:
        return {}, [f"v106_assembled:{path}:{error}"]
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return {}, [f"v106_assembled_rows_missing:{path}"]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("scene"), str):
            out[str(row["scene"])] = row
    return out, missing


def _method_metrics_from_sources(
    report: dict[str, Any],
    detached_results: dict[str, Any],
    explicit_results_path: Path | None,
    method_name: str | None,
    missing: list[str],
) -> tuple[dict[str, float | None], str]:
    report_metrics = _as_dict(report.get("metrics"))
    gated = _metrics(report_metrics.get("gated_method"))
    if _has_all_metrics(gated):
        return gated, "orchestration_report.metrics.gated_method"

    all_methods = _as_dict(report_metrics.get("all_methods_in_eval_output"))
    if method_name and method_name in all_methods:
        values = _metrics(all_methods.get(method_name))
        if _has_all_metrics(values):
            return values, "orchestration_report.metrics.all_methods_in_eval_output"

    if method_name and method_name in detached_results:
        values = _metrics(detached_results.get(method_name))
        if _has_all_metrics(values):
            return values, "detached_model/results.json"

    if explicit_results_path is not None:
        payload, error = _read_json(explicit_results_path)
        if error:
            missing.append(f"method_results:{explicit_results_path}:{error}")
        elif method_name and method_name in payload:
            values = _metrics(payload.get(method_name))
            if _has_all_metrics(values):
                return values, str(explicit_results_path)
            missing.append(f"method_metrics_incomplete:{explicit_results_path}:{method_name}")
        elif len(payload) == 1:
            only_key = next(iter(payload))
            values = _metrics(payload.get(only_key))
            if _has_all_metrics(values):
                return values, f"{explicit_results_path}:{only_key}"
            missing.append(f"method_metrics_incomplete:{explicit_results_path}:{only_key}")
        else:
            missing.append(f"method_missing_in_results:{explicit_results_path}:{method_name or ''}")

    if method_name:
        missing.append(f"method_metrics_missing:{method_name}")
    else:
        missing.append("method_name_missing")
    return _metrics({}), ""


def _detached_results(detached_root: Path, scene: str, missing: list[str]) -> tuple[dict[str, Any], str]:
    path = detached_root / scene / "detached_model" / "results.json"
    payload, error = _read_json(path)
    if error:
        missing.append(f"detached_results:{path}:{error}")
    return payload, str(path)


def _artifact_status(paths: dict[str, Path | None], missing: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in paths.items():
        if path is None:
            out[name] = {"path": "", "present": False}
            missing.append(f"{name}:path_missing")
            continue
        present = path.is_file()
        out[name] = {"path": str(path), "present": present}
        if not present:
            missing.append(f"{name}:{path}:missing")
    return out


def _row_from_report(
    *,
    label: str,
    branch: str,
    scene: str,
    report_path: Path,
    default_method_name: str,
    field_path: Path | None,
    manifest_path: Path | None,
    extra_artifacts: dict[str, Path | None] | None = None,
    detached_root: Path,
    clean_root: Path,
    v106_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing: list[str] = []
    report, report_error = _read_json(report_path)
    if report_error:
        missing.append(f"report:{report_path}:{report_error}")
    method_name = str(report.get("gate_method_name") or report.get("candidate_method_name") or default_method_name)
    eval_path = None
    metrics_section = _as_dict(report.get("metrics"))
    if isinstance(metrics_section.get("results_path"), str):
        eval_path = Path(str(metrics_section["results_path"]))
    elif method_name:
        eval_path = report_path.parent / f"{scene}_{method_name}_test_results.json"

    detached, detached_path = _detached_results(detached_root, scene, missing)
    clean_key, clean, clean_path = _load_clean(clean_root, scene, missing)
    v106 = _load_v106(v106_rows, scene, missing)
    method_metrics, method_metric_source = _method_metrics_from_sources(report, detached, eval_path, method_name, missing)

    artifact_paths = {
        "report": report_path,
        "field": field_path,
        "manifest": manifest_path,
        "eval_results": eval_path,
    }
    if extra_artifacts:
        artifact_paths.update(extra_artifacts)
    artifacts = _artifact_status(artifact_paths, missing)
    status = str(report.get("status") or "MISSING")
    complete = status == "COMPLETE" and _has_all_metrics(method_metrics)
    if report_error:
        status = "MISSING_REPORT"
    elif not _has_all_metrics(method_metrics):
        status = f"{status}_NO_METRICS"
    return {
        "label": label,
        "branch": branch,
        "scene": scene,
        "status": status,
        "complete": complete,
        "method_name": method_name,
        "clean_method": clean_key,
        "paths": {
            "detached_results": detached_path,
            "clean_results": clean_path,
        },
        "artifacts": artifacts,
        "metric_source": method_metric_source,
        "metrics": {
            "clean": clean,
            "v106_parent": v106,
            "method": method_metrics,
        },
        "deltas": {
            "method_minus_clean": _delta(method_metrics, clean),
            "method_minus_v106": _delta(method_metrics, v106),
        },
        "missing": missing,
    }


def _row_v114(
    *,
    v114_root: Path,
    detached_root: Path,
    clean_root: Path,
    v106_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scene = "garden"
    method_name = "ours_26000_v114_oof_refit_podmoe_garden"
    field_path = v114_root / scene / "fields" / f"{method_name}_field.pt"
    manifest_path = v114_root / scene / "fields" / f"{method_name}_field.manifest.json"
    eval_path = v114_root / scene / "reports" / f"{scene}_{method_name}_test_results.json"
    missing: list[str] = []
    detached, detached_path = _detached_results(detached_root, scene, missing)
    clean_key, clean, clean_path = _load_clean(clean_root, scene, missing)
    v106 = _load_v106(v106_rows, scene, missing)
    method_metrics, method_metric_source = _method_metrics_from_sources({}, detached, eval_path, method_name, missing)
    artifacts = _artifact_status(
        {
            "field": field_path,
            "manifest": manifest_path,
            "eval_results": eval_path,
            "eval_per_view": v114_root / scene / "reports" / f"{scene}_{method_name}_test_per_view.json",
        },
        missing,
    )
    status = "COMPLETE" if _has_all_metrics(method_metrics) else "PENDING_OR_MISSING_METRICS"
    return {
        "label": "v114_garden",
        "branch": "v114_oof_refit_pod_moe",
        "scene": scene,
        "status": status,
        "complete": _has_all_metrics(method_metrics),
        "method_name": method_name,
        "clean_method": clean_key,
        "paths": {
            "detached_results": detached_path,
            "clean_results": clean_path,
        },
        "artifacts": artifacts,
        "metric_source": method_metric_source,
        "metrics": {
            "clean": clean,
            "v106_parent": v106,
            "method": method_metrics,
        },
        "deltas": {
            "method_minus_clean": _delta(method_metrics, clean),
            "method_minus_v106": _delta(method_metrics, v106),
        },
        "missing": missing,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    v106_rows, v106_missing = _v106_rows(args.v106_assembled)
    rows = [
        _row_from_report(
            label="v110_counter",
            branch="v110_strict_split_parent_gate",
            scene="counter",
            report_path=args.v110_root / "counter" / "reports" / "counter_v110_strict_split_parent_gate_report.json",
            default_method_name="ours_26000_v110_strict_train_even_odd_parent_gate_counter",
            field_path=args.v110_root / "counter" / "fields" / "ours_26000_v110_strict_train_even_candidate_counter_field.pt",
            manifest_path=args.v110_root / "counter" / "fields" / "ours_26000_v110_strict_train_even_candidate_counter_field.manifest.json",
            extra_artifacts=None,
            detached_root=args.detached_root,
            clean_root=args.clean_baseline_root,
            v106_rows=v106_rows,
        ),
        _row_from_report(
            label="v110_bonsai",
            branch="v110_strict_split_parent_gate",
            scene="bonsai",
            report_path=args.v110_root / "bonsai" / "reports" / "bonsai_v110_strict_split_parent_gate_report.json",
            default_method_name="ours_26000_v110_strict_train_even_odd_parent_gate_bonsai",
            field_path=args.v110_root / "bonsai" / "fields" / "ours_26000_v110_strict_train_even_candidate_bonsai_field.pt",
            manifest_path=args.v110_root / "bonsai" / "fields" / "ours_26000_v110_strict_train_even_candidate_bonsai_field.manifest.json",
            extra_artifacts=None,
            detached_root=args.detached_root,
            clean_root=args.clean_baseline_root,
            v106_rows=v106_rows,
        ),
        _row_from_report(
            label="v111_flowers",
            branch="v111_end_to_end_strict_parent_gate",
            scene="flowers",
            report_path=args.v111_root / "flowers" / "reports" / "flowers_v111_end_to_end_strict_parent_gate_report.json",
            default_method_name="ours_26000_v111_train_even_odd_parent_gate_flowers",
            field_path=args.v111_root / "flowers" / "fields" / "ours_26000_v111_train_even_candidate_flowers_field.pt",
            manifest_path=args.v111_root / "flowers" / "fields" / "ours_26000_v111_train_even_candidate_flowers_field.manifest.json",
            extra_artifacts={
                "parent_field": args.v111_root / "flowers" / "fields" / "ours_26000_v111_train_all_parent_flowers_field.pt",
                "parent_manifest": args.v111_root / "flowers" / "fields" / "ours_26000_v111_train_all_parent_flowers_field.manifest.json",
            },
            detached_root=args.detached_root,
            clean_root=args.clean_baseline_root,
            v106_rows=v106_rows,
        ),
        _row_v114(
            v114_root=args.v114_root,
            detached_root=args.detached_root,
            clean_root=args.clean_baseline_root,
            v106_rows=v106_rows,
        ),
    ]
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "detached_root": str(args.detached_root),
            "clean_baseline_root": str(args.clean_baseline_root),
            "v106_assembled": str(args.v106_assembled),
            "v110_root": str(args.v110_root),
            "v111_root": str(args.v111_root),
            "v114_root": str(args.v114_root),
        },
        "v106_load_missing": v106_missing,
        "rows": rows,
        "complete_rows": sum(1 for row in rows if row.get("complete")),
        "row_count": len(rows),
        "all_complete": all(bool(row.get("complete")) for row in rows),
        "missing_count": sum(len(row.get("missing", [])) for row in rows) + len(v106_missing),
    }


def write_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# SPCarNet v110/v111/v114 Strict Branch Package",
        "",
        f"- created_at: `{summary.get('created_at')}`",
        f"- complete_rows: `{summary.get('complete_rows')} / {summary.get('row_count')}`",
        f"- all_complete: `{summary.get('all_complete')}`",
        f"- missing_count: `{summary.get('missing_count')}`",
        "",
        "Metric triplets are `PSNR / SSIM / LPIPS`. Deltas are method minus reference; negative LPIPS deltas are better.",
        "",
        "| label | status | method | method metrics | delta vs clean | delta vs v106 | metric source | missing |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in summary.get("rows", []):
        metrics = _as_dict(row.get("metrics"))
        deltas = _as_dict(row.get("deltas"))
        missing = row.get("missing", [])
        lines.append(
            "| {label} | {status} | `{method}` | {method_metrics} | {dclean} | {dv106} | {source} | {missing} |".format(
                label=row.get("label", ""),
                status=row.get("status", ""),
                method=row.get("method_name", ""),
                method_metrics=_triplet(_as_dict(metrics.get("method"))),
                dclean=_triplet(_as_dict(deltas.get("method_minus_clean")), signed=True),
                dv106=_triplet(_as_dict(deltas.get("method_minus_v106")), signed=True),
                source=row.get("metric_source", ""),
                missing="; ".join(str(item) for item in missing[:6]) + ("; ..." if len(missing) > 6 else ""),
            )
        )
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
        ]
    )
    for row in summary.get("rows", []):
        lines.append(f"### {row.get('label', '')}")
        artifacts = _as_dict(row.get("artifacts"))
        for key, value in sorted(artifacts.items()):
            item = _as_dict(value)
            lines.append(f"- {key}: `{item.get('path', '')}` present=`{item.get('present')}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detached_root", type=Path, default=DEFAULT_DETACHED_ROOT)
    parser.add_argument("--clean_baseline_root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--v106_assembled", type=Path, default=DEFAULT_V106_ASSEMBLED)
    parser.add_argument("--v110_root", type=Path, default=DEFAULT_V110_ROOT)
    parser.add_argument("--v111_root", type=Path, default=DEFAULT_V111_ROOT)
    parser.add_argument("--v114_root", type=Path, default=DEFAULT_V114_ROOT)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=Path("docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.json"),
    )
    parser.add_argument(
        "--out_md",
        type=Path,
        default=Path("docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(summary, args.out_md)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "complete_rows": summary["complete_rows"],
                "row_count": summary["row_count"],
                "missing_count": summary["missing_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
