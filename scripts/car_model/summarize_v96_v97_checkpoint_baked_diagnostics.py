#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
import re
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
LOWER_IS_BETTER = {"LPIPS", "depth_mae", "depth_rmse", "depth_abs_rel", "normal_mean_ang_deg"}
SKIP_SEARCH_DIRS = {
    ".git",
    "__pycache__",
    "media",
    "point_cloud",
    "tmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MeshSplatting/SPCarNet checkpoint-baked diagnostics from "
            "results.json, per_view.json, contract JSONs, geometry_eval_colmap, "
            "and wandb offline run directories."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Run specs. May be passed once with many pairs or repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output path. If it ends in .json, JSON is written there and Markdown "
            "beside it; otherwise Markdown is written there and JSON beside it."
        ),
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional baseline JSON for metric deltas. Accepts this script's JSON, results.json, or row summaries.",
    )
    parser.add_argument(
        "--print",
        dest="print_markdown",
        action="store_true",
        help="Echo the generated Markdown to stdout after writing files.",
    )
    return parser.parse_args()


def flatten_run_specs(groups: list[list[str]]) -> list[str]:
    specs: list[str] = []
    for group in groups:
        specs.extend(group)
    return specs


def parse_run_specs(groups: list[list[str]]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in flatten_run_specs(groups):
        if "=" not in item:
            raise ValueError(f"run spec must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name:
            raise ValueError(f"run spec has an empty NAME: {item!r}")
        if not raw_path:
            raise ValueError(f"run spec has an empty PATH: {item!r}")
        if name in seen:
            raise ValueError(f"duplicate run name: {name}")
        seen.add(name)
        runs.append((name, Path(raw_path).expanduser()))
    return runs


def read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    except OSError as exc:
        return None, f"read_error:{exc}"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def metric_value(mapping: dict[str, Any], metric: str) -> float | None:
    if metric in mapping:
        return as_float(mapping[metric])
    wanted = metric.lower()
    for key, value in mapping.items():
        if str(key).lower() == wanted:
            return as_float(value)
    return None


def metrics_from_mapping(mapping: Any) -> dict[str, float]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, float] = {}
    for metric in METRICS:
        value = metric_value(mapping, metric)
        if value is not None:
            out[metric] = value
    return out


def first_method_metrics(payload: Any) -> tuple[str, dict[str, float]]:
    direct = metrics_from_mapping(payload)
    if direct:
        return "metrics", direct
    if not isinstance(payload, dict):
        return "", {}
    candidates: list[tuple[str, dict[str, float]]] = []
    for key, value in payload.items():
        metrics = metrics_from_mapping(value)
        if metrics:
            candidates.append((str(key), metrics))
    if not candidates:
        return "", {}
    candidates.sort(key=lambda item: natural_sort_key(item[0]))
    return candidates[-1]


def natural_sort_key(text: str) -> list[Any]:
    parts: list[Any] = []
    for piece in re.split(r"(\d+)", text):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            parts.append(piece)
    return parts


def walk_files_named(root: Path, filename: str) -> list[Path]:
    matches: list[Path] = []
    if not root.is_dir():
        return matches
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            item
            for item in dirnames
            if item not in SKIP_SEARCH_DIRS and not item.startswith(".")
        ]
        if filename in filenames:
            matches.append(Path(dirpath) / filename)
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches


def choose_file(root: Path, preferred: list[Path], filename: str) -> Path | None:
    for item in preferred:
        candidate = item if item.is_absolute() else root / item
        if candidate.is_file():
            return candidate
    matches = walk_files_named(root, filename)
    return matches[0] if matches else None


def contract_preferred_paths(root: Path, filename: str) -> list[Path]:
    paths = [Path("contract") / filename, Path(filename)]
    if root.name == "recovery_model":
        paths.append(root.parent / "contract" / filename)
    return paths


def geometry_files(root: Path) -> list[Path]:
    preferred_dirs = [
        root / "recovery_model" / "geometry_eval_colmap",
        root / "geometry_eval_colmap",
    ]
    seen: set[Path] = set()
    found: list[Path] = []
    for directory in preferred_dirs:
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(path)
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                item
                for item in dirnames
                if item not in SKIP_SEARCH_DIRS and not item.startswith(".")
            ]
            if Path(dirpath).name != "geometry_eval_colmap":
                continue
            for filename in sorted(f for f in filenames if f.endswith(".json")):
                path = Path(dirpath) / filename
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(path)
    found.sort(key=lambda path: (len(path.parts), natural_sort_key(path.name), str(path)))
    return found


def summarize_per_view(payload: Any, method: str) -> dict[str, Any]:
    method_payload = payload
    if isinstance(payload, dict) and method in payload and isinstance(payload[method], dict):
        method_payload = payload[method]
    if not isinstance(method_payload, dict):
        return {}

    summary: dict[str, Any] = {}
    for metric in METRICS:
        values = metric_values_by_view(method_payload.get(metric))
        if not values:
            continue
        numbers = list(values.values())
        if metric in LOWER_IS_BETTER:
            worst_view = max(values, key=values.get)
        else:
            worst_view = min(values, key=values.get)
        summary[metric] = {
            "count": len(numbers),
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "worst_view": worst_view,
            "worst_value": values[worst_view],
        }
    return summary


def metric_values_by_view(value: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            number = as_float(item)
            if number is not None:
                values[str(key)] = number
    elif isinstance(value, list):
        for index, item in enumerate(value):
            number = as_float(item)
            if number is not None:
                values[str(index)] = number
    return values


def compact_contract(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    keys = (
        "preset",
        "source_path",
        "output_path",
        "load_iteration",
        "final_iteration",
        "wandb_project",
        "wandb_group",
        "wandb_name",
        "train_seed",
        "milestone_iterations",
        "split_strategy",
        "teacher_render_lambda",
        "teacher_render_mask_mode",
        "parent_render_rollback_lambda",
        "parent_render_rollback_aggregation",
        "parent_render_rollback_cvar_fraction",
        "checkpoint_geometry_anchor_lambda",
        "checkpoint_render_depth_anchor_lambda",
        "checkpoint_render_normal_anchor_lambda",
        "sparse_depth_parent_rollback_lambda",
        "sparse_depth_parent_rollback_loss_space",
        "sparse_depth_parent_rollback_regressed_only",
    )
    out = {key: summary.get(key) for key in keys if key in summary}
    source_path = out.get("source_path")
    if isinstance(source_path, str) and source_path:
        out["scene"] = Path(source_path).name
    return out


def compact_topology(audit: Any) -> dict[str, Any]:
    if not isinstance(audit, dict):
        return {}
    load = audit.get("load") if isinstance(audit.get("load"), dict) else {}
    final = audit.get("final") if isinstance(audit.get("final"), dict) else {}
    load_vertices = as_int(load.get("vertices"))
    final_vertices = as_int(final.get("vertices"))
    load_triangles = as_int(load.get("triangles"))
    final_triangles = as_int(final.get("triangles"))
    return {
        "topology_unchanged": audit.get("topology_unchanged"),
        "load_iteration": as_int(load.get("iteration")),
        "final_iteration": as_int(final.get("iteration")),
        "load_vertices": load_vertices,
        "final_vertices": final_vertices,
        "delta_vertices": (
            final_vertices - load_vertices if final_vertices is not None and load_vertices is not None else None
        ),
        "load_triangles": load_triangles,
        "final_triangles": final_triangles,
        "delta_triangles": (
            final_triangles - load_triangles if final_triangles is not None and load_triangles is not None else None
        ),
        "required_flags": audit.get("required_flags") if isinstance(audit.get("required_flags"), list) else [],
        "sparse_depth_enabled": audit.get("sparse_depth_enabled"),
    }


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def compact_geometry(path: Path, payload: Any, root: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"path": rel(path, root), "error": "not_a_json_object"}
    depth = payload.get("depth") if isinstance(payload.get("depth"), dict) else {}
    normal = payload.get("normal") if isinstance(payload.get("normal"), dict) else {}
    return {
        "path": rel(path, root),
        "iteration": as_int(payload.get("iteration")),
        "num_test_views": as_int(payload.get("num_test_views")),
        "num_views_evaluated": as_int(payload.get("num_views_evaluated")),
        "point_error_max": as_float(payload.get("point_error_max")),
        "max_points_per_view": as_int(payload.get("max_points_per_view")),
        "depth_count": as_int(depth.get("count")),
        "depth_mae": as_float(depth.get("mae")),
        "depth_rmse": as_float(depth.get("rmse")),
        "depth_abs_rel": as_float(depth.get("abs_rel")),
        "normal_count": as_int(normal.get("count")),
        "normal_mean_abs_cos": as_float(normal.get("mean_abs_cos")),
        "normal_mean_ang_deg": as_float(normal.get("mean_ang_deg")),
        "normal_median_ang_deg": as_float(normal.get("median_ang_deg")),
    }


def choose_geometry_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    return sorted(
        items,
        key=lambda item: (
            item.get("iteration") if item.get("iteration") is not None else -1,
            item.get("num_views_evaluated") if item.get("num_views_evaluated") is not None else -1,
            str(item.get("path", "")),
        ),
    )[-1]


def find_wandb_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not root.is_dir():
        return runs
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [item for item in dirnames if item not in SKIP_SEARCH_DIRS and not item.startswith(".")]
        directory = Path(dirpath)
        if directory.name.startswith("offline-run-"):
            runs.append(compact_wandb_dir(directory, root))
            dirnames[:] = []
    runs.sort(key=lambda item: str(item.get("path", "")))
    return runs


def compact_wandb_dir(path: Path, root: Path) -> dict[str, Any]:
    match = re.match(r"offline-run-(\d{8}_\d{6})-(.+)$", path.name)
    run_id = match.group(2) if match else ""
    started_at = match.group(1) if match else ""
    files_dir = path / "files"
    media_dir = files_dir / "media" / "images"
    wandb_files = sorted(path.glob("run-*.wandb"))
    summary_path = files_dir / "wandb-summary.json"
    summary_payload, summary_error = read_json(summary_path) if summary_path.is_file() else ({}, None)
    return {
        "path": rel(path, root),
        "run_id": run_id,
        "started_at": started_at,
        "wandb_file_count": len(wandb_files),
        "requirements_exists": (files_dir / "requirements.txt").is_file(),
        "debug_log_exists": (path / "logs" / "debug.log").is_file(),
        "image_file_count": count_image_files(media_dir),
        "summary_path": rel(summary_path, root) if summary_path.is_file() else "",
        "summary_error": summary_error,
        "summary": summary_payload if isinstance(summary_payload, dict) else {},
    }


def count_image_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                count += 1
    return count


def summarize_run(name: str, root: Path, baselines: dict[str, dict[str, float]]) -> dict[str, Any]:
    warnings: list[str] = []
    out: dict[str, Any] = {
        "name": name,
        "path": str(root),
        "exists": root.exists(),
        "warnings": warnings,
    }
    if not root.exists():
        warnings.append("run_path_missing")
        return out
    if not root.is_dir():
        warnings.append("run_path_not_directory")

    results_path = choose_file(root, [Path("recovery_model/results.json"), Path("results.json")], "results.json")
    per_view_path = choose_file(root, [Path("recovery_model/per_view.json"), Path("per_view.json")], "per_view.json")
    recovery_path = choose_file(
        root,
        contract_preferred_paths(root, "recovery_summary.json"),
        "recovery_summary.json",
    )
    topology_path = choose_file(
        root,
        contract_preferred_paths(root, "topology_audit.json"),
        "topology_audit.json",
    )

    out["artifact_paths"] = {
        "results": rel(results_path, root) if results_path else "",
        "per_view": rel(per_view_path, root) if per_view_path else "",
        "recovery_summary": rel(recovery_path, root) if recovery_path else "",
        "topology_audit": rel(topology_path, root) if topology_path else "",
    }

    metrics: dict[str, float] = {}
    method = ""
    if results_path is None:
        warnings.append("missing_results_json")
    else:
        payload, error = read_json(results_path)
        if error:
            warnings.append(f"results_json_{error}")
        else:
            method, metrics = first_method_metrics(payload)
            if not metrics:
                warnings.append("results_json_no_metrics")
    out["method"] = method
    out["metrics"] = metrics

    if per_view_path is None:
        warnings.append("missing_per_view_json")
        out["per_view"] = {}
    else:
        payload, error = read_json(per_view_path)
        if error:
            warnings.append(f"per_view_json_{error}")
            out["per_view"] = {}
        else:
            out["per_view"] = summarize_per_view(payload, method)

    if recovery_path is None:
        warnings.append("missing_recovery_summary_json")
        out["contract"] = {}
    else:
        payload, error = read_json(recovery_path)
        if error:
            warnings.append(f"recovery_summary_json_{error}")
            out["contract"] = {}
        else:
            out["contract"] = compact_contract(payload)

    if topology_path is None:
        warnings.append("missing_topology_audit_json")
        out["topology"] = {}
    else:
        payload, error = read_json(topology_path)
        if error:
            warnings.append(f"topology_audit_json_{error}")
            out["topology"] = {}
        else:
            out["topology"] = compact_topology(payload)

    geometry: list[dict[str, Any]] = []
    for path in geometry_files(root):
        payload, error = read_json(path)
        if error:
            geometry.append({"path": rel(path, root), "error": error})
        else:
            geometry.append(compact_geometry(path, payload, root))
    if not geometry:
        warnings.append("missing_geometry_eval_colmap_json")
    out["geometry_eval_colmap"] = geometry
    out["selected_geometry"] = choose_geometry_summary(geometry)

    wandb_runs = find_wandb_runs(root)
    if not wandb_runs:
        warnings.append("missing_wandb_offline_run")
    out["wandb_offline_runs"] = wandb_runs

    baseline = baselines.get(name) or baselines.get("__default__")
    if baseline:
        out["baseline_metrics"] = baseline
        out["delta_vs_baseline"] = metric_delta(metrics, baseline)
    else:
        out["baseline_metrics"] = {}
        out["delta_vs_baseline"] = {}
    return out


def metric_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for metric in METRICS:
        if metric in metrics and metric in baseline:
            delta[metric] = metrics[metric] - baseline[metric]
    return delta


def extract_baselines(path: Path | None) -> tuple[dict[str, dict[str, float]], list[str]]:
    if path is None:
        return {}, []
    payload, error = read_json(path)
    if error:
        return {}, [f"baseline_json_{error}:{path}"]
    baselines: dict[str, dict[str, float]] = {}
    collect_baseline_metrics(payload, baselines, "__default__")
    return baselines, []


def collect_baseline_metrics(value: Any, out: dict[str, dict[str, float]], fallback_name: str) -> None:
    metrics = metrics_from_mapping(value)
    if metrics:
        out.setdefault(fallback_name, metrics)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            name = row_name(item) or f"{fallback_name}_{index}"
            collect_baseline_metrics(item, out, name)
        return
    if not isinstance(value, dict):
        return

    if isinstance(value.get("metrics"), dict):
        name = row_name(value) or fallback_name
        metrics = metrics_from_mapping(value["metrics"])
        if metrics:
            out[name] = metrics
            if "__default__" not in out:
                out["__default__"] = metrics
    if isinstance(value.get("runs"), list):
        collect_baseline_metrics(value["runs"], out, fallback_name)
    if isinstance(value.get("rows"), list):
        collect_baseline_metrics(value["rows"], out, fallback_name)

    for key, item in value.items():
        if key in {"runs", "rows", "metrics"}:
            continue
        if isinstance(item, dict):
            metrics = metrics_from_mapping(item.get("metrics") if isinstance(item.get("metrics"), dict) else item)
            if metrics:
                out[str(key)] = metrics
                if "__default__" not in out:
                    out["__default__"] = metrics


def row_name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("name", "run", "scene", "scene_name", "id", "method"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() == ".json":
        return output.with_suffix(".md"), output
    if output.suffix:
        return output, output.with_suffix(".json")
    return output, output.with_name(output.name + ".json")


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["runs"]
    title = "# SPCarNet Checkpoint-Baked Diagnostics"
    lines = [
        title,
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
    ]
    baseline = summary.get("baseline_json")
    if baseline:
        lines.extend([f"Baseline JSON: `{baseline}`", ""])

    headers = [
        "run",
        "status",
        "iter",
        "PSNR",
        "SSIM",
        "LPIPS",
        "dPSNR",
        "dSSIM",
        "dLPIPS",
        "views",
        "worst view",
        "topology",
        "V/T",
        "depth MAE",
        "normal deg",
        "teacher",
        "rollback",
        "geo anchor",
        "sparse depth",
        "wandb",
        "warnings",
    ]
    table = [headers]
    for row in rows:
        table.append(markdown_row(row))
    lines.extend(render_table(table))
    lines.extend(
        [
            "",
            "Metric deltas are current minus baseline when `--baseline-json` supplies matching metrics. For LPIPS, lower is better.",
            "",
            "JSON summary: `" + str(summary["output_json"]) + "`",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_row(row: dict[str, Any]) -> list[str]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    delta = row.get("delta_vs_baseline") if isinstance(row.get("delta_vs_baseline"), dict) else {}
    contract = row.get("contract") if isinstance(row.get("contract"), dict) else {}
    topology = row.get("topology") if isinstance(row.get("topology"), dict) else {}
    geometry = row.get("selected_geometry") if isinstance(row.get("selected_geometry"), dict) else {}
    per_view = row.get("per_view") if isinstance(row.get("per_view"), dict) else {}
    warnings = row.get("warnings") if isinstance(row.get("warnings"), list) else []
    load_iter = contract.get("load_iteration") or topology.get("load_iteration")
    final_iter = contract.get("final_iteration") or topology.get("final_iteration")
    wandb_runs = row.get("wandb_offline_runs") if isinstance(row.get("wandb_offline_runs"), list) else []
    return [
        str(row.get("name", "")),
        row_status(row),
        compact_iter(load_iter, final_iter),
        fmt(metrics.get("PSNR")),
        fmt(metrics.get("SSIM")),
        fmt(metrics.get("LPIPS")),
        fmt_delta(delta.get("PSNR")),
        fmt_delta(delta.get("SSIM")),
        fmt_delta(delta.get("LPIPS")),
        view_counts(per_view),
        worst_views(per_view),
        topology_status(topology),
        vertices_triangles(topology),
        fmt(geometry.get("depth_mae")),
        fmt(geometry.get("normal_mean_ang_deg")),
        fmt(contract.get("teacher_render_lambda")),
        fmt(contract.get("parent_render_rollback_lambda")),
        geo_anchor(contract),
        fmt(contract.get("sparse_depth_parent_rollback_lambda")),
        wandb_label(wandb_runs),
        ", ".join(str(item) for item in warnings[:4]) + ("..." if len(warnings) > 4 else ""),
    ]


def row_status(row: dict[str, Any]) -> str:
    if not row.get("exists", False):
        return "missing"
    warnings = row.get("warnings") if isinstance(row.get("warnings"), list) else []
    has_metrics = bool(row.get("metrics"))
    if not warnings and has_metrics:
        return "ok"
    if has_metrics:
        return "partial"
    return "no metrics"


def compact_iter(load_iter: Any, final_iter: Any) -> str:
    if load_iter is None and final_iter is None:
        return ""
    if load_iter is None:
        return str(final_iter)
    if final_iter is None:
        return str(load_iter)
    return f"{load_iter}->{final_iter}"


def fmt(value: Any, digits: int = 6) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}g}"


def fmt_delta(value: Any, digits: int = 6) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:+.{digits}g}"


def view_counts(per_view: dict[str, Any]) -> str:
    counts = []
    for metric in METRICS:
        item = per_view.get(metric)
        if isinstance(item, dict) and item.get("count") is not None:
            counts.append(f"{metric}:{item['count']}")
    return "; ".join(counts)


def worst_views(per_view: dict[str, Any]) -> str:
    items = []
    for metric in METRICS:
        item = per_view.get(metric)
        if isinstance(item, dict) and item.get("worst_view"):
            items.append(f"{metric}:{item['worst_view']}={fmt(item.get('worst_value'))}")
    return "; ".join(items)


def topology_status(topology: dict[str, Any]) -> str:
    if not topology:
        return ""
    unchanged = topology.get("topology_unchanged")
    if unchanged is True:
        return "unchanged"
    if unchanged is False:
        return "changed"
    return "unknown"


def vertices_triangles(topology: dict[str, Any]) -> str:
    vertices = topology.get("final_vertices")
    triangles = topology.get("final_triangles")
    if vertices is None and triangles is None:
        return ""
    return f"{compact_int(vertices)}/{compact_int(triangles)}"


def compact_int(value: Any) -> str:
    number = as_int(value)
    if number is None:
        return ""
    return str(number)


def geo_anchor(contract: dict[str, Any]) -> str:
    parts = []
    depth = contract.get("checkpoint_render_depth_anchor_lambda")
    normal = contract.get("checkpoint_render_normal_anchor_lambda")
    geom = contract.get("checkpoint_geometry_anchor_lambda")
    if depth is not None:
        parts.append(f"d={fmt(depth)}")
    if normal is not None:
        parts.append(f"n={fmt(normal)}")
    if geom is not None:
        parts.append(f"g={fmt(geom)}")
    return "/".join(parts)


def wandb_label(runs: list[Any]) -> str:
    if not runs:
        return ""
    labels = []
    for item in runs[:2]:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id") or Path(str(item.get("path", ""))).name
        images = item.get("image_file_count")
        labels.append(f"{run_id}({images} img)")
    if len(runs) > 2:
        labels.append(f"+{len(runs) - 2}")
    return "; ".join(labels)


def render_table(rows: list[list[str]]) -> list[str]:
    escaped = [[escape_cell(cell) for cell in row] for row in rows]
    widths = [max(len(row[index]) for row in escaped) for index in range(len(escaped[0]))]
    lines = []
    header = escaped[0]
    lines.append("| " + " | ".join(pad(header[index], widths[index]) for index in range(len(header))) + " |")
    lines.append("| " + " | ".join("-" * widths[index] for index in range(len(header))) + " |")
    for row in escaped[1:]:
        lines.append("| " + " | ".join(pad(row[index], widths[index]) for index in range(len(row))) + " |")
    return lines


def escape_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def pad(value: str, width: int) -> str:
    return value + " " * (width - len(value))


def aggregate_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_count": len(runs),
        "runs_with_metrics": sum(1 for row in runs if row.get("metrics")),
        "runs_with_geometry_eval": sum(1 for row in runs if row.get("geometry_eval_colmap")),
        "runs_with_wandb_offline": sum(1 for row in runs if row.get("wandb_offline_runs")),
        "topology_unchanged": sum(
            1
            for row in runs
            if isinstance(row.get("topology"), dict) and row["topology"].get("topology_unchanged") is True
        ),
        "missing_results": [
            row.get("name")
            for row in runs
            if isinstance(row.get("warnings"), list) and "missing_results_json" in row["warnings"]
        ],
    }


def main() -> None:
    args = parse_args()
    try:
        run_specs = parse_run_specs(args.runs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    md_path, json_path = output_paths(args.output)
    baselines, baseline_warnings = extract_baselines(args.baseline_json)
    runs = [summarize_run(name, path, baselines) for name, path in run_specs]
    for warning in baseline_warnings:
        if runs:
            runs[0].setdefault("warnings", []).append(warning)

    summary = {
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "baseline_json": str(args.baseline_json) if args.baseline_json else "",
        "output_markdown": str(md_path),
        "output_json": str(json_path),
        "runs": runs,
        "aggregate": aggregate_summary(runs),
    }
    markdown = render_markdown(summary)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_markdown:
        print(markdown, end="")


if __name__ == "__main__":
    main()
