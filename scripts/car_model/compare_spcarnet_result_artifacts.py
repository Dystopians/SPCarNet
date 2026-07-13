#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ("PSNR", "SSIM", "LPIPS")
KNOWN_SCENES = (
    "bicycle",
    "flowers",
    "garden",
    "stump",
    "treehill",
    "room",
    "counter",
    "kitchen",
    "bonsai",
    "courtyard",
    "parking",
    "parking_phone_tiny",
)
METRIC_CONTAINER_KEYS = (
    "selected_metrics",
    "metrics",
    "v84_metrics",
    "v82_metrics",
    "v64_metrics",
    "v56_metrics",
    "v52_metrics",
    "v50_metrics",
    "v49_metrics",
    "v48_metrics",
)
LOWER_IS_BETTER_PATTERNS = (
    "lpips",
    "mae",
    "mse",
    "rmse",
    "loss",
    "absrel",
    "abs_rel",
    "error",
    "l1",
)


@dataclass(frozen=True)
class ExtractedRow:
    scene: str
    metrics: dict[str, float]
    method: str = ""
    source: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SPCarNet baseline/current/improved JSON artifacts and emit a "
            "Markdown metric table with deltas."
        )
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline result or summary JSON.")
    parser.add_argument("--current", type=Path, required=True, help="Current result or summary JSON.")
    parser.add_argument("--improved", type=Path, required=True, help="Improved result or summary JSON.")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--improved-label", default="improved")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to include. Repeatable. Defaults to PSNR, SSIM, LPIPS.",
    )
    parser.add_argument(
        "--scene",
        default="",
        help="Force the scene/key name for single-row artifacts that do not carry a scene field.",
    )
    parser.add_argument(
        "--metric-source",
        default="auto",
        choices=("auto",) + METRIC_CONTAINER_KEYS,
        help="Metric container key to prefer for row-based summaries.",
    )
    parser.add_argument(
        "--baseline-method",
        default="",
        help="Top-level method key to select when the baseline JSON is a method->metrics map.",
    )
    parser.add_argument(
        "--current-method",
        default="",
        help="Top-level method key to select when the current JSON is a method->metrics map.",
    )
    parser.add_argument(
        "--improved-method",
        default="",
        help="Top-level method key to select when the improved JSON is a method->metrics map.",
    )
    parser.add_argument("--digits", type=int, default=9, help="Decimal places for numeric output.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Win/tie epsilon for summary counts.")
    parser.add_argument("--title", default="SPCarNet Result Artifact Comparison")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_metric_lookup(row: dict[str, Any], metric: str) -> Any:
    if metric in row:
        return row[metric]
    lower_metric = metric.lower()
    for key, value in row.items():
        if str(key).lower() == lower_metric:
            return value
    return None


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null", "na"}:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def metrics_from_mapping(row: dict[str, Any], metrics: list[str]) -> dict[str, float]:
    found: dict[str, float] = {}
    for metric in metrics:
        value = as_float(canonical_metric_lookup(row, metric))
        if value is not None:
            found[metric] = value
    return found


def preferred_metric_container(row: dict[str, Any], metric_source: str) -> dict[str, Any] | None:
    if metric_source != "auto":
        value = row.get(metric_source)
        return value if isinstance(value, dict) else None
    for key in METRIC_CONTAINER_KEYS:
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return row


def row_metrics(row: dict[str, Any], metrics: list[str], metric_source: str) -> dict[str, float]:
    container = preferred_metric_container(row, metric_source)
    if container is None:
        return {}
    extracted = metrics_from_mapping(container, metrics)
    if extracted:
        return extracted
    if container is not row:
        return metrics_from_mapping(row, metrics)
    return {}


def infer_scene_from_path(path: Path) -> str:
    normalized_parts = [part.lower() for part in path.parts]
    for scene in sorted(KNOWN_SCENES, key=len, reverse=True):
        scene_lower = scene.lower()
        for part in normalized_parts:
            tokens = [token for token in re.split(r"[^a-z0-9]+", part) if token]
            if scene_lower in tokens or part == scene_lower:
                return scene
    parent = path.parent.name
    if parent:
        return parent
    return path.stem or "artifact"


def row_scene(row: dict[str, Any], fallback: str) -> str:
    for key in ("scene", "scene_name", "name", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def row_method(row: dict[str, Any]) -> str:
    for key in ("method", "method_name", "selected_source", "v82_method", "v63b_method", "v55d_method"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def metric_map_like(value: Any, metrics: list[str]) -> bool:
    return isinstance(value, dict) and bool(metrics_from_mapping(value, metrics))


def extract_rows_from_list(
    rows: list[Any],
    *,
    path: Path,
    metrics: list[str],
    metric_source: str,
    fallback_scene: str,
) -> list[ExtractedRow]:
    extracted: list[ExtractedRow] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        values = row_metrics(item, metrics, metric_source)
        if not values:
            continue
        scene = row_scene(item, fallback_scene)
        if scene == fallback_scene and len(rows) > 1:
            scene = f"{fallback_scene}/{index}"
        extracted.append(ExtractedRow(scene=scene, metrics=values, method=row_method(item), source=str(path)))
    return extracted


def extract_method_map_rows(
    payload: dict[str, Any],
    *,
    path: Path,
    metrics: list[str],
    forced_scene: str,
    method_key: str,
) -> list[ExtractedRow]:
    fallback_scene = forced_scene or infer_scene_from_path(path)
    method_items = [(str(key), value) for key, value in payload.items() if metric_map_like(value, metrics)]
    if method_key:
        if method_key not in payload or not metric_map_like(payload.get(method_key), metrics):
            return []
        method_items = [(method_key, payload[method_key])]
    if not method_items:
        return []
    extracted: list[ExtractedRow] = []
    multiple_methods = len(method_items) > 1
    for method, value in method_items:
        metrics_found = metrics_from_mapping(value, metrics)
        scene = fallback_scene if not multiple_methods else f"{fallback_scene}/{method}"
        extracted.append(ExtractedRow(scene=scene, metrics=metrics_found, method=method, source=str(path)))
    return extracted


def extract_rows(
    payload: Any,
    *,
    path: Path,
    metrics: list[str],
    metric_source: str,
    forced_scene: str,
    method_key: str,
) -> list[ExtractedRow]:
    fallback_scene = forced_scene or infer_scene_from_path(path)
    if isinstance(payload, list):
        return extract_rows_from_list(
            payload,
            path=path,
            metrics=metrics,
            metric_source=metric_source,
            fallback_scene=fallback_scene,
        )
    if not isinstance(payload, dict):
        return []
    rows = payload.get("rows")
    if isinstance(rows, list):
        extracted = extract_rows_from_list(
            rows,
            path=path,
            metrics=metrics,
            metric_source=metric_source,
            fallback_scene=fallback_scene,
        )
        if extracted:
            return extracted
    direct_metrics = row_metrics(payload, metrics, metric_source)
    if direct_metrics:
        return [
            ExtractedRow(
                scene=row_scene(payload, fallback_scene),
                metrics=direct_metrics,
                method=row_method(payload),
                source=str(path),
            )
        ]
    return extract_method_map_rows(
        payload,
        path=path,
        metrics=metrics,
        forced_scene=forced_scene,
        method_key=method_key,
    )


def rows_to_scene_map(rows: list[ExtractedRow]) -> dict[str, ExtractedRow]:
    scene_map: dict[str, ExtractedRow] = {}
    for row in rows:
        scene_map[row.scene] = row
    return scene_map


def metric_lower_is_better(metric: str) -> bool:
    metric_lower = metric.lower()
    return any(pattern in metric_lower for pattern in LOWER_IS_BETTER_PATTERNS)


def is_win(delta: float, metric: str, eps: float) -> bool:
    return delta < -eps if metric_lower_is_better(metric) else delta > eps


def fmt_value(value: float | None, digits: int, signed: bool = False) -> str:
    if value is None:
        return "NA"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def sort_scenes(scenes: set[str]) -> list[str]:
    known_order = {scene: index for index, scene in enumerate(KNOWN_SCENES)}

    def key(scene: str) -> tuple[int, int, str]:
        base = scene.split("/", 1)[0]
        return (0 if base in known_order else 1, known_order.get(base, 999), scene)

    return sorted(scenes, key=key)


def table_header(columns: list[str]) -> list[str]:
    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]


def value_for(rows: dict[str, ExtractedRow], scene: str, metric: str) -> float | None:
    row = rows.get(scene)
    if row is None:
        return None
    return row.metrics.get(metric)


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def build_metric_table(
    baseline: dict[str, ExtractedRow],
    current: dict[str, ExtractedRow],
    improved: dict[str, ExtractedRow],
    *,
    metrics: list[str],
    labels: dict[str, str],
    digits: int,
) -> list[str]:
    columns = [
        "scene",
        "metric",
        labels["baseline"],
        labels["current"],
        labels["improved"],
        f"{labels['current']} - {labels['baseline']}",
        f"{labels['improved']} - {labels['baseline']}",
        f"{labels['improved']} - {labels['current']}",
    ]
    lines = table_header(columns)
    scenes = set(baseline) | set(current) | set(improved)
    for scene in sort_scenes(scenes):
        for metric in metrics:
            base_value = value_for(baseline, scene, metric)
            current_value = value_for(current, scene, metric)
            improved_value = value_for(improved, scene, metric)
            row = [
                markdown_escape(scene),
                markdown_escape(metric),
                fmt_value(base_value, digits),
                fmt_value(current_value, digits),
                fmt_value(improved_value, digits),
                fmt_value(delta(current_value, base_value), digits, signed=True),
                fmt_value(delta(improved_value, base_value), digits, signed=True),
                fmt_value(delta(improved_value, current_value), digits, signed=True),
            ]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_summary_table(
    baseline: dict[str, ExtractedRow],
    current: dict[str, ExtractedRow],
    improved: dict[str, ExtractedRow],
    *,
    metrics: list[str],
    labels: dict[str, str],
    digits: int,
    eps: float,
) -> list[str]:
    columns = [
        "metric",
        "shared scenes",
        f"mean {labels['current']} - {labels['baseline']}",
        f"mean {labels['improved']} - {labels['baseline']}",
        f"mean {labels['improved']} - {labels['current']}",
        f"{labels['current']} wins {labels['baseline']}",
        f"{labels['improved']} wins {labels['baseline']}",
        f"{labels['improved']} wins {labels['current']}",
    ]
    lines = table_header(columns)
    scenes = sort_scenes(set(baseline) | set(current) | set(improved))
    for metric in metrics:
        current_base_deltas: list[float] = []
        improved_base_deltas: list[float] = []
        improved_current_deltas: list[float] = []
        for scene in scenes:
            base_value = value_for(baseline, scene, metric)
            current_value = value_for(current, scene, metric)
            improved_value = value_for(improved, scene, metric)
            current_base = delta(current_value, base_value)
            improved_base = delta(improved_value, base_value)
            improved_current = delta(improved_value, current_value)
            if current_base is not None:
                current_base_deltas.append(current_base)
            if improved_base is not None:
                improved_base_deltas.append(improved_base)
            if improved_current is not None:
                improved_current_deltas.append(improved_current)
        shared_count = len(improved_current_deltas)
        row = [
            markdown_escape(metric),
            str(shared_count),
            fmt_value(mean(current_base_deltas), digits, signed=True),
            fmt_value(mean(improved_base_deltas), digits, signed=True),
            fmt_value(mean(improved_current_deltas), digits, signed=True),
            f"{sum(is_win(value, metric, eps) for value in current_base_deltas)}/{len(current_base_deltas)}",
            f"{sum(is_win(value, metric, eps) for value in improved_base_deltas)}/{len(improved_base_deltas)}",
            f"{sum(is_win(value, metric, eps) for value in improved_current_deltas)}/{len(improved_current_deltas)}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_missing_notes(
    baseline: dict[str, ExtractedRow],
    current: dict[str, ExtractedRow],
    improved: dict[str, ExtractedRow],
    *,
    metrics: list[str],
    labels: dict[str, str],
) -> list[str]:
    role_rows = {
        labels["baseline"]: baseline,
        labels["current"]: current,
        labels["improved"]: improved,
    }
    scenes = sort_scenes(set(baseline) | set(current) | set(improved))
    missing: list[str] = []
    for role, rows in role_rows.items():
        for scene in scenes:
            row = rows.get(scene)
            if row is None:
                missing.append(f"- `{role}` missing scene `{scene}`")
                continue
            absent_metrics = [metric for metric in metrics if metric not in row.metrics]
            if absent_metrics:
                missing.append(
                    f"- `{role}` scene `{scene}` missing metrics: `{', '.join(absent_metrics)}`"
                )
    return missing


def build_source_notes(role: str, rows: dict[str, ExtractedRow]) -> list[str]:
    if not rows:
        return [f"- `{role}`: no metric rows extracted"]
    sources = sorted({row.source for row in rows.values()})
    methods = sorted({row.method for row in rows.values() if row.method})
    source_text = ", ".join(f"`{source}`" for source in sources)
    if methods:
        method_text = ", ".join(f"`{method}`" for method in methods[:8])
        if len(methods) > 8:
            method_text += f", ... ({len(methods)} total)"
        return [f"- `{role}`: {len(rows)} rows from {source_text}; methods {method_text}"]
    return [f"- `{role}`: {len(rows)} rows from {source_text}"]


def build_markdown(
    *,
    title: str,
    baseline: dict[str, ExtractedRow],
    current: dict[str, ExtractedRow],
    improved: dict[str, ExtractedRow],
    metrics: list[str],
    labels: dict[str, str],
    digits: int,
    eps: float,
) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        "## Metric Table",
        "",
        *build_metric_table(
            baseline,
            current,
            improved,
            metrics=metrics,
            labels=labels,
            digits=digits,
        ),
        "",
        "## Delta Summary",
        "",
        *build_summary_table(
            baseline,
            current,
            improved,
            metrics=metrics,
            labels=labels,
            digits=digits,
            eps=eps,
        ),
        "",
        "## Inputs",
        "",
    ]
    for role, rows in (
        (labels["baseline"], baseline),
        (labels["current"], current),
        (labels["improved"], improved),
    ):
        lines.extend(build_source_notes(role, rows))
    missing = build_missing_notes(
        baseline,
        current,
        improved,
        metrics=metrics,
        labels=labels,
    )
    if missing:
        lines.extend(["", "## Missing Data", "", *missing])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    metrics = args.metrics or list(DEFAULT_METRICS)
    labels = {
        "baseline": args.baseline_label,
        "current": args.current_label,
        "improved": args.improved_label,
    }
    role_paths = {
        "baseline": args.baseline,
        "current": args.current,
        "improved": args.improved,
    }
    role_methods = {
        "baseline": args.baseline_method,
        "current": args.current_method,
        "improved": args.improved_method,
    }
    extracted: dict[str, dict[str, ExtractedRow]] = {}
    for role, path in role_paths.items():
        rows = extract_rows(
            read_json(path),
            path=path,
            metrics=metrics,
            metric_source=args.metric_source,
            forced_scene=args.scene,
            method_key=role_methods[role],
        )
        extracted[role] = rows_to_scene_map(rows)
    markdown = build_markdown(
        title=args.title,
        baseline=extracted["baseline"],
        current=extracted["current"],
        improved=extracted["improved"],
        metrics=metrics,
        labels=labels,
        digits=args.digits,
        eps=args.eps,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
