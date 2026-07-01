#!/usr/bin/env python3
"""Summarize strict per-view dominance gaps from apply reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_BASELINE_ROOT = Path("outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701")
DEFAULT_OUTPUT_JSON = Path("docs/car_model/results/v323_strict_oracle_gap_summary.json")
DEFAULT_SCENES = (
    "bicycle",
    "flowers",
    "garden",
    "stump",
    "treehill",
    "room",
    "counter",
    "kitchen",
    "bonsai",
)
REPORT_NAME = "support_transport_apply_report.json"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute strict oracle gaps relative to selected per-view outputs in "
            "support-transport apply reports."
        )
    )
    parser.add_argument(
        "--baseline_root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help=f"Baseline apply report root. Default: {DEFAULT_BASELINE_ROOT}",
    )
    parser.add_argument(
        "--method_root",
        type=Path,
        default=None,
        help="Optional second apply report root to summarize and compare against baseline.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional scene list. Accepts space-separated values and/or comma-separated groups.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_JSON}",
    )
    parser.add_argument("--eps", type=float, default=EPS, help="Tolerance for strict non-degradation checks.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def parse_scenes(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    scenes: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in value.split(","):
            scene = item.strip()
            if scene and scene not in seen:
                scenes.append(scene)
                seen.add(scene)
    return scenes


def discover_scenes(root: Path) -> list[str]:
    present = {path.parent.name for path in root.glob(f"*/{REPORT_NAME}") if path.is_file()}
    scenes = [scene for scene in DEFAULT_SCENES if scene in present]
    scenes.extend(sorted(present.difference(scenes)))
    return scenes


def metric(metrics: dict[str, Any], key: str) -> float | None:
    return as_float(metrics.get(key))


def selected_metrics(view: dict[str, Any]) -> dict[str, Any] | None:
    selected = view.get("selected")
    if isinstance(selected, dict):
        return selected
    selected_variant = view.get("selected_variant")
    candidates = view.get("candidate_metrics")
    if isinstance(selected_variant, str) and isinstance(candidates, dict):
        candidate = candidates.get(selected_variant)
        if isinstance(candidate, dict):
            return candidate
    return None


def candidate_metrics(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = view.get("candidate_metrics")
    if not isinstance(raw, dict):
        return {}
    candidates: dict[str, dict[str, Any]] = {}
    for name, metrics_value in raw.items():
        if isinstance(name, str) and isinstance(metrics_value, dict):
            candidates[name] = metrics_value
    return candidates


def choose_strict_oracle(
    *,
    candidates: dict[str, dict[str, Any]],
    selected_variant: str | None,
    selected_psnr_gain: float,
    selected_ssim_gain: float,
    eps: float,
) -> tuple[str | None, dict[str, Any] | None]:
    eligible: list[tuple[str, dict[str, Any], float, float]] = []
    for name, candidate in candidates.items():
        psnr_gain = metric(candidate, "psnr_gain")
        ssim_gain = metric(candidate, "ssim_gain")
        if psnr_gain is None or ssim_gain is None:
            continue
        if psnr_gain + eps >= selected_psnr_gain and ssim_gain + eps >= selected_ssim_gain:
            eligible.append((name, candidate, psnr_gain, ssim_gain))
    if not eligible:
        return None, None

    max_psnr = max(item[2] for item in eligible)
    top = [item for item in eligible if abs(item[2] - max_psnr) <= eps]
    max_ssim = max(item[3] for item in top)
    top = [item for item in top if abs(item[3] - max_ssim) <= eps]
    if selected_variant is not None:
        for item in top:
            if item[0] == selected_variant:
                return item[0], item[1]
    chosen = sorted(top, key=lambda item: item[0])[0]
    return chosen[0], chosen[1]


def summarize_scene(root: Path, scene: str, eps: float) -> dict[str, Any]:
    report_path = root / scene / REPORT_NAME
    report = read_json(report_path)
    per_view = report.get("per_view")
    if not isinstance(per_view, list):
        raise ValueError(f"missing per_view list in {report_path}")

    selected_variant_counts: Counter[str] = Counter()
    oracle_variant_counts: Counter[str] = Counter()
    improved_oracle_variant_counts: Counter[str] = Counter()
    selected_psnr_gains: list[float] = []
    selected_ssim_gains: list[float] = []
    oracle_psnr_gains: list[float] = []
    oracle_ssim_gains: list[float] = []
    psnr_deltas: list[float] = []
    ssim_deltas: list[float] = []
    improved_psnr_deltas: list[float] = []
    improved_ssim_deltas: list[float] = []
    improved_views: list[dict[str, Any]] = []
    warnings: list[str] = []
    eligible_view_count = 0

    for index, raw_view in enumerate(per_view):
        if not isinstance(raw_view, dict):
            warnings.append(f"view_{index}:not_object")
            continue

        view_name = str(raw_view.get("view", index))
        selected = selected_metrics(raw_view)
        candidates = candidate_metrics(raw_view)
        selected_variant = raw_view.get("selected_variant")
        selected_variant = selected_variant if isinstance(selected_variant, str) else None
        if selected is None:
            warnings.append(f"{view_name}:missing_selected_metrics")
            continue

        selected_psnr_gain = metric(selected, "psnr_gain")
        selected_ssim_gain = metric(selected, "ssim_gain")
        if selected_psnr_gain is None or selected_ssim_gain is None:
            warnings.append(f"{view_name}:missing_selected_gain")
            continue

        selected_variant_counts[selected_variant or "__unknown__"] += 1
        oracle_variant, oracle = choose_strict_oracle(
            candidates=candidates,
            selected_variant=selected_variant,
            selected_psnr_gain=selected_psnr_gain,
            selected_ssim_gain=selected_ssim_gain,
            eps=eps,
        )
        if oracle is None or oracle_variant is None:
            warnings.append(f"{view_name}:no_strict_eligible_candidate")
            oracle_variant = selected_variant or "__selected__"
            oracle = selected
        else:
            eligible_view_count += 1

        oracle_psnr_gain = metric(oracle, "psnr_gain")
        oracle_ssim_gain = metric(oracle, "ssim_gain")
        if oracle_psnr_gain is None or oracle_ssim_gain is None:
            warnings.append(f"{view_name}:missing_oracle_gain")
            continue

        psnr_delta = oracle_psnr_gain - selected_psnr_gain
        ssim_delta = oracle_ssim_gain - selected_ssim_gain
        improved = psnr_delta > eps or ssim_delta > eps

        oracle_variant_counts[oracle_variant] += 1
        selected_psnr_gains.append(selected_psnr_gain)
        selected_ssim_gains.append(selected_ssim_gain)
        oracle_psnr_gains.append(oracle_psnr_gain)
        oracle_ssim_gains.append(oracle_ssim_gain)
        psnr_deltas.append(psnr_delta)
        ssim_deltas.append(ssim_delta)

        if improved:
            improved_oracle_variant_counts[oracle_variant] += 1
            improved_psnr_deltas.append(psnr_delta)
            improved_ssim_deltas.append(ssim_delta)
            improved_views.append(
                {
                    "view": view_name,
                    "selected_variant": selected_variant,
                    "oracle_variant": oracle_variant,
                    "selected_psnr_gain": selected_psnr_gain,
                    "selected_ssim_gain": selected_ssim_gain,
                    "oracle_psnr_gain": oracle_psnr_gain,
                    "oracle_ssim_gain": oracle_ssim_gain,
                    "psnr_delta": psnr_delta,
                    "ssim_delta": ssim_delta,
                }
            )

    view_count = len(selected_psnr_gains)
    return {
        "scene": scene,
        "report_path": str(report_path),
        "view_count": view_count,
        "eligible_view_count": eligible_view_count,
        "improved_view_count": len(improved_views),
        "improved_view_fraction": (len(improved_views) / view_count) if view_count else None,
        "selected_variant_counts": dict(sorted(selected_variant_counts.items())),
        "oracle_variant_counts": dict(sorted(oracle_variant_counts.items())),
        "improved_oracle_variant_counts": dict(sorted(improved_oracle_variant_counts.items())),
        "selected_mean_psnr_gain": mean_or_none(selected_psnr_gains),
        "selected_mean_ssim_gain": mean_or_none(selected_ssim_gains),
        "oracle_mean_psnr_gain": mean_or_none(oracle_psnr_gains),
        "oracle_mean_ssim_gain": mean_or_none(oracle_ssim_gains),
        "mean_psnr_delta": mean_or_none(psnr_deltas),
        "mean_ssim_delta": mean_or_none(ssim_deltas),
        "improved_mean_psnr_delta": mean_or_none(improved_psnr_deltas),
        "improved_mean_ssim_delta": mean_or_none(improved_ssim_deltas),
        "improved_views": improved_views,
        "warnings": warnings,
    }


def add_counts(target: Counter[str], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] += int(value)


def summarize_root(root: Path, scenes: list[str], eps: float) -> dict[str, Any]:
    scene_summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for scene in scenes:
        report_path = root / scene / REPORT_NAME
        if not report_path.is_file():
            missing.append(scene)
            continue
        scene_summaries.append(summarize_scene(root, scene, eps))

    selected_counts: Counter[str] = Counter()
    oracle_counts: Counter[str] = Counter()
    improved_counts: Counter[str] = Counter()
    for scene_summary in scene_summaries:
        add_counts(selected_counts, scene_summary["selected_variant_counts"])
        add_counts(oracle_counts, scene_summary["oracle_variant_counts"])
        add_counts(improved_counts, scene_summary["improved_oracle_variant_counts"])

    total_views = sum(int(scene["view_count"]) for scene in scene_summaries)
    total_improved = sum(int(scene["improved_view_count"]) for scene in scene_summaries)
    all_psnr_deltas = [
        view["psnr_delta"]
        for scene in scene_summaries
        for view in scene["improved_views"]
        if isinstance(view.get("psnr_delta"), (int, float))
    ]
    all_ssim_deltas = [
        view["ssim_delta"]
        for scene in scene_summaries
        for view in scene["improved_views"]
        if isinstance(view.get("ssim_delta"), (int, float))
    ]
    aggregate = {
        "scene_count": len(scene_summaries),
        "missing_scenes": missing,
        "view_count": total_views,
        "improved_view_count": total_improved,
        "improved_view_fraction": (total_improved / total_views) if total_views else None,
        "macro_selected_psnr_gain": mean_or_none(
            [scene["selected_mean_psnr_gain"] for scene in scene_summaries if scene["selected_mean_psnr_gain"] is not None]
        ),
        "macro_selected_ssim_gain": mean_or_none(
            [scene["selected_mean_ssim_gain"] for scene in scene_summaries if scene["selected_mean_ssim_gain"] is not None]
        ),
        "macro_oracle_psnr_gain": mean_or_none(
            [scene["oracle_mean_psnr_gain"] for scene in scene_summaries if scene["oracle_mean_psnr_gain"] is not None]
        ),
        "macro_oracle_ssim_gain": mean_or_none(
            [scene["oracle_mean_ssim_gain"] for scene in scene_summaries if scene["oracle_mean_ssim_gain"] is not None]
        ),
        "macro_psnr_delta": mean_or_none(
            [scene["mean_psnr_delta"] for scene in scene_summaries if scene["mean_psnr_delta"] is not None]
        ),
        "macro_ssim_delta": mean_or_none(
            [scene["mean_ssim_delta"] for scene in scene_summaries if scene["mean_ssim_delta"] is not None]
        ),
        "micro_improved_psnr_delta": mean_or_none(all_psnr_deltas),
        "micro_improved_ssim_delta": mean_or_none(all_ssim_deltas),
        "selected_variant_counts": dict(sorted(selected_counts.items())),
        "oracle_variant_counts": dict(sorted(oracle_counts.items())),
        "improved_oracle_variant_counts": dict(sorted(improved_counts.items())),
    }
    return {
        "root": str(root),
        "aggregate": aggregate,
        "scenes": scene_summaries,
    }


def numeric_delta(lhs: Any, rhs: Any) -> float | None:
    lhs_float = as_float(lhs)
    rhs_float = as_float(rhs)
    if lhs_float is None or rhs_float is None:
        return None
    return lhs_float - rhs_float


def compare_summaries(baseline: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "macro_selected_psnr_gain",
        "macro_selected_ssim_gain",
        "macro_oracle_psnr_gain",
        "macro_oracle_ssim_gain",
        "macro_psnr_delta",
        "macro_ssim_delta",
        "improved_view_count",
        "improved_view_fraction",
    )
    baseline_aggregate = baseline["aggregate"]
    method_aggregate = method["aggregate"]
    aggregate = {
        f"{key}_method_minus_baseline": numeric_delta(method_aggregate.get(key), baseline_aggregate.get(key))
        for key in keys
    }

    baseline_by_scene = {scene["scene"]: scene for scene in baseline["scenes"]}
    method_by_scene = {scene["scene"]: scene for scene in method["scenes"]}
    scenes: list[dict[str, Any]] = []
    for scene in sorted(set(baseline_by_scene).intersection(method_by_scene)):
        baseline_scene = baseline_by_scene[scene]
        method_scene = method_by_scene[scene]
        scenes.append(
            {
                "scene": scene,
                "mean_psnr_delta_method_minus_baseline": numeric_delta(
                    method_scene.get("mean_psnr_delta"), baseline_scene.get("mean_psnr_delta")
                ),
                "mean_ssim_delta_method_minus_baseline": numeric_delta(
                    method_scene.get("mean_ssim_delta"), baseline_scene.get("mean_ssim_delta")
                ),
                "improved_view_count_method_minus_baseline": numeric_delta(
                    method_scene.get("improved_view_count"), baseline_scene.get("improved_view_count")
                ),
                "selected_mean_psnr_gain_method_minus_baseline": numeric_delta(
                    method_scene.get("selected_mean_psnr_gain"), baseline_scene.get("selected_mean_psnr_gain")
                ),
                "selected_mean_ssim_gain_method_minus_baseline": numeric_delta(
                    method_scene.get("selected_mean_ssim_gain"), baseline_scene.get("selected_mean_ssim_gain")
                ),
            }
        )
    return {
        "aggregate": aggregate,
        "scenes": scenes,
        "baseline_only_scenes": sorted(set(baseline_by_scene).difference(method_by_scene)),
        "method_only_scenes": sorted(set(method_by_scene).difference(baseline_by_scene)),
    }


def main() -> None:
    args = parse_args()
    scenes = parse_scenes(args.scenes)
    if scenes is None:
        scenes = discover_scenes(args.baseline_root)

    baseline = summarize_root(args.baseline_root, scenes, args.eps)
    method = summarize_root(args.method_root, scenes, args.eps) if args.method_root is not None else None
    output = {
        "baseline_root": str(args.baseline_root),
        "method_root": str(args.method_root) if args.method_root is not None else None,
        "scenes": scenes,
        "strict_rule": {
            "eligible_candidate": "candidate.psnr_gain >= selected.psnr_gain and candidate.ssim_gain >= selected.ssim_gain",
            "oracle_selection": "max psnr_gain, tie-break by max ssim_gain, prefer selected variant on exact ties",
            "eps": args.eps,
        },
        "baseline": baseline,
        "method": method,
        "comparison": compare_summaries(baseline, method) if method is not None else None,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    aggregate = baseline["aggregate"]
    print(f"wrote {args.output_json}")
    print(
        "baseline strict oracle gap: "
        f"scenes={aggregate['scene_count']} views={aggregate['view_count']} "
        f"improved={aggregate['improved_view_count']} "
        f"macro_psnr_delta={aggregate['macro_psnr_delta']:.9f} "
        f"macro_ssim_delta={aggregate['macro_ssim_delta']:.9f}"
    )
    if method is not None:
        method_aggregate = method["aggregate"]
        print(
            "method strict oracle gap: "
            f"scenes={method_aggregate['scene_count']} views={method_aggregate['view_count']} "
            f"improved={method_aggregate['improved_view_count']} "
            f"macro_psnr_delta={method_aggregate['macro_psnr_delta']:.9f} "
            f"macro_ssim_delta={method_aggregate['macro_ssim_delta']:.9f}"
        )


if __name__ == "__main__":
    main()
