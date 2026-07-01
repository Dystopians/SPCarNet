#!/usr/bin/env python3
"""Audit replay/current report consistency against an archived v322c root.

This script intentionally reads only report JSON files. It does not import or
run rendering/training pipeline code.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ARCHIVE_ROOT = Path("outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701")
DEFAULT_OUTPUT_JSON = Path("docs/car_model/results/v322c_replay_consistency_audit.json")
REPORT_NAME = "support_transport_apply_report.json"
SCENE_ORDER = (
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return payload


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def infer_scene(report_path: Path, root: Path, payload: dict[str, Any] | None = None) -> str:
    try:
        relative = report_path.relative_to(root)
    except ValueError:
        relative = report_path

    if len(relative.parts) >= 2 and relative.name == REPORT_NAME:
        first = relative.parts[0]
        if first not in ("logs", "wandb", "visuals", "renders", "gt"):
            return first

    candidates: list[str] = []
    lower_parts = [part.lower() for part in report_path.parts]
    lower_root = root.name.lower()
    for scene in SCENE_ORDER:
        if scene in lower_parts or scene in lower_root:
            candidates.append(scene)
    if len(candidates) == 1:
        return candidates[0]

    if payload:
        for key in ("scene", "target_scene", "source_scene"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value

    return report_path.parent.name


def find_reports(root: Path) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    if not root.exists():
        return reports

    paths = sorted(root.rglob(REPORT_NAME))
    for path in paths:
        payload: dict[str, Any] | None = None
        scene = infer_scene(path, root)
        if scene == root.name or scene in ("logs", "wandb", "visuals", "renders", "gt"):
            payload = read_json(path)
            scene = infer_scene(path, root, payload)

        current = reports.get(scene)
        if current is None or len(path.parts) < len(current.parts):
            reports[scene] = path
    return reports


def ordered_scenes(scenes: set[str]) -> list[str]:
    known = [scene for scene in SCENE_ORDER if scene in scenes]
    unknown = sorted(scene for scene in scenes if scene not in SCENE_ORDER)
    return known + unknown


def count_strings(values: list[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def merge_counts(*counts: dict[str, int] | None) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for item in counts:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                merged[str(key)] += int(value)
    return dict(sorted(merged.items()))


def selected_metric(report: dict[str, Any], metric: str) -> float | None:
    lower = metric.lower()
    upper = metric.upper()
    selected = report.get("selected_summary")
    if isinstance(selected, dict):
        for key in (f"candidate_{lower}", lower, upper):
            value = as_float(selected.get(key))
            if value is not None:
                return value

    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        for key in (upper, lower):
            value = as_float(metrics.get(key))
            if value is not None:
                return value

    summary = report.get("summary")
    if isinstance(summary, dict):
        for key in (f"selected_{lower}", f"candidate_{lower}", lower, upper):
            value = as_float(summary.get(key))
            if value is not None:
                return value

    return None


def selected_summary(report: dict[str, Any]) -> dict[str, float | None]:
    selected = report.get("selected_summary")
    if not isinstance(selected, dict):
        selected = {}
    return {
        "candidate_psnr": selected_metric(report, "PSNR"),
        "candidate_ssim": selected_metric(report, "SSIM"),
        "psnr_gain": as_float(selected.get("psnr_gain")),
        "ssim_gain": as_float(selected.get("ssim_gain")),
    }


def policy_counts_from_report(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    policy = report.get("policy")
    if not isinstance(policy, dict):
        policy = {}
    per_view = report.get("per_view")
    if not isinstance(per_view, list):
        per_view = []

    return {
        "scene_policy_selected_variant": count_strings([policy.get("selected_variant")]),
        "scene_policy_output_variant": count_strings([policy.get("output_variant")]),
        "per_view_selected_variant": count_strings(
            [row.get("selected_variant") for row in per_view if isinstance(row, dict)]
        ),
        "per_view_output_variant": count_strings(
            [row.get("output_variant") for row in per_view if isinstance(row, dict)]
        ),
    }


def summarize_policy(report: dict[str, Any], key: str) -> dict[str, Any] | None:
    policy = report.get(key)
    if not isinstance(policy, dict):
        return None

    source_counts = policy.get("source_selected_counts")
    if not isinstance(source_counts, dict):
        entries = policy.get("entries_by_variant")
        if isinstance(entries, dict):
            source_counts = {
                str(variant): len(rows)
                for variant, rows in entries.items()
                if isinstance(rows, list)
            }

    summary: dict[str, Any] = {
        "enabled": policy.get("enabled"),
        "verdict": policy.get("verdict"),
        "selected_variant": policy.get("selected_variant"),
        "scene_selected_variant": policy.get("scene_selected_variant"),
        "reject_variant": policy.get("reject_variant"),
        "source_selected_counts": merge_counts(source_counts if isinstance(source_counts, dict) else None),
    }
    return summary


def report_summary(report_path: Path | None) -> dict[str, Any] | None:
    if report_path is None:
        return None

    report = read_json(report_path)
    policy = report.get("policy")
    if not isinstance(policy, dict):
        policy = {}
    per_view = report.get("per_view")
    if not isinstance(per_view, list):
        per_view = []

    return {
        "report_path": str(report_path),
        "final_status": report.get("final_status"),
        "verdict": report.get("verdict"),
        "policy_selected_variant": policy.get("selected_variant"),
        "policy_output_variant": policy.get("output_variant"),
        "selected_summary": selected_summary(report),
        "selected_variant_counts": policy_counts_from_report(report),
        "per_view_count": len([row for row in per_view if isinstance(row, dict)]),
        "source_reliability_policy": summarize_policy(report, "source_reliability_policy"),
        "knn_policy": summarize_policy(report, "per_view_knn_policy"),
        "_raw_per_view": per_view,
    }


def per_view_by_id(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}
    rows = summary.get("_raw_per_view")
    if not isinstance(rows, list):
        return {}

    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        view = row.get("view", index)
        indexed[str(view)] = row
    return indexed


def compare_per_view(archive: dict[str, Any] | None, replay: dict[str, Any] | None) -> dict[str, Any]:
    archive_rows = per_view_by_id(archive)
    replay_rows = per_view_by_id(replay)
    common = sorted(set(archive_rows) & set(replay_rows))

    mismatches: list[dict[str, Any]] = []
    for view in common:
        archive_row = archive_rows[view]
        replay_row = replay_rows[view]
        archive_output = archive_row.get("output_variant")
        replay_output = replay_row.get("output_variant")
        if archive_output != replay_output:
            mismatches.append(
                {
                    "view": view,
                    "archive_output_variant": archive_output,
                    "replay_output_variant": replay_output,
                    "archive_selected_variant": archive_row.get("selected_variant"),
                    "replay_selected_variant": replay_row.get("selected_variant"),
                }
            )

    return {
        "archive_per_view_count": len(archive_rows),
        "replay_per_view_count": len(replay_rows),
        "common_view_count": len(common),
        "archive_only_views": sorted(set(archive_rows) - set(replay_rows)),
        "replay_only_views": sorted(set(replay_rows) - set(archive_rows)),
        "output_variant_mismatch_count": len(mismatches),
        "output_variant_mismatches": mismatches,
    }


def delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def compare_scene(archive: dict[str, Any] | None, replay: dict[str, Any] | None) -> dict[str, Any]:
    archive_selected = archive.get("selected_summary", {}) if archive else {}
    replay_selected = replay.get("selected_summary", {}) if replay else {}
    archive_psnr = archive_selected.get("candidate_psnr")
    archive_ssim = archive_selected.get("candidate_ssim")
    replay_psnr = replay_selected.get("candidate_psnr")
    replay_ssim = replay_selected.get("candidate_ssim")
    archive_psnr_gain = archive_selected.get("psnr_gain")
    archive_ssim_gain = archive_selected.get("ssim_gain")
    replay_psnr_gain = replay_selected.get("psnr_gain")
    replay_ssim_gain = replay_selected.get("ssim_gain")

    return {
        "has_both_reports": archive is not None and replay is not None,
        "delta_psnr_replay_minus_archive": delta(replay_psnr, archive_psnr),
        "delta_ssim_replay_minus_archive": delta(replay_ssim, archive_ssim),
        "delta_psnr_gain_replay_minus_archive": delta(replay_psnr_gain, archive_psnr_gain),
        "delta_ssim_gain_replay_minus_archive": delta(replay_ssim_gain, archive_ssim_gain),
        "per_view": compare_per_view(archive, replay),
    }


def strip_private(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def macro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    psnr_archive: list[float] = []
    psnr_replay: list[float] = []
    psnr_deltas: list[float] = []
    psnr_gain_archive: list[float] = []
    psnr_gain_replay: list[float] = []
    psnr_gain_deltas: list[float] = []
    ssim_archive: list[float] = []
    ssim_replay: list[float] = []
    ssim_deltas: list[float] = []
    ssim_gain_archive: list[float] = []
    ssim_gain_replay: list[float] = []
    ssim_gain_deltas: list[float] = []

    for row in rows:
        archive = row.get("archive") or {}
        replay = row.get("replay") or {}
        comparison = row.get("comparison") or {}
        archive_selected = archive.get("selected_summary") or {}
        replay_selected = replay.get("selected_summary") or {}

        apsnr = as_float(archive_selected.get("candidate_psnr"))
        rpsnr = as_float(replay_selected.get("candidate_psnr"))
        dpsnr = as_float(comparison.get("delta_psnr_replay_minus_archive"))
        apsnr_gain = as_float(archive_selected.get("psnr_gain"))
        rpsnr_gain = as_float(replay_selected.get("psnr_gain"))
        dpsnr_gain = as_float(comparison.get("delta_psnr_gain_replay_minus_archive"))
        assim = as_float(archive_selected.get("candidate_ssim"))
        rssim = as_float(replay_selected.get("candidate_ssim"))
        dssim = as_float(comparison.get("delta_ssim_replay_minus_archive"))
        assim_gain = as_float(archive_selected.get("ssim_gain"))
        rssim_gain = as_float(replay_selected.get("ssim_gain"))
        dssim_gain = as_float(comparison.get("delta_ssim_gain_replay_minus_archive"))

        if apsnr is not None and rpsnr is not None and dpsnr is not None:
            psnr_archive.append(apsnr)
            psnr_replay.append(rpsnr)
            psnr_deltas.append(dpsnr)
        if apsnr_gain is not None and rpsnr_gain is not None and dpsnr_gain is not None:
            psnr_gain_archive.append(apsnr_gain)
            psnr_gain_replay.append(rpsnr_gain)
            psnr_gain_deltas.append(dpsnr_gain)
        if assim is not None and rssim is not None and dssim is not None:
            ssim_archive.append(assim)
            ssim_replay.append(rssim)
            ssim_deltas.append(dssim)
        if assim_gain is not None and rssim_gain is not None and dssim_gain is not None:
            ssim_gain_archive.append(assim_gain)
            ssim_gain_replay.append(rssim_gain)
            ssim_gain_deltas.append(dssim_gain)

    return {
        "paired_psnr_scene_count": len(psnr_deltas),
        "archive_psnr_mean": mean(psnr_archive),
        "replay_psnr_mean": mean(psnr_replay),
        "delta_psnr_mean_replay_minus_archive": mean(psnr_deltas),
        "paired_psnr_gain_scene_count": len(psnr_gain_deltas),
        "archive_psnr_gain_mean": mean(psnr_gain_archive),
        "replay_psnr_gain_mean": mean(psnr_gain_replay),
        "delta_psnr_gain_mean_replay_minus_archive": mean(psnr_gain_deltas),
        "paired_ssim_scene_count": len(ssim_deltas),
        "archive_ssim_mean": mean(ssim_archive),
        "replay_ssim_mean": mean(ssim_replay),
        "delta_ssim_mean_replay_minus_archive": mean(ssim_deltas),
        "paired_ssim_gain_scene_count": len(ssim_gain_deltas),
        "archive_ssim_gain_mean": mean(ssim_gain_archive),
        "replay_ssim_gain_mean": mean(ssim_gain_replay),
        "delta_ssim_gain_mean_replay_minus_archive": mean(ssim_gain_deltas),
    }


def aggregate_selected_counts(rows: list[dict[str, Any]], side: str) -> dict[str, dict[str, int]]:
    buckets = (
        "scene_policy_selected_variant",
        "scene_policy_output_variant",
        "per_view_selected_variant",
        "per_view_output_variant",
    )
    totals: dict[str, Counter[str]] = {bucket: Counter() for bucket in buckets}
    for row in rows:
        summary = row.get(side)
        if not isinstance(summary, dict):
            continue
        counts = summary.get("selected_variant_counts")
        if not isinstance(counts, dict):
            continue
        for bucket in buckets:
            bucket_counts = counts.get(bucket)
            if isinstance(bucket_counts, dict):
                totals[bucket].update(merge_counts(bucket_counts))
    return {bucket: dict(sorted(counter.items())) for bucket, counter in totals.items()}


def aggregate_policy_counts(rows: list[dict[str, Any]], side: str, policy_key: str) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        summary = row.get(side)
        if not isinstance(summary, dict):
            continue
        policy = summary.get(policy_key)
        if not isinstance(policy, dict):
            continue
        totals.update(merge_counts(policy.get("source_selected_counts")))
    return dict(sorted(totals.items()))


def choose_default_replay_root(archive_root: Path, scene_filter: set[str] | None) -> Path | None:
    outputs_root = archive_root.parent
    if not outputs_root.is_dir():
        return None

    archive_reports = find_reports(archive_root)
    target_scenes = set(scene_filter or archive_reports.keys())
    candidates: list[tuple[int, float, str, Path]] = []
    for root in sorted(outputs_root.iterdir()):
        if not root.is_dir() or root == archive_root:
            continue
        lower = root.name.lower()
        if "replay" not in lower and "current" not in lower:
            continue
        reports = find_reports(root)
        if not reports:
            continue
        overlap = len(set(reports) & target_scenes) if target_scenes else len(reports)
        if overlap <= 0:
            continue
        candidates.append((overlap, root.stat().st_mtime, root.name, root))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[-1][3]


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    archive_root = args.archive_root
    scene_filter = set(args.scene or []) or None
    replay_root = args.replay_root
    if replay_root is None:
        replay_root = choose_default_replay_root(archive_root, scene_filter)
    if replay_root is None:
        raise RuntimeError("no replay root supplied and no replay/current root could be auto-detected")

    archive_reports = find_reports(archive_root)
    replay_reports = find_reports(replay_root)
    all_scenes = set(archive_reports) | set(replay_reports)
    if scene_filter is not None:
        all_scenes &= scene_filter

    rows: list[dict[str, Any]] = []
    for scene in ordered_scenes(all_scenes):
        archive_summary = report_summary(archive_reports.get(scene))
        replay_summary = report_summary(replay_reports.get(scene))
        rows.append(
            {
                "scene": scene,
                "archive": strip_private(archive_summary),
                "replay": strip_private(replay_summary),
                "comparison": compare_scene(archive_summary, replay_summary),
            }
        )

    return {
        "archive_root": str(archive_root),
        "replay_root": str(replay_root),
        "scene_filter": sorted(scene_filter) if scene_filter else None,
        "archive_report_count": len(archive_reports),
        "replay_report_count": len(replay_reports),
        "scene_count": len(rows),
        "missing_archive_scenes": [row["scene"] for row in rows if row.get("archive") is None],
        "missing_replay_scenes": [row["scene"] for row in rows if row.get("replay") is None],
        "macro": macro_summary(rows),
        "selected_variant_counts": {
            "archive": aggregate_selected_counts(rows, "archive"),
            "replay": aggregate_selected_counts(rows, "replay"),
        },
        "source_reliability_source_selected_counts": {
            "archive": aggregate_policy_counts(rows, "archive", "source_reliability_policy"),
            "replay": aggregate_policy_counts(rows, "replay", "source_reliability_policy"),
        },
        "knn_policy_source_selected_counts": {
            "archive": aggregate_policy_counts(rows, "archive", "knn_policy"),
            "replay": aggregate_policy_counts(rows, "replay", "knn_policy"),
        },
        "scenes": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare archived v322c support-transport reports against replay/current reports."
    )
    parser.add_argument(
        "--archive_root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help=f"Archived method root containing scene report JSONs. Default: {DEFAULT_ARCHIVE_ROOT}",
    )
    parser.add_argument(
        "--replay_root",
        type=Path,
        default=None,
        help="Replay/current method root. If omitted, the newest overlapping *replay*/*current* root is used.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output audit JSON path. Default: {DEFAULT_OUTPUT_JSON}",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Optional scene filter. May be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

    macro = audit["macro"]
    print(f"wrote {args.output_json}")
    print(f"archive_root: {audit['archive_root']}")
    print(f"replay_root: {audit['replay_root']}")
    print(f"scenes: {audit['scene_count']}")
    print(f"missing_archive_scenes: {len(audit['missing_archive_scenes'])}")
    print(f"missing_replay_scenes: {len(audit['missing_replay_scenes'])}")
    print(f"macro_delta_psnr_gain: {macro['delta_psnr_gain_mean_replay_minus_archive']}")
    print(f"macro_delta_ssim_gain: {macro['delta_ssim_gain_mean_replay_minus_archive']}")
    print(f"macro_delta_candidate_psnr: {macro['delta_psnr_mean_replay_minus_archive']}")
    print(f"macro_delta_candidate_ssim: {macro['delta_ssim_mean_replay_minus_archive']}")


if __name__ == "__main__":
    main()
