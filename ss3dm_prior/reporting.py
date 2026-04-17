"""Reporting helpers for SS3DM prior evaluation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_KEYS = [
    "recon_chamfer_l1",
    "recon_normal_cosine",
    "denoise_gain_chamfer",
    "intrinsic_difficulty_mae",
    "intrinsic_difficulty_spearman",
    "occupancy_iou_visible",
    "free_space_violation_rate",
    "score_mae",
    "score_spearman",
    "point_defect_mae",
    "retrieval_top1_self_aligned",
    "retrieval_top5_self_aligned",
    "retrieval_top1_nonself",
    "retrieval_top5_nonself",
    "retrieval_top1_cross_sequence",
    "prototype_usage_entropy",
]


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _entropy_from_indices(indices: list[int]) -> float:
    if not indices:
        return float("nan")
    valid = np.asarray([index for index in indices if int(index) >= 0], dtype=np.int64)
    if valid.size == 0:
        return float("nan")
    _, counts = np.unique(valid, return_counts=True)
    probs = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-8, None))))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def aggregate_global_metrics(patch_rows: list[dict[str, Any]], retrieval_metrics: dict[str, float]) -> dict[str, Any]:
    summary = {
        "recon_chamfer_l1": _safe_mean([float(row["recon_chamfer_l1"]) for row in patch_rows]),
        "recon_normal_cosine": _safe_mean([float(row["recon_normal_cosine"]) for row in patch_rows]),
        "denoise_gain_chamfer": _safe_mean([float(row["denoise_gain_chamfer"]) for row in patch_rows]),
        "intrinsic_difficulty_mae": _safe_mean([float(row.get("intrinsic_difficulty_abs_error", float("nan"))) for row in patch_rows]),
        "occupancy_iou_visible": _safe_mean([float(row.get("occupancy_iou_visible", float("nan"))) for row in patch_rows]),
        "free_space_violation_rate": _safe_mean([float(row.get("free_space_violation_rate", float("nan"))) for row in patch_rows]),
        "score_mae": _safe_mean([float(row["score_abs_error"]) for row in patch_rows]),
        "point_defect_mae": _safe_mean([float(row["point_defect_mae"]) for row in patch_rows]),
        "mean_visible_support": _safe_mean([float(row.get("visible_support_count", float("nan"))) for row in patch_rows]),
        "mean_intrinsic_difficulty": _safe_mean([float(row.get("intrinsic_difficulty_target", float("nan"))) for row in patch_rows]),
    }
    summary.update(retrieval_metrics)
    summary["patch_count"] = len(patch_rows)
    summary["sequence_count"] = len({row["sequence_id"] for row in patch_rows})
    summary["town_count"] = len({row["town_id"] for row in patch_rows})
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return summary


def aggregate_per_group(
    patch_rows: list[dict[str, Any]],
    *,
    group_key: str,
    include_sequence_count: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in patch_rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    rows: list[dict[str, Any]] = []
    for group_value, items in sorted(grouped.items()):
        row = {
            group_key: group_value,
            "patch_count": len(items),
            "recon_chamfer_l1": _safe_mean([float(item["recon_chamfer_l1"]) for item in items]),
            "recon_normal_cosine": _safe_mean([float(item["recon_normal_cosine"]) for item in items]),
            "denoise_gain_chamfer": _safe_mean([float(item["denoise_gain_chamfer"]) for item in items]),
            "intrinsic_difficulty_mae": _safe_mean([float(item.get("intrinsic_difficulty_abs_error", float("nan"))) for item in items]),
            "occupancy_iou_visible": _safe_mean([float(item.get("occupancy_iou_visible", float("nan"))) for item in items]),
            "free_space_violation_rate": _safe_mean([float(item.get("free_space_violation_rate", float("nan"))) for item in items]),
            "score_mae": _safe_mean([float(item["score_abs_error"]) for item in items]),
            "point_defect_mae": _safe_mean([float(item["point_defect_mae"]) for item in items]),
            "mean_visible_support": _safe_mean([float(item.get("visible_support_count", float("nan"))) for item in items]),
            "mean_intrinsic_difficulty": _safe_mean([float(item.get("intrinsic_difficulty_target", float("nan"))) for item in items]),
            "prototype_usage_entropy": _entropy_from_indices([int(item.get("code_index", -1)) for item in items]),
        }
        if group_key == "town_id":
            row["sequence_count"] = len({item["sequence_id"] for item in items})
        if group_key == "sequence_id":
            row["town_id"] = items[0]["town_id"]
            row["mean_corruption_severity"] = _safe_mean([float(item["corruption_score_target"]) for item in items])
            row["mean_denoise_gain"] = row["denoise_gain_chamfer"]
        rows.append(row)
    return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body]) if body else "\n".join([header, sep])


def write_report_md(
    report_path: str | Path,
    *,
    checkpoint_path: str,
    split_config_path: str,
    summary_metrics: dict[str, Any],
    protocol_audit: dict[str, Any],
    per_town_rows: list[dict[str, Any]],
    per_sequence_rows: list[dict[str, Any]],
    qualitative_paths: list[Path],
) -> Path:
    report_path = Path(report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rel_paths = [path.relative_to(report_path.parent) for path in qualitative_paths if path.exists()]
    lines = [
        "# SS3DM Prior Evaluation Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Split config: `{split_config_path}`",
        f"- Split towns: `train={protocol_audit.get('protocol_summary', {}).get('expected_train_towns')}` / `val={protocol_audit.get('protocol_summary', {}).get('expected_val_towns')}` / `test={protocol_audit.get('protocol_summary', {}).get('expected_test_towns')}`",
        f"- Patch count: `{summary_metrics.get('patch_count')}`",
        f"- Protocol audit result: `{'clean' if protocol_audit.get('protocol_valid') else 'warnings_present'}`",
        f"- Strict valid: `{protocol_audit.get('protocol_valid')}`",
        f"- Legacy retrieval top1: `{summary_metrics.get('retrieval_top1_self_aligned')}`",
        f"- Non-self retrieval top1: `{summary_metrics.get('retrieval_top1_nonself')}`",
        f"- Non-self retrieval top5: `{summary_metrics.get('retrieval_top5_nonself')}`",
        "",
        "## Protocol Audit",
        "",
        f"- Protocol valid: `{protocol_audit.get('protocol_valid')}`",
        f"- debug_use_all_patches_for_train_val: `{protocol_audit.get('protocol_summary', {}).get('debug_use_all_patches_for_train_val')}`",
        f"- allow_debug_split_override: `{protocol_audit.get('protocol_summary', {}).get('allow_debug_split_override')}`",
        f"- allow_split_fallback: `{protocol_audit.get('protocol_summary', {}).get('allow_split_fallback')}`",
        f"- Split config path: `{protocol_audit.get('protocol_summary', {}).get('split_config_path')}`",
        f"- Expected train towns: `{protocol_audit.get('protocol_summary', {}).get('expected_train_towns')}`",
        f"- Expected val towns: `{protocol_audit.get('protocol_summary', {}).get('expected_val_towns')}`",
        f"- Expected test towns: `{protocol_audit.get('protocol_summary', {}).get('expected_test_towns')}`",
        "- Protocol warnings:",
    ]
    warnings_list = protocol_audit.get("protocol_warnings", [])
    if warnings_list:
        lines.extend(f"  - {warning}" for warning in warnings_list)
    else:
        lines.append("  - none")
    lines.extend(
        [
            "",
            "## Global Metrics",
            "",
            _markdown_table(
                [summary_metrics],
                [key for key in ["patch_count", "town_count", "sequence_count", *SUMMARY_KEYS] if key in summary_metrics],
            ),
            "",
            "## Per Town",
            "",
            _markdown_table(
                per_town_rows,
                [
                    "town_id",
                    "patch_count",
                    "sequence_count",
                    "recon_chamfer_l1",
                    "recon_normal_cosine",
                    "denoise_gain_chamfer",
                    "intrinsic_difficulty_mae",
                    "occupancy_iou_visible",
                    "free_space_violation_rate",
                    "mean_visible_support",
                    "mean_intrinsic_difficulty",
                    "prototype_usage_entropy",
                ],
            ),
            "",
            "## Per Sequence",
            "",
            _markdown_table(
                per_sequence_rows,
                [
                    "sequence_id",
                    "town_id",
                    "patch_count",
                    "mean_corruption_severity",
                    "mean_denoise_gain",
                    "recon_chamfer_l1",
                    "recon_normal_cosine",
                    "intrinsic_difficulty_mae",
                    "occupancy_iou_visible",
                    "free_space_violation_rate",
                    "mean_visible_support",
                    "mean_intrinsic_difficulty",
                    "prototype_usage_entropy",
                ],
            ),
            "",
            "## Qualitative Figures",
            "",
        ]
    )
    for rel_path in rel_paths:
        lines.append(f"- `{rel_path}`")
    lines.extend(
        [
            "",
            "## Conclusion Template",
            "",
            "- Strict protocol validity was `TODO summarize whether this run is strict-valid and whether warnings matter for interpretation`.",
            "- Reconstruction and denoise gain were `TODO summarize overall geometry restoration quality`.",
            "- Visible/free-space behavior was `TODO summarize occupancy IoU and free-space violation behavior`.",
            "- Intrinsic difficulty prediction was `TODO summarize ranking/regression quality`.",
            "- Legacy vs non-self retrieval suggests `TODO summarize whether prototypes recover reusable local geometry rather than self-only matches`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
