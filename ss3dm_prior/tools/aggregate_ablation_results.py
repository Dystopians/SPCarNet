"""Aggregate SS3DM prior ablation-suite outputs into paper-style tables."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ss3dm_prior.utils.io import load_json


SUMMARY_COLUMNS = [
    "variant",
    "status",
    "description",
    "selected_checkpoint",
    "recon_chamfer_l1",
    "visible_recon_chamfer_l1",
    "hidden_completion_chamfer_l1",
    "visible_recon_normal_cosine",
    "hidden_completion_gain",
    "denoise_gain_chamfer",
    "intrinsic_difficulty_calibration_mae",
    "intrinsic_difficulty_spearman",
    "occupancy_iou_visible",
    "free_space_violation_rate",
    "free_space_fp_rate",
    "retrieval_top1_nonself",
    "retrieval_top5_nonself",
    "protocol_valid",
]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Ablation manifest must be a JSON object: {path}")
    return manifest


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = entry.get("summary_metrics") or {}
    row = {
        "variant": entry.get("variant"),
        "status": entry.get("status", "unknown"),
        "description": entry.get("description", ""),
        "selected_checkpoint": entry.get("selected_checkpoint", ""),
        "recon_chamfer_l1": metrics.get("recon_chamfer_l1", ""),
        "visible_recon_chamfer_l1": metrics.get("visible_recon_chamfer_l1", ""),
        "hidden_completion_chamfer_l1": metrics.get("hidden_completion_chamfer_l1", ""),
        "visible_recon_normal_cosine": metrics.get("visible_recon_normal_cosine", ""),
        "hidden_completion_gain": metrics.get("hidden_completion_gain", ""),
        "denoise_gain_chamfer": metrics.get("denoise_gain_chamfer", ""),
        "intrinsic_difficulty_calibration_mae": metrics.get("intrinsic_difficulty_calibration_mae", ""),
        "intrinsic_difficulty_spearman": metrics.get("intrinsic_difficulty_spearman", ""),
        "occupancy_iou_visible": metrics.get("occupancy_iou_visible", ""),
        "free_space_violation_rate": metrics.get("free_space_violation_rate", ""),
        "free_space_fp_rate": metrics.get("free_space_fp_rate", ""),
        "retrieval_top1_nonself": metrics.get("retrieval_top1_nonself", ""),
        "retrieval_top5_nonself": metrics.get("retrieval_top5_nonself", ""),
        "protocol_valid": metrics.get("protocol_valid", ""),
    }
    return row


def aggregate_suite_results(
    manifest_path: str | Path,
    *,
    csv_path: str | Path | None = None,
    md_path: str | Path | None = None,
) -> tuple[Path, Path]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    suite_dir = manifest_path.parent
    csv_target = Path(csv_path).expanduser().resolve() if csv_path else suite_dir / "ablation_summary.csv"
    md_target = Path(md_path).expanduser().resolve() if md_path else suite_dir / "ablation_summary.md"
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)

    rows = [_row_from_entry(entry) for entry in manifest.get("variants", [])]
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# SS3DM Prior Ablation Summary",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Suite name: `{manifest.get('suite_name', 'ablation_suite')}`",
        f"- Manifest: `{manifest_path}`",
        f"- Variant count: `{len(rows)}`",
        "",
        "## Core Metrics",
        "",
        _markdown_table(rows, SUMMARY_COLUMNS),
        "",
        "## Notes",
        "",
        "- `status=completed` means both train and eval finished and `metrics_summary.json` was loaded.",
        "- Missing metrics remain blank or `nan` when a variant does not expose or define that measurement.",
        "- `selected_checkpoint` records which checkpoint was chosen for the eval summary, so the table is reproducible.",
    ]
    md_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_target, md_target


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate ablation-suite eval outputs into CSV/Markdown.")
    parser.add_argument("--suite_manifest", required=True, help="Path to `suite_manifest.json` from the ablation runner.")
    parser.add_argument("--csv_out", default=None, help="Optional output CSV path.")
    parser.add_argument("--md_out", default=None, help="Optional output Markdown path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argparser().parse_args(argv)
    csv_path, md_path = aggregate_suite_results(
        args.suite_manifest,
        csv_path=args.csv_out,
        md_path=args.md_out,
    )
    print(f"ablation_summary_csv: {csv_path}")
    print(f"ablation_summary_md: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
