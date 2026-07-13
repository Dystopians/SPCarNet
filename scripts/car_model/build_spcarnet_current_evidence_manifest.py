#!/usr/bin/env python3
"""Build a compact evidence manifest for the current SPCarNet paper loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvidenceItem:
    category: str
    label: str
    path: str
    note: str
    required: bool = True


DEFAULT_ITEMS = [
    EvidenceItem(
        "headline_phasej",
        "Phase-J full9 selected-clean summary",
        "outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md",
        "Main local same-protocol Phase-J headline table.",
    ),
    EvidenceItem(
        "headline_phasej",
        "Phase-J full9 selected-clean JSON",
        "outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json",
        "Machine-readable scene aggregate for the headline endpoint.",
    ),
    EvidenceItem(
        "headline_phasej",
        "Phase-J closure audit",
        "outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md",
        "Scene-level RGB, per-view, and geometry closure audit.",
    ),
    EvidenceItem(
        "headline_phasej",
        "Phase-J closure audit CSV",
        "outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv",
        "Tabular closure audit for paper-table ingestion.",
    ),
    EvidenceItem(
        "stage2_shape_prior",
        "Stage2 v3 full-val MAP-fit eval",
        "outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json",
        "v3 held-out z-only MAP-fit baseline; strict JSON.",
    ),
    EvidenceItem(
        "stage2_shape_prior",
        "Stage2 v4 epoch50 full-val MAP-fit eval",
        "outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json",
        "Best available v4 normal-band checkpoint eval.",
    ),
    EvidenceItem(
        "stage2_shape_prior",
        "Stage2 v4 final full-val MAP-fit eval",
        "outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json",
        "Final checkpoint eval; kept as late-training degradation evidence.",
    ),
    EvidenceItem(
        "stage2_shape_prior",
        "Stage2 v4 checkpoint selection JSON",
        "outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.json",
        "Deterministic report-side best-checkpoint selector output.",
    ),
    EvidenceItem(
        "stage2_shape_prior",
        "Stage2 v4 checkpoint selection Markdown",
        "outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md",
        "Slide/doc-ready checkpoint selection table.",
    ),
    EvidenceItem(
        "stage2_shape_prior",
        "Stage2 v4 selected epoch50 checkpoint",
        "outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_epoch50_probe.pt",
        "Selector-chosen best available v4 checkpoint.",
    ),
    EvidenceItem(
        "stage2_shape_prior",
        "Stage2 v4 final checkpoint",
        "outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt",
        "Full 300-epoch v4 checkpoint kept as late-training degradation evidence.",
    ),
    EvidenceItem(
        "reports",
        "Current mentor/PPT technical report",
        "docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CurrentMethod-Full.zh.md",
        "Primary Chinese technical report for mentor slides.",
    ),
    EvidenceItem(
        "reports",
        "Paper-loop closure audit and slide plan",
        "docs/car_model/6-24-SPCarNet-PaperLoop-Closure-Audit-and-SlidePlan.zh.md",
        "Current honest closure audit.",
    ),
    EvidenceItem(
        "reports",
        "Stage2 v4 normal-band log",
        "docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md",
        "Method-change and eval log for object-prior Stage2 v4.",
    ),
    EvidenceItem(
        "reports",
        "Stage2 implementation report",
        "docs/car_model/spcarnet_stage2_shape_field_implementation_report.md",
        "Stage2 implementation and gate status.",
    ),
    EvidenceItem(
        "representation_diagnostics",
        "v82 capacity-prerank hard-triad summary",
        "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_triad_20260624/summary.md",
        "Hard-triad validation showing raw v82/v82b is not promoted.",
    ),
    EvidenceItem(
        "representation_diagnostics",
        "v83 patchmix hybrid counter summary",
        "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/summary.md",
        "Completed mixed counter diagnostic: PSNR/LPIPS improve, SSIM regresses.",
    ),
    EvidenceItem(
        "representation_diagnostics",
        "v83 patchmix hybrid log",
        "docs/car_model/6-24-v83-PatchMixFaceAlphaLocalPatch-Hybrid-Log.md",
        "Documented v83 method, command evidence, metrics, and non-promotion verdict.",
    ),
    EvidenceItem(
        "representation_diagnostics",
        "v84 strict selector full9 summary",
        "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md",
        "Report-only selector over v82b counter and v64 fallback.",
    ),
    EvidenceItem(
        "reports",
        "Claim boundary and paper gap",
        "docs/car_model/6-24-SPCarNet-Claim-Boundary-And-Paper-Gap.zh.md",
        "Current safe claim boundary for mentor/PPT and paper planning.",
    ),
    EvidenceItem(
        "qualitative",
        "Phase-J where-it-helps panel",
        "assets/spcarnet_phasej_where_it_helps_showcase_20260622.png",
        "Preferred local-crop/error-map qualitative panel.",
        required=False,
    ),
    EvidenceItem(
        "qualitative",
        "M360 outdoor detail panel",
        "assets/spcarnet_m360_outdoor_detail_showcase.png",
        "Outdoor qualitative support panel.",
        required=False,
    ),
    EvidenceItem(
        "qualitative",
        "M360 full9 fair gallery",
        "assets/spcarnet_m360_full9_qualitative_gallery.png",
        "Full-frame fair comparison gallery.",
        required=False,
    ),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _item_record(item: EvidenceItem) -> dict[str, Any]:
    path = REPO_ROOT / item.path
    exists = path.is_file()
    stat = path.stat() if exists else None
    return {
        "category": item.category,
        "label": item.label,
        "path": item.path,
        "note": item.note,
        "required": item.required,
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "mtime_utc": (
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat() if stat else None
        ),
        "sha256": _sha256(path) if exists else None,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SPCarNet Current Evidence Manifest",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        f"Repo root: `{report['repo_root']}`",
        f"Status: `{report['status']}`",
        "",
        "This manifest verifies file existence, sizes, and hashes only. It does not claim paper-loop completion or metric superiority.",
        "",
        "## Summary",
        "",
        f"- items: `{report['summary']['n_items']}`",
        f"- existing: `{report['summary']['n_existing']}`",
        f"- missing required: `{report['summary']['n_missing_required']}`",
        f"- missing optional: `{report['summary']['n_missing_optional']}`",
        "",
        "## Items",
        "",
        "| category | label | exists | size | sha256 | path |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in report["items"]:
        sha = item["sha256"][:12] + "..." if item["sha256"] else "NA"
        lines.append(
            f"| {item['category']} | {item['label']} | `{item['exists']}` | "
            f"{item['size_bytes'] if item['size_bytes'] is not None else 'NA'} | "
            f"`{sha}` | `{item['path']}` |"
        )
    missing = [item for item in report["items"] if item["required"] and not item["exists"]]
    if missing:
        lines.extend(["", "## Missing Required", ""])
        for item in missing:
            lines.append(f"- `{item['path']}` ({item['label']})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report() -> dict[str, Any]:
    items = [_item_record(item) for item in DEFAULT_ITEMS]
    missing_required = [item for item in items if item["required"] and not item["exists"]]
    missing_optional = [item for item in items if not item["required"] and not item["exists"]]
    return {
        "schema": "spcarnet_current_evidence_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "status": "ALL_REQUIRED_FILES_PRESENT" if not missing_required else "MISSING_REQUIRED",
        "summary": {
            "n_items": len(items),
            "n_existing": sum(1 for item in items if item["exists"]),
            "n_missing_required": len(missing_required),
            "n_missing_optional": len(missing_optional),
        },
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_md", required=True, type=Path)
    args = parser.parse_args()

    report = build_report()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(report, args.output_md)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    if report["status"] != "ALL_REQUIRED_FILES_PRESENT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
