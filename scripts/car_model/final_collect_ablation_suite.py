#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/carnet/meshsplatopt/final_ablation_suite"
DOC = ROOT / "docs/car_model/final_stageF11_ablation_suite_report.md"


ROWS = [
    {
        "group": "compact_recovery",
        "row": "clean_long",
        "scene": "parking_phone_tiny",
        "status": "baseline",
        "evidence": "docs/car_model/parking_clean_to_compact_repair_report.md",
        "finding": "strongest clean-long reference for parking compact-recovery",
    },
    {
        "group": "compact_recovery",
        "row": "compaction_only",
        "scene": "parking_phone_tiny",
        "status": "diagnostic",
        "evidence": "docs/car_model/parking_clean_to_compact_repair_report.md",
        "finding": "70/80/90 percent prune-only checkpoints define topology endpoints but are not headline rows without recovery",
    },
    {
        "group": "compact_recovery",
        "row": "compaction_plus_strict_recovery",
        "scene": "parking_phone_tiny",
        "status": "PASS_AREA_CONTROL_SUPERSEDED_BY_CSEF",
        "evidence": "docs/car_model/final_stageF7_parking_pareto_report.md",
        "finding": "R53.01 area70 dominates clean 22k while removing about 70 percent of triangles, but F7 CSEF70 slightly improves PSNR, LPIPS, AbsRel, Depth MAE, and normal at the same triangle count",
    },
    {
        "group": "compact_recovery",
        "row": "csef70_strict_recovery",
        "scene": "parking_phone_tiny",
        "status": "PASS_HEADLINE",
        "evidence": "docs/car_model/final_stageF7_parking_pareto_report.md",
        "finding": "CSEF70 is the strongest parking same-topology row: it beats clean 22k on render and sparse geometry and slightly supersedes R53 area70",
    },
    {
        "group": "compact_recovery",
        "row": "extended_fixed_topology_recovery",
        "scene": "parking_phone_tiny",
        "status": "FAIL",
        "evidence": "docs/car_model/parking_clean_to_compact_repair_report.md",
        "finding": "R56/R50-style continuation does not improve the accepted 26k compact row",
    },
    {
        "group": "compact_recovery",
        "row": "cross_scene_csef50",
        "scene": "bonsai",
        "status": "PASS_SUPERSEDED_BY_QEM_SPARSE",
        "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
        "finding": "50 percent CSEF compact-recovery beats fair clean-long, but F22/F28 show QEM50 variants are stronger on render and sparse-depth QEM is stronger on most geometry/perceptual metrics",
    },
    {
        "group": "posthoc_simplification",
        "row": "open3d_qem70_target",
        "scene": "parking_phone_tiny",
        "status": "FAIL_UNMATCHED_COMPRESSION",
        "evidence": "docs/car_model/final_stageF25_parking_posthoc_qem_baseline_report.md",
        "finding": "Open3D QEM was requested to match the 2,564,473-triangle parking target but stopped at 8,125,970 triangles; no recovery was launched because the compression level is not comparable",
    },
    {
        "group": "compact_recovery",
        "row": "area_smallest_50",
        "scene": "bonsai",
        "status": "PASS_GEOMETRY_PARETO",
        "evidence": "docs/car_model/final_stageF26_bonsai_selector_ablation_report.md",
        "finding": "area50 is slightly behind QEM variants on render, but is a strong structured selector control and remains best on AbsRel/Depth MAE among bonsai same-count rows",
    },
    {
        "group": "compact_recovery",
        "row": "random_same_count_50",
        "scene": "bonsai",
        "status": "FAIL_CONTROL_SUPPORTS_STRUCTURED_SELECTION",
        "evidence": "docs/car_model/final_stageF26_bonsai_selector_ablation_report.md",
        "finding": "random50 fails clean-long on render and AbsRel and loses badly to structured selectors at the same triangle count",
    },
    {
        "group": "posthoc_simplification",
        "row": "open3d_qem50_strict_recovery",
        "scene": "bonsai",
        "status": "PASS_STRONG_RENDER_ROW_SUPERSEDED_BY_SPARSE_DEPTH_PARETO",
        "evidence": "docs/car_model/final_stageF22_bonsai_posthoc_qem_baseline_report.md",
        "finding": "Open3D QEM50 plus strict topology-frozen recovery is the strongest pure-render bonsai row, but F28 sparse-depth QEM improves LPIPS and sparse geometry at negligible PSNR/SSIM cost",
    },
    {
        "group": "compact_recovery",
        "row": "no_freeze_qem50",
        "scene": "bonsai",
        "status": "FAIL_CONTROL_SUPPORTS_FREEZE",
        "evidence": "docs/car_model/final_stageF27_bonsai_qem_no_freeze_control_report.md",
        "finding": "removing strict topology freeze collapses the QEM50 compact row from 44,230 to 17,962 triangles and sharply worsens independent render/geometry metrics",
    },
    {
        "group": "sparse_depth",
        "row": "qem50_sparse_depth_strict_recovery",
        "scene": "bonsai",
        "status": "PASS_GEOMETRY_PERCEPTUAL_PARETO",
        "evidence": "docs/car_model/final_stageF28_bonsai_qem_sparse_depth_report.md",
        "finding": "explicit sparse COLMAP depth loss improves LPIPS, AbsRel, Depth MAE, and normal relative to QEM50 at identical topology with negligible PSNR/SSIM cost",
    },
    {
        "group": "compact_recovery",
        "row": "cross_scene_csef70",
        "scene": "bonsai",
        "status": "FAIL",
        "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
        "finding": "70 percent CSEF compact-recovery is too aggressive and fails the SSIM gate",
    },
    {
        "group": "compact_recovery",
        "row": "cross_scene_csef50",
        "scene": "courtyard",
        "status": "PASS",
        "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
        "finding": "50 percent CSEF compact-recovery improves render and sparse geometry while halving topology",
    },
    {
        "group": "compact_recovery",
        "row": "area_smallest_50",
        "scene": "courtyard",
        "status": "PASS_TIE_RENDER_GEOMETRY_SLIGHTLY_WORSE",
        "evidence": "docs/car_model/final_stageF17_courtyard_selector_ablation_report.md",
        "finding": "area50 nearly ties CSEF50 on render but has slightly weaker sparse geometry; CSEF50 remains geometry-balanced",
    },
    {
        "group": "posthoc_simplification",
        "row": "open3d_qem50_strict_recovery",
        "scene": "courtyard",
        "status": "MIXED_STRONG_CONTROL_CSEF_REMAINS_MAIN",
        "evidence": "docs/car_model/final_stageF23_courtyard_posthoc_qem_baseline_report.md",
        "finding": "Open3D QEM50 improves SSIM, LPIPS, and normal relative to CSEF50, but CSEF50 remains stronger on PSNR, AbsRel, and Depth MAE",
    },
    {
        "group": "sparse_depth",
        "row": "csef50_sparse_depth_strict_recovery",
        "scene": "courtyard",
        "status": "MIXED_NORMAL_ABSREL_PASS_CSEF_REMAINS_MAIN",
        "evidence": "docs/car_model/final_stageF30_F31_courtyard_sparse_depth_controls_report.md",
        "finding": "CSEF50 plus sparse-depth fixes the courtyard normal regression and improves AbsRel, but gives back small PSNR, LPIPS, and Depth MAE margins",
    },
    {
        "group": "sparse_depth",
        "row": "qem50_sparse_depth_strict_recovery",
        "scene": "courtyard",
        "status": "MIXED_QEM_CONTROL_CSEF_REMAINS_MAIN",
        "evidence": "docs/car_model/final_stageF30_F31_courtyard_sparse_depth_controls_report.md",
        "finding": "QEM50 plus lighter sparse-depth improves QEM50 on PSNR, SSIM, AbsRel, and Depth MAE, but remains weaker than CSEF50 on PSNR and sparse depth",
    },
    {
        "group": "compact_recovery",
        "row": "random_same_count_50",
        "scene": "courtyard",
        "status": "FAIL_CONTROL_SUPPORTS_STRUCTURED_SELECTION",
        "evidence": "docs/car_model/final_stageF17_courtyard_selector_ablation_report.md",
        "finding": "random50 fails clean-long and is far worse than CSEF50/area50 at the same triangle count",
    },
    {
        "group": "compact_recovery",
        "row": "cross_scene_csef50",
        "scene": "room",
        "status": "PASS",
        "evidence": "docs/car_model/final_stageF9_third_scene_room_and_qualitative_report.md",
        "finding": "50 percent CSEF compact-recovery improves render metrics, but F19 shows area50 is stronger on room",
    },
    {
        "group": "compact_recovery",
        "row": "area_smallest_50",
        "scene": "room",
        "status": "PASS_SELECTOR_BEST_SUPERSEDED_BY_QEM",
        "evidence": "docs/car_model/final_stageF19_room_selector_ablation_report.md",
        "finding": "area50 beats clean-long, CSEF50, and random50 on all tracked independent metrics, but F20 QEM50 is stronger on render and depth",
    },
    {
        "group": "compact_recovery",
        "row": "no_freeze_qem50",
        "scene": "room",
        "status": "FAIL_CONTROL_SUPPORTS_FREEZE",
        "evidence": "docs/car_model/final_stageF24_room_qem_no_freeze_control_report.md",
        "finding": "removing strict topology freeze collapses the QEM50 compact row from 42,253 to 20,742 triangles and sharply worsens independent render/geometry metrics",
    },
    {
        "group": "posthoc_simplification",
        "row": "open3d_qem50_strict_recovery",
        "scene": "room",
        "status": "PASS_STRONG_BASELINE_OR_OPERATOR",
        "evidence": "docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md",
        "finding": "Open3D QEM50 plus strict topology-frozen recovery is the strongest room row on render, AbsRel, and Depth MAE; this baseline must be reported honestly",
    },
    {
        "group": "sparse_depth",
        "row": "qem50_sparse_depth_strict_recovery",
        "scene": "room",
        "status": "MIXED_GEOMETRY_PERCEPTUAL_PASS_QEM_REMAINS_MAIN",
        "evidence": "docs/car_model/final_stageF29_room_qem_sparse_depth_report.md",
        "finding": "QEM50 plus sparse-depth improves SSIM, LPIPS, AbsRel, Depth MAE, and normal at identical room topology, but gives back 0.001 dB PSNR",
    },
    {
        "group": "compact_recovery",
        "row": "random_same_count_50",
        "scene": "room",
        "status": "FAIL_CONTROL_SUPPORTS_STRUCTURED_SELECTION",
        "evidence": "docs/car_model/final_stageF19_room_selector_ablation_report.md",
        "finding": "random50 fails clean-long and is far worse than area50 at the same triangle count",
    },
    {
        "group": "compact_recovery",
        "row": "cross_scene_csef50",
        "scene": "counter",
        "status": "BORDERLINE",
        "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md",
        "finding": "50 percent CSEF is near the gate but misses SSIM by 0.003827",
    },
    {
        "group": "compact_recovery",
        "row": "cross_scene_csef40",
        "scene": "counter",
        "status": "PASS",
        "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md",
        "finding": "40 percent CSEF improves PSNR, SSIM, LPIPS, and normal versus clean-long, but F16 shows area40 is stronger on counter",
    },
    {
        "group": "compact_recovery",
        "row": "area_smallest_40",
        "scene": "counter",
        "status": "PASS_SELECTOR_BEST_SUPERSEDED_BY_QEM",
        "evidence": "docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md",
        "finding": "area40 beats clean-long and CSEF40 on independent metrics, but F21 QEM40 is stronger on render and depth",
    },
    {
        "group": "posthoc_simplification",
        "row": "open3d_qem40_strict_recovery",
        "scene": "counter",
        "status": "PASS_STRONG_BASELINE_SUPERSEDED_BY_SPARSE_DEPTH_PARETO",
        "evidence": "docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md",
        "finding": "Open3D QEM40 plus strict topology-frozen recovery is the strongest counter PSNR/Depth row, but F32 sparse-depth improves SSIM, LPIPS, AbsRel, and normal at negligible PSNR/Depth cost",
    },
    {
        "group": "sparse_depth",
        "row": "qem40_sparse_depth_strict_recovery",
        "scene": "counter",
        "status": "PASS_GEOMETRY_PERCEPTUAL_PARETO",
        "evidence": "docs/car_model/final_stageF32_counter_qem_sparse_depth_report.md",
        "finding": "QEM40 plus sparse-depth improves SSIM, LPIPS, AbsRel, and normal relative to QEM40 at identical topology, while remaining an all-metric clean-long win",
    },
    {
        "group": "compact_recovery",
        "row": "no_freeze_area40",
        "scene": "counter",
        "status": "FAIL_CONTROL_SUPPORTS_FREEZE",
        "evidence": "docs/car_model/final_stageF18_counter_no_freeze_control_report.md",
        "finding": "removing strict topology freeze collapses the area40 compact row from 50,300 to 18,693 triangles and sharply worsens independent render/geometry metrics",
    },
    {
        "group": "compact_recovery",
        "row": "random_same_count_40",
        "scene": "counter",
        "status": "FAIL_CONTROL_SUPPORTS_CSEF",
        "evidence": "docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md",
        "finding": "random 40 percent compaction at the same triangle count loses badly to CSEF40 and clean-long on independent render/geometry metrics",
    },
    {
        "group": "sparse_depth",
        "row": "sparse_depth_recovery",
        "scene": "parking/courtyard/bonsai",
        "status": "PASS",
        "evidence": "docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md",
        "finding": "earlier sparse-depth recovery branch is useful but separate from the final compact-recovery main rows, which use independent COLMAP sparse geometry evaluation",
    },
    {
        "group": "repair_operations",
        "row": "snap_only",
        "scene": "real checkpoints",
        "status": "SAFETY_PASS_QUALITY_UNPROVEN",
        "evidence": "docs/car_model/meshsplatopt_stageR17_02_checkpoint_local_snap_gate_report.md",
        "finding": "local snap can pass safety gates but is not a headline quality-improving method",
    },
    {
        "group": "repair_operations",
        "row": "fill_only",
        "scene": "parking_phone_tiny",
        "status": "FAIL",
        "evidence": "docs/car_model/meshsplatopt_stageR24_R26_fill_init_and_grid_report.md",
        "finding": "grid fill plus sparse recovery does not beat matched sparse-depth controls at full budget",
    },
    {
        "group": "counterfactual_certification",
        "row": "rollback_and_gate",
        "scene": "implementation",
        "status": "MECHANISM_PASS_LOAD_BEARING_PARTIAL",
        "evidence": "docs/car_model/meshprior_stage28_adaptive_schedule_smoke_report.md",
        "finding": "rollback and gate infrastructure is implemented; full no-gate ablation remains missing",
    },
]

MISSING = [
    "replicate no-freeze compact-recovery control beyond completed bonsai, counter, and room rows",
    "final CSEF selector versus area-only selector on every public scene beyond the completed bonsai, courtyard, room, and counter controls",
    "selector ablation on any additional scenes added to the final benchmark",
    "a matched 70 percent parking posthoc simplification baseline that can actually reach the R53/F7 triangle target; Open3D QEM failed this target",
    "replicate explicit sparse-depth compact-recovery beyond the completed bonsai, room, courtyard, and counter rows if the manuscript wants a universal sparse-depth-guided recovery claim across every final scene",
    "full no-render-gate/no-geometry-gate/no-rollback counterfactual ablations",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "existing_ablation_registry.csv", ROWS)
    (OUT / "existing_ablation_registry.json").write_text(json.dumps({"rows": ROWS, "missing": MISSING}, indent=2))

    lines = [
        "# Final Stage F11 - Ablation Suite Report",
        "",
        "Decision: `FINAL_F11_EXISTING_EVIDENCE_SOFT_PASS_MISSING_FULL_ABLATIONS`.",
        "",
        "This report is an auditable registry of completed ablation evidence. It does not claim that the full F11 training matrix has been run.",
        "",
        "## Existing Evidence",
        "",
        "| group | row | scene | status | finding | evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in ROWS:
        lines.append(
            f"| {row['group']} | {row['row']} | {row['scene']} | {row['status']} | {row['finding']} | `{row['evidence']}` |"
        )
    lines += [
        "",
        "## Missing Rows Required For A Strict NeurIPS Ablation Claim",
        "",
    ]
    lines += [f"- {item}" for item in MISSING]
    lines += [
        "",
        "## Gate",
        "",
        "Soft pass only. The current evidence identifies load-bearing components: compact-recovery, strict topology freezing, structured selection versus random pruning, and strong bonsai/courtyard/room/counter Open3D-QEM recovery baselines/operators. F28/F29/F30/F31/F32 now replicate explicit sparse-depth compact recovery on bonsai, room, courtyard, and counter, supporting a geometry/perceptual regularizer claim but not a universal PSNR-improvement claim. Snap/fill are explicitly not load-bearing headline rows. A strict F11 PASS still requires the missing matched ablations above.",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(OUT / "existing_ablation_registry.csv")
    print(OUT / "existing_ablation_registry.json")
    print(DOC)


if __name__ == "__main__":
    main()
