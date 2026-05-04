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
        "status": "PASS",
        "evidence": "docs/car_model/parking_clean_to_compact_repair_report.md",
        "finding": "R53.01 dominates clean 22k while removing about 70 percent of triangles",
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
        "status": "PASS",
        "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
        "finding": "50 percent CSEF compact-recovery beats fair clean-long on PSNR, SSIM, depth, and normal",
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
        "row": "cross_scene_csef50",
        "scene": "room",
        "status": "PASS",
        "evidence": "docs/car_model/final_stageF9_third_scene_room_and_qualitative_report.md",
        "finding": "50 percent CSEF compact-recovery improves render metrics; depth tradeoff stays inside gate",
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
        "finding": "40 percent CSEF is the recommended counter Pareto point and improves PSNR, SSIM, LPIPS, and normal",
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
        "finding": "sparse COLMAP depth is load-bearing for recovery; trusted sampling is scene dependent",
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
    "full no-freeze matched compact-recovery row on the final CSEF selector",
    "final CSEF selector versus area-only selector on every public scene",
    "random same-count compaction control beyond counter, ideally courtyard plus one more public scene",
    "posthoc QEM/decimation baseline with equal recovery budget",
    "full no-sparse-depth compact-recovery row on at least parking plus one public scene",
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
        "Soft pass only. The current evidence identifies load-bearing components: compact-recovery, strict topology freezing, and sparse COLMAP recovery. Snap/fill are explicitly not load-bearing headline rows. A strict F11 PASS still requires the missing matched ablations above.",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(OUT / "existing_ablation_registry.csv")
    print(OUT / "existing_ablation_registry.json")
    print(DOC)


if __name__ == "__main__":
    main()
