# Final Stage F11 - Ablation Suite Report

Decision: `FINAL_F11_EXISTING_EVIDENCE_SOFT_PASS_MISSING_FULL_ABLATIONS`.

This report is an auditable registry of completed ablation evidence. It does not claim that the full F11 training matrix has been run.

## Existing Evidence

| group | row | scene | status | finding | evidence |
| --- | --- | --- | --- | --- | --- |
| compact_recovery | clean_long | parking_phone_tiny | baseline | strongest clean-long reference for parking compact-recovery | `docs/car_model/parking_clean_to_compact_repair_report.md` |
| compact_recovery | compaction_only | parking_phone_tiny | diagnostic | 70/80/90 percent prune-only checkpoints define topology endpoints but are not headline rows without recovery | `docs/car_model/parking_clean_to_compact_repair_report.md` |
| compact_recovery | compaction_plus_strict_recovery | parking_phone_tiny | PASS | R53.01 dominates clean 22k while removing about 70 percent of triangles | `docs/car_model/parking_clean_to_compact_repair_report.md` |
| compact_recovery | extended_fixed_topology_recovery | parking_phone_tiny | FAIL | R56/R50-style continuation does not improve the accepted 26k compact row | `docs/car_model/parking_clean_to_compact_repair_report.md` |
| compact_recovery | cross_scene_csef50 | bonsai | PASS_SUPERSEDED_BY_QEM | 50 percent CSEF compact-recovery beats fair clean-long, but F22 QEM50 is stronger on render, AbsRel, and normal | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| posthoc_simplification | open3d_qem50_strict_recovery | bonsai | PASS_STRONG_BASELINE_OR_OPERATOR | Open3D QEM50 plus strict topology-frozen recovery becomes the strongest bonsai row on render, AbsRel, and normal, with Depth MAE slightly weaker than CSEF50 | `docs/car_model/final_stageF22_bonsai_posthoc_qem_baseline_report.md` |
| compact_recovery | cross_scene_csef70 | bonsai | FAIL | 70 percent CSEF compact-recovery is too aggressive and fails the SSIM gate | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| compact_recovery | cross_scene_csef50 | courtyard | PASS | 50 percent CSEF compact-recovery improves render and sparse geometry while halving topology | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| compact_recovery | area_smallest_50 | courtyard | PASS_TIE_RENDER_GEOMETRY_SLIGHTLY_WORSE | area50 nearly ties CSEF50 on render but has slightly weaker sparse geometry; CSEF50 remains geometry-balanced | `docs/car_model/final_stageF17_courtyard_selector_ablation_report.md` |
| compact_recovery | random_same_count_50 | courtyard | FAIL_CONTROL_SUPPORTS_STRUCTURED_SELECTION | random50 fails clean-long and is far worse than CSEF50/area50 at the same triangle count | `docs/car_model/final_stageF17_courtyard_selector_ablation_report.md` |
| compact_recovery | cross_scene_csef50 | room | PASS | 50 percent CSEF compact-recovery improves render metrics, but F19 shows area50 is stronger on room | `docs/car_model/final_stageF9_third_scene_room_and_qualitative_report.md` |
| compact_recovery | area_smallest_50 | room | PASS_SELECTOR_BEST_SUPERSEDED_BY_QEM | area50 beats clean-long, CSEF50, and random50 on all tracked independent metrics, but F20 QEM50 is stronger on render and depth | `docs/car_model/final_stageF19_room_selector_ablation_report.md` |
| posthoc_simplification | open3d_qem50_strict_recovery | room | PASS_STRONG_BASELINE_OR_OPERATOR | Open3D QEM50 plus strict topology-frozen recovery is the strongest room row on render, AbsRel, and Depth MAE; this baseline must be reported honestly | `docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md` |
| compact_recovery | random_same_count_50 | room | FAIL_CONTROL_SUPPORTS_STRUCTURED_SELECTION | random50 fails clean-long and is far worse than area50 at the same triangle count | `docs/car_model/final_stageF19_room_selector_ablation_report.md` |
| compact_recovery | cross_scene_csef50 | counter | BORDERLINE | 50 percent CSEF is near the gate but misses SSIM by 0.003827 | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| compact_recovery | cross_scene_csef40 | counter | PASS | 40 percent CSEF improves PSNR, SSIM, LPIPS, and normal versus clean-long, but F16 shows area40 is stronger on counter | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| compact_recovery | area_smallest_40 | counter | PASS_SELECTOR_BEST_SUPERSEDED_BY_QEM | area40 beats clean-long and CSEF40 on independent metrics, but F21 QEM40 is stronger on render and depth | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| posthoc_simplification | open3d_qem40_strict_recovery | counter | PASS_STRONG_BASELINE_OR_OPERATOR | Open3D QEM40 plus strict topology-frozen recovery becomes the strongest counter row on PSNR, SSIM, LPIPS, AbsRel, and Depth MAE | `docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md` |
| compact_recovery | no_freeze_area40 | counter | FAIL_CONTROL_SUPPORTS_FREEZE | removing strict topology freeze collapses the area40 compact row from 50,300 to 18,693 triangles and sharply worsens independent render/geometry metrics | `docs/car_model/final_stageF18_counter_no_freeze_control_report.md` |
| compact_recovery | random_same_count_40 | counter | FAIL_CONTROL_SUPPORTS_CSEF | random 40 percent compaction at the same triangle count loses badly to CSEF40 and clean-long on independent render/geometry metrics | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| sparse_depth | sparse_depth_recovery | parking/courtyard/bonsai | PASS | earlier sparse-depth recovery branch is useful but separate from the final compact-recovery main rows, which use independent COLMAP sparse geometry evaluation | `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md` |
| repair_operations | snap_only | real checkpoints | SAFETY_PASS_QUALITY_UNPROVEN | local snap can pass safety gates but is not a headline quality-improving method | `docs/car_model/meshsplatopt_stageR17_02_checkpoint_local_snap_gate_report.md` |
| repair_operations | fill_only | parking_phone_tiny | FAIL | grid fill plus sparse recovery does not beat matched sparse-depth controls at full budget | `docs/car_model/meshsplatopt_stageR24_R26_fill_init_and_grid_report.md` |
| counterfactual_certification | rollback_and_gate | implementation | MECHANISM_PASS_LOAD_BEARING_PARTIAL | rollback and gate infrastructure is implemented; full no-gate ablation remains missing | `docs/car_model/meshprior_stage28_adaptive_schedule_smoke_report.md` |

## Missing Rows Required For A Strict NeurIPS Ablation Claim

- replicate no-freeze compact-recovery control beyond counter
- final CSEF selector versus area-only selector on every public scene
- selector ablation on remaining scenes beyond completed counter, courtyard, and room controls
- replicate posthoc QEM/decimation baseline beyond completed bonsai, room, and counter rows
- separate final compact-recovery rows that explicitly enable sparse-depth loss, if the manuscript wants to claim sparse-depth-guided recovery
- full no-render-gate/no-geometry-gate/no-rollback counterfactual ablations

## Gate

Soft pass only. The current evidence identifies load-bearing components: compact-recovery, strict topology freezing, structured selection versus random pruning, and strong bonsai/room/counter Open3D-QEM recovery baselines/operators. The earlier sparse-depth branch is useful but should not be conflated with the final compact-recovery main rows unless new rows explicitly enable that loss. Snap/fill are explicitly not load-bearing headline rows. A strict F11 PASS still requires the missing matched ablations above.
