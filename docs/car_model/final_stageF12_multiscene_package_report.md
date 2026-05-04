# Final Stage F12 - Multi-Scene Package Report

Decision: `FINAL_F12_MULTISCENE_PACKAGE_PASS_WITH_ABLATION_GAPS`.

Scenes with compact-recovery pass decisions: `5/5`.
Scenes with PSNR+SSIM improvement and LPIPS non-regression tolerance: `5/5`.

## Main Quantitative Table

| scene | clean triangles | ours triangles | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| parking_phone_tiny | 8,548,242 | 2,564,473 | 70.0% | 0.232330 | 0.012730 | -0.008741 | -0.002929 | -0.013985 | -1.072292 | PASS_PARETO |
| bonsai | 88,460 | 44,230 | 50.0% | 0.137266 | 0.020400 | -0.016500 | -0.012551 | -0.036627 | -2.932622 | PASS |
| courtyard | 1,677,484 | 838,742 | 50.0% | 0.452301 | 0.041625 | -0.024231 | -0.032415 | -0.220612 | 0.008508 | PASS |
| room | 84,506 | 42,253 | 50.0% | 0.802811 | 0.080218 | -0.062114 | -0.025153 | -0.135009 | -0.541874 | PASS |
| counter | 83,834 | 50,300 | 40.0% | 0.272587 | 0.034768 | -0.031847 | -0.008982 | -0.030858 | -0.701820 | PASS_PARETO |

## Per-Scene Evidence

| scene | clean row | best row | W&B | evidence |
| --- | --- | --- | --- | --- |
| parking_phone_tiny | clean-long 22k | CSEF70 + sparse-depth strict recovery 26k | `x6rmhhlp` | `docs/car_model/final_stageF33_parking_csef_sparse_depth_report.md` |
| bonsai | clean-long 22k | Open3D QEM50 + sparse-depth strict recovery 26k | `07k1ii1d` | `docs/car_model/final_stageF28_bonsai_qem_sparse_depth_report.md` |
| courtyard | clean-long 22k | CSEF50 strict recovery 26k | `jz93wrbc` | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| room | clean-long 22k | Open3D QEM50 strict recovery 26k | `9wri3owt` | `docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md` |
| counter | clean-long 22k | Open3D QEM40 + sparse-depth strict recovery 26k | `x9b89ssf` | `docs/car_model/final_stageF32_counter_qem_sparse_depth_report.md` |

## Negative Result Table

| scene | row | finding | evidence |
| --- | --- | --- | --- |
| bonsai | CSEF70 | 70 percent compaction fails SSIM gate | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| bonsai | CSEF50 | passes clean-long but is superseded by QEM/sparse-depth on render and by area50 or sparse-depth QEM on geometry/perceptual metrics | `docs/car_model/final_stageF28_bonsai_qem_sparse_depth_report.md` |
| bonsai | QEM50 frozen | strong render row, but QEM50 + sparse-depth is better on LPIPS, AbsRel, Depth MAE, and normal at the same topology with negligible PSNR/SSIM cost | `docs/car_model/final_stageF28_bonsai_qem_sparse_depth_report.md` |
| bonsai | QEM50 no-freeze | omitting strict topology freeze collapses topology to 17,962 triangles and loses badly to frozen QEM50 | `docs/car_model/final_stageF27_bonsai_qem_no_freeze_control_report.md` |
| bonsai | area50 | strong structured Pareto row: slightly behind QEM50 on PSNR/SSIM, but better on LPIPS, AbsRel, Depth MAE, and normal | `docs/car_model/final_stageF26_bonsai_selector_ablation_report.md` |
| bonsai | random50 | same-count random compaction loses badly to clean-long, area50, CSEF50, and QEM50 on render and AbsRel | `docs/car_model/final_stageF26_bonsai_selector_ablation_report.md` |
| courtyard | Open3D QEM50 | improves SSIM, LPIPS, and normal but is weaker than CSEF50 on PSNR, AbsRel, and Depth MAE | `docs/car_model/final_stageF23_courtyard_posthoc_qem_baseline_report.md` |
| courtyard | CSEF50 + sparse-depth | fixes the courtyard normal regression and slightly improves AbsRel, but gives back small PSNR, LPIPS, and Depth MAE margins, so CSEF50 remains the balanced headline | `docs/car_model/final_stageF30_F31_courtyard_sparse_depth_controls_report.md` |
| courtyard | QEM50 + sparse-depth | improves QEM50 on PSNR, SSIM, AbsRel, and Depth MAE, but remains weaker than CSEF50 on PSNR and sparse depth | `docs/car_model/final_stageF30_F31_courtyard_sparse_depth_controls_report.md` |
| courtyard | CSEF50 no-freeze | omitting strict topology freeze makes CSEF50 drift to 1,317,435 triangles and lose badly on render and sparse depth, supporting the frozen recovery contract | `docs/car_model/final_stageF35_courtyard_csef_no_freeze_control_report.md` |
| counter | CSEF50 | 50 percent compaction is a boundary case and misses SSIM by 0.003827 | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| counter | CSEF50 30k | extended recovery worsens SSIM and LPIPS | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| counter | random40 | same-count random compaction loses badly to area40 and CSEF40 | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| counter | CSEF40 | passes clean-long but is not the strongest selector on counter; area40 and QEM40 are better | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| counter | area40 | strong structured selector row but superseded by Open3D QEM40 on render, AbsRel, and Depth MAE | `docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md` |
| counter | QEM40 frozen | strong PSNR/Depth row, but QEM40 + sparse-depth improves SSIM, LPIPS, AbsRel, and normal at the same topology with negligible PSNR/Depth cost | `docs/car_model/final_stageF32_counter_qem_sparse_depth_report.md` |
| counter | area40 no-freeze | omitting strict topology freeze collapses topology to 18,693 triangles and loses badly to frozen area40 | `docs/car_model/final_stageF18_counter_no_freeze_control_report.md` |
| parking_phone_tiny | R53.01 area70 | strong area-only control at the same triangle count; superseded by CSEF70 on PSNR, LPIPS, AbsRel, Depth MAE, and normal with negligible SSIM loss | `docs/car_model/final_stageF7_parking_pareto_report.md` |
| parking_phone_tiny | CSEF70 | strong all-metric clean-long win, but CSEF70 + sparse-depth improves PSNR, LPIPS, AbsRel, and normal at identical topology with negligible SSIM cost and a small Depth MAE tradeoff | `docs/car_model/final_stageF33_parking_csef_sparse_depth_report.md` |
| parking_phone_tiny | F34 CSEF70 sparse-depth 30k continuation | longer fixed-topology sparse-depth continuation slightly improves sparse depth proxies but sharply worsens PSNR, SSIM, LPIPS, and normal, so F33 26k remains the validated parking row | `docs/car_model/final_stageF34_parking_long_continuation_report.md` |
| parking_phone_tiny | CSEF70 no-freeze | omitting strict topology freeze makes CSEF70 drift to 3,533,325 triangles and lose badly to frozen CSEF70/sparse-depth on render and sparse depth | `docs/car_model/final_stageF36_parking_csef_no_freeze_control_report.md` |
| parking_phone_tiny | fast-QEM70 matched | matched QEM reaches the F7/F33 topology within 9 triangles and improves sparse geometry proxies, but collapses PSNR, SSIM, and LPIPS relative to clean-long and F33 | `docs/car_model/final_stageF37_parking_fast_qem_matched_baseline_report.md` |
| parking_phone_tiny | Open3D QEM70 | Open3D QEM did not reach the matched 2,564,473-triangle target on the 8.55M-triangle parking mesh, stopping at 8,125,970 triangles, so it is rejected as an unmatched compression control | `docs/car_model/final_stageF25_parking_posthoc_qem_baseline_report.md` |
| synthetic_edit_suite | no-gate/no-rollback counterfactual | F38 applies identical bad edits with and without MeshSplatOpt gate/rollback; the gate rejects and exactly restores all unsafe edits, while the unsafe path commits an unobserved floater, a 5m free-space snap, and supported-surface deletion | `docs/car_model/final_stageF38_counterfactual_gate_ablation_report.md` |
| parking_phone_tiny | real gate-removed ratio0.04 | F39 same-schedule 500-iteration ablation shows the gated run rolls back a 2579-triangle candidate set while the gate-removed run commits it and is slightly worse on PSNR, SSIM, LPIPS, AbsRel, and Depth MAE | `docs/car_model/final_stageF39_real_gate_removed_ablation_report.md` |
| parking_phone_tiny | real gate-removed ratio0.04 long | F41 same-schedule 2000-iteration ablation again shows gate-on rolling back the no-accept 2579-triangle candidate set while gate-off commits it; final metrics are mixed, so this supports unsafe-edit rejection rather than monotonic final-metric dominance | `docs/car_model/final_stageF41_gate_removed_ratio004_long_report.md` |
| parking_phone_tiny | real gate-removed ratio0.04 7000 | F42 same-schedule 7000-iteration ablation again shows gate-on rolling back the no-accept 2579-triangle candidate set while gate-off commits it; gated wins PSNR, SSIM, and LPIPS, while sparse geometry proxies are still slightly better for no-gate | `docs/car_model/final_stageF42_gate_removed_ratio004_7000_report.md` |
| bonsai | real gate-removed ratio0.02 7000 | F43 is a negative long-budget gate-generalization result: gate-on rolls back all six candidate rounds while gate-off commits all six, and no-gate wins PSNR, SSIM, LPIPS, AbsRel, Depth MAE, and normal with fewer triangles | `docs/car_model/final_stageF43_bonsai_gate_removed_7000_report.md` |
| bonsai | calibrated gate ratio0.02 7000 | F44 repairs the F43 strict-gate weakness: calibrated gate commits three recoverable rounds, rejects three later rounds, beats strict gate by large margins on every tracked metric, and finishes close to no-gate with fewer triangles | `docs/car_model/final_stageF44_bonsai_calibrated_gate_7000_report.md` |
| bonsai | CSEF50 + sparse-depth + LPIPS validation-budget repair | all-metric clean-long win with 50 percent topology reduction; this repairs the fixed-CSEF50 LPIPS miss and the CSEF20 Depth MAE miss without switching to QEM; F49 strengthens PSNR, SSIM, AbsRel, and Depth MAE margins over F47 | `docs/car_model/final_stageF47_F48_csef_family_all_metric_repair_report.md` |
| room | QEM50 no-freeze | omitting strict topology freeze collapses topology to 20,742 triangles and loses badly to frozen QEM50 | `docs/car_model/final_stageF24_room_qem_no_freeze_control_report.md` |
| room | QEM50 + sparse-depth | improves SSIM, LPIPS, AbsRel, Depth MAE, and normal at identical topology, but gives back 0.001 dB PSNR, so QEM50 remains the room PSNR headline | `docs/car_model/final_stageF29_room_qem_sparse_depth_report.md` |
| room | CSEF20 + sparse-depth validation-budget repair | all-metric clean-long win with 20 percent topology reduction; this repairs the fixed-CSEF50 room depth weakness without switching to QEM | `docs/car_model/final_stageF46_unified_csef_sparse_depth_report.md` |
| room | random50 | same-count random compaction loses badly to area50 and clean-long | `docs/car_model/final_stageF19_room_selector_ablation_report.md` |
| room | CSEF50 | passes clean-long but is superseded by area50 on all tracked independent metrics | `docs/car_model/final_stageF19_room_selector_ablation_report.md` |
| room | area50 | strong structured selector row but superseded by Open3D QEM50 on render, AbsRel, and Depth MAE | `docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md` |
| counter | CSEF20 + sparse-depth validation-budget repair | all-metric clean-long win with 20 percent topology reduction; this repairs the fixed-CSEF50 counter failure without switching to QEM | `docs/car_model/final_stageF46_unified_csef_sparse_depth_report.md` |
| parking_phone_tiny | grid fill full-budget | fill branch does not beat matched sparse-depth control | `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md` |

## Gate

PASS with ablation gaps. Five scenes show compact-recovery benefit over fair clean-long baselines, and all five have auditable long-baseline comparisons. F34 adds a parking long-continuation rejection showing that simply training the best sparse-depth row from 26k to 30k hurts visual quality, so F33 is the validated stopping point. F36 extends no-freeze failure evidence to all five final-package scenes, including the largest parking scene. F37 closes the matched parking posthoc simplification gap: fast-QEM reaches the target topology and is strong on sparse geometry proxies, but collapses visual render quality. F38 closes the implementation-level no-gate/no-rollback counterfactual with identical-edit evidence. F39 adds a real-scene gate-removed same-schedule ablation where gate-on rolls back an aggressive candidate set and gate-off commits it with worse PSNR, SSIM, LPIPS, AbsRel, and Depth MAE. F41 adds the requested longer ratio0.04 counterpart with the same rollback/commit mechanism but mixed metrics. F42 extends the real gate-removed ablation to 7000 iterations: gate-on again blocks the no-accept candidate commit and wins PSNR, SSIM, and LPIPS, while no-gate remains slightly better on sparse geometry proxies. F43 adds a multi-scene long-budget bonsai stress test and is negative for strict broad gate superiority. F44 repairs most of that weakness with calibrated gate thresholds: it preserves counterfactual gating, accepts recoverable edits, rejects later edits, and finishes close to no-gate with a smaller mesh. F45/F46 close an important fairness hole: fixed CSEF50 is not universal, but the same CSEF selector family plus sparse-depth strict recovery and conservative validation-selected budgets gives all-metric clean-long wins on room and counter and a matched CSEF50 parking win. F47 closes the remaining bonsai CSEF-family gap with a small LPIPS-loss recovery term, yielding an all-metric clean-long win at 50 percent reduction. F49 strengthens that bonsai row by improving PSNR, SSIM, AbsRel, and Depth MAE margins while keeping LPIPS and normal wins. F48 consolidates the CSEF-family package and uses the earlier F30 courtyard sparse-depth row, giving five of five all-metric clean-long wins without relying on QEM rows. The remaining NeurIPS risk is now mainly replicating calibrated gate behavior beyond bonsai and wording the method as validation-selected compact-recovery rather than universal fixed hyperparameters.
