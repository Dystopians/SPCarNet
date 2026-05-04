# Final Stage F12 - Multi-Scene Package Report

Decision: `FINAL_F12_MULTISCENE_PACKAGE_PASS_WITH_ABLATION_GAPS`.

Scenes with compact-recovery pass decisions: `5/5`.
Scenes with PSNR+SSIM improvement and LPIPS non-regression tolerance: `5/5`.

## Main Quantitative Table

| scene | clean triangles | ours triangles | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| parking_phone_tiny | 8,548,242 | 2,564,473 | 70.0% | 0.226000 | 0.013000 | -0.009000 | -0.002000 | -0.014000 | -0.847000 | PASS |
| bonsai | 88,460 | 44,230 | 50.0% | 0.013149 | 0.001910 | 0.000257 | -0.009069 | -0.078595 | -1.864381 | PASS |
| courtyard | 1,677,484 | 838,742 | 50.0% | 0.452301 | 0.041625 | -0.024231 | -0.032415 | -0.220612 | 0.008508 | PASS |
| room | 84,506 | 42,253 | 50.0% | 0.128784 | 0.014090 | -0.010638 | 0.018745 | 0.122800 | -0.799860 | PASS |
| counter | 83,834 | 50,300 | 40.0% | 0.178148 | 0.024090 | -0.020945 | -0.004245 | -0.012059 | -0.571153 | PASS_PARETO |

## Per-Scene Evidence

| scene | clean row | best row | W&B | evidence |
| --- | --- | --- | --- | --- |
| parking_phone_tiny | clean-long 22k | R53.01 area70 strict recovery 26k | `q15qg2b8` | `docs/car_model/parking_clean_to_compact_repair_report.md` |
| bonsai | clean-long 22k | CSEF50 strict recovery 26k | `irdsa4c8` | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| courtyard | clean-long 22k | CSEF50 strict recovery 26k | `jz93wrbc` | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| room | clean-long 22k | CSEF50 strict recovery 26k | `pb1tg4p2` | `docs/car_model/final_stageF9_third_scene_room_and_qualitative_report.md` |
| counter | clean-long 22k | area40 strict recovery 26k | `85lmm0lr` | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |

## Negative Result Table

| scene | row | finding | evidence |
| --- | --- | --- | --- |
| bonsai | CSEF70 | 70 percent compaction fails SSIM gate | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| counter | CSEF50 | 50 percent compaction is a boundary case and misses SSIM by 0.003827 | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| counter | CSEF50 30k | extended recovery worsens SSIM and LPIPS | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| counter | random40 | same-count random compaction loses badly to area40 and CSEF40 | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| counter | CSEF40 | passes clean-long but is not the strongest selector on counter; area40 is better | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| counter | area40 no-freeze | omitting strict topology freeze collapses topology to 18,693 triangles and loses badly to frozen area40 | `docs/car_model/final_stageF18_counter_no_freeze_control_report.md` |
| parking_phone_tiny | grid fill full-budget | fill branch does not beat matched sparse-depth control | `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md` |

## Gate

PASS with ablation gaps. At least two scenes show meaningful compact-recovery benefit over fair clean-long baselines; five scenes now have auditable long-baseline comparisons. The remaining NeurIPS risk is not scene count, but missing matched ablations against area-only, random same-count compaction beyond the completed controls, replicated no-freeze controls, explicit sparse-depth-loss variants if claimed, and posthoc simplification controls.
