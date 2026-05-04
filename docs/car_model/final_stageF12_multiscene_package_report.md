# Final Stage F12 - Multi-Scene Package Report

Decision: `FINAL_F12_MULTISCENE_PACKAGE_PASS_WITH_ABLATION_GAPS`.

Scenes with compact-recovery pass decisions: `5/5`.
Scenes with PSNR+SSIM improvement and LPIPS non-regression tolerance: `5/5`.

## Main Quantitative Table

| scene | clean triangles | ours triangles | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| parking_phone_tiny | 8,548,242 | 2,564,473 | 70.0% | 0.226079 | 0.012764 | -0.008718 | -0.002596 | -0.015184 | -0.903503 | PASS |
| bonsai | 88,460 | 44,230 | 50.0% | 0.138057 | 0.020401 | -0.015981 | -0.011283 | -0.022558 | -2.469017 | PASS |
| courtyard | 1,677,484 | 838,742 | 50.0% | 0.452301 | 0.041625 | -0.024231 | -0.032415 | -0.220612 | 0.008508 | PASS |
| room | 84,506 | 42,253 | 50.0% | 0.802811 | 0.080218 | -0.062114 | -0.025153 | -0.135009 | -0.541874 | PASS |
| counter | 83,834 | 50,300 | 40.0% | 0.273252 | 0.034654 | -0.031194 | -0.008920 | -0.031309 | -0.571028 | PASS_PARETO |

## Per-Scene Evidence

| scene | clean row | best row | W&B | evidence |
| --- | --- | --- | --- | --- |
| parking_phone_tiny | clean-long 22k | CSEF70 strict recovery 26k | `oqpkykcw` | `docs/car_model/final_stageF7_parking_pareto_report.md` |
| bonsai | clean-long 22k | Open3D QEM50 strict recovery 26k | `bsed9ik1` | `docs/car_model/final_stageF22_bonsai_posthoc_qem_baseline_report.md` |
| courtyard | clean-long 22k | CSEF50 strict recovery 26k | `jz93wrbc` | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| room | clean-long 22k | Open3D QEM50 strict recovery 26k | `9wri3owt` | `docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md` |
| counter | clean-long 22k | Open3D QEM40 strict recovery 26k | `kr8565st` | `docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md` |

## Negative Result Table

| scene | row | finding | evidence |
| --- | --- | --- | --- |
| bonsai | CSEF70 | 70 percent compaction fails SSIM gate | `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md` |
| bonsai | CSEF50 | passes clean-long but is superseded by Open3D QEM50 on render, AbsRel, and normal | `docs/car_model/final_stageF22_bonsai_posthoc_qem_baseline_report.md` |
| courtyard | Open3D QEM50 | improves SSIM, LPIPS, and normal but is weaker than CSEF50 on PSNR, AbsRel, and Depth MAE | `docs/car_model/final_stageF23_courtyard_posthoc_qem_baseline_report.md` |
| counter | CSEF50 | 50 percent compaction is a boundary case and misses SSIM by 0.003827 | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| counter | CSEF50 30k | extended recovery worsens SSIM and LPIPS | `docs/car_model/final_stageF10_fourth_scene_counter_report.md` |
| counter | random40 | same-count random compaction loses badly to area40 and CSEF40 | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| counter | CSEF40 | passes clean-long but is not the strongest selector on counter; area40 and QEM40 are better | `docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md` |
| counter | area40 | strong structured selector row but superseded by Open3D QEM40 on render, AbsRel, and Depth MAE | `docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md` |
| counter | area40 no-freeze | omitting strict topology freeze collapses topology to 18,693 triangles and loses badly to frozen area40 | `docs/car_model/final_stageF18_counter_no_freeze_control_report.md` |
| parking_phone_tiny | R53.01 area70 | strong area-only control at the same triangle count; superseded by CSEF70 on PSNR, LPIPS, AbsRel, Depth MAE, and normal with negligible SSIM loss | `docs/car_model/final_stageF7_parking_pareto_report.md` |
| parking_phone_tiny | Open3D QEM70 | Open3D QEM did not reach the matched 2,564,473-triangle target on the 8.55M-triangle parking mesh, stopping at 8,125,970 triangles, so it is rejected as an unmatched compression control | `docs/car_model/final_stageF25_parking_posthoc_qem_baseline_report.md` |
| room | QEM50 no-freeze | omitting strict topology freeze collapses topology to 20,742 triangles and loses badly to frozen QEM50 | `docs/car_model/final_stageF24_room_qem_no_freeze_control_report.md` |
| room | random50 | same-count random compaction loses badly to area50 and clean-long | `docs/car_model/final_stageF19_room_selector_ablation_report.md` |
| room | CSEF50 | passes clean-long but is superseded by area50 on all tracked independent metrics | `docs/car_model/final_stageF19_room_selector_ablation_report.md` |
| room | area50 | strong structured selector row but superseded by Open3D QEM50 on render, AbsRel, and Depth MAE | `docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md` |
| parking_phone_tiny | grid fill full-budget | fill branch does not beat matched sparse-depth control | `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md` |

## Gate

PASS with ablation gaps. At least two scenes show meaningful compact-recovery benefit over fair clean-long baselines; five scenes now have auditable long-baseline comparisons. The remaining NeurIPS risk is not scene count, but missing matched ablations against area-only and random same-count compaction beyond the completed controls, further replicated no-freeze controls beyond the completed counter/room controls, explicit sparse-depth-loss variants if claimed, and the fact that Open3D QEM is strong on small/medium meshes but cannot reach the matched 70 percent parking target.
