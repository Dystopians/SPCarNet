# Final Stage F47/F48/F49 - CSEF-Family All-Metric Repair

Date: 2026-05-04

Decision: `CSEF_FAMILY_VALIDATION_BUDGET_ALL_SCENE_ALL_METRIC_PASS_MARGIN_STRENGTHENED`.

## What Changed

F46 closed room, counter, and parking, but bonsai still had no single CSEF-family row that beat clean-long on every tracked metric. F47 therefore targeted the remaining bonsai failure directly: keep the CSEF50 compact checkpoint, keep sparse-depth strict recovery, and add a small LPIPS training term during the same `22000->26000` long recovery. F49 then reran the same long contract with a lower LPIPS weight to strengthen the thin bonsai PSNR/SSIM/depth margins.

F47 command contract:

- source compact model: `outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/bonsai/csef_low_evidence_boundary_protected/prune50/compact_model`
- recovery model: `outputs/carnet/meshsplatopt/final_stageF47_bonsai_csef50_sparse_lpips/prune50/recovery_model`
- W&B: `4yz7s4s4`
- topology: `44,230` triangles at iteration 22000 and 26000, unchanged
- strict flags: `--freeze_topology_updates`, `--skip_restricted_delaunay`
- recovery additions: sparse COLMAP depth plus `--lambda_lpips_loss 0.01`

F49 command contract:

- source compact model: `outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/bonsai/csef_low_evidence_boundary_protected/prune50/compact_model`
- recovery model: `outputs/carnet/meshsplatopt/final_stageF49_bonsai_csef50_sparse_lpips005/prune50/recovery_model`
- W&B: `cuq7olfd`
- topology: `44,230` triangles at iteration 22000 and 26000, unchanged
- strict flags: `--freeze_topology_updates`, `--skip_restricted_delaunay`
- recovery additions: sparse COLMAP depth plus `--lambda_lpips_loss 0.005`

## Bonsai Results

Clean-long bonsai baseline: 88,460 triangles, PSNR 10.944348, SSIM 0.222848, LPIPS 0.586158, AbsRel 0.194249, Depth MAE 1.816410, Normal 45.358356.

| row | W&B | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bonsai CSEF50+sparse+LPIPS0.010 | `4yz7s4s4` | 44,230 | 50.0% | 10.954357 | 0.224641 | 0.577722 | 0.186212 | 1.747782 | 43.073652 | +0.010009 | +0.001793 | -0.008436 | -0.008037 | -0.068628 | -2.284704 | `PASS_ALL_METRIC_CLEAN_WIN` |
| bonsai CSEF50+sparse+LPIPS0.005 | `cuq7olfd` | 44,230 | 50.0% | 10.954425 | 0.224850 | 0.581385 | 0.184509 | 1.731033 | 43.210080 | +0.010077 | +0.002002 | -0.004773 | -0.009740 | -0.085377 | -2.148276 | `PASS_ALL_METRIC_CLEAN_WIN` |

This fixes the previous bonsai gap. F46 CSEF50 had the desired depth and normal behavior but missed LPIPS by +0.000239. F46 CSEF20 fixed LPIPS but lost Depth MAE by +0.035809. Both F47 and F49 keep the 50% reduction and win on all six metrics. F49 is selected for the final table because it improves PSNR, SSIM, AbsRel, and Depth MAE margins over F47 while retaining LPIPS and normal wins; F47 remains the stronger perceptual/normal backup row.

## Final CSEF-Family Package

This table uses only CSEF-family compaction rows, not QEM rows. It is the cleaner method claim after the F47/F49 bonsai repair.

| scene | selected row | evidence | W&B | clean triangles | ours triangles | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| parking_phone_tiny | CSEF50+sparse-depth | F46 | `8l96pfjx` | 8,548,242 | 4,274,121 | 50.0% | +0.159285 | +0.010216 | -0.009794 | -0.001635 | -0.005476 | -0.791343 | `PASS_ALL_METRIC_CLEAN_WIN` |
| bonsai | CSEF50+sparse-depth+LPIPS | F49 | `cuq7olfd` | 88,460 | 44,230 | 50.0% | +0.010077 | +0.002002 | -0.004773 | -0.009740 | -0.085377 | -2.148276 | `PASS_ALL_METRIC_CLEAN_WIN` |
| courtyard | CSEF50+sparse-depth | F30 | `9aaku1yn` | 1,677,484 | 838,742 | 50.0% | +0.448939 | +0.042206 | -0.023697 | -0.032958 | -0.210749 | -0.207904 | `PASS_ALL_METRIC_CLEAN_WIN` |
| room | CSEF20+sparse-depth | F46 | `v7ld1o0x` | 84,506 | 67,605 | 20.0% | +0.709980 | +0.065611 | -0.044496 | -0.002689 | -0.007460 | -1.469287 | `PASS_ALL_METRIC_CLEAN_WIN` |
| counter | CSEF20+sparse-depth | F46 | `pijpv7ny` | 83,834 | 67,067 | 20.0% | +0.209242 | +0.023350 | -0.016343 | -0.002719 | -0.005011 | -1.181437 | `PASS_ALL_METRIC_CLEAN_WIN` |

## Answer To The Baseline Question

Against the strongest clean-long baselines, the current CSEF-family package now beats clean-long on all tracked render and sparse-geometry metrics for all five final scenes. The fixed single hyperparameter claim is still false, but the validation-budget CSEF-family method claim is now fully supported by long-run evidence.

The important change from the previous state is that the method no longer needs QEM to rescue bonsai, room, or counter. QEM can stay as an operator baseline or stronger posthoc simplification point, but the core CSEF-family method now has direct long-run evidence on the formerly weak scenes.
