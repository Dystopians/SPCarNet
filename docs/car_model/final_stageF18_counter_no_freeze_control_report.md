# Final Stage F18 - Counter No-Freeze Control

Decision: `FINAL_F18_COUNTER_NO_FREEZE_CONTROL_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`.

## Goal

Test whether the strict topology-freeze mechanism is load-bearing for the final compact-recovery recipe. This control starts from the same `counter` area40 compact checkpoint used by the current best counter row, but resumes training without `--freeze_topology_updates`.

## Run

- scene: `mipnerf360/counter`
- source compact checkpoint: `outputs/carnet/meshsplatopt/final_stageF16_counter_area_selector_control/prune40/compact_model`
- control recovery checkpoint: `outputs/carnet/meshsplatopt/final_stageF18_counter_no_freeze_control/area40/recovery_model`
- schedule: `22000 -> 26000`
- topology-control flags: `--skip_restricted_delaunay` only, deliberately omitting `--freeze_topology_updates`
- W&B: `g5pmw9lk`

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 83,834 | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| area40 frozen 26k | 50,300 | 14.314330 | 0.536892 | 0.431104 | 0.072751 | 0.357914 | 43.715882 |
| area40 no-freeze 26k | 18,693 | 13.641099 | 0.467266 | 0.483981 | 0.104043 | 0.442218 | 45.148206 |

## Finding

`--skip_restricted_delaunay` alone is not enough to preserve the compact topology. The no-freeze control continues standard topology updates and collapses the area40 model from `50,300` to `18,693` triangles. That hidden topology change also destroys the counter win: relative to frozen area40, no-freeze loses `0.673231` PSNR, `0.069626` SSIM, worsens LPIPS by `0.052877`, worsens AbsRel by `0.031292`, worsens Depth MAE by `0.084304`, and worsens Normal by `1.432324` degrees.

## Gate

FAIL control. This is useful evidence: strict topology freezing is not cosmetic; it is required for the accepted compact-recovery contract.
