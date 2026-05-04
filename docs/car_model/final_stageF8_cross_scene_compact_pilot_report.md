# Final Stage F8 Cross-Scene Compact Pilot Report

Date: 2026-05-04

## Decision

`FINAL_F8_CROSS_SCENE_COMPACT_PILOT_PASS`.

F8 now has two non-parking scenes with fair long-run clean baselines and strict topology-frozen compact recoveries. Both passing rows use the same method setting, `csef_low_evidence_boundary_protected/prune50`, and both compare against their own best available 22k clean-long baseline rather than against a shorter run.

This does not replace the need for broader full-paper evaluation, but it closes the previous core weakness: the compact method is no longer supported only by parking.

## Implemented

```text
scripts/car_model/final_run_cross_scene_compact_pilot.py
scripts/car_model/final_collect_cross_scene_compact_pilot.py
outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/
```

The manifest covers `bonsai`, `courtyard`, `room`, and `counter` with conservative 50/60/70/80 percent pruning for `csef_low_evidence_boundary_protected` and `area_smallest`. The collector marks scenes without clean-long baselines as `MISSING_BASELINE` and only counts compact rows that satisfy the long-baseline gate.

## W&B Runs

| scene | role | W&B run | note |
| --- | --- | --- | --- |
| bonsai | clean-long 9k->22k | `r8ozggn1` | completed |
| bonsai | CSEF50 22k->26k | `irdsa4c8` | completed |
| bonsai | CSEF70 22k->26k | `ou72x2zw` | completed, failed SSIM gate |
| courtyard | clean-long retry 1 | `eqjygth6` | resource OOM near final stage |
| courtyard | clean-long retry 2 | `5ptlupv8` | completed with online scalar W&B, inline image eval disabled |
| courtyard | CSEF50 22k->26k | `jz93wrbc` | completed with online scalar W&B, inline image eval disabled |

Courtyard's first retry failed because another process occupied roughly 36GB on the same GPU. The successful retry disabled training-loop image logging and deferred render/metrics/geometry to independent commands, while keeping online W&B scalar logging enabled.

## Results

| scene | method | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bonsai | clean-long 22k | 88,460 | - | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 | baseline |
| bonsai | CSEF50 26k | 44,230 | 50.0% | 10.957497 | 0.224758 | 0.586415 | 0.185180 | 1.737815 | 43.493975 | PASS |
| bonsai | CSEF70 26k | 26,538 | 70.0% | 10.779668 | 0.197751 | 0.603040 | 0.197295 | 1.645116 | 44.388508 | fail SSIM gate |
| courtyard | clean-long 22k | 1,677,484 | - | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 | baseline |
| courtyard | CSEF50 26k | 838,742 | 50.0% | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 | PASS |

## Deltas

| scene | method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonsai | CSEF50 | +0.013148 | +0.001910 | +0.000257 | -0.009069 | -0.078595 | -1.864381 |
| courtyard | CSEF50 | +0.452301 | +0.041626 | -0.024231 | -0.032415 | -0.220612 | +0.008507 |

## Gate

PASS requires at least two scenes with fair clean-long comparisons and compact rows satisfying:
- at least 50 percent triangle reduction;
- PSNR drop no worse than 0.2 dB;
- SSIM drop no worse than 0.01;
- LPIPS increase no worse than 0.02;
- no severe sparse geometry regression.

Current gate state: `PASSED`, with passing scenes `bonsai` and `courtyard`.

## Residual Weakness

Bonsai's clean-long baseline is unexpectedly compact at 88,460 triangles and has low absolute PSNR. It is still a fair same-scene long-baseline comparison, but it should not be the only public-scene evidence in the final paper. Courtyard is the stronger F8 evidence because its clean-long baseline is high-complexity and CSEF50 improves both render and sparse geometry while removing half the triangles.

Recommended next milestone: add one more non-parking scene (`room` or `counter`) with the same clean-long plus CSEF50 protocol, then run a consolidated qualitative montage across parking, bonsai, and courtyard.
