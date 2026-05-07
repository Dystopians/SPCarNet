# Stage OUT2 Parking Local Parent-Gated ELA Report

Date: 2026-05-06

## Decision

OUT2 supersedes OUT1 as the parking outdoor render-repair layer. The core change is to move from an all-or-nothing frame gate to a two-stage gate:

1. A train-derived low-risk frame floor skips views whose F33 render is already very close to the clean parent.
2. A local multi-scale parent-consistency mask applies ELA only in high-disagreement regions inside the remaining views.

The frame floor is not selected from test GT. It is the 15% quantile of the train split safe-parent RGB distance (`0.013329030945897102`). The local mask uses the maximum of 1, 9, and 25 pixel average-pooled safe-parent disagreement maps with threshold `0.018`, softness `0.008`, and full candidate blend.

## Result

Against the corrected held-out-test-selected clean22000 Mesh Splatting baseline on `parking_phone_tiny`:

| method | PSNR | SSIM | LPIPS | per-view full-pass |
|---|---:|---:|---:|---:|
| clean22000 | 18.4800 | 0.6346 | 0.3469 | reference |
| clean30000 | 18.4088 | 0.6315 | 0.3510 | reference |
| F33 CSEF70 sparse-depth | 18.7123 | 0.6477 | 0.3383 | 50 / 54 |
| OUT1-v7 frame parent gate | 18.9528 | 0.6593 | 0.3137 | 54 / 54 |
| OUT2-v6 local parent gate | **18.9767** | **0.6613** | **0.3133** | **53 / 54 vs clean22000; 54 / 54 vs clean30000** |

OUT2-v6 keeps the same F33 topology and sparse-geometry evaluation:

| metric vs clean22000 | OUT2-v6 delta |
|---|---:|
| PSNR | +0.4967 dB |
| SSIM | +0.02672 |
| LPIPS | -0.03358 |
| sparse AbsRel | -0.00311 |
| sparse Depth MAE | -0.01438 |
| sparse normal angle | -1.0727 deg |
| triangle reduction | 70.00 % |

Against OUT1-v7, OUT2-v6 adds +0.0239 dB PSNR, +0.00203 SSIM, and -0.000353 LPIPS. The corrected audit discloses the remaining tail: versus clean22000, `parking_phone_tiny/00001.png` misses PSNR by 0.0497 dB while still improving SSIM and LPIPS.

## Why It Helps Visually

The previous OUT1 gate could only choose between the safe render and the ELA candidate for an entire frame. That avoided regressions but also discarded useful local repairs in otherwise safe views. OUT2 keeps the same safety principle but applies it locally: vegetation, road boundaries, and reflective building patches can receive ELA correction while smooth low-disagreement regions stay close to the compact parent.

## Artifacts

- Local gate script: `scripts/car_model/meshsplatopt_local_parent_consistency_gate.py`
- Output model: `outputs/carnet/meshsplatopt/stageOUT2_parking_local_parentgate/f33_local_parentgate_v6_eval`
- Local gate report: `outputs/carnet/meshsplatopt/stageOUT2_parking_local_parentgate/f33_local_parentgate_v6_eval/test/ours_26000_outdoor_local_parentgate_v6/local_parent_gate_report.json`
- Full montage: `assets/parking_outdoor_local_parentgate_v6_full_montage.png`
- Crop/error montage: `assets/parking_outdoor_local_parentgate_v6_crop_error_montage.png`
- Corrected fair audit W&B run: `bn55syns`

## Remaining Risk

OUT2 is still a render-repair layer. It improves the visible outdoor tail without changing the compact mesh, but it is not a universal per-view dominance result under the corrected clean22000 baseline. The next stronger research step is to move this local parent-consistency mask into recovery training as a loss weight or as a selector feature, so the corrected appearance emerges from the optimized checkpoint rather than from a guarded render adapter.
