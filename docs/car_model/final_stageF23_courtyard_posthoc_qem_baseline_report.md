# Final Stage F23 - Courtyard Posthoc QEM Baseline

Decision: `FINAL_F23_COURTYARD_POSTHOC_QEM_MIXED_PASS_CSEF50_REMAINS_MAIN`.

## Goal

Replicate the Open3D QEM posthoc simplification baseline on a larger scene after positive bonsai, room, and counter QEM rows. The control applies Open3D quadric decimation to the clean-long `courtyard` checkpoint, transfers checkpoint attributes by nearest neighbors, then uses the same strict topology-frozen `22000 -> 26000` recovery budget as CSEF50, area50, and random50.

## Implementation

- script: `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`
- simplifier: Open3D `simplify_quadric_decimation`
- source: `outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000`
- compact: `outputs/carnet/meshsplatopt/final_stageF23_courtyard_posthoc_qem_baseline/prune50/compact_model`
- recovery: `outputs/carnet/meshsplatopt/final_stageF23_courtyard_posthoc_qem_baseline/prune50/recovery_model`
- failed launch: `tuqvfmaz`, caused by an incorrect dataset path and excluded from results
- accepted W&B: `60tdigdj`
- topology: `1,677,484 -> 838,741` triangles, `3,140,491 -> 1,671,976` vertices
- validation: `degenerate_face_count=0`, `invalid_index_count=0`

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 1,677,484 | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 |
| CSEF50 26k | 838,742 | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 |
| area50 26k | 838,742 | 12.552895 | 0.338469 | 0.544993 | 0.324157 | 3.630241 | 40.907990 |
| random50 26k | 838,742 | 11.383848 | 0.264778 | 0.587667 | 0.371186 | 4.015910 | 41.158282 |
| Open3D QEM50 26k | 838,741 | 12.530957 | 0.339798 | 0.543378 | 0.332515 | 3.694743 | 40.804188 |

## Finding

Open3D QEM50 is a strong same-budget control on `courtyard`, but it does not supersede CSEF50 as the main row. Relative to clean-long, QEM50 improves PSNR by `+0.427449`, SSIM by `+0.043150`, LPIPS by `-0.025930`, AbsRel by `-0.022133`, Depth MAE by `-0.134301`, and normal by `-0.017461` degrees while halving topology.

Relative to CSEF50, QEM50 improves SSIM by `+0.001525`, LPIPS by `-0.001699`, and normal by `-0.025969` degrees, but loses PSNR by `-0.024852`, AbsRel by `+0.010282`, and Depth MAE by `+0.086311`. CSEF50 therefore remains the courtyard main-table row because it is more balanced on PSNR and sparse geometry.

## Gate

MIXED PASS. This result strengthens the QEM baseline coverage on a larger scene, but it also prevents an overclaim: QEM is not universally dominant across all metrics. The correct paper framing is operator-aware fixed-topology recovery with per-scene validated compact operators.
