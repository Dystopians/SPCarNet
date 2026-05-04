# Final Stage F22 - Bonsai Posthoc QEM Baseline

Decision: `FINAL_F22_BONSAI_POSTHOC_QEM_STRONG_PASS_SUPERSEDES_CSEF50_ON_RENDER`.

## Goal

Replicate the Open3D QEM posthoc simplification baseline on a third scene after F20 `room` and F21 `counter`. The control applies Open3D quadric decimation to the clean-long `bonsai` checkpoint, transfers checkpoint attributes by nearest neighbors, then uses the same strict topology-frozen `22000 -> 26000` recovery budget as the accepted CSEF50 row.

## Implementation

- script: `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`
- simplifier: Open3D `simplify_quadric_decimation`
- source: `outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000`
- compact: `outputs/carnet/meshsplatopt/final_stageF22_bonsai_posthoc_qem_baseline/prune50/compact_model`
- recovery: `outputs/carnet/meshsplatopt/final_stageF22_bonsai_posthoc_qem_baseline/prune50/recovery_model`
- W&B: `bsed9ik1`
- topology: `88,460 -> 44,230` triangles, `152,485 -> 87,101` vertices
- validation: `degenerate_face_count=0`, `invalid_index_count=0`

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 88,460 | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 |
| CSEF50 26k | 44,230 | 10.957497 | 0.224758 | 0.586415 | 0.185180 | 1.737815 | 43.493975 |
| Open3D QEM50 26k | 44,230 | 11.082405 | 0.243249 | 0.570177 | 0.182966 | 1.793852 | 42.889339 |

## Finding

Open3D QEM50 plus strict topology-frozen recovery is the strongest `bonsai` row on PSNR, SSIM, LPIPS, AbsRel, and normal. Relative to clean-long, it improves PSNR by `+0.138057`, SSIM by `+0.020401`, LPIPS by `-0.015981`, AbsRel by `-0.011283`, Depth MAE by `-0.022558`, and normal by `-2.469017` degrees while halving triangles.

Relative to CSEF50, QEM50 improves PSNR by `+0.124908`, SSIM by `+0.018491`, LPIPS by `-0.016238`, AbsRel by `-0.002214`, and normal by `-0.604636` degrees. CSEF50 remains better on Depth MAE by `0.056037`, so this is not a universal geometry dominance claim.

## Gate

PASS. QEM is now replicated on `bonsai`, `room`, and `counter`, which substantially reduces the posthoc simplification missing-baseline risk and upgrades the bonsai main-table row.
