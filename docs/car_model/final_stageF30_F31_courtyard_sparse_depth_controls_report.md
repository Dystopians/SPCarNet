# Final Stage F30/F31 - Courtyard Sparse-Depth Controls

Date: 2026-05-04

Decision: `FINAL_F30_F31_COURTYARD_SPARSE_DEPTH_MIXED_CONTROLS_CSEF_REMAINS_MAIN`.

## Goal

Close the courtyard sparse-depth replication gap and directly address the remaining
normal-angle weakness in the accepted CSEF50 courtyard row. Both controls use the same
strict topology-frozen `22000->26000` recovery budget, online W&B logging, independent
rendering, independent image metrics, and COLMAP sparse geometry evaluation.

## Runs

- F30 CSEF50 + sparse-depth: `outputs/carnet/meshsplatopt/final_stageF30_courtyard_csef_sparse_depth/prune50/recovery_model`, W&B `9aaku1yn`, `lambda=0.001`
- F31 QEM50 + sparse-depth: `outputs/carnet/meshsplatopt/final_stageF31_courtyard_qem_sparse_depth/prune50_lam0p0005/recovery_model`, W&B `hbt9x0kg`, `lambda=0.0005`
- both runs: `--freeze_topology_updates`, `--skip_restricted_delaunay`, low-error sparse COLMAP samples, `0.5` low-error fraction

## Results

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 1,677,484 | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 |
| CSEF50 26k | 838,742 | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 |
| QEM50 26k | 838,741 | 12.530957 | 0.339798 | 0.543378 | 0.332515 | 3.694743 | 40.804188 |
| CSEF50 + sparse-depth 26k | 838,742 | 12.552447 | 0.338854 | 0.545612 | 0.321690 | 3.618295 | 40.613745 |
| QEM50 + sparse-depth 26k | 838,741 | 12.531974 | 0.340074 | 0.543645 | 0.330244 | 3.689526 | 40.810260 |

## Interpretation

F30 directly fixes the main CSEF50 row's only clean-long regression: normal angle improves
from `40.830157` to `40.613745`, and AbsRel also improves from `0.322233` to `0.321690`.
However, it gives back `0.003362 dB` PSNR, worsens LPIPS by `0.000535`, and worsens Depth
MAE by `0.009863`, so it is a targeted geometry control rather than the courtyard main row.

F31 confirms that a lighter sparse-depth weight can preserve QEM50's perceptual/SSIM
strength: it improves QEM50 on PSNR, SSIM, AbsRel, and Depth MAE, but not enough to beat
CSEF50 on PSNR, AbsRel, or Depth MAE. It also does not improve QEM50 normal angle.

The correct package decision is to keep CSEF50 as the balanced courtyard headline and
record F30/F31 as sparse-depth controls. The sparse-depth mechanism is now replicated on
bonsai, room, and courtyard, but the evidence supports a geometry/perceptual regularizer
claim, not a universal PSNR-improvement claim.
