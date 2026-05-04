# Final Stage F32 - Counter QEM Sparse-Depth Replication

Date: 2026-05-04

Decision: `FINAL_F32_COUNTER_QEM_SPARSE_DEPTH_PARETO_PASS_PROMOTE_GEOMETRY_PERCEPTUAL`.

## Goal

Replicate the final sparse-depth compact-recovery branch on a fourth accepted final scene.
The run starts from the accepted counter Open3D QEM40 compact checkpoint and uses the same
strict topology-frozen `22000->26000` recovery budget, online W&B logging, independent
rendering, independent image metrics, and COLMAP sparse geometry evaluation.

## Setup

- scene: `counter`
- dataset: `/data/peilincai/mesh_datasets/mipnerf360/counter`
- source compact checkpoint: `outputs/carnet/meshsplatopt/final_stageF21_counter_posthoc_qem_baseline/prune40/compact_model`
- recovery checkpoint: `outputs/carnet/meshsplatopt/final_stageF32_counter_qem_sparse_depth/prune40/recovery_model`
- W&B: `x9b89ssf`
- schedule: `22000->26000`
- topology: `50,300` triangles / `102,638` vertices at recovery
- sparse-depth flags: `--enable_sparse_colmap_depth_loss`, `lambda=0.001`, low-error sample mode, `0.5` low-error fraction, `22000->26000` warmup/decay schedule

## Results

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 83,834 | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| QEM40 frozen 26k | 50,300 | 14.409434 | 0.547456 | 0.420855 | 0.068076 | 0.338664 | 43.716007 |
| QEM40 + sparse-depth 26k | 50,300 | 14.408769 | 0.547570 | 0.420202 | 0.068014 | 0.339115 | 43.585215 |

## Deltas

| comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse-depth - clean | +0.272587 | +0.034768 | -0.031847 | -0.008982 | -0.030858 | -0.701820 |
| sparse-depth - QEM40 frozen | -0.000665 | +0.000114 | -0.000653 | -0.000062 | +0.000451 | -0.130792 |

## Interpretation

F32 is the strongest counter geometry/perceptual row at the accepted 40 percent topology.
It improves SSIM, LPIPS, AbsRel, and normal angle relative to QEM40, while giving back only
`0.000665 dB` PSNR and `0.000451` Depth MAE. It remains a clean-long all-metric win.

This is enough to promote the counter package row to QEM40 + sparse-depth for the same
reason F28 promotes bonsai: the image PSNR cost is negligible and the replicated
geometry/perceptual gains are more relevant to the sparse-depth-guided recovery claim.
The package wording must still report the tiny PSNR/Depth tradeoff relative to pure QEM40.
