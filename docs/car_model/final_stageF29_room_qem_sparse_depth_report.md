# Final Stage F29 - Room QEM Sparse-Depth Replication

Date: 2026-05-04

Decision: `FINAL_F29_ROOM_QEM_SPARSE_DEPTH_MIXED_GEOMETRY_PASS_QEM_REMAINS_MAIN`.

## Goal

Replicate the F28 sparse-depth compact-recovery branch on a second public scene using the
accepted room QEM50 compact checkpoint. The run uses the same long strict topology-frozen
`22000->26000` recovery budget, online W&B logging, independent rendering, independent image
metrics, and COLMAP sparse geometry evaluation.

## Setup

- scene: `room`
- dataset: `/data/peilincai/mesh_datasets/mipnerf360/room`
- source compact checkpoint: `outputs/carnet/meshsplatopt/final_stageF20_room_posthoc_qem_baseline/prune50/compact_model`
- recovery checkpoint: `outputs/carnet/meshsplatopt/final_stageF29_room_qem_sparse_depth/prune50/recovery_model`
- schedule: `22000->26000`
- W&B: `wl94n5bp`
- topology: `42,253` triangles / `84,806` vertices at both compact and recovered checkpoints
- sparse-depth flags: `--enable_sparse_colmap_depth_loss`, `lambda=0.001`, low-error sample mode, `0.5` low-error fraction, `22000->26000` warmup/decay schedule

## Results

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| QEM50 frozen 26k | 42,253 | 15.061190 | 0.481082 | 0.516805 | 0.181129 | 1.345221 | 54.900779 |
| QEM50 + sparse-depth 26k | 42,253 | 15.060190 | 0.481189 | 0.516350 | 0.181065 | 1.344086 | 54.841056 |

## Deltas

| comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse-depth - clean | +0.801811 | +0.080325 | -0.062569 | -0.025217 | -0.136144 | -0.601597 |
| sparse-depth - QEM50 frozen | -0.001000 | +0.000107 | -0.000455 | -0.000064 | -0.001135 | -0.059723 |

## Interpretation

F29 is a useful replication of the sparse-depth mechanism, but not a headline replacement
for the room row. It preserves the exact QEM50 topology and improves SSIM, LPIPS, AbsRel,
Depth MAE, and normal angle by small margins, while giving back `0.001000 dB` PSNR.

The main room table should therefore keep the pure QEM50 frozen row as the PSNR-headline
result, while the ablation suite records F29 as a geometry/perceptual sparse-depth pass.
This matches the F28 finding: sparse depth is a reliable geometry/perceptual recovery
regularizer, but it should not be overclaimed as a universal PSNR improver.
