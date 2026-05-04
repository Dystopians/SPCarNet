# Final Stage F33 - Parking CSEF Sparse-Depth Replication

Date: 2026-05-04

Decision: `FINAL_F33_PARKING_CSEF_SPARSE_DEPTH_PARETO_PASS_PROMOTE`.

## Goal

Run the explicit sparse COLMAP depth compact-recovery branch on the final remaining
headline scene. The run starts from the accepted parking CSEF70 compact checkpoint and
uses the same strict topology-frozen `22000->26000` long recovery budget, online W&B
logging, independent rendering, independent image metrics, and COLMAP sparse geometry
evaluation.

## Setup

- scene: `parking_phone_tiny`
- dataset: `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- source compact checkpoint: `outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/csef_low_evidence_boundary_protected/prune70/recovery_model/point_cloud/iteration_22000`
- recovery checkpoint: `outputs/carnet/meshsplatopt/final_stageF33_parking_csef_sparse_depth/prune70/recovery_model`
- W&B: `x6rmhhlp`
- schedule: `22000->26000`
- topology: `2,564,473` triangles / `1,661,616` vertices at recovery
- sparse-depth flags: `--enable_sparse_colmap_depth_loss`, `lambda=0.001`, low-error sample mode, `0.5` low-error fraction, `22000->26000` warmup/decay schedule

## Results

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 8,548,242 | 18.480000 | 0.635000 | 0.347000 | 0.082000 | 1.868000 | 45.108000 |
| CSEF70 26k | 2,564,473 | 18.706079 | 0.647764 | 0.338282 | 0.079404 | 1.852816 | 44.204497 |
| CSEF70 + sparse-depth 26k | 2,564,473 | 18.712330 | 0.647730 | 0.338259 | 0.079071 | 1.854015 | 44.035708 |

## Deltas

| comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse-depth - clean | +0.232330 | +0.012730 | -0.008741 | -0.002929 | -0.013985 | -1.072292 |
| sparse-depth - CSEF70 | +0.006251 | -0.000034 | -0.000023 | -0.000333 | +0.001199 | -0.168789 |

## Interpretation

F33 is a stronger parking Pareto row than F7 at identical topology. It improves PSNR,
LPIPS, AbsRel, and normal angle relative to CSEF70, with a negligible SSIM cost and a
small Depth MAE tradeoff. It remains a clean-long all-metric win while removing `70.0%`
of triangles.

This closes the explicit sparse-depth replication gap for the five-scene final package:
bonsai, room, courtyard, counter, and parking all now have W&B-logged long strict-recovery
sparse-depth rows with independent render and sparse-geometry evaluation. The defensible
claim is still "geometry/perceptual Pareto regularizer", not universal dominance over
every matched control metric.
