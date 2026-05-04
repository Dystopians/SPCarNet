# Final Stage F34 - Parking Sparse-Depth Long Continuation Control

Date: 2026-05-04

Decision: `FINAL_F34_PARKING_LONG_CONTINUATION_FAIL_KEEP_F33_26K`.

## Goal

Stress-test the current strongest parking row with a longer fixed-topology continuation.
The run starts from the accepted F33 CSEF70 + sparse-depth checkpoint at iteration
`26000` and continues to `30000` with online W&B logging, strict topology freeze,
independent rendering, independent image metrics, and COLMAP sparse-geometry
evaluation.

This is a fairness control for the long-budget question: if simply training longer is
better, F34 should replace F33. If render quality collapses, F33 is the validated
stopping point.

## Setup

- scene: `parking_phone_tiny`
- dataset: `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- source checkpoint: `outputs/carnet/meshsplatopt/final_stageF33_parking_csef_sparse_depth/prune70/recovery_model/point_cloud/iteration_26000`
- continuation checkpoint: `outputs/carnet/meshsplatopt/final_stageF34_parking_sparse_depth_long_continuation/prune70/recovery_model`
- W&B: `d3nyktd4`
- schedule: `26000->30000`
- topology: `2,564,473` triangles / `1,661,616` vertices
- sparse-depth flags: `--enable_sparse_colmap_depth_loss`, `lambda=0.0005`, low-error sample mode, `0.5` low-error fraction, strict topology freeze

## Results

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 8,548,242 | 18.480000 | 0.635000 | 0.347000 | 0.082000 | 1.868000 | 45.108000 |
| F33 CSEF70 + sparse-depth 26k | 2,564,473 | 18.712330 | 0.647730 | 0.338259 | 0.079071 | 1.854015 | 44.035708 |
| F34 continuation 30k | 2,564,473 | 18.510445 | 0.632080 | 0.354338 | 0.079023 | 1.847455 | 44.427875 |

## Deltas

| comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F34 - clean | +0.030445 | -0.002920 | +0.007338 | -0.002977 | -0.020545 | -0.680125 |
| F34 - F33 | -0.201885 | -0.015650 | +0.016079 | -0.000048 | -0.006560 | +0.392167 |

## Interpretation

F34 is a useful negative long-continuation control, not a promoted method row. It
preserves the F33 topology and slightly improves sparse depth proxies, but it gives back
the visual gains that matter for the paper: PSNR drops by `0.201885`, SSIM drops by
`0.015650`, LPIPS worsens by `0.016079`, and normal angle regresses by `0.392167`
relative to F33.

The conclusion is that the parking sparse-depth schedule needs the F33 `26k` stopping
point. More fixed-topology training is not automatically better, and a fair paper table
should keep F33 as the headline while reporting F34 as a long-budget rejection.
