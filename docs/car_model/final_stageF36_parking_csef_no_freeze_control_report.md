# Final Stage F36 - Parking CSEF No-Freeze Control

Date: 2026-05-04

Decision: `FINAL_F36_PARKING_CSEF_NO_FREEZE_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`.

## Goal

Close the largest-scene freeze-control gap. This run starts from the accepted
`parking_phone_tiny` CSEF70 compact checkpoint and matches the F7/F33 `22000->26000`
recovery budget, but deliberately omits `--freeze_topology_updates` while keeping
`--skip_restricted_delaunay` and online W&B logging.

## Run

| field | value |
| --- | --- |
| source compact checkpoint | `outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/csef_low_evidence_boundary_protected/prune70/recovery_model/point_cloud/iteration_22000` |
| recovery checkpoint | `outputs/carnet/meshsplatopt/final_stageF36_parking_csef_no_freeze_control/prune70/recovery_model` |
| schedule | `22000->26000` |
| W&B | `ist00zs5` |
| deliberate control | omitted `--freeze_topology_updates`; kept `--skip_restricted_delaunay` |
| start topology | `2,564,473` triangles / `1,661,616` vertices |
| final topology | `3,533,325` triangles / `4,935,615` vertices |

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 8,548,242 | 18.480000 | 0.635000 | 0.347000 | 0.082000 | 1.868000 | 45.108000 |
| CSEF70 frozen 26k | 2,564,473 | 18.706079 | 0.647764 | 0.338282 | 0.079404 | 1.852816 | 44.204497 |
| CSEF70 + sparse-depth frozen 26k | 2,564,473 | 18.712330 | 0.647730 | 0.338259 | 0.079071 | 1.854015 | 44.035708 |
| CSEF70 no-freeze 26k | 3,533,325 | 17.367449 | 0.589928 | 0.363591 | 0.097893 | 1.873921 | 44.812861 |

## Deltas

| comparison | dTriangles | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-freeze - clean | -5,014,917 | -1.112551 | -0.045072 | +0.016591 | +0.015893 | +0.005921 | -0.295139 |
| no-freeze - frozen CSEF70 | +968,852 | -1.338630 | -0.057836 | +0.025309 | +0.018489 | +0.021105 | +0.608364 |
| no-freeze - frozen sparse-depth | +968,852 | -1.344881 | -0.057802 | +0.025332 | +0.018822 | +0.019906 | +0.777153 |

## Finding

On the largest final scene, strict topology freeze is also load-bearing. The no-freeze
control increases topology from `2,564,473` to `3,533,325` triangles and loses badly to
the frozen CSEF70 and frozen sparse-depth rows on independent render quality and sparse
depth. It also falls below the clean-long baseline on PSNR, SSIM, LPIPS, AbsRel, and
Depth MAE.

Together with bonsai, courtyard, room, and counter no-freeze controls, F36 closes the
replication gap: strict topology freezing is validated as a load-bearing mechanism on
all five final-package scenes.
