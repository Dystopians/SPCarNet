# Final Stage F37 - Parking Matched Fast-QEM Baseline

Date: 2026-05-04

Decision: `FINAL_F37_PARKING_FAST_QEM_MATCHED_MIXED_RENDER_FAIL_GEOMETRY_STRONG_CONTROL`.

## Goal

Close the parking posthoc simplification fairness gap. F25 showed that Open3D QEM could
not reach the accepted parking target of `2,564,473` triangles, stopping at
`8,125,970`. F37 adds a `fast_simplification` backend to the QEM checkpoint script and
applies repeated QEM passes until the target topology is matched.

## Matched Compaction

| pass | input triangles | output triangles | target | valid |
| --- | ---: | ---: | ---: | --- |
| Open3D F25 | 8,548,242 | 8,125,970 | 2,564,473 | yes, unmatched |
| fast-QEM pass 1 | 8,548,242 | 4,967,085 | 2,564,473 | yes |
| fast-QEM pass 2 | 4,967,085 | 3,766,935 | 2,564,473 | yes |
| fast-QEM pass 3 | 3,766,935 | 3,172,305 | 2,564,473 | yes |
| fast-QEM pass 4 | 3,172,305 | 2,835,089 | 2,564,473 | yes |
| fast-QEM pass 5 | 2,835,089 | 2,634,880 | 2,564,473 | yes |
| fast-QEM pass 6 | 2,634,880 | 2,564,464 | 2,564,473 | yes |

Final matched topology differs from the F7/F33 target by only `-9` triangles
(`0.000351%`). The final checkpoint has `0` degenerate faces and `0` invalid indices.

## Run

| field | value |
| --- | --- |
| source clean checkpoint | `outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model/point_cloud/iteration_22000` |
| matched compact checkpoint | `outputs/carnet/meshsplatopt/final_stageF37_parking_fast_qem_matched_baseline/prune70_pass6/compact_model` |
| recovery checkpoint | `outputs/carnet/meshsplatopt/final_stageF37_parking_fast_qem_matched_baseline/prune70_pass6/recovery_model` |
| schedule | `22000->26000` |
| W&B | `23bqvu1k` |
| topology | `2,564,464` triangles / `284,830` vertices |
| recovery contract | strict topology freeze + skip restricted Delaunay |

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 8,548,242 | 18.480000 | 0.635000 | 0.347000 | 0.082000 | 1.868000 | 45.108000 |
| F33 CSEF70 + sparse-depth 26k | 2,564,473 | 18.712330 | 0.647730 | 0.338259 | 0.079071 | 1.854015 | 44.035708 |
| F37 matched fast-QEM70 26k | 2,564,464 | 15.263329 | 0.500210 | 0.485292 | 0.076425 | 1.252606 | 40.286478 |

## Deltas

| comparison | dTriangles | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F37 - clean | -5,983,778 | -3.216671 | -0.134790 | +0.138292 | -0.005575 | -0.615394 | -4.821522 |
| F37 - F33 | -9 | -3.449001 | -0.147520 | +0.147032 | -0.002646 | -0.601409 | -3.749230 |

## Interpretation

F37 closes the fairness gap that F25 left open: we now have a topology-matched parking
posthoc QEM baseline at the same 70 percent compression level as F7/F33. The result is
not a render competitor. It is much worse than clean-long and F33 on PSNR, SSIM, and
LPIPS, despite being strong on sparse COLMAP depth and normal proxies.

This supports a narrower and more accurate claim: our final row is the better
appearance-preserving compact-recovery method at matched topology, while aggressive QEM
can improve sparse geometry proxies by sacrificing visual fidelity. The paper should
report F37 as a strong geometry-biased baseline/control, not hide it and not overclaim
universal metric dominance.
