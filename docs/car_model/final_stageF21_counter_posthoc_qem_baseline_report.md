# Final Stage F21 - Counter Posthoc QEM Baseline

Decision: `FINAL_F21_COUNTER_POSTHOC_QEM_STRONG_PASS_SUPERSEDES_AREA40_ON_RENDER_DEPTH`.

## Goal

Replicate the F20 Open3D QEM posthoc simplification baseline beyond `room`, using the `counter` scene where area40 had previously been the strongest compact row. The control applies Open3D quadric decimation to the clean-long checkpoint, transfers checkpoint attributes by nearest neighbors, then uses the same strict topology-frozen `22000 -> 26000` recovery budget as CSEF40, area40, and random40.

## Implementation

- script: `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`
- simplifier: Open3D `simplify_quadric_decimation`
- source: `outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000`
- compact: `outputs/carnet/meshsplatopt/final_stageF21_counter_posthoc_qem_baseline/prune40/compact_model`
- recovery: `outputs/carnet/meshsplatopt/final_stageF21_counter_posthoc_qem_baseline/prune40/recovery_model`
- W&B: `kr8565st`
- topology: `83,834 -> 50,300` triangles, `155,104 -> 102,638` vertices
- validation: `degenerate_face_count=0`, `invalid_index_count=0`

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 83,834 | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| CSEF40 26k | 50,300 | 14.212033 | 0.518401 | 0.450481 | 0.085542 | 0.406373 | 43.476972 |
| area40 26k | 50,300 | 14.314330 | 0.536892 | 0.431104 | 0.072751 | 0.357914 | 43.715882 |
| random40 26k | 50,300 | 13.875822 | 0.482349 | 0.485052 | 0.099779 | 0.444684 | 43.941494 |
| Open3D QEM40 26k | 50,300 | 14.409434 | 0.547456 | 0.420855 | 0.068076 | 0.338664 | 43.716007 |

## Finding

Open3D QEM40 plus strict topology-frozen recovery is the strongest `counter` row on PSNR, SSIM, LPIPS, AbsRel, and Depth MAE. Relative to clean-long, it improves PSNR by `+0.273252`, SSIM by `+0.034654`, LPIPS by `-0.031194`, AbsRel by `-0.008920`, and Depth MAE by `-0.031309` while removing 40 percent of triangles. It also improves normal versus clean by `-0.571028` degrees.

Relative to the previous area40 best row, QEM40 improves PSNR by `+0.095104`, SSIM by `+0.010564`, LPIPS by `-0.010249`, AbsRel by `-0.004675`, and Depth MAE by `-0.019250`; normal is effectively tied, with QEM worse by only `0.000125` degrees.

This second QEM pass changes the paper risk profile. QEM is no longer just a one-scene missing baseline; it is a strong compact operator under the fixed-topology recovery framework on two scenes. The honest claim is that MeshSplatOpt can evaluate and recover compact topology operators reliably, with random pruning rejected and QEM emerging as a strong collapse-style operator.

## Gate

PASS. The posthoc QEM row upgrades the counter main-table row and reduces the classical-simplification baseline risk. More QEM replications are still needed before making broad claims that the framework outperforms classical simplification in general.
