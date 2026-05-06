# Final Stage SCE6 Targeted Rollback Recovery Report

Date: 2026-05-06

Decision: `SCE_TARGETED_ROLLBACK_PARTIAL_NEEDS_DENSE_GEOMETRY_PHASE`

## Goal

Run principled courtyard recovery experiments that preserve F95 visual/normal gains while fixing F95's sparse-depth regression against the F82 parent. All valid runs use strict topology freeze and online W&B logging.

## Fixed References

| row | Iter | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal angle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F82 parent | 26000 | 12.198611 | 0.308649 | 0.566687 | 0.301884 | 3.339873 | 40.215702 |
| F95 rejected candidate | 27000 | 12.276576 | 0.315319 | 0.565402 | 0.303441 | 3.378707 | 40.167017 |

Lower is better for LPIPS, AbsRel, Depth MAE, and normal angle.

## Runs

| run | W&B id | cache | lambda | loss | Iter | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | Decision |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `f95_absrelrollback0p01_27000to28000_seed0` | `dpcqn150` | res4 historical | 0.01 | absrel | 28000 | 12.314113 | 0.319029 | 0.565611 | 0.307682 | 3.417613 | 40.085460 | invalid evidence: cache resolution mismatch |
| `f95_res8_absrelrollback0p05_27000to28000_seed0` | `omp7409e` | res8 corrected | 0.05 | absrel | 28000 | 12.313207 | 0.319149 | 0.565527 | 0.308244 | 3.421661 | 40.150194 | reject: sparse depth worse |
| `f95_res8_absrelrollback0p5_27000to28000_seed0` | `xhvmsv8m` | res8 corrected | 0.5 | absrel | 28000 | 12.313520 | 0.319199 | 0.565644 | 0.307620 | 3.423940 | 40.117207 | reject: sparse depth worse |

## Interpretation

The SCE3 loss is correctly wired and logs online, but the first valid SCE6 settings do not yet solve the parent-Pareto blocker.

What improved:

- PSNR and SSIM improve strongly over F82 and F95.
- LPIPS remains better than F82, though slightly worse than F95.
- Normal angle improves over F82 and F95 in the stronger lambda run.
- Topology remains unchanged.

What still fails:

- AbsRel and Depth MAE are still worse than F82.
- Increasing lambda from `0.05` to `0.5` did not materially fix test sparse depth.
- W&B showed sparse rollback active points are too few relative to the image/teacher/render-anchor forces, so the current signal is insufficiently dense.

## Root Cause Learned

The blocker is not a missing interface anymore. It is a training-signal balance problem:

1. The cache must be resolution-aware, or the rollback loss samples the wrong depth pixels.
2. Even after fixing resolution, the 500-point cache gives too sparse a per-view gradient.
3. F95-style teacher/RGB recovery continues to pull the model toward visible quality while only a small set of sparse sentinels pushes back.

## Next Required Step

Build a denser resolution-8 train sentinel cache, then run a geometry-first rollback phase:

- `max_points_per_view=2000`
- corrected resolution-aware cache
- no teacher render loss in the geometry-first phase
- no LPIPS in the geometry-first phase
- keep sparse COLMAP depth and render-normal anchor
- use one-sided parent rollback with stronger lambda and more active points

Only after the sentinel gate passes should an appearance recovery phase be run. This is the cleanest path toward an automatic SCE policy rather than another parameter game.
