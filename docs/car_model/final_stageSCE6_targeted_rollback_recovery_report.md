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

## Extended Dense-Cache Recovery Results

After the initial partial result, a denser train-only resolution-8 cache was built:

- `sentinel_cache_res8_dense2k`: `55634` sentinels, `21462` F95-regressed sentinels.
- `sentinel_cache_res8_hardfar4k`: `93386` sentinels, `33880` regressed sentinels, biased toward harder correspondences.

The key change was not just higher rollback lambda. The successful direction required increasing the decayed vertex learning rate for the short geometry phase. With the default late-stage vertex LR around `2e-5`, dense rollback signals are visible in W&B but do not move geometry enough. With `--lr_triangles_points_init 0.015`, the effective late LR is about `2e-4`, which finally moves sparse geometry while keeping topology frozen.

| run | W&B id | cache | rollback | vertex LR init | iter | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `f95_res8_dense2k_geometryfirst2p0_27000to28000` | `mrlzfkj9` | dense2k | absrel 2.0 | default | 28000 | 12.311295 | 0.319002 | 0.565928 | 0.308618 | 3.426251 | 40.150542 | reject: strong loss, weak geometry movement |
| `f95_res8_dense2k_highvlr1p0_27000to27500` | `h7tn8on7` | dense2k | absrel 1.0 | 0.015 | 27500 | 12.476927 | 0.328088 | 0.562784 | 0.304513 | 3.393839 | 39.869039 | partial: large RGB/normal gain, depth closer |
| `f95_res8_dense2k_highvlr1p0_continue_27500to28000` | `l7g5cxqa` | dense2k | absrel 1.0 | 0.015 | 28000 | 12.567472 | 0.334938 | 0.560788 | 0.300637 | 3.374279 | 39.258385 | best balanced: AbsRel passes, MAE still short |
| `f95_res8_dense2k_highvlr_absrel1p0_28000to28500` | `jgvk6zfe` | dense2k | absrel 1.0 | 0.015 | 28500 | 12.606700 | 0.337344 | 0.560571 | 0.298651 | 3.353155 | 39.392915 | current best, MAE gap `+0.013282` |
| `f95_res8_dense2k_highvlr_absrel1p0_28500to29000` | `fhahhra4` | dense2k | absrel 1.0 | 0.015 | 29000 | 12.576760 | 0.334900 | 0.564261 | 0.302179 | 3.367962 | 39.377600 | reject: over-continued |
| `f95_res8_dense2k_highvlr_mae0p05_28000to28500` | `vryqsbsb` | dense2k | mae 0.05 | 0.015 | 28500 | 12.599410 | 0.337597 | 0.560614 | 0.302402 | 3.379025 | 39.200141 | reject: MAE-only destabilizes relative depth |
| `f95_res8_dense2k_highvlr_absrel1p0_sparse0p01_28500to29000` | `l4ekxdsn` | dense2k | absrel 1.0 + sparse 0.01 | 0.0075 | 29000 | 12.592743 | 0.336374 | 0.562328 | 0.302982 | 3.379816 | 39.242347 | reject: stronger sparse global loss hurts |
| `f95_res8_hardfar4k_absrel0p5_28500to28800` | `bdcer742` | hardfar4k | absrel 0.5 | 0.015 | 28800 | 12.593544 | 0.336485 | 0.562117 | 0.303245 | 3.382493 | 39.635445 | reject: hard/far cache over-focus hurts test geometry |

Against F82 (`PSNR 12.198611`, `SSIM 0.308649`, `LPIPS 0.566687`, `AbsRel 0.301884`, `Depth MAE 3.339872`, `Normal 40.215702`), the current best `jgvk6zfe` strongly improves PSNR, SSIM, LPIPS, AbsRel, and normal angle. The remaining blocker is Depth MAE: `3.353155`, still `+0.013282` above F82.

### Lessons

1. Dense sentinel rollback plus high late-stage vertex LR is the first configuration that genuinely crosses the AbsRel parent-Pareto blocker.
2. MAE-only rollback is not the missing piece; it degrades the relative-depth structure that made AbsRel pass.
3. A hard/far-biased train cache does not automatically improve held-out MAE; it can over-focus on train hard correspondences.
4. The policy must include early stopping. Continuing the best 28.5k model to 29k regresses render and depth.
5. The honest current status is `SCE_TARGETED_ROLLBACK_STRONG_PARTIAL`: the method is now much stronger than F95 and mostly beyond F82, but the last Depth MAE margin is not closed yet.
