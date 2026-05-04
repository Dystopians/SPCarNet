# MeshSplatOpt Stage R57-R58 Cross-Scene Matched Clean-to-Compact Report

## Purpose

R53/R55 established the parking `clean22k -> compact -> recovery` result. R57/R58 test whether the same clean-to-compact principle transfers to public scenes under a matched continuation screen:

- start from the clean 7000-iteration checkpoint,
- prune the smallest-area 70% of triangles,
- recover from 7000 to 9000 iterations with topology updates frozen,
- compare against the matched clean 7000-to-9000 continuation,
- use online W&B logging and independent render, metrics, and COLMAP sparse-geometry evaluation.

## Runs

| row | scene | role | W&B | triangles |
|---|---|---|---|---:|
| R57.clean9k | courtyard | clean continuation baseline | `ucqyn1ym` | 410254 |
| R57.compact70 | courtyard | prune70 compact recovery | `kgazucjj` | 123076 |
| R58.clean9k | bonsai | clean continuation baseline | `ulv6dpku` | 2487474 |
| R58.compact70 | bonsai | prune70 compact recovery | `82v2cg9z` | 746242 |

## Independent Metrics

| row | scene | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal deg | triangles |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R57.clean9k | courtyard | 18.494551 | 0.602439 | 0.423865 | 0.130293 | 1.592247 | 37.729855 | 410254 |
| R57.compact70 | courtyard | 18.492825 | 0.601917 | 0.451669 | 0.165717 | 1.801261 | 36.696892 | 123076 |
| R58.clean9k | bonsai | 18.541124 | 0.463496 | 0.483265 | 0.201539 | 2.191790 | 40.656060 | 2487474 |
| R58.compact70 | bonsai | 18.821461 | 0.480972 | 0.475725 | 0.194957 | 2.129675 | 40.140394 | 746242 |

## Matched Deltas

| candidate | baseline | pass | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepthMAE | dNormal | triangle reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R57.compact70 | R57.clean9k | false | -0.001726 | -0.000522 | +0.027805 | +0.035424 | +0.209014 | -1.032962 | 0.700000 |
| R58.compact70 | R58.clean9k | true | +0.280336 | +0.017475 | -0.007539 | -0.006582 | -0.062115 | -0.515667 | 0.700000 |

## Decision

`PUBLIC_SCENE_REPLICATION_PARTIAL_PASS`.

The bonsai result is a real public-scene replication: R58.compact70 beats the matched clean continuation on independent PSNR, SSIM, LPIPS, sparse depth, and sparse normal proxy while using 70% fewer triangles.

The courtyard result is a controlled negative: R57.compact70 keeps the normal proxy better and reduces topology by 70%, but loses render quality and depth against the matched clean continuation. It should be reported as a failure mode, not hidden.

The current evidence is therefore stronger than single-scene parking, but still not a finished NeurIPS main claim. The next threshold is either one more public-scene positive or a selector that predicts when area-based compaction will be accepted.

## Artefacts

- `outputs/carnet/meshsplatopt/cross_scene_clean_to_compact_tables/cross_scene_clean_to_compact_results.md`
- `outputs/carnet/meshsplatopt/cross_scene_clean_to_compact_tables/cross_scene_clean_to_compact_results.json`
- `outputs/carnet/meshsplatopt/full_budget_sweep/full_budget_jobs.json`
