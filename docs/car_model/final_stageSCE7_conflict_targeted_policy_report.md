# Final Stage SCE7 Conflict-Targeted Policy Report

Date: 2026-05-06

Decision: `SCE7_STRONG_PARTIAL_MAE_GAP_REDUCED_TO_0P0018`

## Goal

Test whether the new fixed SCE policy controls can close the remaining courtyard Depth MAE gap without using test correspondences for training.

## Best Current Result

Best run:

`outputs/carnet/meshsplatopt/final_stageSCE7_conflict_targeted_policy/courtyard/combined_beta0p02_regressed_28500to28600_seed0/recovery_model`

W&B: `lyhtoty4`

Configuration:

- start checkpoint: SCE6 best `28500`
- final iteration: `28600`
- dense train-only cache: `sentinel_cache_res8_dense2k`
- rollback: `combined`, `beta=0.02`, `lambda=0.5`
- selection: `regressed_only=true`, `cluster_top_k=0`, `cluster_balance=true`
- sparse COLMAP lambda: `0.003`
- vertex LR init: `0.005`
- topology unchanged: `true`

Metrics against F82 courtyard:

| metric | F82 | SCE7 best | delta |
|---|---:|---:|---:|
| PSNR | 12.198611 | 12.610288 | +0.411677 |
| SSIM | 0.308649 | 0.338174 | +0.029525 |
| LPIPS | 0.566687 | 0.560069 | -0.006619 |
| AbsRel | 0.301884 | 0.298901 | -0.002983 |
| Depth MAE | 3.339872 | 3.341660 | +0.001787 |
| Normal angle | 40.215702 | 39.368305 | -0.847397 |

## Negative Controls

| run | key change | Depth MAE | decision |
|---|---|---:|---|
| `combined_beta0p02_topk16_28500to28800_seed0` | top-16 conflict clusters only | 3.360691 | reject: too narrow |
| `combined_beta0p02_regressed_28600to28650_seed0` | continue best 50 steps | 3.343629 | reject: past early-stop knee |
| `combined_beta0p05_regressed_28500to28600_seed0` | higher MAE beta | 3.341664 | tie, no gain |
| `sparse0p006_beta0p02_regressed_28500to28600_seed0` | stronger sparse loss | 3.342886 | reject |
| `combined_beta0p02_regressed_28500to28600_seed1` | same fixed policy, seed 1 | 3.342438 | strong but not best |
| `hardfar_mild_beta0p02_28500to28600_seed0` | train-only hard/far cache | 3.349863 | reject |

## Bottleneck Diagnosis

The independent test analyzer for SCE7 best vs F82 reports:

- global AbsRel improves: `0.331580 -> 0.326868`
- global analyzer MAE still regresses: `3.593810 -> 3.610025`
- `DSC_0318` is the dominant failing held-out view:
  - MAE delta: `+0.419859`
  - AbsRel delta: `+0.032703`
  - gate-critical correspondences: `51`

All other fixed test views improve MAE. This means the remaining global MAE gap is localized and held-out, not a broad geometry collapse.

## Interpretation

SCE7 fixed policy materially improves the previous SCE6 best and nearly closes strict all-metric F82 parent-Pareto. It is not yet a clean full pass because Depth MAE is still `+0.001787` above F82 under the official geometry evaluator. The main lesson is that local conflict overfitting and hard/far train proxies do not reliably transfer to `DSC_0318`; the next method step should represent conflicts explicitly as an Evidence Conflict Graph and plan certificate-carrying actions rather than continuing global loss sweeps.

