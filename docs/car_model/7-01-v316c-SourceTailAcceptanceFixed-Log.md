# v316c Source-Tail Acceptance Fixed Log

Date: 2026-07-01

## Purpose

v315d was the best mean-quality policy, but it still did not fully close the
tail comparison against v305: it beat v305 on macro PSNR, macro SSIM, and mean
min PSNR, but trailed v305 mean CVaR20 PSNR by `0.000173`.

The remaining gap was mostly from `garden`. v315d let source-heldout KNN choose
learned branches because they improved source mean score, even though the same
source-heldout policy was tail-negative:

```text
garden source KNN vs scene-selected:
mean PSNR delta  +0.003530
CVaR20 delta     -0.004184
min PSNR delta   -0.005679
```

v316c fixes this as a protocol bug and a method update: fixed-threshold KNN
acceptance now enforces the same source-heldout PSNR / SSIM / CVaR / min /
positive-view constraints that were already used by auto-threshold search.

## Implementation

Main code:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Changes:

- added source-neighbor local KNN diagnostics and optional local-tail guard;
- recorded `per_view_knn_diagnostics` in target per-view reports;
- added CLI flags for local-tail KNN diagnostics;
- fixed final KNN policy acceptance so non-auto/fixed threshold mode enforces:
  - `--per_view_knn_min_source_psnr_delta`;
  - `--per_view_knn_min_source_ssim_delta`;
  - `--per_view_knn_min_source_cvar_delta`;
  - `--per_view_knn_min_source_min_delta`;
  - `--per_view_knn_min_source_positive_fraction_delta`.

v316c promoted flags:

```text
--per_view_knn_min_source_cvar_delta 0.0
--per_view_knn_min_source_min_delta 0.0
--per_view_knn_min_source_positive_fraction_delta 0.0
--per_view_knn_min_score_delta_vs_scene 0.0005
--per_view_knn_forbid_fixed_when_scene_nonfixed
--per_view_knn_reject_variant scene
```

This is target-blind: target/test GT is still read only after selected images
are saved for evaluation. The KNN admission decision uses only source-heldout
leave-one-out summaries.

## Artifacts

Full9 output root:

```text
outputs/carnet/spcarnet_v316c_source_tail_acceptance_fixed_multiscene_20260701
```

Summary:

```text
docs/car_model/results/v316c_source_tail_acceptance_fixed_multiscene_summary.json
```

Focused exploratory outputs:

```text
outputs/carnet/spcarnet_v316a_local_tail_guard_focused_20260701
outputs/carnet/spcarnet_v316b_source_tail_eligible_focused_20260701
outputs/carnet/spcarnet_v316c_source_tail_acceptance_fixed_focused_20260701
```

All v316c runs used `WANDB_MODE=offline` and wrote per-scene W&B logs under
each scene output directory.

## Full9 Result

| method | PSNR | SSIM | safe scene rate | positive-view fraction | mean min PSNR | mean CVaR PSNR | negative views |
|---|---:|---:|---:|---:|---:|---:|---:|
| v305 | +0.266578 | +0.003701 | 1.00 | 0.954228 | +0.013917 | +0.082173 | 8 |
| v309 | +0.267843 | +0.003711 | 1.00 | 0.949784 | +0.013817 | +0.081414 | 9 |
| v310c | +0.267134 | +0.003704 | 1.00 | 0.954228 | +0.014003 | +0.081866 | 8 |
| v314 | +0.268348 | +0.003715 | 1.00 | 0.949784 | +0.001562 | +0.078339 | 9 |
| v315d | +0.269175 | +0.003718 | 1.00 | 0.954228 | +0.014301 | +0.082000 | 8 |
| v316c | +0.268444 | +0.003710 | 1.00 | 0.954228 | +0.013917 | +0.082235 | 8 |

## Main Deltas

v316c vs v305:

```text
macro PSNR      +0.001866
macro SSIM      +0.00000961
mean min PSNR   +0.000000
mean CVaR20     +0.0000621
negative views  +0
```

v316c vs v315d:

```text
macro PSNR      -0.000732
macro SSIM      -0.00000789
mean min PSNR   -0.000385
mean CVaR20     +0.000235
negative views  +0
```

## Interpretation

v316c is not a simple upgrade over v315d. It is a different frontier point:

- v315d remains the mean-quality frontier.
- v316c is the source-tail-safe frontier and closes the tiny CVaR gap against
  v305 while preserving v305's negative-view count and min tail.

The gain is scientifically useful because it converts a hidden protocol
inconsistency into a clean source-heldout acceptance rule. It also explains why
blindly chasing mean PSNR caused repeated garden regressions: source mean can be
positive while source tail is negative.

## Remaining Weaknesses

Final status: NOT COMPLETE.

The engineering loop is stronger, but the paper loop is still not fully closed:

- v316c does not beat v315d on macro PSNR/SSIM or mean min PSNR;
- v315d and v316c should be presented as a two-point frontier, not one dominant
  final method;
- perceptual metrics, geometry/triangle accounting, and qualitative crop/error
  panels are still incomplete;
- Phase-J remains a stronger render-time residual endpoint, so the paper story
  must emphasize target-blind policy reliability rather than claiming raw
  residual superiority.

Next step: build a frontier selector that can report both `mean-frontier`
v315d and `tail-safe-frontier` v316c, then add LPIPS/DISTS and qualitative
error-map panels for the scenes where the policies differ (`flowers`, `garden`).
