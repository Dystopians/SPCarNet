# v310 Tail-Risk KNN with Scene Fallback Log

Date: 2026-07-01

## Goal

v309 improved macro PSNR/SSIM over v305, but it worsened per-view tail behavior:

- mean positive-view fraction dropped from `0.954228` to `0.949784`;
- total negative-PSNR target views increased from `8` to `9`;
- bicycle gained mean PSNR but added one extra negative view.

v310 tests whether a source-heldout tail-risk controller can keep v309's mean
benefit while recovering v305-like tail behavior.

## Implementation

Main file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New v310 controls:

```text
--per_view_knn_auto_threshold
--per_view_knn_reject_variant {noop,scene}
--per_view_knn_min_accept_fraction
--per_view_knn_max_accept_fraction
--per_view_knn_min_source_ssim_delta
--per_view_knn_min_source_cvar_delta
--per_view_knn_min_source_min_delta
--per_view_knn_min_source_positive_fraction_delta
--per_view_knn_source_cvar_weight
--per_view_knn_source_min_weight
--per_view_knn_source_positive_weight
```

Mechanism:

1. keep the v305 scene-level source-heldout selector;
2. fit a leave-one-out source-heldout KNN policy over fixed/learned/hybrid
   branch scores;
3. search source-heldout KNN acceptance thresholds;
4. require source-heldout PSNR/CVaR/min-gain constraints before enabling KNN;
5. for v310c, rejected target views fall back to the scene-selected branch
   instead of no-oping to the base render.

The target/test GT is still read only after selected images are written.

## Experiments

All runs used:

```text
--output_variant source_heldout_auto
--enable_per_view_knn_policy
--per_view_knn_k 3
--per_view_knn_auto_threshold
--per_view_knn_allow_when_scene_fixed
--per_view_knn_min_source_psnr_delta 0.0
--per_view_knn_min_source_ssim_delta -0.0002
--per_view_knn_min_source_cvar_delta 0.0
--per_view_knn_min_source_min_delta 0.0
--per_view_knn_min_accept_fraction 0.10
```

Focused roots:

```text
outputs/carnet/spcarnet_v310_tailrisk_knn_strict_focused_20260701
outputs/carnet/spcarnet_v310_tailrisk_knn_relaxedpos_focused_20260701
outputs/carnet/spcarnet_v310_tailrisk_knn_scenefallback_focused_20260701
```

Full9 root:

```text
outputs/carnet/spcarnet_v310_tailrisk_knn_scenefallback_multiscene_20260701
```

Machine-readable summary:

```text
docs/car_model/results/v310_tailrisk_knn_scenefallback_multiscene_summary.json
```

## Negative Ablations

v310a strict required source-heldout mean, CVaR, min, and positive-view
fraction to be non-decreasing. It was safe but mostly disabled KNN, reverting
to v305 on the focused scenes.

v310b relaxed the positive-view constraint but rejected low-confidence target
views by no-oping to the base render. This failed badly:

| scene | selected PSNR gain | selected SSIM gain | no-op fraction | verdict |
|---|---:|---:|---:|---|
| stump | +0.000000 | +0.000000 | 1.000000 | unsafe vs fixed |
| treehill | +0.048926 | +0.000835 | 0.444444 | unsafe vs fixed |

Lesson: low-confidence KNN should not fall back to the base render. The safe
fallback is the scene-level source-heldout branch.

## v310c Full9 Result

v310c uses `--per_view_knn_reject_variant scene`.

| method | PSNR gain | SSIM gain | positive-view fraction | mean min PSNR gain | mean CVaR20 PSNR gain | total negative views |
|---|---:|---:|---:|---:|---:|---:|
| v305 | +0.266578 | +0.003701 | 0.954228 | +0.013917 | +0.082173 | 8 |
| v309 | +0.267843 | +0.003711 | 0.949784 | +0.013817 | +0.081414 | 9 |
| v310c | +0.267134 | +0.003704 | 0.954228 | +0.014003 | +0.081866 | 8 |

Comparison:

```text
v310c - v305: +0.000556 PSNR / +0.000003 SSIM
v310c - v309: -0.000710 PSNR / -0.000007 SSIM
```

Scene-level deltas versus v309:

| scene | PSNR delta | SSIM delta | positive-view delta | tail note |
|---|---:|---:|---:|---|
| bicycle | -0.000500 | +0.000012 | +0.040000 | removes v309's extra negative view |
| flowers | -0.002564 | -0.000059 | +0.000000 | falls back to v305 |
| garden | -0.001149 | -0.000020 | +0.000000 | keeps source-tail-safe subset |
| kitchen | -0.002173 | +0.000006 | +0.000000 | source equality was too weak |
| other scenes | +0.000000 | +0.000000 | +0.000000 | unchanged |

## Verdict

v310c is not the new mean-quality main result. v309 remains the best macro
PSNR/SSIM policy.

v310c is still useful as a documented tail-balanced frontier:

- it stays above v305 on macro PSNR/SSIM;
- it recovers v305's mean positive-view fraction;
- it reduces v309's total negative target views from `9` to `8`;
- it improves mean min PSNR gain over both v305 and v309.

The remaining blocker is that source-heldout tail-risk thresholds are too weak
to repair fixed fallback scenes, and can reject useful v309 changes on flowers
and garden. The next real method step should use a learned tail-risk predictor
or calibration model, not threshold search alone.

Current verdict:

```text
Final status: NOT COMPLETE.
```
