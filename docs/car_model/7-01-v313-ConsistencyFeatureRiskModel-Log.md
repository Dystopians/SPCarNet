# 2026-07-01 v313 Consistency-Feature Risk Model Log

## Purpose

v311 showed that learned risk selection is unsafe. v312 showed that ordinary
source-feature OOD distance does not catch the harmful target switches. v313
tests whether residual-consistency features can improve the learned risk model.

## Implementation

File:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New target-blind proxy features:

- `delta_signal_cosine`
- `opposition_fraction`
- `aligned_fraction`
- `delta_to_signal_ratio`
- `std_to_signal_ratio`
- `support_confidence`

These features measure whether a candidate correction is aligned with the
transported multi-source residual signal, rather than only measuring magnitude,
coverage, and residual variance.

## Experiments

Focused scenes:

```text
bicycle, counter, stump, treehill
```

Runs:

- `v313a`: consistency features in the learned risk model.
- `v313b`: v313a plus a source-heldout min-tail guard
  (`--per_view_risk_model_min_source_min_delta 0.0`).

Machine-readable summaries:

```text
docs/car_model/results/v313_consistency_feature_risk_model_focused_summary.json
docs/car_model/results/v313_consistency_tailguard_risk_model_focused_summary.json
```

Output roots:

```text
outputs/carnet/spcarnet_v313a_consistency_features_risk_model_focused_20260701
outputs/carnet/spcarnet_v313b_consistency_source_min_guard_focused_20260701
```

## Results

| method | macro PSNR gain | macro SSIM gain | safe scene rate | positive-view fraction | mean min PSNR gain | negative views |
|---|---:|---:|---:|---:|---:|---:|
| v309 selective KNN | +0.173055 | +0.003173 | 1.00 | 0.887014 | -0.031668 | 9 |
| v310c tail-risk scene fallback | +0.172930 | +0.003176 | 1.00 | 0.897014 | -0.031668 | 8 |
| v311c dual-guard risk model | +0.165518 | +0.003099 | 0.50 | 0.905347 | -0.061860 | 7 |
| v312a OOD-guard risk model | +0.165518 | +0.003099 | 0.50 | 0.905347 | -0.061860 | 7 |
| v313a consistency features | +0.167239 | +0.003093 | 0.75 | 0.905347 | -0.061176 | 7 |
| v313b consistency + source min guard | +0.170377 | +0.003166 | 1.00 | 0.905347 | -0.056508 | 7 |

## Interpretation

Positive findings:

- Consistency features fixed the `treehill` all-axis failure seen in v311c/v312a.
- The source min-tail guard correctly predicted and disabled the unsafe `stump`
  risk model.
- v313b recovered all-scene safety on the focused set and reduced negative views
  versus v309/v310c.

Limitations:

- v313b still does not beat v309/v310c on macro PSNR or SSIM.
- The mean min-tail is worse than v309/v310c.
- `counter` loses mean PSNR relative to the scene-level learned choice.
- Therefore v313b is not a paper-final main method and should not be expanded
  as a claimed full9 improvement without further changes.

## Verdict

v313 is the first learned-risk branch to recover focused all-scene safety after
the v311/v312 failures, but it does not solve the main objective. It is a useful
ablation showing that residual-consistency features are more meaningful than
plain OOD distance, while also proving that reliability alone is insufficient
without preserving mean quality.

Current main frontier remains:

- `v309`: mean-quality frontier.
- `v310c`: tail-balanced frontier.
- `v313b`: reliability ablation, not main method.

Current status:

```text
Final status: NOT COMPLETE.
```
