# 2026-07-01 v311 Learned Risk Model Audit

## Scope

This audit tests whether a learned source-heldout per-view risk model can replace
the current selective KNN / source-heldout policy frontier.

The tested variants are:

- `v311a`: strict source-heldout risk model gate.
- `v311b`: relaxed source gate, risk model allowed to select per-view variants.
- `v311c`: relaxed source gate plus predicted PSNR/SSIM dominance guard versus
  the scene-level selected variant.

Machine-readable focused comparison:

```text
docs/car_model/results/v311_risk_model_focused_comparison_summary.json
```

Focused output roots:

```text
outputs/carnet/spcarnet_v311_learned_risk_model_focused_20260701
outputs/carnet/spcarnet_v311b_learned_risk_model_relaxed_focused_20260701
outputs/carnet/spcarnet_v311c_dual_guard_risk_model_focused_20260701
```

## Implementation Changes

File:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New experimental mechanism:

- fits a ridge predictor on source-heldout candidate proxies;
- predicts objective, PSNR gain, and SSIM gain for `fixed`, `learned`, and
  `hybrid` variants;
- selects a per-view variant only from source-heldout evidence, without target
  GT in the decision path;
- supports scene fallback or no-op rejection;
- now includes optional predicted PSNR/SSIM dominance constraints versus the
  source-heldout scene-level selected variant.

Additional audit fields were added to per-view JSON:

- `risk_model_raw_output_variant`
- `risk_model_decision`
- `selected_proxy_variant`
- `selected_proxy`

The `compute_ssim` handling in the target risk chooser was also fixed so SSIM
constraints are only enforced when SSIM was requested.

## Focused Results

Focused scenes:

```text
bicycle, counter, stump, treehill
```

| method | macro PSNR gain | macro SSIM gain | safe scene rate | positive-view fraction | mean min PSNR gain | negative views |
|---|---:|---:|---:|---:|---:|---:|
| v305 source-heldout auto | +0.171559 | +0.003166 | 1.00 | 0.897014 | -0.031668 | 8 |
| v309 selective KNN | +0.173055 | +0.003173 | 1.00 | 0.887014 | -0.031668 | 9 |
| v310c tail-risk KNN scene fallback | +0.172930 | +0.003176 | 1.00 | 0.897014 | -0.031668 | 8 |
| v311a strict risk model | +0.171559 | +0.003166 | 1.00 | 0.897014 | -0.031668 | 8 |
| v311b relaxed risk model | +0.172679 | +0.003045 | 0.25 | 0.891458 | -0.070348 | 8 |
| v311c dual-guard risk model | +0.165518 | +0.003099 | 0.50 | 0.905347 | -0.061860 | 7 |

Per-scene notes:

- `v311a` usually disabled the risk model and fell back to existing scene-level
  selection.
- `v311b` enabled the model, but it traded SSIM/tail safety for PSNR and became
  unsafe on `bicycle`, `stump`, and `treehill`.
- `v311c` added predicted dual-metric constraints, but the predictor still
  accepted harmful target switches on `stump` and `treehill`.
- `counter` benefited in tail safety under v311b/v311c, but the scene-level
  `learned` choice still had higher mean PSNR.

## Verdict

v311 is a useful diagnostic and an implementation milestone, but it is not a
new main method. The current main method remains:

- `v309` for mean-quality reporting;
- `v310c` as the tail-balanced frontier / ablation.

The learned risk model should remain an ablation until its proxy-to-target
generalization is fixed.

## Main Lesson

The bottleneck is not merely a missing gate. The source-heldout risk predictor
can appear safe on leave-one-out source views while still selecting harmful
variants on target views. This is a source-to-target proxy shift problem:

```text
source-heldout proxy ranking != target per-view perceptual/PSNR risk ranking
```

This explains why loosening gates creates unsafe gains and why adding predicted
dual-axis constraints still does not reliably protect target views.

## Next Required Work

Do not expand v311 to full9 as a claimed method. The next real upgrade should
target the proxy shift directly:

1. Build a target-blind reliability model that predicts whether source-heldout
   evidence is in-distribution for the target view, using agreement, source
   diversity, support geometry, residual variance, and view/normal consistency.
2. Separate `selection` from `evaluation`: selection runs without target GT;
   evaluation loads target GT only after all outputs are written.
3. Report v309/v310c/v311 side by side as an ablation story: naive learned
   risk selection fails, tail-risk scene fallback remains safer, and future work
   needs representation/reliability improvement rather than parameter scans.

Current status:

```text
Final status: NOT COMPLETE.
```
