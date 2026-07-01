# v323 Local Support Reflection and Negative Evidence

Date: 2026-07-01

## Summary

v323 implements a source-heldout local support policy in
`scripts/car_model/apply_source_heldout_support_transport_calibrator.py`.
The policy was intended to unlock the strict oracle gap left by v322C without
using target/test GT for selection. It is a real train/eval pipeline interface
change, but focused experiments show it is not yet a successful replacement for
v322C.

Current status: v323 is useful as a diagnostic and ablation module, not as the
paper-final method.

## Implemented Changes

- Added `--enable_local_support_policy`.
- Added candidate-space support for fixed/learned/hybrid plus ladder variants.
- Added source-heldout local KNN summaries over proxy features.
- Added source gates for PSNR, SSIM, CVaR, min-tail, accept fraction, and
  positive-view fraction.
- Added `--local_support_post_incumbent_fallback_only` so local support can act
  only after the incumbent policy falls back to the scene-selected output.
- Added incumbent-aware source validation: in post-fallback mode, source LOO
  rows already refined by the incumbent policy are skipped and the source
  summary is compared against the source incumbent summary, not only against the
  raw scene-selected variant.
- Added per-view diagnostics and report/W&B fields for local support.

## Important Fairness Corrections

Two early v323 runs were not fair against v322C:

- They accidentally used `evidence_max_side=512`, while the v322C full9 reports
  used `evidence_max_side=256`.
- They omitted v322C source-reliability replay knobs such as OOD guard and KNN
  reject mode.

The fair replay knobs that must be kept for future v322C/v323 comparisons are:

```text
--evidence_max_side 256
--per_view_knn_reject_variant scene
--source_reliability_calibration_quantile 0.5
--source_reliability_calibration_scale 0.5
--source_reliability_enable_ood_guard
--source_reliability_ood_quantile 0.8
--source_reliability_min_source_min_delta 0.0
--source_reliability_fixed_scene_min_source_ssim_delta -0.00002
```

## Focused Experiment Paths

- Non-fair diagnostic, `evidence_max_side=512`:
  `outputs/carnet/spcarnet_v323a_local_support_psnr_focused_20260701`
- Fair256 but incomplete replay knobs:
  `outputs/carnet/spcarnet_v323a_local_support_psnr_focused_fair256_20260701`
- Post-fallback local support with partial replay knobs:
  `outputs/carnet/spcarnet_v323b_fallback_local_allaxis_focused_20260701`
- Incumbent-aware local support with partial replay knobs:
  `outputs/carnet/spcarnet_v323c_incumbentaware_local_focused_20260701`
- Best fair replay attempt so far:
  `outputs/carnet/spcarnet_v323d_exactreplay_incumbentlocal_focused_20260701`

All medium focused runs used W&B offline logs under each scene output directory.

## v323d Focused Result vs v322C

Baseline root:
`outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701`

v323d root:
`outputs/carnet/spcarnet_v323d_exactreplay_incumbentlocal_focused_20260701`

| scene | PSNR delta vs v322C | SSIM delta vs v322C | selected all-axis safe vs fixed | local status |
|---|---:|---:|---|---|
| treehill | -0.013086 | -0.000665 | true | disabled: source SSIM gate |
| stump | +0.000000 | -0.000344 | true | enabled but target fallback only |
| bicycle | -0.000938 | -0.000832 | true | disabled: source SSIM gate |
| room | +0.000795 | -0.000482 | true | disabled: source PSNR gate |

Macro over these four focused scenes:

- PSNR delta vs v322C: `-0.003307`
- SSIM delta vs v322C: `-0.000581`
- safety vs fixed: `4/4`

This is not an improvement over v322C.

## Strict Oracle Still Exists

Using target GT only for post-hoc diagnosis, the existing candidate set still
contains a strict per-view oracle that improves PSNR without lowering per-view
SSIM relative to v322C selected output:

| scene | strict oracle PSNR delta | strict oracle SSIM delta | improved views |
|---|---:|---:|---:|
| treehill | +0.029372 | +0.000177 | 8/18 |
| stump | +0.021858 | +0.000097 | 7/16 |
| bicycle | +0.010344 | +0.000128 | 13/25 |
| room | +0.013123 | +0.000147 | 18/39 |
| bonsai | +0.015262 | +0.000134 | 6/37 |
| garden | +0.004707 | +0.000029 | 7/24 |
| flowers | +0.004452 | +0.000055 | 9/22 |
| counter | +0.000696 | +0.000020 | 2/22 |
| kitchen | +0.002224 | +0.000022 | 3/25 |

So the bottleneck is not candidate capacity. The bottleneck is target-GT-free
selection.

## Lessons Learned

1. Local PSNR-only opportunity certificates are unsafe. They can raise PSNR but
   often lower SSIM, especially on stump, bicycle, and treehill.
2. A source-heldout policy compared only against the raw scene-selected variant
   is not enough. The real incumbent is v322C's per-view output, not fixed or
   hybrid alone.
3. Incumbent-aware fallback-only local support is safer but too conservative.
   It mostly disables itself under source SSIM gates and does not capture the
   strict oracle.
4. Fair replay matters. Missing OOD guard or using `evidence_max_side=512`
   changes source policy behavior enough to invalidate direct comparison.
5. The next policy must be pairwise candidate-vs-incumbent, not candidate-vs-scene.

## Next Required Method Direction

The next real attempt should implement a pairwise dominance policy:

- Build source LOO incumbents from the exact v322C policy.
- For each candidate and source view, train/predict deltas relative to that
  incumbent, not relative to the scene-selected variant.
- Accept a target candidate only when source-heldout local evidence predicts:
  PSNR delta >= 0, SSIM delta >= 0, CVaR delta >= 0, and min-tail delta not
  worse than the incumbent.
- Keep v322C as the default output whenever the pairwise dominance certificate
  is absent.

This is the correct next step because strict oracle evidence shows candidate
capacity exists, but v323 local support does not learn the right
candidate-vs-incumbent decision boundary.

Final status: NOT COMPLETE.
