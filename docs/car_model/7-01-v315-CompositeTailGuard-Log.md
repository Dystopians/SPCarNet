# 2026-07-01 v315 Composite Tail Guard Log

## Purpose

v314 became the full9 mean-quality frontier, but its tail behavior was not
acceptable: `treehill` learned-risk repair improved the mean while collapsing
one worst view, and `bicycle` KNN made one low-margin branch switch from hybrid
to fixed. v315 turns those lessons into a target-blind composite policy.

## Method

Main file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New v315 KNN controls:

```text
--per_view_knn_min_score_delta_vs_scene
--per_view_knn_forbid_fixed_when_scene_nonfixed
```

Existing risk controls used by the final v315d policy:

```text
--per_view_risk_model_only_when_scene_fixed
--per_view_risk_model_enable_ood_guard
--per_view_risk_model_ood_quantile 0.8
```

Final v315d policy:

1. keep the source-heldout scene selector from v305/v309;
2. allow KNN only when its predicted score beats the scene branch by at least
   `0.0005`;
3. when the scene branch is non-fixed, forbid KNN from downgrading to `fixed`;
4. allow learned risk only for scene-level `fixed` fallbacks;
5. reject fixed-scene learned-risk candidates whose feature distance is outside
   the source-heldout OOD quantile `0.8`.

The policy is target-blind. Target/test GT is used only after rendering for
metrics.

## Artifacts

Focused summaries:

```text
docs/car_model/results/v315b_scene_margin_risk_ssim_focused_summary.json
```

Full9 summaries:

```text
docs/car_model/results/v315b_scene_margin_risk_ssim_multiscene_summary.json
docs/car_model/results/v315c_margin_oodq80_multiscene_summary.json
docs/car_model/results/v315d_no_fixed_downgrade_multiscene_summary.json
```

Output roots:

```text
outputs/carnet/spcarnet_v315b_scene_margin_risk_ssim_multiscene_20260701
outputs/carnet/spcarnet_v315c_margin_oodq80_multiscene_20260701
outputs/carnet/spcarnet_v315d_no_fixed_downgrade_multiscene_20260701
```

For v315d, `bicycle`, `flowers`, and `garden` were rerun with the final
no-fixed-downgrade KNN guard. `treehill` uses the v315c OOD-q0.8 risk run.
`bonsai`, `counter`, `kitchen`, `room`, and `stump` are copied from v315c because
the final v315d controls do not alter their active branch decisions.

All new scene runs used W&B offline under the corresponding scene output
directory.

## Full9 Result

| method | macro PSNR | macro SSIM | safe scene rate | positive-view fraction | mean min PSNR | mean CVaR PSNR | negative views |
|---|---:|---:|---:|---:|---:|---:|---:|
| v305 source-heldout auto | +0.266578 | +0.003701 | 1.00 | 0.954228 | +0.013917 | +0.082173 | 8 |
| v309 selective KNN | +0.267843 | +0.003711 | 1.00 | 0.949784 | +0.013817 | +0.081414 | 9 |
| v310c tail-risk KNN fallback | +0.267134 | +0.003704 | 1.00 | 0.954228 | +0.014003 | +0.081866 | 8 |
| v314 scene-fixed risk + KNN | +0.268348 | +0.003715 | 1.00 | 0.949784 | +0.001562 | +0.078339 | 9 |
| v315b SSIM dominance + KNN margin | +0.268434 | +0.003714 | 1.00 | 0.954228 | +0.013817 | +0.081940 | 8 |
| v315c OOD-q0.8 + KNN margin | +0.268894 | +0.003717 | 1.00 | 0.954228 | +0.013817 | +0.081929 | 8 |
| v315d no fixed downgrade | +0.269175 | +0.003718 | 1.00 | 0.954228 | +0.014301 | +0.082000 | 8 |

v315d deltas:

| baseline | PSNR delta | SSIM delta | min-tail delta | CVaR delta | negative-view delta |
|---|---:|---:|---:|---:|---:|
| v305 | +0.002597 | +0.0000175 | +0.000385 | -0.000173 | 0 |
| v309 | +0.001332 | +0.0000073 | +0.000485 | +0.000586 | -1 |
| v310c | +0.002041 | +0.0000142 | +0.000299 | +0.000134 | 0 |
| v314 | +0.000828 | +0.0000034 | +0.012739 | +0.003661 | -1 |

## Interpretation

Positive evidence:

- v315d is the current best full9 method by macro PSNR and SSIM.
- It fixes the v314 `treehill` tail collapse while keeping a stronger mean gain.
- It fixes the v314/v309 `bicycle` extra negative view.
- It beats v309/v310c/v314 on the listed mean and tail metrics.
- Against v305 it improves PSNR, SSIM, and mean-min tail while matching the
  negative-view count.

Remaining weakness:

- v315d still trails v305 on mean CVaR by `0.000173`. The gap is tiny, but it
  means the current method should not be advertised as a mathematically complete
  all-axis closure.
- The improvements are still image-space support-transport policy gains. They
  do not yet prove a geometry/triangle-count or perceptual-metric paper closure.

## Verdict

v315d is a significant milestone and the best current main policy. It is strong
enough to replace v314 as the headline method for internal reports. It is not
yet a 100% paper-closed solution because v305 retains a slight CVaR advantage
and geometry/perceptual/qualitative evidence still needs to be tightened.

Current status:

```text
Final status: NOT COMPLETE.
```
