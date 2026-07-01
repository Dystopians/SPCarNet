# 2026-07-01 v314 Scene-Fixed Risk + KNN Policy Log

## Purpose

v311-v313 proved that the learned risk model is only partially reliable:

- unrestricted learned risk can override strong scene-level choices and create
  unsafe target switches;
- OOD distance alone does not catch those failures;
- residual-consistency features and a source min-tail guard recover safety, but
  still underperform the v309/v310 mean-quality frontier.

v314 therefore tests a stricter composition rule: use the learned risk model
only when the source-heldout scene selector falls back to `fixed`; otherwise let
the stronger scene-level selector and source-heldout KNN refinement dominate.

## Implementation

File:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New CLI:

```text
--per_view_risk_model_only_when_scene_fixed
```

Behavior:

- If the scene-level source-heldout selector chooses `learned` or `hybrid`, the
  risk model disables itself and records:

```text
disabled because scene-level source-heldout selector was not fixed
```

- If the scene-level selector chooses `fixed`, the risk model may run, but still
  must satisfy the v313 source-heldout tail guard:

```text
--per_view_risk_model_min_source_min_delta 0.0
```

- KNN remains available for non-fixed scene selections when it clears the
  configured source-heldout thresholds.

This is a target-blind policy composition. Target/test GT is used only after
rendering for evaluation.

## Experiment

Focused scenes:

```text
bicycle, counter, stump, treehill
```

Full multiscene set:

```text
bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill
```

Main output root:

```text
outputs/carnet/spcarnet_v314_scene_fixed_risk_knn_multiscene_20260701
```

Machine-readable summaries:

```text
docs/car_model/results/v314_scene_fixed_risk_knn_focused_summary.json
docs/car_model/results/v314_scene_fixed_risk_knn_multiscene_summary.json
```

W&B was run in offline mode inside each scene output directory.

## Full9 Results

| method | macro PSNR gain | macro SSIM gain | selected-fixed PSNR | selected-fixed SSIM | safe scene rate | positive-view fraction | mean min PSNR | mean CVaR PSNR | negative views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v305 source-heldout auto | +0.266578 | +0.003701 | +0.036542 | +0.000286 | 1.00 | 0.954228 | +0.013917 | +0.082173 | 8 |
| v309 selective KNN | +0.267843 | +0.003711 | +0.037808 | +0.000296 | 1.00 | 0.949784 | +0.013817 | +0.081414 | 9 |
| v310c tail-risk KNN fallback | +0.267134 | +0.003704 | +0.037098 | +0.000289 | 1.00 | 0.954228 | +0.014003 | +0.081866 | 8 |
| v314 scene-fixed risk + KNN | +0.268348 | +0.003715 | +0.038312 | +0.000300 | 1.00 | 0.949784 | +0.001562 | +0.078339 | 9 |

Compared with v309:

- macro PSNR gain improves by `+0.000504`;
- macro SSIM gain improves by `+0.00000392`;
- safe scene rate stays `1.0`;
- negative views stay `9`;
- mean min PSNR worsens by `-0.012254`;
- mean CVaR PSNR worsens by `-0.003075`.

Compared with v310c:

- macro PSNR gain improves by `+0.001214`;
- macro SSIM gain improves by `+0.00001082`;
- safe scene rate stays `1.0`;
- negative views worsen from `8` to `9`;
- mean min PSNR worsens by `-0.012440`;
- mean CVaR PSNR worsens by `-0.003528`.

## Scene-Level Interpretation

Useful v314 behavior:

- `bicycle`, `flowers`, and `garden` use source-heldout KNN where KNN clears the
  source evidence.
- `bonsai`, `counter`, `kitchen`, and `room` keep the strong scene-level choice
  because KNN/risk does not add enough source-heldout evidence.
- `stump` remains fixed because the learned risk model fails the source min-tail
  guard.

Remaining failure:

- `treehill` enables the learned risk model and improves mean PSNR from
  `+0.090757` to `+0.095295`, but worsens the worst-view tail from `-0.049846`
  to `-0.160136` and the CVaR tail from `+0.002068` to `-0.025606`.

That `treehill` tradeoff is the current bottleneck. v314 is the current mean
quality frontier, but it is not an all-axis or paper-final winner.

## Verdict

v314 is a real policy improvement over v309/v310c on full9 mean PSNR/SSIM and
keeps all scenes mean-safe. It also shows that learned risk should be used as a
specialized fixed-fallback repair mechanism, not as a universal override.

However, the method is still not complete:

- tail metrics regress versus v305/v310c;
- negative-view count does not improve;
- qualitative improvements can remain subtle;
- this cannot yet be claimed as comprehensive superiority over MeshSplatting.

Next required step:

```text
v315 should preserve v314 mean gains while adding a target-blind tail guard that
blocks treehill-like risk switches when source-heldout evidence predicts a
possible worst-view collapse.
```

Current status:

```text
Final status: NOT COMPLETE.
```
