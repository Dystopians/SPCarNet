# v319 Incumbent Reliability Policy Log

Date: 2026-07-01

## Status

`v319c` is the current best engineering version in this branch. It is a small
but real improvement over `v315d` on full9 apply PSNR/SSIM while preserving the
same tail/safety profile. It is not a paper-final breakthrough: LPIPS is still
slightly worse than `v315d`, and the qualitative visual gap remains subtle.

`v319d` is a negative ablation. It tested a source-heldout perceptual hard
guard and should not replace `v319c`.

Final status: NOT COMPLETE.

## Method Change

The code change is in:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

The new policy is a source-only relative reliability model. For each target
view it predicts candidate-vs-incumbent deltas from proxy evidence only:

- candidate proxy features;
- incumbent scene-selected proxy features;
- candidate-minus-incumbent proxy differences;
- candidate/incumbent variant indicators;
- OOD distance to source-heldout evidence.

The crucial design correction is that the new model is an override on top of
the existing incumbent, not a replacement. If reliability rejects the override,
the pipeline falls back to the v315d incumbent path: scene selector, fixed-scene
risk repair, or KNN policy as applicable.

## Reflection Outcome

The reflection was useful in two concrete ways:

1. v319 initial replacement policy was wrong. It replaced the strong incumbent
   too often and damaged bicycle/treehill.
2. v319b used hard source-tail gates on KNN and accidentally disabled the
   flowers KNN path that v315d needed.

v319c fixes both issues: reliability is an abstaining override, and KNN keeps
the v315d acceptance behavior.

The reflection was not sufficient to solve the full paper bottleneck. v319d
showed that source-heldout LPIPS/DISTS predictions are not reliable enough as a
hard target-time gate with the current proxy features.

## Full9 Apply Results

Result file:

```text
docs/car_model/results/v319c_full9_apply_metrics_vs_prior_summary.json
```

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 PSNR | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v305 | +0.266578 | +0.003701 | +0.013917 | +0.039504 | 8 | 9/9 |
| v315d | +0.269175 | +0.003718 | +0.014301 | +0.039726 | 8 | 9/9 |
| v316c | +0.268444 | +0.003710 | +0.013917 | +0.039504 | 8 | 9/9 |
| v318e | +0.268629 | +0.003715 | +0.013917 | +0.039504 | 8 | 9/9 |
| v319c | +0.269725 | +0.003720 | +0.014301 | +0.039726 | 8 | 9/9 |

v319c minus v315d:

- mean PSNR gain: `+0.000550`;
- mean SSIM gain: `+0.00000197`;
- mean min PSNR: tie;
- mean CVaR10 PSNR: tie;
- negative views: tie;
- safe scenes: tie.

## Clean Baseline Frontier

Result files:

```text
docs/car_model/results/v319c_frontier_lpips_qualitative_summary.json
docs/car_model/results/v319c_frontier_lpips_qualitative_summary.md
docs/car_model/results/v319c_frontier_panels/
```

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v315d | 27.582989 | 0.028182 | 0.087739 | 0.057679 |
| v318e | 27.581262 | 0.028185 | 0.087743 | 0.057674 |
| v319c | 27.583642 | 0.028181 | 0.087746 | 0.057678 |

v319c is best on PSNR and MAE, essentially tied on DISTS, but still slightly
worse than v315d on LPIPS. This is why it is a current best engineering
version, not a closed paper-final method.

## Negative Ablation: v319d

v319d added source-heldout LPIPS/DISTS computation and required predicted
perceptual deltas to be non-negative before allowing a reliability override.

Result files:

```text
docs/car_model/results/v319d_full9_apply_metrics_vs_prior_summary.json
docs/car_model/results/v319d_frontier_lpips_qualitative_summary.json
docs/car_model/results/v319d_frontier_lpips_qualitative_summary.md
docs/car_model/results/v319d_frontier_panels/
```

Apply result:

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 PSNR | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v315d | +0.269175 | +0.003718 | +0.014301 | +0.039726 | 8 | 9/9 |
| v319c | +0.269725 | +0.003720 | +0.014301 | +0.039726 | 8 | 9/9 |
| v319d | +0.267239 | +0.003702 | +0.014301 | +0.039696 | 8 | 8/9 |

Frontier result:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| v315d | 27.582989 | 0.028182 | 0.087739 | 0.057679 |
| v319c | 27.583642 | 0.028181 | 0.087746 | 0.057678 |
| v319d | 27.580252 | 0.028191 | 0.087746 | 0.057673 |

v319d improves DISTS slightly but loses PSNR/MAE and breaks fixed safety on
stump. It did not repair LPIPS versus v315d. Treat this as a failed ablation.

## Commands And Outputs

Main v319c full9 output:

```text
outputs/carnet/spcarnet_v319c_incumbent_reliability_full9_20260701
```

Main v319d full9 output:

```text
outputs/carnet/spcarnet_v319d_perceptual_reliability_full9_20260701
```

Frontier outputs:

```text
outputs/carnet/spcarnet_v319c_incumbent_reliability_frontier_comparison_full9_20260701
outputs/carnet/spcarnet_v319d_perceptual_reliability_frontier_comparison_full9_20260701
```

Each apply run used W&B offline logging under the scene output directory:

```text
outputs/carnet/spcarnet_v319c_incumbent_reliability_full9_20260701/*/wandb/offline-run-*
outputs/carnet/spcarnet_v319d_perceptual_reliability_full9_20260701/*/wandb/offline-run-*
```

The frontier runs used:

```text
scripts/car_model/build_support_transport_frontier_comparison.py
```

with methods `clean26000`, `v315d`, `v319c`, and `v319d` on all nine selected
scenes. W&B offline logs are in the corresponding frontier output directories.

## Next Step

Do not continue by making source perceptual prediction a harder gate. The next
credible direction is a calibrated reliability model with an explicit
abstention objective and source-heldout validation target that includes
tail-risk and perceptual metrics jointly, or a paper story that frames v319c as
a quality-complexity Pareto improvement while honestly acknowledging LPIPS and
qualitative subtlety.
