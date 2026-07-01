# v329b Fixed Rollback Certificate Full9 Log

Date: 2026-07-01

## Question

This pass asks whether the post-v327 reflection can produce a real method
change beyond parameter scanning. The specific failure pattern was that source
reliability sometimes knew a `fixed` output was safer than the current
scene-selected `learned` or `hybrid` output, but the incumbent preservation rule
rejected it with `fixed_when_scene_nonfixed`.

Short answer: v329b is a useful milestone. It adds a target-blind fixed rollback
certificate, improves full9 over both v322C and v327b, and preserves the
non-changing scenes exactly. It is still not a paper-level closed loop because
the macro gain is small and the qualitative difference is subtle.

## Implemented Change

Main file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

The apply/eval pipeline now has two opt-in additions.

1. Pairwise adaptive blend-step guard.

```text
--pairwise_dominance_enable_adaptive_blend_step
--pairwise_dominance_adaptive_max_blend_step
--pairwise_dominance_large_step_min_local_psnr_delta
--pairwise_dominance_large_step_min_local_ssim_delta
--pairwise_dominance_large_step_min_local_cvar_delta
--pairwise_dominance_large_step_min_local_min_delta
--pairwise_dominance_large_step_min_positive_fraction
```

This was tested in v328 and did not become the main result. It is retained as
an opt-in diagnostic, but the paper-facing candidate remains the stricter v327b
blend-step guard.

2. Source reliability fixed rollback certificate.

```text
--source_reliability_enable_fixed_rollback_certificate
--source_reliability_fixed_rollback_min_objective_margin
--source_reliability_fixed_rollback_min_psnr_margin
--source_reliability_fixed_rollback_min_ssim_margin
--source_reliability_fixed_rollback_min_best_psnr_delta
--source_reliability_fixed_rollback_min_best_ssim_delta
--source_reliability_fixed_rollback_max_scene_opposition_fraction
--source_reliability_fixed_rollback_min_scene_aligned_fraction
```

The certificate is only evaluated when the ordinary source-reliability policy
would choose `fixed` but the scene-level incumbent is non-fixed. It is
target-blind: it uses source-heldout prediction margins and scene-consistency
statistics, not held-out target/test metrics. If the fixed rollback candidate
does not clear all margins, the original `fixed_when_scene_nonfixed` rejection
still applies.

v329b uses the v322C/v327b preserved profile plus:

```text
--policy_profile v322c_incumbent
--enable_pairwise_dominance_policy
--pairwise_dominance_enable_ood_guard
--pairwise_dominance_min_local_ssim_delta -0.001
--pairwise_dominance_min_local_min_delta -0.005
--pairwise_dominance_min_source_ssim_delta -0.0002
--pairwise_dominance_min_source_min_delta -0.005
--pairwise_dominance_max_blend_step 0.25
--source_reliability_enable_fixed_rollback_certificate
--source_reliability_fixed_rollback_min_objective_margin 0.005
--source_reliability_fixed_rollback_min_psnr_margin 0.005
--source_reliability_fixed_rollback_min_ssim_margin 0.0
--source_reliability_fixed_rollback_min_best_psnr_delta 0.005
--source_reliability_fixed_rollback_min_best_ssim_delta 0.0
--source_reliability_fixed_rollback_max_scene_opposition_fraction 0.05
--source_reliability_fixed_rollback_min_scene_aligned_fraction 0.9
--enable_wandb
```

This is a fixed policy across scenes, not per-scene tuning.

## Full9 Result Versus v322C

Audit file:

```text
docs/car_model/results/v329b_fixed_rollback_strict_full9_vs_v322c_audit.json
```

Replay root:

```text
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701
```

Incumbent archive:

```text
outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701
```

Full9 macro result:

| metric | v322C | v329b | delta |
|---|---:|---:|---:|
| selected PSNR gain mean | 0.271334337119 | 0.272522652479 | +0.001188315360 |
| selected SSIM gain mean | 0.003727241355 | 0.003736660673 | +0.000009419319 |
| selected PSNR mean | 25.411728563384 | 25.412916878744 | +0.001188315360 |
| selected SSIM mean | 0.840481245798 | 0.840490665117 | +0.000009419319 |

Per-scene delta versus v322C:

| scene | PSNR gain delta | SSIM gain delta | output mismatches |
|---|---:|---:|---:|
| bicycle | +0.000000000000 | +0.000000000000 | 0 |
| flowers | +0.000000000000 | +0.000000000000 | 0 |
| garden | +0.000541668618 | +0.000003593663 | 1 |
| stump | +0.000000000000 | +0.000000000000 | 0 |
| treehill | +0.000820402117 | +0.000008841356 | 7 |
| room | +0.001369726094 | +0.000009183700 | 1 |
| counter | +0.000000000000 | +0.000000000000 | 0 |
| kitchen | +0.000000000000 | +0.000000000000 | 0 |
| bonsai | +0.007963041411 | +0.000063155148 | 3 |

## Full9 Result Versus v327b

Audit file:

```text
docs/car_model/results/v329b_fixed_rollback_strict_full9_vs_v327b_audit.json
```

| metric | v327b | v329b | delta |
|---|---:|---:|---:|
| selected PSNR gain mean | 0.271425492910 | 0.272522652479 | +0.001097159569 |
| selected SSIM gain mean | 0.003728223728 | 0.003736660673 | +0.000008436946 |

The new fixed rollback certificate contributes `bonsai`, `room`, and `garden`.
The `treehill` improvement is inherited from the v327b pairwise blend-step
guard. The other scenes replay exactly.

## Changed Views

Strict v329b fixed rollback accepted these high-confidence source-backed
rollbacks:

| scene/view | change | PSNR gain delta | SSIM gain delta |
|---|---|---:|---:|
| bonsai/00017 | learned -> fixed | +0.102373814711 | +0.000340580940 |
| bonsai/00019 | learned -> fixed | +0.107126042486 | +0.001131176949 |
| bonsai/00026 | learned -> fixed | +0.085132675013 | +0.000864982605 |
| room/00002 | hybrid -> fixed | +0.053419317668 | +0.000358164310 |
| garden/00016 | hybrid -> fixed | +0.013000046828 | +0.000086247921 |

Pairwise blend-step changed treehill views from `fixed` to `mix0250`. The scene
mean is positive, but not every target view improves:

| treehill view | PSNR gain delta | SSIM gain delta |
|---|---:|---:|
| 00002 | +0.007259106762 | +0.000268816948 |
| 00004 | +0.002510692409 | -0.000219762325 |
| 00007 | -0.026469408413 | -0.000042915344 |
| 00008 | -0.004945773279 | -0.000321209431 |
| 00009 | -0.012384636218 | -0.000003993511 |
| 00011 | +0.033720386903 | +0.000268816948 |
| 00015 | +0.015076869936 | +0.000209391117 |

This is a real weakness: treehill is scene-positive but still has local
negative views.

## Qualitative Panel

Panel:

```text
docs/car_model/results/v329b_fixed_rollback_panels/v329b_key_changed_views_panel.png
```

Manifest:

```text
docs/car_model/results/v329b_fixed_rollback_panels/v329b_key_changed_views_panel_manifest.json
```

The panel rows are `bonsai/00019`, `bonsai/00017`, `bonsai/00026`,
`room/00002`, `garden/00016`, and `treehill/00011`, with columns for v322C
incumbent, v329b rollback, and GT. The visual changes are visible mostly as
small color/structure corrections. They are not a dramatic qualitative
breakthrough.

## Ablations and Failed Variants

v328 adaptive blend-step:

```text
docs/car_model/results/v328a_adaptive_blend_treehill_vs_v322c_audit.json
docs/car_model/results/v328a_adaptive_blend_focused6_vs_v322c_audit.json
docs/car_model/results/v328b_riskbudget_adaptive_focused2_vs_v322c_audit.json
```

Result: v328a reproduced the v327b treehill gain but did not unlock more
scenes. v328b improved `stump` and slightly improved `room` PSNR, but `room`
SSIM was slightly negative, so it is not promoted.

v329a loose fixed rollback:

```text
docs/car_model/results/v329a_fixed_rollback_focused3_vs_v322c_audit.json
```

Result: v329a improved `bonsai` and `room`, but harmed `garden`
(`-0.000699067991` PSNR gain and `-0.000007919967` SSIM gain) by also accepting
`garden/00008`. Tightening `min_best_psnr_delta` to `0.005` rejects that bad
rollback while keeping `garden/00016`, producing v329b.

## W&B Offline Runs

Full9 scenes run directly in the full9 root:

```text
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701/bicycle/wandb/offline-run-20260701_023601-3nr4haky
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701/flowers/wandb/offline-run-20260701_023643-grdlj0bm
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701/kitchen/wandb/offline-run-20260701_023716-e9kkrkxd
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701/stump/wandb/offline-run-20260701_023749-8kmovpff
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701/counter/wandb/offline-run-20260701_023814-tc24617f
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701/treehill/wandb/offline-run-20260701_023825-d13arsg4
```

Focused runs copied into the full9 report root for the scenes they cover:

```text
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_focused3_20260701/bonsai/wandb/offline-run-20260701_023313-zbkd2kr6
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_focused3_20260701/room/wandb/offline-run-20260701_023449-vyke8vzr
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_garden_20260701/wandb/offline-run-20260701_023115-vnfrx2do
```

## Verification

Static checks:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/apply_source_heldout_support_transport_calibrator.py
git diff --check -- scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Both checks passed before this report was written.

## Interpretation

v329b is the first post-v327 method change that expands the gain beyond
treehill while preserving the v322C/v327b replay discipline. The mechanism is
scientifically cleaner than a scene-specific parameter search: the rollback
certificate makes an explicit target-blind claim about when a non-fixed scene
decision should be overridden by a safer fixed rendering.

However, the current evidence does not justify claiming a completed top-tier
paper result:

- full9 macro gain over v322C is only `+0.001188` PSNR and `+0.0000094` SSIM;
- only `bonsai`, `room`, `garden`, and `treehill` change;
- treehill still has negative changed views;
- no new LPIPS/DISTS/frontier or triangle-count table has been produced for
  v329b;
- the qualitative panel is technically correct but visually subtle;
- the method is still mostly a conservative policy/certificate improvement,
  not a large representation-capacity breakthrough.

## Verdict

Final status: NOT COMPLETE.

v329b should be kept as a real milestone because it fixes a clear weakness in
the source reliability policy and gives a verified full9 improvement. It should
not be treated as the final paper endpoint. The next work should attack the
effect-size bottleneck directly: stronger target-blind residual reliability,
larger but certified view-dependent surface capacity, and perceptual/geometry
audits that can demonstrate visually meaningful gains rather than micro-metric
movement.
