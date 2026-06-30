# v257-v258 Policy-Calibrated Deferred Residual Gain Log

Date: 2026-06-29

This log follows `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.  The hard rule remains unchanged: do not launch full9 until flowers exact beats the Phase-J flowers reference on all three axes:

- PSNR > `20.304358`
- SSIM > `0.557770`
- LPIPS < `0.329222`

## Status

`NOT COMPLETE`.

v257-v258 are real train/eval pipeline changes, not only report edits.  They improve the deferred source-feature residual renderer over v256/v257 on flowers target exact mean metrics, but they still fail the Phase-J PSNR gate by about `0.466` PSNR.  Therefore no full9 promotion was launched.

## Storage / GPU Preflight

Before v258 exact experiments:

```text
/data:    28T total, 27T used, 132G available
/dev/shm: 252G total, 251G used, 1.7G available
/tmp (/): 14T total, 7.2T used, 6.0T available
GPU5:     2444 MiB used, 0% utilization
```

The runs used `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/...` outputs and reused the existing frozen v253 source bank rather than duplicating evidence caches.

## Method Changes

### v257 Patch/Perceptual Reliability

Implemented in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`.

New `--policy_reliability_mode patch_perceptual_v1` learns a train-policy-val face/UV reliability map from:

- local RGB L1 improvement;
- local luma patch improvement;
- luma-gradient improvement.

The map is frozen before target/test no-GT apply.  Target GT is only loaded after apply for evaluation.

### v258 Policy-Calibrated Residual Gain

v258 adds `--policy_gain_mode positive_soft`.

Instead of using policy-val only to suppress risky face/UV bins, v258 also learns a bounded per-bin residual gain from positive policy-val evidence:

```text
policy_gain = 1 + (policy_gain_max - 1) * reliability * clipped_positive_gain
```

This is a representation/policy change because the baked source bank now carries both:

- `policy_reliability`: whether a residual bin should be trusted;
- `policy_gain`: how much teacher residual energy a trusted bin may retain.

The checkpoint now saves both maps, and `_predict_delta` applies them during policy-val, no-GT target preview, and target exact evaluation.

## Commands / Artifacts

Full command lines are preserved in each audit Markdown/JSON:

| run | audit Markdown | audit JSON | W&B offline dir |
|---|---|---|---|
| v257a patch perceptual reliability | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v257a_patch_perceptual_reliability_targetexact/v253_deferred_source_renderer_audit.md` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v257a_patch_perceptual_reliability_targetexact/v253_deferred_source_renderer_audit.json` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v257a_patch_perceptual_reliability_targetexact/wandb` |
| v258a gain max 2.0 | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258a_policy_gain_patch_perceptual_targetexact/v253_deferred_source_renderer_audit.md` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258a_policy_gain_patch_perceptual_targetexact/v253_deferred_source_renderer_audit.json` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258a_policy_gain_patch_perceptual_targetexact/wandb` |
| v258b gain max 1.5 | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258b_policy_gain15_patch_perceptual_targetexact/v253_deferred_source_renderer_audit.md` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258b_policy_gain15_patch_perceptual_targetexact/v253_deferred_source_renderer_audit.json` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258b_policy_gain15_patch_perceptual_targetexact/wandb` |
| v258c gain max 1.5 + source agreement | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258c_policy_gain15_source_agreement_targetexact/v253_deferred_source_renderer_audit.md` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258c_policy_gain15_source_agreement_targetexact/v253_deferred_source_renderer_audit.json` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258c_policy_gain15_source_agreement_targetexact/wandb` |

Machine summary:

- `docs/car_model/results/v257_v258_policy_calibrated_gain_summary.json`

## Quantitative Results

### Policy-Val

| run | alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR tail CVaR | SSIM tail CVaR | LPIPS tail CVaR | active teacher energy | cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v256c local L1 reliability | 0.5 | +0.010844 | +0.000343 | +0.000144 | +0.007959 | +0.000251 | +0.000060 | 0.034799 | 0.352913 |
| v257a patch perceptual reliability | 0.5 | +0.011102 | +0.000351 | +0.000154 | +0.008089 | +0.000255 | +0.000070 | 0.035923 | 0.355540 |
| v258a gain max 2.0 | 1.0 | +0.030404 | +0.000942 | +0.000450 | +0.022339 | +0.000562 | +0.000089 | 0.467043 | 0.343132 |
| v258b gain max 1.5 | 1.0 | +0.026003 | +0.000812 | +0.000392 | +0.019119 | +0.000525 | +0.000127 | 0.281217 | 0.348178 |
| v258c gain max 1.5 + source agreement | 1.0 | +0.022564 | +0.000696 | +0.000363 | +0.016288 | +0.000455 | +0.000100 | 0.241122 | 0.328454 |

Policy-val interpretation:

- v257 fixed the reliability objective to include structure/perceptual evidence but only slightly improved v256.
- v258 restored much more teacher residual energy: active energy retention increased from `0.035923` in v257a to `0.467043` in v258a.
- This confirms that the previous v256/v257 bottleneck was partly residual-energy suppression, not only source-bank content.

### Target Exact

| run | PSNR | SSIM | LPIPS | PSNR gain | SSIM gain | LPIPS gain | PSNR tail CVaR | SSIM tail CVaR | LPIPS tail CVaR | changed | Phase-J PSNR gap | Phase-J all-axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v256c local L1 reliability | 19.835239 | 0.620001 | 0.180285 | +0.003185 | +0.000091 | +0.000050 | +0.000313 | -0.000005 | -0.000056 | 0.007788 | -0.469119 | false |
| v257a patch perceptual reliability | 19.835336 | 0.620004 | 0.180285 | +0.003282 | +0.000093 | +0.000050 | +0.000309 | -0.000006 | -0.000060 | 0.008096 | -0.469022 | false |
| v258a gain max 2.0 | 19.838304 | 0.620019 | 0.180196 | +0.006250 | +0.000108 | +0.000139 | -0.002007 | -0.000258 | -0.000380 | 0.010295 | -0.466054 | false |
| v258b gain max 1.5 | 19.838286 | 0.620047 | 0.180217 | +0.006232 | +0.000137 | +0.000118 | -0.000816 | -0.000151 | -0.000245 | 0.010148 | -0.466072 | false |
| v258c gain max 1.5 + source agreement | 19.837588 | 0.620037 | 0.180235 | +0.005534 | +0.000126 | +0.000100 | -0.000674 | -0.000137 | -0.000203 | 0.008089 | -0.466770 | false |

Target exact interpretation:

- v258a is the best target mean PSNR/LPIPS result in this line.
- v258b is the best target mean SSIM result in this line.
- v258c reduces target tail damage relative to v258a/v258b, but loses mean gain.
- None of the runs passes the Phase-J all-axis gate because PSNR remains about `0.466` below the Phase-J flowers reference, even though SSIM and LPIPS are numerically better than that reference under this local metric scale.

## No-Target-GT Audit

All v257-v258 runs used:

- fit evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target apply evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- target eval evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`

The script verifies target apply evidence against forbidden GT/residual keys before target preview/exact.  Target GT is loaded only after no-GT apply for evaluation.

## Qualitative Outputs

Target exact render triplets are saved under:

- v258a: `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258a_policy_gain_patch_perceptual_targetexact/target_exact_fixed_policy`
- v258b: `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258b_policy_gain15_patch_perceptual_targetexact/target_exact_fixed_policy`
- v258c: `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v258c_policy_gain15_source_agreement_targetexact/target_exact_fixed_policy`

Each directory contains candidate, `_parent`, and `_gt` PNGs for the 22 target views.

## Current Bottleneck

v258 proves that local policy-calibrated gain can recover teacher residual energy and improve target means.  The remaining blocker is not "no signal"; it is target-tail/OOD safety and the still-large Phase-J PSNR gap.

Observed tradeoff:

- More residual energy improves target mean metrics.
- More residual energy also produces negative target tail CVaR on a few target views.
- Source-agreement confidence reduces tail damage somewhat but also weakens mean gains.

## Next Step

Do not launch full9 yet.

The next useful method should learn a target-support/OOD-aware gain predictor from train/policy-val evidence, not a fixed manual gain cap.  A reasonable v259 direction is:

1. keep v258 policy gain as the energy-recovery mechanism;
2. add an OOD confidence term based on target view distance to source cameras, source residual variance, and policy-val tail risk;
3. certify policy-val mean and tail;
4. run flowers target exact only if policy-val remains all-axis positive;
5. promote to full9 only if flowers exact beats Phase-J all-axis.
