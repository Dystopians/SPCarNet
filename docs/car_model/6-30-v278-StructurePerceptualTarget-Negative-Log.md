# v278 Structure/Perceptual Target Negative Log

Date: 2026-06-30

Prompt:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

## Motivation

v275-v277 showed that a learned surface decoder can pass policy-val, but target
exact still has negative SSIM and LPIPS. The next hypothesis was that the raw
RGB teacher-parent residual target itself is unsafe. v278 therefore changes the
training target, not only the apply policy.

## Implementation

File changed:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

New CLI:

```text
--residual_target_mode {raw,gain_soft,structure_safe,structure_gain}
--residual_target_gain_floor
--residual_target_gain_scale
--residual_target_structure_strength
--residual_target_structure_floor
--residual_target_structure_eps
--residual_target_chroma_scale
```

The v278a target used:

```text
residual_target_mode=structure_gain
gain_floor=0.003
gain_scale=0.04
structure_strength=1.5
structure_eps=0.02
chroma_scale=0.35
```

This transforms train-fit teacher residuals using only train evidence:

1. soft scale from `teacher_gain_l1`;
2. parent/residual luma-gradient support;
3. chroma shrink.

The same transformed target is used by both pixel residual loss and image proxy
loss, so the method is a true target change rather than an apply-only gate.

## v278a Command

The exact command is stored in:

```text
outputs/carnet/spcarnet_v278_structure_perceptual_target_20260630/v278a_mid_structure_gain_targetexact/v180_perceptual_surface_decoder_audit.json
```

W&B offline:

```text
outputs/carnet/spcarnet_v278_structure_perceptual_target_20260630/v278a_mid_structure_gain_targetexact/wandb/offline-run-20260630_063101-882pw77m
```

## Flowers Results

| run | policy alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v278a | 0.75 | +0.016578 | +0.000043 | +0.000299 | +0.008341 | -0.001218 | -0.000904 | fail |

The target exact candidate metrics are:

```text
PSNR 19.840395
SSIM 0.618692
LPIPS 0.181239
```

Phase-J flowers reference remains:

```text
PSNR 20.304358 / SSIM 0.557770 / LPIPS 0.329222
```

The candidate still fails Phase-J PSNR by about `0.463963`, and more importantly
fails parent-relative target SSIM/LPIPS.

## Interpretation

v278a is negative but useful. The structure/gain transformed target makes
policy-val stronger than v277 in all three axes, but target exact gets worse:

```text
v277a target: +0.010690 PSNR / -0.001008 SSIM / -0.000456 LPIPS
v278a target: +0.008341 PSNR / -0.001218 SSIM / -0.000904 LPIPS
```

This means a simple train-fit gain/structure shrink is not enough. It can make
the held-out policy split look better while still failing out-of-trajectory
target structure.

## Next Bottleneck

The next attempt should not be another scalar residual target transform. It
should explicitly learn target-safety from multi-view agreement or held-out-view
structure/perceptual gains. A useful next route is a face/bin/view reliability
model trained on a separate calibration split, then certified on a disjoint
policy-val split before target exact.

## Verdict

```text
Final status: NOT COMPLETE.
Full9 allowed: false.
```
