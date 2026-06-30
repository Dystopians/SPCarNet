# 2026-06-30 v294 Cross-View Residual Direction Synthesis

This note combines two pieces of evidence:

1. `v294` teacher projection upper-bound diagnostic.
2. Existing `v285/v286` source-heldout residual direction calibration results.

The purpose is to decide whether the current SPCarNet surface carrier should be
promoted, patched, or replaced.

## Evidence A: v294 Projection Upper Bound

Artifact:

```text
docs/car_model/6-30-v294-TeacherProjectionUpperBound-Diagnostic.md
docs/car_model/results/v294_teacher_projection_upper_bound_summary.json
```

Best candidate:

| item | value |
|---|---:|
| texture size | 8 |
| low-rank rank | 4 |
| alpha | 0.03125 |
| full-image PSNR gain | +0.000163820 dB |
| SSIM gain | +0.000000392 |
| LPIPS gain | +0.000000956 |
| SSIM positive-view fraction | 0.500000 |
| LPIPS positive-view fraction | 0.666667 |
| robust all-axis pass | false |

Interpretation:

The current face/UV/low-rank carrier can project a tiny nonzero teacher residual
onto policy-val, but the gain is practically negligible. Raising alpha increases
PSNR a little, but quickly turns SSIM/LPIPS negative. This is not a promotion
signal for flowers exact or full9.

## Evidence B: v285/v286 Source-Heldout Calibration

Artifact:

```text
docs/car_model/results/v285_v286_holdout_calibration_summary.json
```

Key heldout direction statistics from the completed flowers exact runs:

| run | target PSNR gain | target SSIM gain | target LPIPS gain | heldout cosine | heldout error ratio | Phase-J PSNR gap |
|---|---:|---:|---:|---:|---:|---:|
| v285b | +0.010698 | +0.000215 | +0.000317 | 0.214671 | 2.078181 | -0.461606 |
| v286b | +0.008856 | +0.000272 | +0.000235 | 0.214671 | 2.078181 | -0.463448 |

Interpretation:

The heldout cosine around `0.21` is too weak for confident cross-view residual
transport. The error ratio around `2.08` means a residual predictor fitted from
part of the source-view evidence is substantially wrong on heldout source
views. v286 makes tails safer than v285, but it loses mean PSNR and remains far
below Phase-J.

## Joint Diagnosis

These two evidence streams point to the same bottleneck:

- The carrier does contain some train-fit teacher signal.
- That signal is not directionally stable across views.
- Conservative reliability gates can suppress harm, but they also suppress the
  useful residual energy.
- Adding rank, alpha, local ridge features, or texture latent capacity without a
  stronger cross-view direction model is expected to keep producing tiny gains.

This explains why recent methods hover around `+0.00` to `+0.02 dB` target PSNR
gain while Phase-J requires roughly `+0.47 dB` over the parent on flowers.

## Decision

Do not promote the current projection carrier to flowers exact/full9 as a main
method. Do not continue rank/alpha scans as the main route.

The next method must change the representation objective:

1. Learn a cross-view residual direction predictor using source-heldout loss,
   not only teacher residual reconstruction.
2. Predict both residual RGB and a calibrated reliability/confidence from
   source-view diversity, heldout cosine, error ratio, view support,
   normal-camera agreement, parent color, and local edge structure.
3. Train/certify on train-fit/policy-val only, then apply to stripped target
   no-GT evidence.
4. Require flowers exact all-axis vs Phase-J before full9.

## Status

```text
Final status: NOT COMPLETE.
```
