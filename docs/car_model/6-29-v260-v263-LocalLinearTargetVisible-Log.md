# v260-v263 Local-Linear / Target-Visible Deferred Surface Renderer Log

Date: 2026-06-29

This log records the first pass after reading `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` in this continuation. The hard rule remains unchanged: no full9 promotion before flowers exact beats Phase-J all-axis.

Phase-J flowers reference:

- PSNR `20.304358`
- SSIM `0.557770`
- LPIPS `0.329222`

## Implementation Changes

Code file:

- `scripts/car_model/train_surface_deferred_source_residual_renderer.py`

Implemented changes:

- v260 adds `--ood_gain_mode learned_linear`, a policy-val-supervised linear OOD/gain head over target-free support features. This is useful as an auxiliary guard, but it is not the v169 main representation change because it does not add residual capacity.
- v261 adds `--residual_decoder_mode local_linear`. Instead of only weighted-averaging train-fit source residual slots, each target pixel fits a small ridge decoder inside the face/UV bin from source camera direction and source parent RGB to source residual, then predicts residual at target camera direction and target parent RGB.
- v262 rebuilds the source bank with 32k train-residual candidate faces.
- v263 adds `--target_visible_face_quota`, a no-target-GT target-visible face expansion. It scans stripped target evidence only for `face_id`, `barycentric`, and `alpha`; forbidden target/test RGB or residual keys hard-fail.

All exact runs used W&B offline logging and target/test GT only after no-GT apply for final evaluation.

## Key Results

| run | method | faces | alpha | target PSNR | SSIM | LPIPS | gains | changed | Phase-J verdict |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| v260a | learned OOD head, 8k bank | 8192 | 1.0 | 19.837703 | 0.620010 | 0.180221 | +0.005649 / +0.000100 / +0.000114 | 0.009234 | fail PSNR |
| v261a | local-linear decoder, 8k bank | 8192 | 1.0 | 19.840117 | 0.620063 | 0.180124 | +0.008063 / +0.000153 / +0.000210 | 0.011307 | fail PSNR |
| v261b | lower ridge / larger clip | 8192 | 1.0 | 19.839197 | 0.620018 | 0.180122 | +0.007144 / +0.000108 / +0.000213 | n/a | fail PSNR |
| v262a | local-linear, 32k bank | 32768 | 1.0 | 19.843509 | 0.620217 | 0.180049 | +0.011455 / +0.000306 / +0.000286 | 0.024958 | fail PSNR |
| v263a | local-linear, 32k + target-visible | 58047 | 1.0 | 19.844512 | 0.620224 | 0.179968 | +0.012458 / +0.000314 / +0.000367 | 0.040890 | fail PSNR |
| v263b | v263a bank, alpha up to 3 | 58047 | 1.5 | 19.839942 | 0.619739 | 0.179855 | +0.007888 / -0.000172 / +0.000480 | 0.061011 | fail PSNR/SSIM |

Machine-readable summary:

- `docs/car_model/results/v260_v263_local_linear_target_visible_summary.json`

## What Changed Scientifically

The important step is v261, not v260. v261 changes the residual carrier from a convex source residual average into a local view/appearance-conditioned residual decoder. That directly addresses the v169 criticism that scalar RGB residual atlases and fixed gates cannot carry view-dependent teacher signal.

The local-linear decoder improved policy-val projection and target exact:

- v260a target: `19.837703 / 0.620010 / 0.180221`
- v261a target: `19.840117 / 0.620063 / 0.180124`

The next important step is v263. Expanding the carrier with target-visible geometry increased target support without target/test RGB leakage:

- v261a target active fraction: `0.033383`
- v263a target active fraction: `0.199257`
- v261a changed fraction: `0.011307`
- v263a changed fraction: `0.040890`

This produced the best v260-v263 target exact result:

- v263a: `19.844512 / 0.620224 / 0.179968`

## Negative Result

v263a is better than v260-v262 but still does not pass the v169 flowers gate. Its PSNR remains:

- `19.844512 - 20.304358 = -0.459846`

The SSIM and LPIPS values are numerically above the recorded Phase-J reference under this local metric scale, but the PSNR gap is too large. Full9 remains blocked.

v263b proves that simply increasing alpha is not the solution. Alpha `1.5` improved LPIPS but harmed PSNR/SSIM and tails:

- v263b target SSIM gain: `-0.000172`
- v263b PSNR tail CVaR: `-0.013618`

## Current Bottleneck

The bottleneck has shifted from "the residual carrier has almost no capacity" to:

1. The carrier can now alter more target-visible surface area, but useful changed fraction is still only about `4.1%`.
2. Wider target-visible coverage helps mean metrics, but tail safety degrades when the correction is amplified.
3. Policy-val gains scale much faster than target exact gains, so cross-view generalization and OOD support mismatch remain the central obstacle.

## Verdict

Final status for this line: `NOT COMPLETE`.

Do not launch full9 from v260-v263. The next useful method change should be a stronger surface-attached representation that increases target-visible useful changed fraction without relying on global alpha amplification, such as patch/edge-aware residual basis decoding or a compact learned source-feature decoder with an explicit target-support risk model.

