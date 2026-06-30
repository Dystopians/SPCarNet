# v281-v282 Low-Rank Texture v169 Gate Log

Date: 2026-06-30

Authoritative prompt: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Objective

Answer the v169 question on flowers before any full9 promotion:

> Can Phase-J teacher residual be baked into a MeshSplatting-compatible surface representation that beats the Phase-J flowers gate without target/test GT leakage?

Phase-J flowers gate:

| PSNR | SSIM | LPIPS |
|---:|---:|---:|
| 20.304358 | 0.557770 | 0.329222 |

The gate is not passed unless PSNR is higher, SSIM is higher, and LPIPS is lower. Full9 remains forbidden until this gate passes.

## Storage And Runtime Preflight

Before this run:

| path | status |
|---|---:|
| `/data` | about 115G free, near full |
| `/dev/shm` | about 1.7G free, near full |
| `/tmp` | about 6.0T free |

Therefore all runs reused the existing low-copy evidence at:

- `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`

No full evidence cache was duplicated. W&B ran offline for the medium training/eval runs.

## Code Changes

Primary file:

- `scripts/car_model/train_perceptual_surface_residual_decoder.py`

Implemented changes:

- Fixed the `surface_feature_texture` runtime interface so fitted textures now preserve `mode`; this matters because v2/lowrank reliability was previously at risk of falling back to v1-style reliability.
- Added `--surface_texture_mode lowrank_v1`.
- Added `--decoder_output_mode lowrank_texture`.
- For each selected face and UV bin, fit a train-fit-only residual texture with:
  - basis 0: mean teacher-parent residual RGB;
  - basis 1-3: PCA residual directions from a 3x3 residual covariance;
  - support/count, direction agreement, sign consistency, residual variance, and low-rank reliability.
- The low-rank decoder predicts mixture coefficients over the baked surface basis instead of directly predicting unconstrained RGB residual.
- PCA/covariance fitting is vectorized for full flowers scale.

This is a real train/eval pipeline method change: the same lowrank texture is used during training, policy-val certification, and stripped target apply.

## Runs

Full commands are stored in each audit JSON under the `command` field.

| run | output | W&B offline run |
|---|---|---|
| v281a | `outputs/carnet/spcarnet_v281_texture_direction_20260630/v281a_surface_texture_v2_direction_confidence_targetexact` | `wandb/offline-run-20260630_083215-cuus1mhg` |
| v282a | `outputs/carnet/spcarnet_v282_lowrank_texture_20260630/v282a_lowrank_confidence_policyauto` | `wandb/offline-run-20260630_090748-sk6eybgn` |
| v282b | `outputs/carnet/spcarnet_v282_lowrank_texture_20260630/v282b_lowrank_no_confidence_ablation_policyauto` | `wandb/offline-run-20260630_090749-p5zfkf03` |
| v282b alpha 0.25 | `outputs/carnet/spcarnet_v282_lowrank_texture_20260630/v282b_fixed_alpha025_exact` | `wandb/offline-run-20260630_093759-b52rnpml` |
| v282b alpha 0.50 | `outputs/carnet/spcarnet_v282_lowrank_texture_20260630/v282b_fixed_alpha050_exact` | `wandb/offline-run-20260630_093759-hlzhzpjy` |

## Texture Diagnostics

| run | mode | feature dim | covered bins | covered faces | lowrank reliability | basis0 L1 | PCA energy |
|---|---|---:|---:|---:|---:|---:|---:|
| v281a | v2 | 22 | 0.407608 | 0.998566 | n/a | n/a | n/a |
| v282a | lowrank_v1 | 38 | 0.407408 | 0.998184 | 0.503100 | 0.025214 | 0.002201 |
| v282b | lowrank_v1 | 38 | 0.407259 | 0.998322 | 0.501438 | 0.025124 | 0.002203 |

The carrier is populated, but the mean residual and PCA energy are small relative to the Phase-J gap. This already suggests a carrier-strength bottleneck.

## Policy-Val Results

| run | alpha | PSNR gain | SSIM gain | LPIPS gain | positive views | SSIM positive | LPIPS positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| v281a all-axis | 0.25 | +0.011055 | +0.000251 | +0.000209 | 1.000 | 0.667 | 0.667 |
| v282a lowrank + confidence | 0.75 | +0.027855 | +0.000774 | +0.000903 | 0.833 | 0.833 | 1.000 |
| v282b lowrank no confidence | 0.75 | +0.030253 | +0.000819 | +0.001119 | 0.833 | 0.750 | 1.000 |
| v282b fixed alpha 0.25 | 0.25 | +0.015793 | +0.000501 | +0.000282 | 1.000 | 1.000 | 1.000 |
| v282b fixed alpha 0.50 | 0.50 | +0.025885 | +0.000767 | +0.000670 | 1.000 | 1.000 | 1.000 |

Policy-val improved substantially over v281, so the low-rank carrier is a real improvement on the train-policy split.

## Flowers Target Exact Results

| run | alpha | candidate PSNR | candidate SSIM | candidate LPIPS | PSNR gain | SSIM gain | LPIPS gain | Phase-J PSNR gap | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v281a | 0.25 | 19.832653 | 0.619412 | 0.180652 | +0.000599 | -0.000498 | -0.000317 | -0.471705 | fail |
| v282a | 0.75 | 19.842574 | 0.618900 | 0.180701 | +0.010520 | -0.001010 | -0.000366 | -0.461784 | fail |
| v282b | 0.75 | 19.847127 | 0.619016 | 0.180564 | +0.015073 | -0.000895 | -0.000229 | -0.457231 | fail |
| v282b fixed alpha 0.25 | 0.25 | 19.845635 | 0.620099 | 0.180523 | +0.013581 | +0.000188 | -0.000188 | -0.458723 | fail |
| v282b fixed alpha 0.50 | 0.50 | 19.850666 | 0.619745 | 0.180620 | +0.018612 | -0.000165 | -0.000286 | -0.453692 | fail |

No-target-GT audit passed for all exact runs. Target/test RGB GT was loaded only after no-GT apply for metric computation.

## Interpretation

v282 is a meaningful representation upgrade over v281:

- policy-val gains roughly doubled or tripled;
- target exact PSNR gain improved from `+0.000599` to up to `+0.018612`;
- fixed-alpha diagnostics show that alpha 0.25 can recover target SSIM positivity.

But v282 still fails the v169 hard gate:

- best target PSNR is only `19.850666`, still `0.453692` below Phase-J flowers;
- alpha 0.25 gives positive SSIM but still negative LPIPS;
- alpha 0.50/0.75 improves PSNR more but hurts SSIM/LPIPS;
- confidence did not help; v282b without confidence is stronger than v282a.

Therefore the blocker is not merely alpha selection or confidence thresholding. The low-rank baked texture can carry a small, coherent part of the teacher correction, but it does not carry enough Phase-J residual energy to close a 0.45 dB PSNR gap on held-out target views.

## Verdict

Status: **NOT COMPLETE**.

Full9: **blocked**.

v169 completion condition A is not met because flowers exact does not beat Phase-J PSNR.

The current evidence supports a B-style negative diagnosis for this carrier family: face/UV low-rank baked teacher residual improves policy-val and small target means, but remains too weak and not sufficiently view-coherent for Phase-J-level target PSNR.

## Next Step

Do not continue low-rank/alpha variants as the main route. The next useful method must change the representation class again:

- a coherent view-dependent deferred surface renderer using train-fit source features rather than per-bin residual bases;
- patch/gradient teacher supervision that directly optimizes perceptual structure on rendered patches;
- target-free uncertainty/visibility estimation based on source-view disagreement, not policy-val scalar confidence alone.

Machine-readable summary:

- `docs/car_model/results/v281_v282_lowrank_texture_summary.json`
