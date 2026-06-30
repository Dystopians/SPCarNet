# v273 Source-Consensus Residual Denoise Gate Log

Date: 2026-06-30

Prompt followed: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Purpose

v273 addresses the v272 lesson: scalar learned confidence can overfit policy-val and fail target exact. Instead of adding another confidence head, v273 changes the residual carrier itself by projecting each train-fit source residual toward a leave-one-out source-view consensus residual. This is a target-free residual-bank denoise mechanism.

The goal was to test whether noisy or view-inconsistent teacher residual samples were the reason target exact lagged policy-val.

## Code Changes

Edited file:

- `scripts/car_model/train_surface_deferred_source_residual_renderer.py`

Implemented:

- New `--source_consistency_mode denoise`.
- New `--source_consistency_denoise_blend`.
- During source-view consistency calibration, each source slot gets a leave-one-out consensus residual from other views with different `source_view_id`.
- In denoise mode, `bank["residual"]` is rewritten as a reliability-weighted blend between the original residual and the source-view consensus residual.
- The checkpoint stores the denoised residual bank directly.
- Audit JSON, Markdown, and W&B record denoised slot fraction, residual-energy ratio, mean shift, relative shift, and original/denoised cosine.

Validation:

- `PYTHONDONTWRITEBYTECODE=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/train_surface_deferred_source_residual_renderer.py`
- `git diff --check -- scripts/car_model/train_surface_deferred_source_residual_renderer.py`
- CLI help confirmed `denoise` and `source_consistency_denoise_blend` are exposed.

## Experiment Matrix

Common evidence paths:

- fit: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target no-GT: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- target eval: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`

All v273 runs used W&B offline logging and wrote outputs under:

- `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v273_consensus_denoise_flowers_20260630`

| run | change | policy PSNR / SSIM / LPIPS | target PSNR / SSIM / LPIPS | target gains | denoise energy ratio | verdict |
|---|---|---:|---:|---:|---:|---|
| v266c | conservative hybrid reference | 20.668309 / 0.719789 / 0.152274 | 19.845698 / 0.620201 / 0.179915 | +0.013644 / +0.000290 / +0.000419 | n/a | reference |
| v270d | texture-lowrank reference | 20.673378 / 0.720244 / 0.152112 | 19.844320 / 0.620226 / 0.179934 | +0.012266 / +0.000315 / +0.000401 | n/a | reference |
| v273a | denoise blend 0.50 on v266c bank | 20.672101 / 0.720139 / 0.152109 | 19.844213 / 0.620207 / 0.179945 | +0.012159 / +0.000297 / +0.000390 | 0.897333 | target fail |
| v273b | denoise blend 0.15 on v266c bank | 20.673607 / 0.720203 / 0.152097 | 19.844259 / 0.620205 / 0.179934 | +0.012206 / +0.000295 / +0.000401 | 0.967005 | target fail |

## Interpretation

The denoise mechanism is working mechanically:

- v273a denoised `626926` source slots, about `69.81%` of valid source slots.
- v273a residual energy fell to `89.73%` of the original bank.
- v273b used the same consensus but a weaker blend, so residual energy stayed at `96.70%`.

However, both runs are worse than v266c target exact. The trend is monotonic toward the no-denoise baseline: lower blend reduces damage but does not create a new gain. This is strong evidence that the current source-slot residual noise is not the main bottleneck, or that simple leave-one-out averaging removes useful high-frequency/view-dependent signal along with noise.

## Phase-J Gate

Phase-J flowers reference:

- PSNR `20.304358`
- SSIM `0.557770`
- LPIPS `0.329222`

v273 still fails the Phase-J PSNR gate by roughly `0.460` PSNR, so full9 remains blocked.

## Lesson

v273 is a real residual-target/carrier modification, not a scalar confidence tweak, but it does not solve the gap. The next route should not continue source-slot denoise strength scans. The likely bottleneck is missing view-dependent/high-frequency carrier capacity, not noisy source residuals. Future attempts should focus on:

- a genuinely coherent cross-UV surface feature texture,
- a decoder that aggregates source features rather than only RGB residual slots,
- patch/gradient residual supervision with target-free OOD certificate,
- or a separate visual/edge reconstruction objective that preserves high-frequency residuals instead of averaging them away.

Current status remains `NOT COMPLETE`.
