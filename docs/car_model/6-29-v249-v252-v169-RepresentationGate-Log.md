# SPCarNet v169 Representation Gate Log

Date: 2026-06-29

This log records the execution of `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` after the v249-v252 runs.  The key constraint was followed: no full9 run was launched because flowers did not pass the Phase-J all-axis gate.

## v169 Gate

Phase-J flowers exact reference:

| metric | required candidate relation | Phase-J |
|---|---:|---:|
| PSNR | higher | 20.304358 |
| SSIM | higher | 0.557770 |
| LPIPS | lower | 0.329222 |

Current status: **FAIL for promotion**.  No v249-v252 candidate produced a policy-val all-axis certificate strong enough to justify flowers exact promotion, so full9 remains blocked.

## Code Changes

Implemented in `scripts/car_model/train_surface_conditioned_residual_unet.py`:

- Added `--teacher_benefit_mask_mode {off,teacher_better,positive_gain,better_and_positive_gain}`.
- Added `--teacher_benefit_min_gain_l1` and `--teacher_benefit_dilate`.
- The teacher-benefit mask is train-fit only.  It uses `teacher_better_mask` and/or `teacher_gain_l1` from the Phase-J teacher evidence and blends target supervision back to parent/no-op outside the mask.
- Added `--policy_allow_noop_alpha`; by default v169 runs now exclude `alpha=0` from policy best selection while still logging alpha-0 rows.
- Added Phase-J flowers numeric reference fields to the report.  The report explicitly says that policy-val numeric comparison is not an official Phase-J exact win.
- Saved `surface_evidence_stats` into checkpoints so standalone checkpoint apply can rebuild the same surface-evidence model.

Implemented in supporting scripts before this log:

- `scripts/car_model/audit_surface_checkpoint_residual_projection.py` now supports alpha-conditioned residual audits correctly.
- `scripts/car_model/train_surface_residual_memory_texture.py` reports tail/min/CVaR metrics and blocks target exact when all-axis/tail gates fail.

## Storage And Runtime Preflight

At run start:

| mount / resource | state |
|---|---|
| `/data` | about 136G available |
| `/dev/shm` | about 1.4G available |
| `/tmp` user quota | about 92G / 100G used |
| GPUs | GPU1 and GPU3 were selected for v252 because they were low-use |

Decision: run only flowers policy-val and projection audits under `/tmp`; do not duplicate evidence caches and do not launch full9.

## Experiments

### v249a: LPIPS No-Harm GT-Assisted U-Net

Artifacts:

- report: `/tmp/peilincai_spcarnet_v249_lpips_noharm_flowers_20260629/v249a_lpips_noharm_gtassist_native1256/v249a_lpips_noharm_gtassist_flowers_report.json`
- projection audit: `/tmp/peilincai_spcarnet_v249_lpips_noharm_flowers_20260629/v249a_projection_audit.json`
- W&B offline: `/tmp/peilincai_spcarnet_v249_lpips_noharm_flowers_20260629/v249a_lpips_noharm_gtassist_native1256/wandb/offline-run-20260629_182205-fgtuxux1`

| alpha | PSNR gain | SSIM gain | LPIPS gain | min SSIM gain | min LPIPS gain | changed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | +0.027357 | +0.000589 | +0.000250 | -0.000152 | -0.001432 | 0.246711 |

Projection: energy retention `0.020147`, cosine `0.127558`.

Verdict: mean gains exist, but LPIPS/SSIM tails fail.  This is not enough for v169.

### v250: Surface Residual Memory Texture

Artifacts:

- v250a report: `/tmp/peilincai_spcarnet_v250_surface_memory_flowers_20260629/v250a_edge_confidence_memory/v223_surface_residual_memory_texture_audit.json`
- v250b report: `/tmp/peilincai_spcarnet_v250_surface_memory_flowers_20260629/v250b_rawrgb_memory/v223_surface_residual_memory_texture_audit.json`

| run | alpha | PSNR gain | SSIM gain | LPIPS gain | active energy retention | active cosine |
|---|---:|---:|---:|---:|---:|---:|
| v250a edge/confidence memory | 0.125 | +0.007847 | -0.000152 | -0.000019 | 0.048002 | 0.284919 |
| v250b raw-RGB memory | 0.125 | +0.007915 | -0.000107 | -0.000004 | 0.031182 | 0.295237 |

Verdict: memory prototypes improve local active residual projection more than v249, but they still damage structure/perceptual GT metrics.  This confirms that simple nearest/prototype memory is not a sufficient v169 representation.

### v251: Low-Rank / Surface Feature Texture

Artifacts:

- v251a report: `/tmp/peilincai_spcarnet_v251_lowrank_confidence_flowers_20260629/v251a_lowrank_k4_confidence/v251a_lowrank_k4_confidence_flowers_report.json`
- v251b report: `/tmp/peilincai_spcarnet_v251_lowrank_confidence_flowers_20260629/v251b_surface_unet_confidence_evidence/v251b_surface_unet_confidence_evidence_flowers_report.json`
- W&B offline:
  - `/tmp/peilincai_spcarnet_v251_lowrank_confidence_flowers_20260629/v251a_lowrank_k4_confidence/wandb/offline-run-20260629_184533-jsymhpm0`
  - `/tmp/peilincai_spcarnet_v251_lowrank_confidence_flowers_20260629/v251b_surface_unet_confidence_evidence/wandb/offline-run-20260629_184559-biys8twv`

Both runs selected `alpha=0` under the strict tail guard.  The teacher signal was strong on policy-val:

| teacher gain | value |
|---|---:|
| PSNR | +0.913279 |
| SSIM | +0.065512 |
| LPIPS | +0.017600 |

Verdict: the failure is not missing teacher signal.  The learned low-rank and surface feature carriers were too conservative or too unstable, so the strict policy preferred no-op.

### v252: Teacher-Benefit Masked Distillation

Motivation:

v251 showed that nonzero residuals have tail risk.  v252 therefore added a train-fit-only teacher-benefit mask: learn Phase-J residual only where teacher evidence says Phase-J improves over the parent; train all other pixels toward parent/no-op.  This was intended to preserve perceptual/SSIM tails.

Commands:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
WANDB_DIR=/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252a_lowrank_benefit_mask/wandb \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_conditioned_residual_unet.py \
  --model_type lowrank_surface_texture \
  --teacher_benefit_mask_mode better_and_positive_gain \
  --teacher_residual_cosine_weight 0.08 \
  --teacher_residual_energy_weight 0.25 \
  --compute_lpips --enable_wandb \
  --output_dir /tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252a_lowrank_benefit_mask
```

```bash
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline \
WANDB_DIR=/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252b_surface_unet_benefit_mask/wandb \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_conditioned_residual_unet.py \
  --model_type surface_texture_unet \
  --enable_surface_evidence_texture \
  --teacher_benefit_mask_mode better_and_positive_gain \
  --teacher_residual_cosine_weight 0.06 \
  --teacher_residual_energy_weight 0.20 \
  --compute_lpips --enable_wandb \
  --output_dir /tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252b_surface_unet_benefit_mask
```

Results:

| run | alpha | PSNR gain | SSIM gain | LPIPS gain | min PSNR | min SSIM | min LPIPS | changed | teacher PSNR recovery | teacher SSIM recovery | teacher LPIPS recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v252a low-rank benefit | 0.0625 | +0.000094 | +0.000002 | +0.000002 | -0.000002 | -0.000000 | -0.000007 | 0.000369 | 0.000102 | 0.000034 | 0.000157 |
| v252b surface U-Net benefit | 0.0625 | +0.000382 | +0.000011 | +0.000004 | -0.000406 | -0.000009 | -0.000046 | 0.003078 | 0.000447 | 0.000181 | 0.000358 |

Projection audits:

```bash
CUDA_VISIBLE_DEVICES=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/audit_surface_checkpoint_residual_projection.py \
  --run_name v252a_lowrank_benefit_mask \
  --checkpoint /tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252a_lowrank_benefit_mask/v252a_lowrank_benefit_mask_flowers.pt \
  --alpha 0.0625 --compute_lpips \
  --output_json /tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252a_lowrank_benefit_mask/v252a_projection_audit.json
```

```bash
CUDA_VISIBLE_DEVICES=3 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/audit_surface_checkpoint_residual_projection.py \
  --run_name v252b_surface_unet_benefit_mask \
  --checkpoint /tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252b_surface_unet_benefit_mask/v252b_surface_unet_benefit_mask_flowers.pt \
  --alpha 0.0625 --compute_lpips \
  --output_json /tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252b_surface_unet_benefit_mask/v252b_projection_audit.json
```

| run | energy retention | cosine | interpretation |
|---|---:|---:|---|
| v252a low-rank benefit | 0.000019 | 0.021462 | effectively no teacher residual is carried |
| v252b surface U-Net benefit | 0.000158 | 0.026398 | still less than 0.02 percent teacher residual energy |

Verdict: teacher-benefit masking reduced damage, but collapsed useful residual magnitude.  It turned the candidate into a near-no-op and did not solve the carrier bottleneck.

## No-GT Status

The train/eval pipeline keeps the intended separation:

- Train-fit can use teacher evidence and, for v252, train-fit teacher-benefit masks derived from teacher-vs-parent GT comparison.
- Policy-val uses GT only for certification and alpha/policy choice.
- Target/test apply is skipped unless policy-val all-axis passes.
- v252 target/test apply was skipped, so no target/test RGB GT leakage occurred.

## Main Bottleneck

The evidence is now consistent across v249-v252:

1. Phase-J teacher signal is strong on held-out policy-val.
2. A broad GT-assisted U-Net can move images, but its tails are unsafe.
3. Prototype memory and surface textures can write locally aligned residuals, but they hurt SSIM/LPIPS tails.
4. Low-rank/surface-feature carriers with learned confidence become too conservative.
5. Teacher-benefit masking prevents large failures but makes residual energy nearly zero.

Therefore, the current MeshSplatting-compatible baked RGB residual carrier is not strong enough to distill Phase-J into a reliable all-axis improvement.  More alpha scans, face gates, or support thresholds are unlikely to cross the v169 gate.

## Recommendation

Do not run full9 from v249-v252.

The next real research step should change representation class, not parameters.  The most plausible next route is a train-view source feature bank or deferred neural surface renderer:

- keep MeshSplatting as the parent geometry/rasterizer;
- attach compact train-view evidence to surface bins;
- at render time, aggregate source-view residual/features conditioned on target view direction and parent buffers;
- predict residual and confidence with a small view-dependent decoder;
- certify on flowers policy-val before any target/test exact or full9.

This is a larger representation-level change than v252.  It directly attacks the measured bottleneck: current carriers retain too little teacher residual energy and lose alignment under conservative no-harm gates.

## Artifact Index

- summary JSON: `docs/car_model/results/v249_v252_v169_representation_gate_summary.json`
- v252a report: `/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252a_lowrank_benefit_mask/v252a_lowrank_benefit_mask_flowers_report.json`
- v252a projection audit: `/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252a_lowrank_benefit_mask/v252a_projection_audit.json`
- v252b report: `/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252b_surface_unet_benefit_mask/v252b_surface_unet_benefit_mask_flowers_report.json`
- v252b projection audit: `/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252b_surface_unet_benefit_mask/v252b_projection_audit.json`
- v252a W&B offline: `/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252a_lowrank_benefit_mask/wandb/offline-run-20260629_191224-df1jgnyj`
- v252b W&B offline: `/tmp/peilincai_spcarnet_v252_teacher_benefit_flowers_20260629/v252b_surface_unet_benefit_mask/wandb/offline-run-20260629_191247-tc42xxty`

Final status: **NOT COMPLETE for paper-level all-axis win**.  v169 diagnostic completion standard B is satisfied for the current carrier family, but the broader paper goal still requires a stronger representation.
