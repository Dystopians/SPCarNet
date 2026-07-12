# v65 Teacher-Distilled Shared Residual Basis Probe Log

Date: 2026-06-24

Purpose: test a stronger representation-level residual model beyond bin/face threshold tuning by fitting a teacher-distilled shared per-face residual basis from Phase-J evidence.

---

## Motivation

v63/v64 showed that bin-level alpha calibration can safely absorb a small kitchen improvement, but its effect is tiny and does not solve the larger persistent-representation gap. The next meaningful method change was therefore not another threshold tweak, but a higher-capacity residual field:

```text
fit-view Phase-J teacher residuals
  -> per-face shared ridge residual model
  -> view/normal/UV-conditioned prediction
  -> policy-val non-regression guard
  -> target application or fallback
```

The goal was to determine whether a compact shared residual basis can recover some of Phase-J's render-time teacher signal in a persistent surface-addressed representation.

---

## Implementation

Changed files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter interface:

```text
--teacher_distilled_basis_mode {none,face_uv_normal_camera_ridge}
--teacher_distilled_basis_guard_mode {none,policy_val_nonregressive}
--teacher_distilled_basis_min_face_samples
--teacher_distilled_basis_ridge
--teacher_distilled_basis_ood_max_z
--teacher_distilled_basis_ood_min_std
--teacher_distilled_basis_apply_mode {replace_supported,blend,fill_empty_only}
--teacher_distilled_basis_blend
```

Feature basis for `face_uv_normal_camera_ridge`:

```text
[1,
 camera_center_x, camera_center_y, camera_center_z,
 normal_x, normal_y, normal_z,
 dot(normal, camera_center),
 u, v, u^2, v^2, u*v]
```

For each face, v65 solves a ridge model:

```text
residual_rgb ~= X @ W_face
```

and stores fit statistics:

- candidate faces;
- supported faces;
- supported-face fraction;
- feature mean/std for OOD z-score filtering;
- apply mode and blend;
- guard decision and fallback reason.

The runner now forwards all teacher-distilled basis flags and logs the following W&B fields:

```text
policy/teacher_basis_enabled
policy/teacher_basis_guard_fallback
policy/teacher_basis_supported_faces
policy/teacher_basis_supported_face_fraction
policy/teacher_basis_candidate_faces
policy/teacher_basis_blend
```

The atlas writer also supports serializing teacher basis arrays. In the two promoted probe outputs below, the final saved atlas intentionally has no teacher basis arrays because the policy-val guard falls back to the legacy atlas before saving.

---

## Validation

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

CLI checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py --help | rg 'teacher_distilled'

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py --help | rg 'teacher_distilled'
```

Trailing whitespace check:

```bash
rg -n "[ \t]+$" \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py || true
```

---

## Probe Runs

Output root:

```text
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624
```

W&B runs:

| scene | GPU | W&B run | status |
|---|---:|---|---|
| kitchen | 5 | `zrqz5kzw` | completed |
| room | 4 | `bfnmewgo` | completed |

Both runs used W&B online logging and mid-occupied GPUs selected from `nvidia-smi`.

Shared command pattern:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <scene> \
  --gpu <gpu> \
  --output_root /dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624 \
  --tag v65_teacher_shared_faceuv_blend05_min512_ridge001_oodz3_bin_alpha_max035_support4096_tex16_nearest_region_texture_adapter \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 16 \
  --view_conditioned_basis_ridge 0.1 \
  --view_conditioned_basis_ood_mode diag_z \
  --view_conditioned_basis_ood_max_z 2.5 \
  --view_conditioned_basis_ood_min_std 0.05 \
  --teacher_distilled_basis_mode face_uv_normal_camera_ridge \
  --teacher_distilled_basis_guard_mode policy_val_nonregressive \
  --teacher_distilled_basis_min_face_samples 512 \
  --teacher_distilled_basis_ridge 0.01 \
  --teacher_distilled_basis_ood_max_z 3.0 \
  --teacher_distilled_basis_ood_min_std 0.05 \
  --teacher_distilled_basis_apply_mode blend \
  --teacher_distilled_basis_blend 0.5 \
  --enable_policy_val_bin_alpha_calibration \
  --bin_alpha_calibration_max_alpha 0.35 \
  --bin_alpha_calibration_min_alpha 0.0 \
  --bin_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --bin_alpha_calibration_min_bin_samples 32 \
  --bin_alpha_calibration_min_positive_view_fraction 0.5 \
  --bin_alpha_calibration_shrink_count_tau 128 \
  --bin_alpha_calibration_shrink_denominator_tau 0.0 \
  --bin_alpha_calibration_shrink_prior fallback \
  --bin_alpha_calibration_max_profile_bins 8192 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v65_teacher_shared_probe \
  --wandb_run_name v65_teacher_shared_<scene>_20260624 \
  --wandb_mode online \
  --force
```

---

## Results

### Kitchen

Artifacts:

```text
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624/kitchen_v65_teacher_shared_faceuv_blend05_min512_ridge001_oodz3_bin_alpha_max035_support4096_tex16_nearest_region_texture_adapter/results.json
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624/kitchen_v65_teacher_shared_faceuv_blend05_min512_ridge001_oodz3_bin_alpha_max035_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624/logs/apply_metrics_kitchen.log
```

Metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v52/v56 reference | 27.818935 | 0.876535 | 0.199019 |
| v64 selected | 27.822626 | 0.876538 | 0.198849 |
| v65 probe | 27.822626 | 0.876538 | 0.198849 |

Deltas:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v65 vs v52/v56 | +0.003691 | +0.000003 | -0.000171 |
| v65 vs v64 | +0.000000 | +0.000000060 | -0.000000015 |

Teacher basis audit:

| field | value |
|---|---:|
| requested mode | `face_uv_normal_camera_ridge` |
| effective mode | `none` |
| guard decision | `fallback_to_legacy` |
| supported faces | `782` |
| candidate faces | `4343` |
| supported-face fraction | `0.180060` |
| selected alpha | `1.0` |
| changed fraction | `0.039585` |
| bin-alpha count | `67` |

Fallback reasons:

```text
ssim_gain 0.00033229 < legacy 0.00036997
image_l1_gain 0.00004583 < legacy 0.00004735
image_l1_cvar20_view_gain 0.00001873 < legacy 0.00002198
image_l1_min_view_gain 0.00001412 < legacy 0.00001479
```

Interpretation:

v65 does not improve kitchen beyond v64. The teacher basis is fit on many faces, but the train-policy-val non-regression guard correctly finds that the shared basis is weaker than the legacy atlas and falls back.

### Room

Artifacts:

```text
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624/room_v65_teacher_shared_faceuv_blend05_min512_ridge001_oodz3_bin_alpha_max035_support4096_tex16_nearest_region_texture_adapter/results.json
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624/room_v65_teacher_shared_faceuv_blend05_min512_ridge001_oodz3_bin_alpha_max035_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624/logs/apply_metrics_room.log
```

Metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v52/v56/v64 reference | 28.740660 | 0.884829 | 0.249897 |
| v65 probe | 28.739618 | 0.884807 | 0.249906 |

Deltas:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v65 vs v52/v56/v64 | -0.001041 | -0.000022 | +0.000008 |

Teacher basis audit:

| field | value |
|---|---:|
| requested mode | `face_uv_normal_camera_ridge` |
| effective mode | `none` |
| guard decision | `fallback_to_legacy` |
| supported faces | `227` |
| candidate faces | `1160` |
| supported-face fraction | `0.195690` |
| selected alpha | `0.125` |
| changed fraction | `0.010602` |
| bin-alpha count | `177` |

Fallback reasons:

```text
ssim_gain 0.00002179 < legacy 0.00002352
image_l1_cvar20_view_gain -0.00000027 < legacy 0.00000010
image_l1_min_view_gain -0.00000144 < legacy -0.00000121
```

Interpretation:

The teacher basis again fails the non-regression guard. The final candidate is accepted by the local adapter policy but is worse than the stronger v64/v56 reference on held-out test, so it must not be promoted. This validates the need for an outer v64-style fixed auto policy before any full9 promotion.

---

## Key Finding

v65 is a real method/interface upgrade, but the tested linear shared per-face teacher basis is not strong enough:

- it can fit hundreds of supported faces;
- the guard detects that it underperforms the legacy atlas on policy-val;
- final saved atlases correctly remove teacher basis arrays after fallback;
- kitchen stays at v64 quality;
- room becomes a negative candidate if promoted directly.

Therefore:

> Do not promote v65 teacher shared basis as a paper endpoint. Keep v64 as the current fixed-policy representation result and treat v65 as a negative diagnostic showing that a linear shared per-face basis is too weak for Phase-J teacher distillation.

---

## Lesson

The failure mode is not just insufficient support. Kitchen has `782` supported teacher-basis faces and room has `227`, yet the shared basis is weaker than the legacy atlas. This suggests the model class is too rigid:

- a single linear face-level mapping blurs UV-local residual structure;
- camera/normal/UV features are not enough to resolve occlusion-boundary and high-frequency appearance residuals;
- policy-val prefers the legacy bin/atlas residual because it preserves localized residual magnitude better;
- stronger teacher distillation likely needs either local basis mixtures, uncertainty-predicted gating, or a small learned residual field with explicit held-out-safe regularization.

---

## Next Commands

Recommended next experiment is not full9 v65 promotion. First test a local mixture or confidence-predicted residual field on `kitchen`, `room`, and `counter`, with v64 fixed policy as outer fallback.

Minimum follow-up:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 5 \
  --output_root /dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624 \
  --tag v65_teacher_shared_faceuv_blend05_min512_ridge001_oodz3_bin_alpha_max035_support4096_tex16_nearest_region_texture_adapter \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 16 \
  --view_conditioned_basis_ridge 0.1 \
  --view_conditioned_basis_ood_mode diag_z \
  --view_conditioned_basis_ood_max_z 2.5 \
  --view_conditioned_basis_ood_min_std 0.05 \
  --teacher_distilled_basis_mode face_uv_normal_camera_ridge \
  --teacher_distilled_basis_guard_mode policy_val_nonregressive \
  --teacher_distilled_basis_min_face_samples 512 \
  --teacher_distilled_basis_ridge 0.01 \
  --teacher_distilled_basis_ood_max_z 3.0 \
  --teacher_distilled_basis_ood_min_std 0.05 \
  --teacher_distilled_basis_apply_mode blend \
  --teacher_distilled_basis_blend 0.5 \
  --enable_policy_val_bin_alpha_calibration \
  --bin_alpha_calibration_max_alpha 0.35 \
  --bin_alpha_calibration_min_alpha 0.0 \
  --bin_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --bin_alpha_calibration_min_bin_samples 32 \
  --bin_alpha_calibration_min_positive_view_fraction 0.5 \
  --bin_alpha_calibration_shrink_count_tau 128 \
  --bin_alpha_calibration_shrink_denominator_tau 0.0 \
  --bin_alpha_calibration_shrink_prior fallback \
  --bin_alpha_calibration_max_profile_bins 8192 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v65_teacher_shared_probe \
  --wandb_run_name v65_teacher_shared_counter_20260624 \
  --wandb_mode online \
  --force
```

But because both kitchen and room already forced fallback, the more valuable next method change is v66:

```text
v66: UV-local mixture residual field or uncertainty-predicted per-bin residual magnitude,
     trained only from fit/policy-val evidence,
     with v64 fixed policy as the outer promotion gate.
```

