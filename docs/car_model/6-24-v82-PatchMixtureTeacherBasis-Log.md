# v82 Patch-Mixture Teacher Basis Log

Date: 2026-06-24

## Purpose

v82 tested whether a stronger teacher-distilled residual representation can break the counter-scene plateau around the v56/v64/v79 anchor. Instead of only tuning alpha or support, it added a new per-face teacher basis mode:

```text
face_uv_patch_mixture_ridge
```

The feature vector has 31 dimensions:

```text
[1, camera3, normal3, normal_dot_camera, u, v, u^2, v^2, u*v]
+ 3x3 local UV RBF patch mixture
+ patch mixture * normal_dot_camera
```

## Implementation

Changed files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

The runner now forwards `--teacher_distilled_basis_mode face_uv_patch_mixture_ridge`; the adapter builds the 31D feature vector and keeps the existing `policy_val_nonregressive` guard.

## Result

W&B run: `6subv75i`

| method | PSNR | SSIM | LPIPS | accepted | selected alpha |
|---|---:|---:|---:|---|---:|
| v82 patchmix teacher counter | `26.753459930` | `0.862114668` | `0.251868337` | yes | `0.125` |

Reference:

| reference | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64/v79 counter anchor | `26.756130219` | `0.862126231` | `0.251691371` |
| v80 near-tie hybrid | `26.756135941` | `0.862126231` | `0.251691461` |

## Diagnosis

The new patch-mixture basis was not promoted by the guard. The audit records:

```text
effective_mode: none
guard decision: fallback_to_legacy
reason: ssim_gain 0.00015930 < legacy 0.00016533
```

This is useful evidence: the pipeline can fit and audit a stronger teacher basis, but policy-val correctly detected that this particular basis did not beat the legacy teacher branch. The final accepted atlas still underperforms the strong counter anchor and should not be expanded to hard-triad or full9.

## Persistent Evidence

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_patchmix_teacher_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_patchmix_teacher_20260624/counter_v82_patchmix_teacher_tex32_support4096_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_patchmix_teacher_20260624/counter_v82_patchmix_teacher_tex32_support4096_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
```
