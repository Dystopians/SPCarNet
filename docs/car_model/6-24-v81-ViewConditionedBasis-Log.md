# v81 View-Conditioned Residual Basis Diagnostic Log

Date: 2026-06-24

Status: `COMPLETED_NEGATIVE_NOT_PROMOTED`

## Purpose

v65-v80 suggested that the remaining bottleneck is no longer only alpha, blend, support cap, or target-footprint auditing. v81 therefore tests a representation-level change with a small view-conditioned residual basis:

```text
view_conditioned_basis_mode = normal_camera_linear
```

The intent is to let each surface bin fit residual variation as a function of normal/camera-view features, rather than using a purely view-agnostic RGB residual atlas.

## Command

```bash
WANDB_DIR=/dev/shm/wandb_spcarnet_v81 WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v81_viewbasis_20260624 \
  --tag v81_viewbasis_tex32_support4096_counter_region_texture_adapter \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 4 \
  --view_conditioned_basis_ridge 0.1 \
  --view_conditioned_basis_ood_mode diag_z \
  --view_conditioned_basis_ood_max_z 2.5 \
  --view_conditioned_basis_ood_min_std 0.05 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.001 \
  --wandb_project SPCarNet \
  --wandb_group v81_viewbasis \
  --wandb_run_name v81_viewbasis_counter_20260624 \
  --wandb_mode online --force
```

W&B run: `q6v1qvz4`.

## Artifacts

Persistent small artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/
```

Key files:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/counter/apply_metrics_counter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/counter/run_counter.log
```

Large checkpoint/render artifacts were left in `/dev/shm` because `/data` had less than 0.5 GB free.

## Result

| row | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v81 view-conditioned basis | `26.753919601` | `0.862121582` | `0.251836061` |
| v79/v56/v64 counter anchor | `26.756130219` | `0.862126231` | `0.251691371` |
| v81 - anchor | `-0.002210618` | `-0.000004649` | `+0.000144690` |
| v80 face-alpha hybrid local-patch | `26.756135941` | `0.862126231` | `0.251691461` |
| v81 - v80 | `-0.002216340` | `-0.000004649` | `+0.000144600` |

## Audit Summary

- Accepted: `true`
- Effective policy: `accepted_atlas`
- Selected alpha: `0.125`
- Selected support added faces: `4096`
- Selected texture size: `32`
- Selected fill mode: `nearest_observed`
- View-conditioned basis mode: `normal_camera_linear`
- View-conditioned basis guard decision: `keep_view_basis`
- View-conditioned basis supported bins: `116171`
- View-conditioned basis supported-bin fraction: `0.020009`
- Policy-val relative gain: `0.026660`
- Policy-val positive-view fraction: `1.000000`
- Policy-val image SSIM gain: `0.000194982`
- Policy-val image L1 positive-view fraction: `1.000000`
- Target changed fraction: `0.065362`

## Decision

v81 is not promoted. The mechanism is active and passes policy-val gates, but held-out metrics regress on all three RGB axes versus the v79/v56/v64 anchor and also underperform v80.

The useful lesson is that a simple per-bin normal/camera linear basis is not enough. The next representation-level attempt should either add stronger uncertainty/coverage certificates for the basis or move to a richer residual model such as local mixture/teacher-distilled patch fields with stricter held-out-safe gating.

