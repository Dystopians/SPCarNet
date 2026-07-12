# v60 View-Basis OOD Guard Log

Date: 2026-06-24

Status: `COMPLETED_DIAGNOSTIC_PROBE_NOT_PROMOTED`.

## Motivation

v59 added a surface-aware view-conditioned residual basis:

```text
[1, camera_center_xyz, normal_xyz, dot(normal, camera_center)]
```

It was a real train/eval pipeline change, but the held-out result was mixed:

- `counter`: PSNR/LPIPS slightly improved over v52, but SSIM regressed and it stayed worse than v56/v57 references.
- `kitchen`: tiny PSNR gain over v52, but SSIM/LPIPS did not strictly improve.

The likely failure mode is not only feature expressiveness. A per-bin linear basis can be non-regressive on train policy-val views while still applying to target-view feature configurations that are poorly represented by the fit-view feature distribution.

v60 therefore adds a local out-of-distribution fallback:

> Use the view-conditioned basis only when the target sample's view feature is inside the fit-view feature envelope for that face/UV bin; otherwise fall back to the legacy mean residual atlas.

## Implementation

Updated files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter/runner flags:

```text
--view_conditioned_basis_ood_mode {none,diag_z}
--view_conditioned_basis_ood_max_z 2.5
--view_conditioned_basis_ood_min_std 0.05
```

New per-bin atlas fields:

```text
view_basis_feature_mean
view_basis_feature_std
view_basis_ood_mode
view_basis_ood_max_z
view_basis_ood_min_std
```

The `diag_z` gate computes the diagonal z-score of the target view feature against the fit-view feature distribution for the same face/UV bin. If any non-bias feature exceeds `ood_max_z`, the target sample uses the mean atlas residual instead of the view-conditioned basis prediction.

## Correctness Fixes After Review

The first v60 probe was interrupted and is not valid evidence:

```text
counter W&B interrupted: bfl3zyoz
kitchen W&B interrupted: 49tdgj0a
```

It was stopped after a read-only review found two correctness risks:

1. Basis support and feature statistics must be based on samples with valid view features, not residual `count_grid`.
2. Target `changed_fraction` should count actual non-zero delta pixels, not every geometrically valid pixel.

Fixes applied before clean rerun:

- Added per-bin `view_feature_counts`.
- Basis support now uses `view_feature_counts >= min_bin_samples`.
- Feature mean/std now divide by `view_feature_counts`.
- `changed_fraction` now uses `np.any(abs(delta) > 1e-8, axis=0)`.
- OOD parameter validation only applies when OOD mode is enabled, preserving default compatibility.

Validation passed:

```text
py_compile adapter + runner
adapter --help exposes OOD flags
runner --help exposes OOD flags
synthetic OOD fallback test passed
synthetic missing-feature support test passed
```

## Clean Probe Commands

### Counter

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v60b_basis_ood_counter_20260624 \
  --tag v60b_normal_camera_oodz25_minstd005_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
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
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v60_surface_aware_view_basis_ood_probe \
  --wandb_run_name v60b_ood_counter_20260624 \
  --wandb_mode online \
  --force
```

### Kitchen

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene kitchen \
  --gpu 3 \
  --output_root /dev/shm/peilincai_spcarnet_v60b_basis_ood_kitchen_20260624 \
  --tag v60b_normal_camera_oodz25_minstd005_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
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
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v60_surface_aware_view_basis_ood_probe \
  --wandb_run_name v60b_ood_kitchen_20260624 \
  --wandb_mode online \
  --force
```

## Promotion Criteria

Do not promote v60 unless it does at least one of the following without opening a new held-out SSIM/LPIPS regression:

- strictly improves `counter` against v56/raw v55d;
- strictly improves `kitchen` against v52;
- produces a clear train-policy-val guard fallback that protects held-out metrics better than v59.

Reference rows:

| scene | reference | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| counter | v52 | `26.7534599304` | `0.8621146679` | `0.2518683374` |
| counter | v56/raw v55d | `26.7561302185` | `0.8621262312` | `0.2516913712` |
| counter | v57a | `26.7550086975` | `0.8621243238` | `0.2517508566` |
| counter | v59 | `26.7536087036` | `0.8621008992` | `0.2518228889` |
| kitchen | v52 | `27.8189353943` | `0.8765353560` | `0.1990194172` |
| kitchen | raw v55d | `27.8234424591` | `0.8764375448` | `0.1987804621` |
| kitchen | v57a | `27.8229331970` | `0.8764361739` | `0.1988590807` |
| kitchen | v59 | `27.8192043304` | `0.8765304089` | `0.1990223229` |

## Results

Both clean v60b probes completed with W&B online logging.

| scene | W&B run | PSNR | SSIM | LPIPS | accepted | changed fraction | effective basis | guard decision | supported-bin fraction |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| counter | `d9tozw7s` | `26.7539958954` | `0.8621192575` | `0.2518530488` | `true` | `0.065630` | `normal_camera_linear` | `keep_view_basis` | `0.013339` |
| kitchen | `924sxfsd` | `27.8191566467` | `0.8765332103` | `0.1990308464` | `true` | `0.039585` | `normal_camera_linear` | `keep_view_basis` | `0.012640` |

Comparison against key references:

| scene | reference | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | `+0.0005359650` | `+0.0000045896` | `-0.0000152886` | strict small positive vs v52 |
| counter | v56/raw v55d | `-0.0021343231` | `-0.0000069737` | `+0.0001616776` | still worse than stronger counter reference |
| counter | v57a | `-0.0010128021` | `-0.0000050663` | `+0.0001021922` | still worse than shrink probe |
| counter | v59 | `+0.0003871918` | `+0.0000183583` | `+0.0000301599` | PSNR/SSIM recover, LPIPS worsens |
| kitchen | v52 | `+0.0002212524` | `-0.0000021457` | `+0.0000114292` | not strict: SSIM/LPIPS regress |
| kitchen | raw v55d | `-0.0042858124` | `+0.0000956655` | `+0.0002503843` | SSIM up, PSNR/LPIPS worse |
| kitchen | v57a | `-0.0037765503` | `+0.0000970364` | `+0.0001717657` | SSIM up, PSNR/LPIPS worse |
| kitchen | v59 | `-0.0000476837` | `+0.0000028014` | `+0.0000085235` | mixed vs v59 |

## Verdict

v60 is a real method change and a useful diagnostic, but it is not promoted as the current representation-level best.

Main lessons:

- The corrected OOD guard infrastructure works and the clean W&B probes completed.
- `counter` becomes a strict three-metric small positive over v52, which is better than v59's mixed-vs-v52 result.
- `counter` still trails the stronger v56/raw-v55d and v57a references, so it does not satisfy the counter promotion criterion.
- `kitchen` still does not strictly beat v52 because SSIM and LPIPS regress, so it does not satisfy the kitchen promotion criterion.
- The train-policy-val guard kept the view basis on both scenes rather than falling back to the mean atlas, so v60 did not produce a protective fallback story.

Recommended next step:

Do not continue with a pure z-threshold sweep as the main path. The next representation-level attempt should add region-level no-regression or uncertainty aggregation above the per-bin OOD gate, because the current per-bin diagonal OOD fallback is too weak to change held-out promotion decisions.
