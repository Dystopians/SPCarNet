# v59 Surface-Aware View-Conditioned Residual Basis Log

Date: 2026-06-24

Status: `COMPLETED_DIAGNOSTIC_PROBE_NOT_PROMOTED`.

## Motivation

v58 proved that a camera-center-only residual basis is not enough:

- `texture_size=16` raised active basis support to `11.5% / 15.3%` on `counter / kitchen`;
- higher support still did not protect held-out SSIM;
- the missing signal is likely surface/view interaction rather than only support density.

v59 therefore upgrades the view-conditioned residual basis from a per-view camera-center feature to a per-pixel surface-aware feature.

## Implementation

Updated files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New basis mode:

```text
--view_conditioned_basis_mode normal_camera_linear
```

Feature per valid evidence pixel:

```text
[1,
 normalized_camera_center_x,
 normalized_camera_center_y,
 normalized_camera_center_z,
 normal_x,
 normal_y,
 normal_z,
 dot(normal, normalized_camera_center)]
```

This uses fields already present in the evidence `.npz` files:

```text
camera_center
normal
face_id
barycentric
barycentric_valid
```

Unsupported or low-support face/UV bins still fall back to the legacy mean residual atlas.

## Safety Guard

v59 also adds a train-only basis fallback guard:

```text
--view_conditioned_basis_guard_mode policy_val_nonregressive
```

For each policy candidate, the adapter evaluates:

1. the requested view-conditioned basis atlas;
2. the same atlas with the basis disabled, i.e. the legacy mean residual atlas.

The basis is kept only if it is non-regressive relative to the mean atlas on the same policy-val metrics used by the atlas selector:

- relative residual gain;
- mean SSIM gain;
- image L1 gain;
- CVaR20 view relative gain;
- min-view relative gain;
- image-L1 CVaR20 gain;
- image-L1 min-view gain.

If the basis fails this train-only comparison and the legacy mean atlas is accepted, the candidate falls back to the mean atlas before held-out test rendering. The audit records:

```text
view_conditioned_basis.guard.decision
view_conditioned_basis.effective_mode
```

## Running Probe

### Counter

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v59_normal_camera_guard_counter_20260624 \
  --tag v59_normal_camera_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 16 \
  --view_conditioned_basis_ridge 0.1 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v59_surface_aware_view_basis_probe \
  --wandb_run_name v59_normal_camera_guard_counter_20260624 \
  --wandb_mode online \
  --force
```

W&B:

```text
counter: oiwl6r88
```

Output:

```text
/dev/shm/peilincai_spcarnet_v59_normal_camera_guard_counter_20260624/counter_v59_normal_camera_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter
```

### Kitchen

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene kitchen \
  --gpu 3 \
  --output_root /dev/shm/peilincai_spcarnet_v59_normal_camera_guard_kitchen_20260624 \
  --tag v59_normal_camera_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 16 \
  --view_conditioned_basis_ridge 0.1 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v59_surface_aware_view_basis_probe \
  --wandb_run_name v59_normal_camera_guard_kitchen_20260624 \
  --wandb_mode online \
  --force
```

W&B:

```text
kitchen: 08bwukw3
```

Output:

```text
/dev/shm/peilincai_spcarnet_v59_normal_camera_guard_kitchen_20260624/kitchen_v59_normal_camera_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter
```

## Completion Criteria

Do not promote v59 unless it improves at least one diagnostic scene against the current representation-level reference without opening a new held-out SSIM/LPIPS regression.

Reference rows:

| scene | reference | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| counter | v52 | `26.7534599304` | `0.8621146679` | `0.2518683374` |
| counter | v56/raw v55d | `26.7561302185` | `0.8621262312` | `0.2516913712` |
| counter | v57a | `26.7550086975` | `0.8621243238` | `0.2517508566` |
| kitchen | v52 | `27.8189353943` | `0.8765353560` | `0.1990194172` |
| kitchen | raw v55d | `27.8234424591` | `0.8764375448` | `0.1987804621` |
| kitchen | v57a | `27.8229331970` | `0.8764361739` | `0.1988590807` |

## Results

Both probes completed with W&B online logging.

| scene | W&B run | PSNR | SSIM | LPIPS | accepted | changed fraction | effective basis | guard decision | supported-bin fraction |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| counter | `oiwl6r88` | `26.7536087036` | `0.8621008992` | `0.2518228889` | `true` | `0.065630` | `normal_camera_linear` | `keep_view_basis` | `0.013339` |
| kitchen | `08bwukw3` | `27.8192043304` | `0.8765304089` | `0.1990223229` | `true` | `0.039585` | `normal_camera_linear` | `keep_view_basis` | `0.012640` |

Comparison against the current representation-level references:

| scene | reference | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | `+0.0001487732` | `-0.0000137687` | `-0.0000454485` | mixed: PSNR/LPIPS up, SSIM down |
| counter | v56/raw v55d | `-0.0025215149` | `-0.0000253320` | `+0.0001315177` | worse than current guarded face-alpha reference |
| counter | v57a | `-0.0013999939` | `-0.0000234246` | `+0.0000720323` | worse than shrink probe |
| counter | v58c | `+0.0008277893` | `+0.0001407862` | `+0.0000927151` | PSNR/SSIM recover vs v58c, LPIPS worse |
| kitchen | v52 | `+0.0002689361` | `-0.0000049471` | `+0.0000029057` | not strict: SSIM/LPIPS slightly worse |
| kitchen | raw v55d | `-0.0042381287` | `+0.0000928641` | `+0.0002418608` | SSIM up, PSNR/LPIPS worse |
| kitchen | v57a | `-0.0037288666` | `+0.0000942350` | `+0.0001632422` | SSIM up, PSNR/LPIPS worse |
| kitchen | v58c | `-0.0002269745` | `+0.0000873804` | `+0.0000213981` | SSIM up, PSNR/LPIPS worse |

## Verdict

v59 is a real train/eval pipeline change and a useful diagnostic, but it is not promoted as the current best endpoint.

Main lessons:

- The train-only guard kept the surface-aware basis on both diagnostic scenes, and policy-val metrics were non-regressive relative to the legacy mean atlas.
- Held-out gains remained too small and mixed. `counter` still trails the stronger v56/v57 references, while `kitchen` does not strictly beat v52 because SSIM and LPIPS are slightly worse.
- The active basis support is still only about `1.3%` of face/UV bins at `texture_size=16`, much lower than v58c's camera-center min-sample-4 support fraction, because the safer `min_bin_samples=16` threshold leaves most bins on the mean residual fallback.
- This confirms the deeper bottleneck: a linear normal/camera feature is not yet enough to replace the strongest render-time Phase-J repair or the current guarded face-alpha representation line.

Recommended next step:

Do not continue by sweeping only ridge/min-bin parameters. The next representation-level attempt should combine surface-aware view features with a region-level no-regression guard and an uncertainty/OOD detector for target views, then rerun full9 only after counter/kitchen show strict diagnostic improvement.
