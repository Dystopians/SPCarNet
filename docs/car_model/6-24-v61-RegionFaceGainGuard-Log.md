# v61 Region-Level Face Gain Guard Log

Date: 2026-06-24

Status: `COMPLETED_NEGATIVE_DIAGNOSTIC_NOT_PROMOTED`.

## Motivation

v60 fixed the local view-basis OOD path and completed clean counter/kitchen probes, but it did not satisfy promotion criteria:

- `counter` became a strict small positive over v52, but still trailed v56/raw-v55d.
- `kitchen` remained mixed and did not strictly beat v52 because SSIM/LPIPS regressed.
- The train-policy-val view-basis guard kept the basis on both scenes, so v60 did not produce a protective fallback story.

The next failure mode is region-level risk: a global train-policy-val candidate can be non-regressive overall while a subset of surface faces still hurts held-out structure.

v61 adds a train-policy-val face-level no-regression allowlist:

> Apply residuals on target views only for faces whose active residual prediction reduces policy-val residual error with enough samples and enough positive-view support.

## Implementation

Updated files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter/runner flags:

```text
--enable_policy_val_face_gain_guard
--face_gain_guard_min_face_samples
--face_gain_guard_min_relative_gain
--face_gain_guard_min_positive_view_fraction
```

Core behavior:

1. Fit the same surface atlas / view-conditioned basis candidate as v60.
2. Select alpha through the existing train policy-val risk gate.
3. On policy-val views, aggregate active residual prediction gains per face:
   - before error: `||teacher_residual||^2`;
   - after error: `||teacher_residual - predicted_delta||^2`;
   - per-face relative gain;
   - per-face positive-view fraction.
4. Keep only faces that pass:
   - `samples >= face_gain_guard_min_face_samples`;
   - `relative_gain >= face_gain_guard_min_relative_gain`;
   - `positive_view_fraction >= face_gain_guard_min_positive_view_fraction`.
5. Re-run the normal policy-val selector with the face allowlist active.
6. Apply target residuals only on the allowed faces if the post-guard policy remains accepted.

The guard is disabled by default, so existing v48/v52/v56/v60 replay commands keep their original behavior unless this flag is explicitly enabled.

## Validation Before Probe

```text
py_compile adapter + runner: passed
adapter --help exposes face_gain_guard flags: passed
runner --help exposes face_gain_guard flags: passed
synthetic face allowlist smoke: passed
```

The synthetic smoke confirmed that a face allowlist suppresses deltas on disallowed faces while preserving deltas on allowed faces.

## Probe Commands

### Counter

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 5 \
  --output_root /dev/shm/peilincai_spcarnet_v61_face_gain_guard_counter_20260624 \
  --tag v61_facegain_min128_pos075_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter \
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
  --enable_policy_val_face_gain_guard \
  --face_gain_guard_min_face_samples 128 \
  --face_gain_guard_min_relative_gain 0.0 \
  --face_gain_guard_min_positive_view_fraction 0.75 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v61_region_face_gain_guard_probe \
  --wandb_run_name v61_facegain_counter_20260624 \
  --wandb_mode online \
  --force
```

W&B run: `a9bf3hbb`

### Kitchen

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene kitchen \
  --gpu 1 \
  --output_root /dev/shm/peilincai_spcarnet_v61_face_gain_guard_kitchen_20260624 \
  --tag v61_facegain_min128_pos075_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter \
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
  --enable_policy_val_face_gain_guard \
  --face_gain_guard_min_face_samples 128 \
  --face_gain_guard_min_relative_gain 0.0 \
  --face_gain_guard_min_positive_view_fraction 0.75 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v61_region_face_gain_guard_probe \
  --wandb_run_name v61_facegain_kitchen_20260624 \
  --wandb_mode online \
  --force
```

W&B run: `nhyjuth5`

## Promotion Criteria

Do not promote v61 unless it does at least one of the following without opening a new held-out SSIM/LPIPS regression:

- strictly improves `counter` against v56/raw-v55d;
- strictly improves `kitchen` against v52;
- or gives a clear protective fallback/no-op story that beats v60 on the failing metric while preserving non-regression against the relevant reference.

Reference rows:

| scene | reference | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| counter | v52 | `26.7534599304` | `0.8621146679` | `0.2518683374` |
| counter | v56/raw v55d | `26.7561302185` | `0.8621262312` | `0.2516913712` |
| counter | v60 | `26.7539958954` | `0.8621192575` | `0.2518530488` |
| kitchen | v52 | `27.8189353943` | `0.8765353560` | `0.1990194172` |
| kitchen | v60 | `27.8191566467` | `0.8765332103` | `0.1990308464` |

## Results

Both counter and kitchen probes completed with W&B online logging.

| scene | W&B run | PSNR | SSIM | LPIPS | accepted | selected alpha | changed fraction | allowed / candidate faces | allowed sample fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `a9bf3hbb` | `26.7510070801` | `0.8620729446` | `0.2519522905` | true | `0.125` | `0.0178885698` | `577 / 5628` | `0.3102756523` |
| kitchen | `nhyjuth5` | `27.8172969818` | `0.8764855266` | `0.1991390735` | true | `0.125` | `0.0120642812` | `817 / 4333` | `0.3138086661` |

Reference deltas:

| scene | reference | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | `-0.0024528503` | `-0.0000417233` | `+0.0000839531` | worse |
| counter | v56/raw v55d | `-0.0051231384` | `-0.0000532866` | `+0.0002609193` | worse |
| counter | v60 | `-0.0029888153` | `-0.0000463129` | `+0.0000992417` | worse |
| kitchen | v52 | `-0.0016384125` | `-0.0000498294` | `+0.0001196563` | worse |
| kitchen | v60 | `-0.0018596649` | `-0.0000476837` | `+0.0001082271` | worse |

## Verdict

Do not promote v61.

The face-level allowlist is active and does reduce target changed fraction, but it does not protect held-out RGB metrics. Both probe scenes are worse than their relevant references across all three metrics. This is a useful negative diagnostic: global policy-val non-regression plus per-face positive-gain filtering is still insufficient for robust view-conditioned residual transfer.

Next method direction should not be another threshold sweep of the same guard. The stronger route is an uncertainty-certified surface residual field that models residual magnitude, multi-view consistency, and view-feature support jointly before applying target deltas.
