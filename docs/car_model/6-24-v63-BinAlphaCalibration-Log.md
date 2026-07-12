# v63 Bin-Level Alpha Calibration Log

Date: 2026-06-24

Purpose: document the v63/v63b attempt to fix the v61/v62 failure mode by calibrating local residual magnitude instead of only shrinking the apply mask.

---

## Motivation

v61 and v62 showed a clear bottleneck:

- face/bin-level allowlists can reduce target changed area;
- but small changed area still does not guarantee held-out improvement;
- the failure is not only “where to apply residual” but also “how much residual to apply”.

v63 changes the residual atlas from a binary apply/no-apply rule into a local magnitude-calibrated residual field.

---

## Method Change

Implemented in:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New mode:

```text
policy_val_bin_alpha
```

For each reliable `(face_id, uv_bin)`, the adapter estimates a local residual alpha from train policy-val evidence:

```text
alpha_bin = argmin_alpha || residual_gt - alpha * residual_pred ||^2
```

The raw least-squares alpha is then guarded by:

- `max_alpha` / `min_alpha`;
- minimum bin samples;
- positive-view fraction;
- global fallback alpha;
- count shrink;
- denominator shrink;
- maximum profile-bin count.

New CLI flags:

```text
--enable_policy_val_bin_alpha_calibration
--bin_alpha_calibration_max_alpha
--bin_alpha_calibration_min_alpha
--bin_alpha_calibration_multipliers
--bin_alpha_calibration_min_bin_samples
--bin_alpha_calibration_min_denominator
--bin_alpha_calibration_min_positive_view_fraction
--bin_alpha_calibration_shrink_count_tau
--bin_alpha_calibration_shrink_denominator_tau
--bin_alpha_calibration_shrink_prior
--bin_alpha_calibration_max_profile_bins
```

W&B/audit additions:

```text
policy/local_alpha_mode_bin
policy/bin_alpha_count
policy/bin_alpha_candidate_bins
policy/fallback_bin_count
```

---

## Static and Smoke Validation

Completed:

- adapter and runner `py_compile` passed;
- adapter/runner `--help` exposes the new bin-alpha flags;
- synthetic smoke test passed: same face, different UV bins learned different local alpha and applied different target residual magnitudes.

---

## Commands

Counter v63:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 5 \
  --output_root /dev/shm/peilincai_spcarnet_v63_bin_alpha_counter_20260624 \
  --tag v63_bin_alpha_min32_pos05_shrink128_profile8192_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter \
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
  --enable_policy_val_bin_alpha_calibration \
  --bin_alpha_calibration_max_alpha 0.5 \
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
  --wandb_group v63_bin_alpha_calibration_probe \
  --wandb_run_name v63_bin_alpha_counter_20260624 \
  --wandb_mode online \
  --force
```

Kitchen v63 used the same settings with `--scene kitchen`.

v63b changed only the bin alpha cap:

```text
--bin_alpha_calibration_max_alpha 0.35
```

---

## Results

| scene | version | W&B | PSNR | SSIM | LPIPS | accepted | selected alpha | changed frac. | bin alpha count | verdict |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| counter | v63 max0.5 | `g4ub52pr` | 26.752016 | 0.862093 | 0.251934 | true | 0.125 | 0.065630 | 664 | better than v61/v62, worse than v52/v56/v60 |
| kitchen | v63 max0.5 | `hu5k9lyu` | 27.823883 | 0.876437 | 0.198897 | true | 1.0 | 0.039585 | 75 | PSNR/LPIPS better than v52/v60, SSIM slightly worse |
| counter | v63b max0.35 | `rlctknlk` | 26.751209 | 0.862078 | 0.251961 | true | 0.125 | 0.065630 | 667 | better than v61/v62, still worse than v52/v56/v60 |
| kitchen | v63b max0.35 | `tyqm9u38` | 27.822626 | 0.876538 | 0.198849 | true | 1.0 | 0.039585 | 77 | strict three-metric improvement over v52 and v60 |

v63b kitchen deltas:

| Reference | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| vs v52 | +0.003691 | +0.0000025 | -0.000171 |
| vs v60 | +0.003469 | +0.0000046 | -0.000182 |
| vs v61 | +0.005329 | +0.0000523 | -0.000290 |
| vs v62 | +0.006237 | +0.0000950 | -0.000352 |

v63b counter deltas:

| Reference | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| vs v52 | -0.002251 | -0.0000370 | +0.000093 |
| vs v56/raw v55d | -0.004921 | -0.0000485 | +0.000270 |
| vs v60 | -0.002787 | -0.0000415 | +0.000108 |
| vs v62 | +0.001326 | +0.0000279 | -0.000035 |

---

## Result Paths

```text
/dev/shm/peilincai_spcarnet_v63_bin_alpha_counter_20260624/counter_v63_bin_alpha_min32_pos05_shrink128_profile8192_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter
/dev/shm/peilincai_spcarnet_v63_bin_alpha_kitchen_20260624/kitchen_v63_bin_alpha_min32_pos05_shrink128_profile8192_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_counter_20260624/counter_v63b_bin_alpha_max035_min32_pos05_shrink128_profile8192_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_kitchen_20260624/kitchen_v63b_bin_alpha_max035_min32_pos05_shrink128_profile8192_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter
```

---

## Verdict

v63/v63b is a real method change, not a report-only edit.

The positive finding is meaningful:

- v63b kitchen is the first strict three-metric win over v52/v60 in this representation-level chain;
- the local alpha profile gives a principled way to calibrate residual magnitude per surface bin.

The blocker is also clear:

- counter still does not beat v52/v56/v60;
- enabling v63b everywhere would not be a safe promotion;
- the next policy must automatically select v63b only when train policy-val evidence is strong enough, and otherwise fallback to v52/v56.

Do not claim v63/v63b as a global endpoint yet. Use it as the latest research milestone and as evidence that magnitude calibration is the correct next axis.
