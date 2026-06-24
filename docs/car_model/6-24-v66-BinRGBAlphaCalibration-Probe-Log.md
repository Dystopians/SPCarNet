# v66 Bin-RGB Alpha Calibration Probe Log

Date: 2026-06-24

Purpose: test whether bin-level residual magnitude calibration should be channel-wise RGB instead of a scalar alpha.

---

## Summary

v66 is a real train/eval pipeline change, not a documentation-only update. It adds a local `policy_val_bin_rgb_alpha` profile to the surface residual region texture adapter. The intent is to fit a separate alpha for R/G/B at each reliable face/UV bin, using only train/policy-val evidence, then apply the calibrated RGB residual field to held-out views.

Result: v66 is not promoted.

| scene | reference | v66 PSNR | v66 SSIM | v66 LPIPS | verdict |
|---|---:|---:|---:|---:|---|
| counter | v56/v64: `26.756130 / 0.862126 / 0.251691` | 26.751209 | 0.862078 | 0.251961 | worse than selected policy |
| kitchen | v64: `27.822626 / 0.876538 / 0.198849` | 27.822626 | 0.876538 | 0.198849 | tied with v64 |

The probe confirms that RGB-wise alpha is mechanically valid and W&B-logged, but it does not solve the current bottleneck. The selected method should remain Phase-J for the headline result and v64 as the best fixed representation-level policy.

---

## Implementation

Files changed:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter mode:

```text
local_alpha_profile.mode = policy_val_bin_rgb_alpha
```

New behavior:

- fit per-bin `alpha_rgb = [alpha_r, alpha_g, alpha_b]`;
- require reliable train/policy-val bin support;
- shrink weak bins toward a fallback RGB alpha;
- keep the same policy-val guard and target audit structure;
- handle either scalar `N` local alpha or RGB `N x 3` local alpha at application time;
- reject simultaneous scalar-bin and RGB-bin alpha calibration.

New key flags:

```text
--enable_policy_val_bin_rgb_alpha_calibration
--bin_rgb_alpha_calibration_max_alpha
--bin_rgb_alpha_calibration_min_alpha
--bin_rgb_alpha_calibration_multipliers
--bin_rgb_alpha_calibration_min_bin_samples
--bin_rgb_alpha_calibration_min_denominator
--bin_rgb_alpha_calibration_min_positive_view_fraction
--bin_rgb_alpha_calibration_shrink_count_tau
--bin_rgb_alpha_calibration_shrink_denominator_tau
--bin_rgb_alpha_calibration_shrink_prior
--bin_rgb_alpha_calibration_max_profile_bins
```

Post-probe runner cleanup:

```text
scripts/car_model/run_l1risk_fairnoop_scene.py
```

also exposes:

```text
--support_expansion_mode
--support_expansion_max_extra_faces
```

The default remains `fit_residual_topk` with `2048` extra faces for backward compatibility, but follow-up isolation runs can now set `--support_expansion_mode none` for base-carrier-only controls or set a specific extra-face budget with `--support_expansion_max_extra_faces`.

The adapter now also rejects ambiguous local-alpha configurations. At most one of bucket-local alpha, face alpha, scalar bin alpha, and RGB bin alpha can be enabled in a single run. For RGB bin alpha, channels with weak per-channel denominator now fall back to the configured prior instead of producing a noisy channel alpha through denominator flooring.

---

## Static Validation

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

CLI exposure was verified with:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py --help | rg 'bin_rgb_alpha|bin alpha|RGB alpha'
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py --help | rg 'bin_rgb_alpha'
```

Mutual exclusion check was verified:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  --source_model x --fit_evidence_dir x --target_evidence_dir x --region_carrier_json x --output_model x \
  --enable_policy_val_bin_alpha_calibration --enable_policy_val_bin_rgb_alpha_calibration
```

Expected result:

```text
error: enable either scalar bin alpha or RGB bin alpha calibration, not both
```

---

## Probe Commands

Output root:

```text
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624
```

W&B:

| scene | GPU | W&B run | status |
|---|---:|---|---|
| counter | 2 | `22zgoxfl` | completed |
| kitchen | 3 | `4qtck8uq` | completed |

Command template:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <SCENE> \
  --gpu <GPU> \
  --output_root /dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624 \
  --tag v66_bin_rgb_alpha_max035_support4096_tex16_nearest_region_texture_adapter \
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
  --enable_policy_val_bin_rgb_alpha_calibration \
  --bin_rgb_alpha_calibration_max_alpha 0.35 \
  --bin_rgb_alpha_calibration_min_alpha 0.0 \
  --bin_rgb_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --bin_rgb_alpha_calibration_min_bin_samples 32 \
  --bin_rgb_alpha_calibration_min_positive_view_fraction 0.5 \
  --bin_rgb_alpha_calibration_shrink_count_tau 128 \
  --bin_rgb_alpha_calibration_shrink_denominator_tau 0.0 \
  --bin_rgb_alpha_calibration_shrink_prior fallback \
  --bin_rgb_alpha_calibration_max_profile_bins 8192 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v66_bin_rgb_alpha_probe \
  --wandb_run_name v66_bin_rgb_alpha_<SCENE>_20260624 \
  --wandb_mode online \
  --force
```

Note: this probe was launched before the runner cleanup above. It used support expansion and the broader default runner candidate policy, so it is a valid W&B probe but not the fastest base-only isolation run. Follow-up isolation controls should set `--support_expansion_mode none` or explicitly set the support mode and budget.

---

## Probe Results

### Counter

Artifacts:

```text
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624/counter_v66_bin_rgb_alpha_max035_support4096_tex16_nearest_region_texture_adapter/results.json
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624/counter_v66_bin_rgb_alpha_max035_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624/logs/apply_metrics_counter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v66_bin_rgb_alpha_probe_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v66_bin_rgb_alpha_probe_20260624/counter/surface_residual_region_texture_adapter_audit.json
```

| field | value |
|---|---:|
| PSNR | 26.751209 |
| SSIM | 0.862078 |
| LPIPS | 0.251961 |
| accepted | true |
| selected alpha | 0.125 |
| changed fraction | 0.065630 |
| local alpha mode | `policy_val_bin_rgb_alpha` |
| RGB alpha bins | 669 |
| candidate bins | 669 |
| fallback alpha | `[0.35, 0.35, 0.35]` |
| fallback bins | 249555 |

Reference comparison:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64 selected | 26.756130 | 0.862126 | 0.251691 |
| v66 RGB-bin alpha | 26.751209 | 0.862078 | 0.251961 |
| delta | -0.004921 | -0.000048 | +0.000270 |

Verdict: v66 should be rejected on counter.

### Kitchen

Artifacts:

```text
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624/kitchen_v66_bin_rgb_alpha_max035_support4096_tex16_nearest_region_texture_adapter/results.json
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624/kitchen_v66_bin_rgb_alpha_max035_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624/logs/apply_metrics_kitchen.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v66_bin_rgb_alpha_probe_20260624/kitchen/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v66_bin_rgb_alpha_probe_20260624/kitchen/surface_residual_region_texture_adapter_audit.json
```

| field | value |
|---|---:|
| PSNR | 27.822626 |
| SSIM | 0.876538 |
| LPIPS | 0.198849 |
| accepted | true |
| selected alpha | 1.0 |
| changed fraction | 0.039585 |
| local alpha mode | `policy_val_bin_rgb_alpha` |
| RGB alpha bins | 76 |
| candidate bins | 76 |
| fallback alpha | `[0.35, 0.35, 0.35]` |
| fallback bins | 217333 |

Reference comparison:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v64 selected | 27.822626 | 0.876538 | 0.198849 |
| v66 RGB-bin alpha | 27.822626 | 0.876538 | 0.198849 |
| delta | +0.000000 | +0.000000 | +0.000000 |

Verdict: v66 ties v64 on kitchen and does not justify promotion.

---

## Interpretation

v66 answers a concrete technical question:

> Is the v63/v64 bottleneck mainly caused by scalar alpha being unable to correct channel-wise residual magnitude?

Current answer: no, not by itself.

The counter regression means that per-channel alpha can overfit bin/color statistics without improving held-out rendering. The kitchen tie means the previous scalar bin-alpha policy was already sufficient for the one accepted scene. The next meaningful direction should be a stronger residual-field model with confidence and local support modeling, not another alpha-only variant.

---

## Promotion Decision

Do not promote v66.

Recommended current stack:

| role | method |
|---|---|
| PPT headline endpoint | Phase-J guarded adaptive Evidence Lumigraph Adapter |
| best fixed representation-level policy | v64 fixed auto bin-alpha policy |
| negative diagnostics | v65 teacher-distilled shared basis, v66 bin-RGB alpha |
