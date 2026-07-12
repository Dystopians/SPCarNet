# v67 Policy-Val Bin Uncertainty Shrink Probe Log

Date: 2026-06-24

Purpose: test whether bin-level residual application should be controlled by an uncertainty-aware shrink field instead of scalar/RGB alpha alone.

---

## Summary

v67 is a real train/eval pipeline change. It adds `policy_val_bin_uncertainty_shrink` to the surface residual atlas adapter and exposes the same controls through the scene runner. The goal is to estimate, from train/policy-val evidence only, which face/UV bins deserve strong residual transfer and which bins should be attenuated because their residual is low-support, low-gain, high-variance, or inconsistent across views.

Result: v67 is not promoted.

| scene | reference | v67 PSNR | v67 SSIM | v67 LPIPS | verdict |
|---|---:|---:|---:|---:|---|
| counter | v56/v64: `26.756130 / 0.862126 / 0.251691` | 26.749853 | 0.862050 | 0.251998 | worse than selected policy |
| kitchen | v64: `27.822626 / 0.876538 / 0.198849` | 27.816389 | 0.876443 | 0.199201 | fallback/no-op and worse than v64 |

The implementation is mechanically valid, W&B-logged, and useful as infrastructure, but the first fixed configuration is too conservative. It selects only a very small shrink mass and does not recover the v64/v56 gains.

---

## Implementation

Changed files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter mode:

```text
local_alpha_profile.mode = policy_val_bin_uncertainty_shrink
```

Core idea:

```text
policy-val residual prediction at alpha = 1.0
  -> per face/bin before-vs-after MSE gain
  -> positive-view fraction
  -> sample-count confidence
  -> atlas variance and sign consistency
  -> local residual shrink in [min_shrink, max_shrink]
```

The shrink field is not another scalar alpha search. It builds a per-bin attenuation profile, then the existing policy-val gate still selects the global alpha and accepts/falls back.

Key new adapter/runner flags:

```text
--enable_policy_val_bin_uncertainty_shrink
--bin_uncertainty_shrink_min_bin_samples
--bin_uncertainty_shrink_min_relative_gain
--bin_uncertainty_shrink_min_positive_view_fraction
--bin_uncertainty_shrink_max_mean_variance
--bin_uncertainty_shrink_min_mean_sign_consistency
--bin_uncertainty_shrink_count_tau
--bin_uncertainty_shrink_gain_tau
--bin_uncertainty_shrink_variance_scale
--bin_uncertainty_shrink_sign_power
--bin_uncertainty_shrink_min_shrink
--bin_uncertainty_shrink_max_shrink
--bin_uncertainty_shrink_fallback_shrink
--bin_uncertainty_shrink_max_profile_bins
```

Mutual exclusion was extended so only one local-alpha family can be active in one run:

```text
bucket alpha, face alpha, scalar bin alpha, RGB bin alpha, or uncertainty shrink
```

W&B logging now includes uncertainty-shrink mode flags and shrink statistics.

---

## Static Validation

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

CLI exposure:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py --help | rg 'bin_uncertainty_shrink|uncertainty shrink'
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py --help | rg 'bin_uncertainty_shrink|support_expansion_mode'
```

Mutual-exclusion check:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  --source_model x --fit_evidence_dir x --target_evidence_dir x --region_carrier_json x --output_model x \
  --enable_policy_val_bin_alpha_calibration --enable_policy_val_bin_uncertainty_shrink
```

Expected error:

```text
error: enable at most one local alpha calibration mode: bucket, face, scalar bin, RGB bin, or uncertainty shrink
```

---

## Probe Commands

Output root:

```text
/dev/shm/peilincai_spcarnet_v67_uncertainty_shrink_probe_20260624
```

Persistent copy:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v67_uncertainty_shrink_probe_20260624
```

W&B:

| scene | GPU | W&B run | status |
|---|---:|---|---|
| counter | 2 | `1p7ov1k3` | completed |
| kitchen | 3 | `u7xb0tu4` | completed |

Command template:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <SCENE> \
  --gpu <GPU> \
  --output_root /dev/shm/peilincai_spcarnet_v67_uncertainty_shrink_probe_20260624 \
  --tag v67_uncertainty_shrink_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 16 \
  --view_conditioned_basis_ridge 0.1 \
  --view_conditioned_basis_ood_mode diag_z \
  --view_conditioned_basis_ood_max_z 2.5 \
  --view_conditioned_basis_ood_min_std 0.05 \
  --enable_policy_val_bin_uncertainty_shrink \
  --bin_uncertainty_shrink_min_bin_samples 32 \
  --bin_uncertainty_shrink_min_relative_gain 0.0 \
  --bin_uncertainty_shrink_min_positive_view_fraction 0.5 \
  --bin_uncertainty_shrink_max_mean_variance -1 \
  --bin_uncertainty_shrink_min_mean_sign_consistency 0.0 \
  --bin_uncertainty_shrink_count_tau 128 \
  --bin_uncertainty_shrink_gain_tau 0.005 \
  --bin_uncertainty_shrink_variance_scale 0.004 \
  --bin_uncertainty_shrink_sign_power 0.5 \
  --bin_uncertainty_shrink_min_shrink 0.0 \
  --bin_uncertainty_shrink_max_shrink 1.0 \
  --bin_uncertainty_shrink_fallback_shrink 0.0 \
  --bin_uncertainty_shrink_max_profile_bins 8192 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v67_uncertainty_shrink_probe \
  --wandb_run_name v67_uncertainty_shrink_<SCENE>_20260624 \
  --wandb_mode online \
  --force
```

---

## Probe Results

### Counter

Artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v67_uncertainty_shrink_probe_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v67_uncertainty_shrink_probe_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v67_uncertainty_shrink_probe_20260624/logs/apply_metrics_counter.log
```

| field | value |
|---|---:|
| PSNR | 26.749853 |
| SSIM | 0.862050 |
| LPIPS | 0.251998 |
| accepted | true |
| selected alpha | 0.125 |
| target changed fraction | 0.001639 |
| local alpha mode | `policy_val_bin_uncertainty_shrink` |
| shrink bins | 546 |
| candidate bins | 63246 |
| fallback bins | 62700 |
| mean selected shrink | 0.071909 |
| min/max selected shrink | 0.006102 / 0.263049 |

Policy-val gate:

| field | value |
|---|---:|
| selected SSIM gain | 0.000000934 |
| selected SSIM positive fraction | 0.916667 |
| selected image-L1 gain | 0.000000107 |
| selected image-L1 positive fraction | 0.916667 |
| selected image-L1 min-view gain | -0.0000000149 |
| selected image-L1 CVaR20 gain | 0.0000000143 |

Reference comparison:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64 selected | 26.756130 | 0.862126 | 0.251691 |
| v67 uncertainty shrink | 26.749853 | 0.862050 | 0.251998 |
| delta | -0.006277 | -0.000076 | +0.000307 |

Verdict: v67 should be rejected on counter. The gate accepts a tiny policy-val signal, but held-out metrics regress.

### Kitchen

Artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v67_uncertainty_shrink_probe_20260624/kitchen/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v67_uncertainty_shrink_probe_20260624/kitchen/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v67_uncertainty_shrink_probe_20260624/logs/apply_metrics_kitchen.log
```

| field | value |
|---|---:|
| PSNR | 27.816389 |
| SSIM | 0.876443 |
| LPIPS | 0.199201 |
| accepted | false |
| effective policy | `fallback_noop` |
| selected alpha | 0.0 |
| target changed fraction | 0.0 |
| local alpha mode | `policy_val_bin_uncertainty_shrink` |
| shrink bins | 42 |
| candidate bins | 217409 |
| fallback bins | 217367 |
| mean selected shrink | 0.035126 |
| min/max selected shrink | 0.005191 / 0.150935 |

Policy-val gate:

| field | value |
|---|---:|
| selected positive view fraction | 0.833333 |
| selected SSIM gain | 0.00000000993 |
| selected SSIM positive fraction | 0.166667 |
| selected image-L1 gain | 0.00000000217 |
| selected image-L1 positive fraction | 0.333333 |
| selected image-L1 min-view gain | -0.00000000186 |
| selected image-L1 CVaR20 gain | -0.000000000621 |

Reference comparison:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v64 selected | 27.822626 | 0.876538 | 0.198849 |
| v67 uncertainty shrink | 27.816389 | 0.876443 | 0.199201 |
| delta | -0.006237 | -0.000095 | +0.000352 |

Verdict: v67 should be rejected on kitchen. It correctly rejects the candidate internally, but the resulting no-op/fallback is below the selected v64 row.

---

## Diagnosis

v67 exposed a useful failure mode:

- the shrink profile is too conservative: `counter` keeps only `546 / 63246` candidate bins with mean shrink `0.0719`;
- `kitchen` keeps only `42 / 217409` candidate bins with mean shrink `0.0351`;
- with `fallback_shrink = 0.0`, most target bins receive no residual transfer;
- the current policy-val gate can accept tiny improvements that are not large enough to survive held-out evaluation;
- the main bottleneck is not just where to shrink, but how to model a stronger uncertainty-certified residual field.

This suggests the next research step should not be another scalar threshold sweep. A stronger direction is to learn or estimate a local residual distribution with confidence, then let the policy gate reason over expected gain and uncertainty mass rather than one hand-composed shrink product.

---

## Decision

Status: `NOT_PROMOTED_NEGATIVE_DIAGNOSTIC`

Keep current presentation hierarchy:

- headline endpoint: Phase-J guarded adaptive ELA;
- best fixed representation-level policy: v64 fixed auto bin-alpha;
- negative diagnostics: v65 teacher basis, v66 RGB-bin alpha, v67 uncertainty shrink.

