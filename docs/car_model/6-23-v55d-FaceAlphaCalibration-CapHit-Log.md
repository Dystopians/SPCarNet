# v55d Policy-Val Face-Alpha Calibration Cap-Hit Log

Date: 2026-06-23

Status: `NOT_PROMOTED_AS_GLOBAL_REPLACEMENT`. v55d is a real train/eval pipeline method change and a useful diagnostic. It strictly improves `counter` over v52, but it does not strictly improve the full cap-hit set: `kitchen` improves PSNR/LPIPS while regressing SSIM, and `bonsai` regresses all three metrics versus v52.

## Motivation

v53 showed that residual amplitude is a real bottleneck, but a single global policy-val least-squares alpha is too blunt. v55d tests a more local version:

```text
fit train policy-val residual evidence
  -> estimate per-face residual alpha where enough samples exist
  -> cap effective local alpha
  -> select global multiplier through existing policy-val risk gates
  -> apply to held-out test views
```

This keeps the selection train/policy-val based. Held-out test metrics are used only for reporting.

## Implementation Changes

Modified scripts:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

Important fixes made before v55d:

- Per-face/local alpha is now effectively capped so `global_alpha * local_alpha` cannot exceed the configured maximum.
- Audit JSON is sanitized before writing with `allow_nan=False`.
- The runner logs the explicit method row instead of assuming the first `results.json` entry.
- W&B logs include local-alpha policy fields such as selected multiplier, face-alpha count, fallback face count, and fallback alpha.

## Fixed Validation Command Pattern

The three cap-hit scenes were run with the same fixed policy:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <counter|kitchen|bonsai> \
  --gpu <gpu> \
  --output_root /dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623 \
  --tag v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --enable_policy_val_face_alpha_calibration \
  --face_alpha_calibration_max_alpha 0.5 \
  --face_alpha_calibration_min_alpha 0.0 \
  --face_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --face_alpha_calibration_min_face_samples 256 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.001 \
  --wandb_project SPCarNet \
  --wandb_group v55d_policyval_face_alpha_l1pos09_caphit \
  --wandb_run_name v55d_face_alpha_<scene>_20260623 \
  --force
```

## W&B Runs

| scene | W&B run id |
|---|---|
| counter | `wwp6tn65` |
| kitchen | `8znw2xhy` |
| bonsai | `6k94f7mm` |

## Results vs v52

v52 references are the reproduced capacity-aware source-rerun rows:

| scene | v52 PSNR | v52 SSIM | v52 LPIPS |
|---|---:|---:|---:|
| counter | `26.753460` | `0.86211467` | `0.25186834` |
| kitchen | `27.818935` | `0.87653536` | `0.19901942` |
| bonsai | `28.868467` | `0.89608848` | `0.25920403` |

v55d held-out results:

| scene | accepted | selected alpha | face-alpha count | changed | dPSNR vs v52 | dSSIM vs v52 | dLPIPS vs v52 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| counter | 1 | `0.5000` | `394` | `6.5362%` | `+0.002670` | `+0.00001156` | `-0.00017697` | strict win |
| kitchen | 1 | `1.0000` | `240` | `3.9361%` | `+0.004507` | `-0.00009782` | `-0.00023896` | PSNR/LPIPS up, SSIM down |
| bonsai | 1 | `0.1250` | `26` | `2.6786%` | `-0.001932` | `-0.00003034` | `+0.00006741` | worse than v52 |

Aggregate cap-hit reading:

| comparison | strict wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| raw v55d vs v52 on cap-hit | `1 / 3` | `1 / 3` | `+0.001749` | `-0.00003886` | `-0.00011617` |

The mean PSNR/LPIPS are positive because `kitchen` improves those metrics, but the SSIM regression violates the current paper-facing strict standard.

## Policy Diagnostics

### counter

`counter` is the successful case.

```text
selected alpha: 0.5
face_alpha_count: 394
fallback_raw_alpha: 1.5018
fallback_alpha: 0.5
selected policy-val SSIM min-view gain: +0.00006020
selected policy-val image-L1 positive view fraction: 0.9167
```

The local alpha has enough support and does not require a high scene-level multiplier. Held-out PSNR, SSIM, and LPIPS all improve versus v52.

### kitchen

`kitchen` exposes the key weakness.

```text
selected alpha: 1.0
face_alpha_count: 240
fallback_raw_alpha: 1.3038
fallback_alpha: 0.5
selected policy-val SSIM min-view gain: +0.00023663
selected policy-val image-L1 positive view fraction: 1.0
```

The train policy-val gates are strongly positive, yet held-out SSIM drops. This means the current policy-val SSIM proxy is still not sufficient to certify high-amplitude residual application on held-out views.

### bonsai

`bonsai` shows a sparse local-alpha support failure.

```text
selected alpha: 0.125
face_alpha_count: 26
fallback_raw_alpha: 2.1484
fallback_alpha: 0.5
selected policy-val SSIM min-view gain: +0.00003451
selected policy-val image-L1 positive view fraction: 0.9167
```

Only 26 faces get direct calibrated alpha. Most target support falls back to the global fallback alpha. The accepted policy-val row is not enough to predict held-out improvement, so this scene should remain v52 under any paper-facing policy.

## Decision

Do not promote raw v55d as a global replacement for v52.

Reasons:

- It fails strict cap-hit validation (`1 / 3` strict wins).
- `kitchen` repeats the v53 pattern: PSNR/LPIPS improve but SSIM regresses.
- `bonsai` demonstrates sparse face-alpha coverage and regresses all metrics.
- A paper-facing method needs non-regressive/tie behavior under a fixed train-only policy, not a metric-mixed mean improvement.

## Next Fixed Policy Candidate

v55d suggests a narrower train-only reliability guard:

```text
use v55d only if
  accepted_atlas
  and local_alpha_profile.enabled
  and face_alpha_count >= 128
  and selected_alpha <= 0.5
  and selected_image_l1_positive_view_fraction >= 0.9
  and selected_ssim_min_view_gain >= 5e-5
else
  fallback to v52
```

This candidate would select `counter` and reject `kitchen/bonsai` using train/policy-val audit fields only. It is not yet promoted because the guard was designed after seeing the cap-hit held-out results. The correct next step is to implement it as v56, rerun or replay it mechanically, and validate on additional scenes or a fresh held-out protocol before making a paper-level claim.

## Evidence Paths

| content | path |
|---|---|
| counter result | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/counter_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter/results.json` |
| counter audit | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/counter_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json` |
| kitchen result | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/kitchen_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter/results.json` |
| kitchen audit | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/kitchen_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json` |
| bonsai result | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/bonsai_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter/results.json` |
| bonsai audit | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/bonsai_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json` |
| run logs | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/logs` |
