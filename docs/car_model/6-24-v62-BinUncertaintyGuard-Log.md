# v62 Bin-Level Uncertainty Guard Log

Date: 2026-06-24

Status: `COMPLETED_NEGATIVE_DIAGNOSTIC_NOT_PROMOTED`.

## Motivation

v61 added a policy-val face-level gain allowlist, but counter and kitchen both regressed against the relevant v52/v56/v60 references. The failure mode is that a whole face can be policy-val positive on average while individual face/UV bins remain unsafe on held-out views.

v62 makes the guard finer and more uncertainty-aware:

> Apply residuals only on face/UV bins that are policy-val positive with enough support, enough positive-view evidence, and acceptable residual uncertainty statistics.

This is a real train/eval pipeline change. It is disabled by default and must be explicitly enabled.

## Implementation

Updated files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter/runner flags:

```text
--enable_policy_val_bin_uncertainty_guard
--bin_uncertainty_guard_min_bin_samples
--bin_uncertainty_guard_min_relative_gain
--bin_uncertainty_guard_min_positive_view_fraction
--bin_uncertainty_guard_max_mean_variance
--bin_uncertainty_guard_min_mean_sign_consistency
```

Core behavior:

1. Fit the same surface atlas / view-conditioned candidate as v60/v61.
2. Select alpha through the existing train policy-val risk gate.
3. On policy-val views, aggregate active residual prediction gains per `(face_id, uv_bin)`:
   - before error: `||teacher_residual||^2`;
   - after error: `||teacher_residual - predicted_delta||^2`;
   - relative gain;
   - positive-view fraction;
   - atlas bin residual variance;
   - atlas bin sign consistency.
4. Keep a bin only if:
   - samples `>= bin_uncertainty_guard_min_bin_samples`;
   - relative gain `>= bin_uncertainty_guard_min_relative_gain`;
   - positive-view fraction `>= bin_uncertainty_guard_min_positive_view_fraction`;
   - optional variance/sign uncertainty checks pass.
5. Re-run the normal train policy-val selector with the bin allowlist active.
6. Apply target residuals only on the allowed face/UV bins if the post-guard policy remains accepted.

## Validation Before Probe

```text
py_compile adapter + runner: passed
adapter --help exposes bin_uncertainty_guard flags: passed
runner --help exposes bin_uncertainty_guard flags: passed
synthetic bin allowlist smoke: passed
```

The synthetic smoke confirmed that, on the same face, only explicitly allowed UV bins receive nonzero deltas while disallowed bins remain unchanged.

## Probe Plan

Initial probe scenes:

- `counter`: needed because v60 was positive vs v52 but behind v56/raw-v55d.
- `kitchen`: needed because v60/v61 both failed strict improvement vs v52.

Promotion criteria:

- Do not promote v62 unless it strictly improves counter against v56/raw-v55d, or strictly improves kitchen against v52, without opening SSIM/LPIPS regression.
- If it only writes a near-no-op, treat it as a diagnostic, not a success.

Reference rows:

| scene | reference | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| counter | v52 | `26.7534599304` | `0.8621146679` | `0.2518683374` |
| counter | v56/raw v55d | `26.7561302185` | `0.8621262312` | `0.2516913712` |
| counter | v60 | `26.7539958954` | `0.8621192575` | `0.2518530488` |
| counter | v61 | `26.7510070801` | `0.8620729446` | `0.2519522905` |
| kitchen | v52 | `27.8189353943` | `0.8765353560` | `0.1990194172` |
| kitchen | v60 | `27.8191566467` | `0.8765332103` | `0.1990308464` |
| kitchen | v61 | `27.8172969818` | `0.8764855266` | `0.1991390735` |

## Results

Both counter and kitchen probes completed with W&B online logging.

| scene | W&B run | PSNR | SSIM | LPIPS | accepted | selected alpha | changed fraction | bin guard decision | allowed / candidate bins | allowed faces | allowed sample fraction |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| counter | `tdu3x70o` | `26.7498836517` | `0.8620498180` | `0.2519963086` | true | `0.125` | `0.0001962647` | `keep_bin_uncertainty_guard` | `74 / 250224` | `21` | `0.0046970623` |
| kitchen | `wnse9uz8` | `27.8163890839` | `0.8764428496` | `0.1992005110` | false | `0.0` | `0.0` | `skipped_candidate_not_accepted` | n/a | n/a | n/a |

Reference deltas:

| scene | reference | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | `-0.0035762787` | `-0.0000648499` | `+0.0001279712` | worse |
| counter | v56/raw v55d | `-0.0062465668` | `-0.0000764132` | `+0.0003049374` | worse |
| counter | v60 | `-0.0041122437` | `-0.0000694395` | `+0.0001432598` | worse |
| counter | v61 | `-0.0011234284` | `-0.0000231266` | `+0.0000440181` | worse |
| kitchen | v52 | `-0.0025463104` | `-0.0000925064` | `+0.0001810938` | worse |
| kitchen | v60 | `-0.0027675628` | `-0.0000903607` | `+0.0001696646` | worse |
| kitchen | v61 | `-0.0009078979` | `-0.0000426770` | `+0.0000614375` | worse |

Output paths:

```text
/dev/shm/peilincai_spcarnet_v62_bin_uncertainty_counter_20260624/counter_v62_binunc_min32_pos075_var004_sign05_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter/results.json
/dev/shm/peilincai_spcarnet_v62_bin_uncertainty_counter_20260624/counter_v62_binunc_min32_pos075_var004_sign05_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
/dev/shm/peilincai_spcarnet_v62_bin_uncertainty_kitchen_20260624/kitchen_v62_binunc_min32_pos075_var004_sign05_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter/results.json
/dev/shm/peilincai_spcarnet_v62_bin_uncertainty_kitchen_20260624/kitchen_v62_binunc_min32_pos075_var004_sign05_normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
```

## Verdict

Do not promote v62.

The bin-level uncertainty guard is a real pipeline change and its allowlist is active on `counter`, but it becomes too conservative and still does not improve held-out metrics. On `kitchen`, the candidate is rejected and writes a no-op fallback, but the no-op row is still weaker than the v52/v60/v61 reference rows under this probe output. The main lesson is that target safety cannot be recovered by only shrinking the application mask after fitting; the residual field itself needs stronger magnitude calibration and uncertainty modeling before target transfer.
