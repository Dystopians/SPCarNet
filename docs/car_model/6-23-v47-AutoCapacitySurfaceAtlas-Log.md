# v47 Auto-Capacity Guarded Surface Atlas Log

Date: 2026-06-23

Status: `REPRESENTATION-LEVEL MILESTONE, NOT FINAL PAPER ENDPOINT`.

v47 upgrades the v46 auto-fill atlas into a train-only auto-capacity policy. It keeps the same v42-calibrated risk gates, but evaluates a fixed texture-size candidate set and promotes a higher-capacity atlas only when train policy-val evidence is non-regressive against the calibrated face-mean baseline.

## Motivation

v42/v46 made the surface residual atlas safe, but the effect size remained tiny because the atlas used a single fixed texture resolution. v43 showed that nearest-observed fill could help some scenes, while hurting others. v46 fixed the fill-mode selection problem, but it still had no mechanism for scene-adaptive representation capacity.

v47 changes the method rather than hand-tuning scenes:

```text
fixed capacity candidates: texture_size in {8, 16, 24, 32}
fixed fill candidates: face_mean, nearest_observed
selection evidence: train policy-val only
promotion guard: non-regressive relative gain, SSIM gain, CVaR20, min-view gain
held-out test GT: report-only
```

## Implementation

Main implementation:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

New CLI:

```text
--texture_size_candidates 8,16,24,32
```

Legacy behavior is preserved. If `--texture_size_candidates` is omitted, the adapter uses the original fixed `--texture_size`.

Selection is over joint candidates:

```text
(texture_size, atlas_empty_bin_fill_mode)
```

The audit records:

- selected texture size;
- selected fill mode;
- selected alpha;
- accepted candidate count;
- train policy-val score order;
- per-candidate risk-gate reasons.

## Fixed v47 Policy

The four-scene validation uses the same fixed policy for `garden/room/counter/bonsai`:

```text
--texture_size 16
--texture_size_candidates 8,16,24,32
--atlas_empty_bin_fill_mode auto_policy
--alpha_grid 0,0.015625,0.03125,0.0625,0.125
--min_l1 0.001
--min_atlas_face_samples 32
--atlas_confidence_mode count_var_sign
--atlas_confidence_count_scale 2.0
--atlas_confidence_empty_bin 0.5
--atlas_confidence_variance_scale 0.004
--atlas_confidence_sign_power 0.5
--atlas_confidence_face_sample_scale 256
--min_atlas_confidence 0.02
--atlas_lowpass_passes 1
--select_alpha_by_risk_gate
--min_policy_val_relative_gain 0.0002
--min_policy_val_positive_view_fraction 1.0
--min_policy_val_cvar20_relative_gain 0.0
--min_policy_val_min_view_relative_gain 0.0
--enable_policy_val_image_ssim_gate
--min_policy_val_ssim_mean_gain 0.0
--min_policy_val_ssim_positive_view_fraction 0.75
--min_policy_val_ssim_min_view_gain -0.000005
--min_target_changed_fraction 0.001
```

## Policy Decisions

| scene | selected texture | selected fill | alpha | changed fraction | accepted candidates |
|---|---:|---|---:|---:|---:|
| garden | 32 | nearest_observed | 0.1250 | 0.3515% | 8 |
| room | 16 | face_mean | 0.1250 | 1.0602% | 8 |
| counter | 32 | nearest_observed | 0.1250 | 1.8701% | 8 |
| bonsai | 16 | nearest_observed | 0.0625 | 0.7277% | 8 |

Reading:

- `garden` and `counter` have enough train evidence to promote `texture_size=32`.
- `room` is deliberately held at the calibrated v42/v46 capacity because high-capacity nearest-fill loses the SSIM/non-regression guard against the baseline.
- `bonsai` remains at the v46 setting; the policy does not promote capacity without sufficient train evidence.

## Held-Out Metrics

| scene | v47 PSNR | v47 SSIM | v47 LPIPS | dPSNR vs no-op | dSSIM | dLPIPS | dPSNR vs v42 | dSSIM | dLPIPS | dPSNR vs v46 | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garden | 24.741262 | 0.75405383 | 0.24801193 | +0.000259 | +0.00000483 | -0.00001128 | +0.000122 | +0.00000262 | -0.00000794 | +0.000120 | +0.00000244 | -0.00000787 |
| room | 28.740660 | 0.88482928 | 0.24989747 | +0.001656 | +0.00003928 | -0.00001849 | +0.000000 | +0.00000000 | +0.00000000 | +0.000000 | +0.00000000 | +0.00000000 |
| counter | 26.751411 | 0.86205649 | 0.25196922 | +0.001575 | +0.00000715 | -0.00002876 | +0.000061 | +0.00000238 | -0.00000843 | +0.000069 | +0.00000221 | -0.00000769 |
| bonsai | 28.865551 | 0.89601415 | 0.25932455 | +0.001171 | +0.00000411 | -0.00000906 | +0.000565 | +0.00000072 | -0.00000691 | +0.000000 | +0.00000000 | +0.00000000 |

Summary:

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v47 vs no-op | 4 / 4 | 4 / 4 | +0.001165 | +0.00001384 | -0.00001690 |
| v47 vs v42 | 3 / 4 | 4 / 4 | +0.000187 | +0.00000143 | -0.00000582 |
| v47 vs v43 | 3 / 4 | 3 / 4 | +0.000387 | +0.00000781 | -0.00000427 |
| v47 vs v46 | 2 / 4 | 4 / 4 | +0.000047 | +0.00000116 | -0.00000389 |

## Evidence Paths

Summary JSON:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v47_autocap_guarded_v42calib_multiscene_summary.json
```

Output models:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_v47_autocap_guarded_v42calib_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_v47_autocap_guarded_v42calib_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_v47_autocap_guarded_v42calib_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_v47_autocap_guarded_v42calib_region_texture_adapter
```

Logs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_garden_v47_autocap_guarded_v42calib_region_texture_adapter_gpu1.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/metrics_garden_v47_autocap_guarded_v42calib_region_texture_adapter_gpu1.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_room_v47_autocap_guarded_v42calib_region_texture_adapter_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/metrics_room_v47_autocap_guarded_v42calib_region_texture_adapter_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_counter_v47_autocap_guarded_v42calib_region_texture_adapter_gpu1.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/metrics_counter_v47_autocap_guarded_v42calib_region_texture_adapter_gpu1.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_bonsai_v47_autocap_guarded_v42calib_region_texture_adapter_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/metrics_bonsai_v47_autocap_guarded_v42calib_region_texture_adapter_gpu4.log
```

Note: the first garden/room wrapper used `status` as a zsh variable name, which is read-only in zsh. The apply jobs had already completed and wrote valid audits/renders; metrics were rerun separately and completed successfully. Counter/bonsai were run without that wrapper issue.

## Interpretation

v47 is a real method improvement over v46:

- it adds representation-capacity selection to the train/eval pipeline;
- it improves held-out metrics on `garden` and `counter`;
- it preserves `room` and `bonsai` by falling back to lower-risk candidates;
- it improves the four-scene mean over no-op, v42, v43, and v46.

This is still not a final paper endpoint. The absolute effect size is larger than v42/v46 but remains small compared with Phase-J render-time ELA. The next representation-level step should increase support beyond residual-hot carrier regions, likely by adding a higher-capacity surface residual basis or view-conditioned residual field while retaining the v47 train-only capacity/guard policy.
