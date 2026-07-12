# v85 SSIM-Safe PreRank PatchMix Log

Date: `2026-06-24`

Status: `COMPLETED_NEGATIVE_DIAGNOSTIC_NOT_PROMOTED`

## Motivation

v83 showed a useful but incomplete signal:

```text
PSNR/LPIPS improved over the v56/v64/v79/v82 counter anchors,
but SSIM regressed by about 8.94e-7 versus the strong anchor.
```

That failure is small numerically but important scientifically: a representation
candidate cannot be promoted if the certificate allows a known SSIM regression.
v85-ssimsafe therefore tightens the train/policy-val certificate before allowing
the patch-mixture + face-alpha + local-patch hybrid candidate to write a non-noop
adapter.

## Method Change Used By This Probe

The runner now forwards explicit policy-val image gate thresholds:

```text
--min_policy_val_ssim_mean_gain
--min_policy_val_ssim_positive_view_fraction
--min_policy_val_ssim_min_view_gain
--min_policy_val_l1_mean_gain
--min_policy_val_l1_min_view_gain
--min_policy_val_l1_cvar20_view_gain
```

These thresholds are intentionally command-line controlled and W&B logged, so the
certificate can be reproduced without hidden per-scene editing.

## Running Command

W&B:

```text
project: SPCarNet
group: v85_ssimsafe_prerank_patchmix
run: 58swzibf
name: v85_ssimsafe_prerank_patchmix_counter_20260624
```

Output root:

```text
/dev/shm/peilincai_spcarnet_v85_ssimsafe_prerank_patchmix_20260624
```

Command:

```text
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v85_ssimsafe_prerank_patchmix_20260624 \
  --tag v85_ssimsafe_prerank_patchmix_facealpha_localpatch_hybrid_tex32_support4096_8192_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --support_expansion_max_extra_faces_candidates 4096,8192 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --max_abs_delta_rgb 0.12 \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_block_sizes 1,2,3 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32.0 \
  --surface_multiscale_prior_blend 1.0 \
  --surface_multiscale_prior_blend_candidates 0,0.5,1.0 \
  --surface_multiscale_prior_gate_mode none \
  --enable_policy_val_prior_bin_gain_hybrid \
  --prior_bin_gain_hybrid_min_bin_samples 4 \
  --prior_bin_gain_hybrid_min_views 1 \
  --prior_bin_gain_hybrid_min_abs_gain 0.0 \
  --prior_bin_gain_hybrid_min_relative_gain 0.0 \
  --prior_bin_gain_hybrid_min_positive_view_fraction 0.5 \
  --prior_bin_gain_hybrid_max_profile_bins 0 \
  --teacher_distilled_basis_mode face_uv_patch_mixture_ridge \
  --teacher_distilled_basis_guard_mode policy_val_nonregressive \
  --teacher_distilled_basis_min_face_samples 1024 \
  --teacher_distilled_basis_ridge 0.05 \
  --teacher_distilled_basis_ood_max_z 2.5 \
  --teacher_distilled_basis_ood_min_std 0.05 \
  --teacher_distilled_basis_apply_mode blend \
  --teacher_distilled_basis_blend 0.5 \
  --enable_policy_val_face_alpha_calibration \
  --face_alpha_calibration_max_alpha 0.5 \
  --face_alpha_calibration_min_alpha 0.0 \
  --face_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --face_alpha_calibration_min_face_samples 256 \
  --face_alpha_calibration_shrink_count_tau 0.0 \
  --face_alpha_calibration_shrink_denominator_tau 0.0 \
  --face_alpha_calibration_shrink_prior fallback \
  --target_support_prerank_top_k 1 \
  --target_support_prerank_max_views 4 \
  --min_policy_val_ssim_mean_gain 0.000294 \
  --min_policy_val_ssim_positive_view_fraction 1.0 \
  --min_policy_val_ssim_min_view_gain 0.000060 \
  --min_policy_val_l1_mean_gain 0.0000268 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_policy_val_l1_min_view_gain -0.00000085 \
  --min_policy_val_l1_cvar20_view_gain 0.0000026 \
  --min_target_changed_fraction 0.001 \
  --wandb_project SPCarNet \
  --wandb_group v85_ssimsafe_prerank_patchmix \
  --wandb_run_name v85_ssimsafe_prerank_patchmix_counter_20260624 \
  --wandb_mode online \
  --force
```

## Promotion Rule

This run is not promoted unless held-out `counter` strictly improves the current
strong counter anchor on all three metrics:

```text
v56/v64/v79 counter anchor:
PSNR  26.756130219
SSIM   0.862126231
LPIPS  0.251691371
```

It should also be compared against the latest richer-capacity probes:

```text
v82 capacity-prerank:
PSNR  26.756137848
SSIM   0.862126350
LPIPS  0.251690656

v83 patchmix hybrid:
PSNR  26.756147385
SSIM   0.862125337
LPIPS  0.251688808
```

If the run rejects all candidates and writes a no-op fallback, the result should
be treated as a useful safety certificate, not a performance improvement.

## Final Result

The run completed on `counter` with W&B online. All three policy candidates were
rejected by the stricter SSIM/L1 risk gate, so the final output is a
`fallback_noop`.

Held-out result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64/v79 counter anchor | `26.756130219` | `0.862126231` | `0.251691371` |
| v82 capacity-prerank | `26.756137848` | `0.862126350` | `0.251690656` |
| v83 patchmix hybrid | `26.756147385` | `0.862125337` | `0.251688808` |
| v85 ssimsafe prerank patchmix | `26.749835968` | `0.862049341` | `0.251998007` |

Delta vs v56/v64/v79 anchor:

| dPSNR | dSSIM | dLPIPS |
|---:|---:|---:|
| `-0.006294251` | `-0.000076890` | `+0.000306636` |

Candidate progress:

| candidate | support | prior blend | accepted by strict gate | alpha | policy-val SSIM gain | policy-val L1 gain |
|---|---|---:|---:|---:|---:|---:|
| 1 / 3 | `fit_residual_topk_8192` | `0` | `false` | `0` | `0.00035386284` | `0.000040038954` |
| 2 / 3 | `fit_residual_topk_8192` | `0.5` | `false` | `0` | `0.0003537039` | `0.000039916951` |
| 3 / 3 | `fit_residual_topk_8192` | `1.0` | `false` | `0` | `0.00035283963` | `0.000039756453` |

Final audit:

| field | value |
|---|---:|
| accepted | `false` |
| effective policy | `fallback_noop` |
| selected alpha | `0.0` |
| changed pixels | `0` |
| changed fraction | `0.0` |
| fallback source | `target_evidence` |

Final reject reason:

```text
ssim_positive_view_fraction 0.833333 < min_policy_val_ssim_positive_view_fraction 1.000000
ssim_min_view_gain -0.000293374 < min_policy_val_ssim_min_view_gain 0.000060000
image_l1_positive_view_fraction 0.833333 < min_policy_val_l1_positive_view_fraction 0.900000
image_l1_min_view_gain -0.000011431 < min_policy_val_l1_min_view_gain -0.000000850
image_l1_cvar20_view_gain -0.000003693 < min_policy_val_l1_cvar20_view_gain 0.000002600
```

## Interpretation

v85-ssimsafe is a useful negative diagnostic. It confirms that the stricter
certificate can block the v83-style SSIM-tail failure mode, but it does not solve
the performance problem because the fallback source is weaker than the
v56/v64/v79 anchor.

Do not promote this run. The next useful direction is not simply tightening gates
further; it is designing a fallback/selector that preserves the strong v64/v84
anchor while allowing only SSIM-tail-safe non-noop improvements.

## Persistent Evidence

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/counter_v85_ssimsafe_prerank_patchmix_facealpha_localpatch_hybrid_tex32_support4096_8192_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/counter_v85_ssimsafe_prerank_patchmix_facealpha_localpatch_hybrid_tex32_support4096_8192_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/logs/apply_metrics_counter.log
```
