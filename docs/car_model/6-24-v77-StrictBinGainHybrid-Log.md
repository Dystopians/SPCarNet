# v77 Strict Multi-View Bin-Gain Hybrid Log

Date: 2026-06-24  
Status: `COMPLETED_NEGATIVE_DIAGNOSTIC_NOT_PROMOTED`  
Scene: `counter`  
W&B run: `3ho2y4s1`  
W&B URL: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/3ho2y4s1`

## Purpose

v76 showed that a policy-val bin-gain hybrid prior could be selected, but its held-out test metrics regressed versus the zero-blend/local-patch line and the v64/v56 `counter` reference. The observed failure mode was weak bin-level certificates: many positive bins were supported by too few samples and too few policy-val views.

v77 tightens the certificate:

- `prior_bin_gain_hybrid_min_bin_samples = 16`
- `prior_bin_gain_hybrid_min_views = 2`
- `prior_bin_gain_hybrid_min_abs_gain = 1e-5`
- `prior_bin_gain_hybrid_min_relative_gain = 0.005`
- `prior_bin_gain_hybrid_min_positive_view_fraction = 0.75`

## Implementation

Changed files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

Added behavior:

- the policy-val prior bin-gain hybrid gate now records and checks absolute bin gain;
- hybrid keep requires minimum samples, minimum policy-val views, minimum absolute gain, minimum relative gain, positive-view fraction, and `prior_after < baseline_after`;
- CLI exposes `--prior_bin_gain_hybrid_min_views` and `--prior_bin_gain_hybrid_min_abs_gain`;
- the scene runner forwards these flags and logs them to W&B.

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

Both checks passed.

## Command

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 3 \
  --output_root /dev/shm/peilincai_spcarnet_v77_strict_bin_gain_hybrid_20260624 \
  --tag v77_strict_bin_gain_hybrid_counter_region_texture_adapter \
  --texture_size_candidates 16 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_block_sizes 1,2,3 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32.0 \
  --surface_multiscale_prior_blend 1.0 \
  --surface_multiscale_prior_blend_candidates 0,0.5,1.0 \
  --surface_multiscale_prior_gate_mode none \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --max_abs_delta_rgb 0.12 \
  --enable_policy_val_prior_bin_gain_hybrid \
  --prior_bin_gain_hybrid_min_bin_samples 16 \
  --prior_bin_gain_hybrid_min_views 2 \
  --prior_bin_gain_hybrid_min_abs_gain 0.00001 \
  --prior_bin_gain_hybrid_min_relative_gain 0.005 \
  --prior_bin_gain_hybrid_min_positive_view_fraction 0.75 \
  --enable_target_support_candidate_selection \
  --target_support_prerank_top_k 1 \
  --target_support_prerank_max_views 8 \
  --min_target_changed_fraction 0.0 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --wandb_project SPCarNet \
  --wandb_group v77_strict_bin_gain_hybrid \
  --wandb_run_name v77_strict_bin_gain_hybrid_counter_20260624 \
  --wandb_mode online \
  --force
```

## Results

| method | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| v77 strict bin-gain hybrid | `26.753528595` | `0.862111032` | `0.251881331` | not promoted |
| v76 policy-val bin-gain hybrid | `26.753532410` | `0.862111092` | `0.251881331` | not promoted |
| v75 local patch prior / zero-blend | `26.753995895` | `0.862119257` | `0.251853049` | stronger |
| v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | stronger |

Selected audit:

```text
accepted = true
effective_policy = accepted_atlas
selected_alpha = 0.125
selected_surface_multiscale_prior_blend = 0.0
selected_policy_val_prior_bin_gain_hybrid = false
selected_support_mode = fit_residual_topk
selected_support_added_faces = 4096
target_changed_fraction = 0.06563028947904326
policy_val_relative_gain = 0.02109431338991033
policy_val_positive_view_fraction = 1.0
policy_val_image_l1_positive_view_fraction = 0.9166666666666666
```

Candidate ordering:

| blend | hybrid | allowed bins | rel gain | SSIM gain | image L1 gain |
|---:|---:|---:|---:|---:|---:|
| `0.0` | `false` | `0` | `0.021094313` | `0.000158509` | `0.000014013` |
| `0.5` | `false` | `0` | `0.020915056` | `0.000156805` | `0.000013874` |
| `1.0` | `false` | `0` | `0.020734755` | `0.000155091` | `0.000013731` |

## Conclusion

v77 is a useful safety correction, not a performance improvement. The stricter multi-view and absolute-gain certificate prevents the weak v76 hybrid from being selected, leaving zero certified hybrid bins under this policy. The final held-out metrics remain below v75 and below the v64/v56 `counter` reference.

For the mentor/PPT story, v77 should be framed as an honest negative diagnostic: it confirms that weak local bin certificates were unsafe and that stricter evidence gates correctly refuse them. The broader bottleneck remains the same: persistent surface residual repair needs a stronger representation and a better target-view generalization certificate, not more scalar tuning.

## Artifacts

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/counter_v77_strict_bin_gain_hybrid_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/counter_v77_strict_bin_gain_hybrid_counter_region_texture_adapter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/counter_v77_strict_bin_gain_hybrid_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/counter_v77_strict_bin_gain_hybrid_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/logs/apply_metrics_counter.log
```

