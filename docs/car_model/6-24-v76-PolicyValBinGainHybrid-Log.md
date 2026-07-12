# v76 Policy-Val Bin-Gain Hybrid Prior Log

Date: 2026-06-24

Status: `COMPLETED_DIAGNOSTIC_NOT_PROMOTED`.

## Motivation

v75 proved that same-face local patch prior can cover many low-support bins, but train-policy selection still preferred `blend=0.0`. v76 tests a stricter idea: keep the zero-blend atlas as the baseline, build a nonzero local-patch prior atlas, and copy only face/UV bins whose prior gives a positive train-policy-val bin-level gain.

The intent is to replace a global prior decision with a local certificate:

```text
zero-blend atlas + locally certified prior bins
```

## Implementation

Main files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter features:

- `--enable_policy_val_prior_bin_gain_hybrid`;
- `--prior_bin_gain_hybrid_min_bin_samples`;
- `--prior_bin_gain_hybrid_min_relative_gain`;
- `--prior_bin_gain_hybrid_min_positive_view_fraction`;
- `--prior_bin_gain_hybrid_max_profile_bins`;
- `build_policy_val_prior_bin_gain_hybrid_atlas(...)`;
- deep-copy helpers for `FaceAtlas` so the hybrid candidate can inherit zero-blend bins and replace only certified prior bins;
- identical policy-val risk gate for ordinary and hybrid candidates.

New runner features:

- forwards the v76 flags to the adapter;
- records v76 config in W&B;
- logs selected/allowed/candidate bin counts as W&B metrics.

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

## Experiment

Scene: `counter`

GPU: `2`

W&B:

```text
project = SPCarNet
group = v76_policyval_bin_gain_hybrid
run = v76_policyval_bin_gain_hybrid_counter_20260624
id = 8qetk7tj
url = https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/8qetk7tj
```

Command:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v76_policyval_bin_gain_hybrid_20260624 \
  --tag v76_policyval_bin_gain_hybrid_counter_region_texture_adapter \
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
  --prior_bin_gain_hybrid_min_bin_samples 4 \
  --prior_bin_gain_hybrid_min_relative_gain 0.0 \
  --prior_bin_gain_hybrid_min_positive_view_fraction 0.5 \
  --enable_target_support_candidate_selection \
  --target_support_prerank_top_k 1 \
  --target_support_prerank_max_views 8 \
  --min_target_changed_fraction 0.0 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --wandb_project SPCarNet \
  --wandb_group v76_policyval_bin_gain_hybrid \
  --wandb_run_name v76_policyval_bin_gain_hybrid_counter_20260624 \
  --wandb_mode online \
  --force
```

## Results

| method | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| v76 policy-val bin-gain hybrid | `26.753532410` | `0.862111092` | `0.251881331` | not promoted |
| v75 local patch prior / zero-blend | `26.753995895` | `0.862119257` | `0.251853049` | stronger |
| v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | stronger |

Selected audit:

```text
accepted = true
effective_policy = accepted_atlas
selected_alpha = 0.125
selected_surface_multiscale_prior_blend = 1.0
selected_policy_val_prior_bin_gain_hybrid = true
selected_support_mode = fit_residual_topk
selected_support_added_faces = 4096
target_changed_fraction = 0.06563028947904326
```

Hybrid profile:

| item | value |
|---|---:|
| candidate bins | `233306` |
| allowed bins | `13708` |
| allowed fraction | `0.058755454` |
| policy-val views | `12` |
| active samples | `500279` |

## Conclusion

v76 is a real method/pipeline addition, but it should not be promoted. The hybrid candidate slightly improves policy-val relative gain over the zero-blend candidate (`0.021140188` vs `0.021094313`), yet held-out test metrics are worse than v75 and still below the v64/v56 `counter` reference.

The likely reason is that the current bin certificate is too weak: many top bins have only `4-6` samples and one policy-val view. A local bin can therefore look positive under policy-val but still fail to generalize. If this direction is revisited, the next fair variant should require stronger multi-view support, higher minimum samples, and a larger policy-val gain margin.

For the mentor/PPT story, v76 should be used as a negative diagnostic: it proves the team has moved beyond scalar parameter sweeps into local evidence certificates, but the measured result reinforces the current bottleneck diagnosis.

## Artifacts

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/counter_v76_policyval_bin_gain_hybrid_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/counter_v76_policyval_bin_gain_hybrid_counter_region_texture_adapter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/counter_v76_policyval_bin_gain_hybrid_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/counter_v76_policyval_bin_gain_hybrid_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/logs/apply_metrics_counter.log
```
