# v74 Delta-Cap Ladder Log

Date: 2026-06-24

Status: `COMPLETED_NOT_PROMOTED`.

v74 adds a real train/eval pipeline change to the surface residual atlas branch: `max_abs_delta_rgb` is no longer only a fixed safety constant. It can now be exposed as a policy-val candidate ladder, and the selected cap is used consistently in policy-val scoring and final target application.

## Motivation

v72 and v73 showed that target support can be made active, but larger target footprint still did not improve held-out metrics. One plausible bottleneck was residual amplitude capacity: the atlas may have been too conservatively clipped at `0.12`, preventing useful residuals from reaching the target view. v74 tests that hypothesis without manual parameter selection by adding a cap ladder:

```text
max_abs_delta_rgb candidates = 0.12, 0.18, 0.24
```

The policy still uses train/policy-val evidence only. Held-out test metrics are used only for reporting.

## Code Changes

Primary files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

Implemented changes:

| item | status |
|---|---|
| `--max_abs_delta_rgb_candidates` in adapter | done |
| `--max_abs_delta_rgb_candidates` in scene runner | done |
| cap-consistent policy-val delta clipping | done |
| cap-consistent final target apply | done |
| audit records selected cap and candidate list | done |
| W&B logs selected cap and candidate count | done |

Static validation:

```bash
python -m py_compile scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_l1risk_fairnoop_scene.py
```

Result: passed.

## Command

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 5 \
  --output_root /dev/shm/peilincai_spcarnet_v74_delta_cap_ladder_20260624 \
  --tag v74_deltacap_ladder_targetsupport_prerank_top1_countpyramid_blendladder_support4096_tex16_nearest_region_texture_adapter \
  --texture_size_candidates 16 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --surface_multiscale_prior_mode count_pyramid \
  --surface_multiscale_prior_block_sizes 2,4,6 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32.0 \
  --surface_multiscale_prior_blend 1.0 \
  --surface_multiscale_prior_blend_candidates 0,1.0 \
  --surface_multiscale_prior_gate_mode evidence_consistent \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --max_abs_delta_rgb 0.12 \
  --max_abs_delta_rgb_candidates 0.12,0.18,0.24 \
  --enable_target_support_candidate_selection \
  --target_support_prerank_top_k 1 \
  --target_support_prerank_max_views 8 \
  --min_target_changed_fraction 0.0 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --wandb_project SPCarNet \
  --wandb_group v74_delta_cap_ladder \
  --wandb_run_name v74_deltacap_ladder_counter_20260624 \
  --wandb_mode online \
  --force
```

W&B:

```text
run id: q9g7b7o9
url: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/q9g7b7o9
```

## Result

Scene: `counter`

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| v74 delta-cap ladder | 26.753995895 | 0.862119257 | 0.251853049 | completed, not promoted |
| v73b target-support pre-rank | 26.753995895 | 0.862119257 | 0.251853049 | tie |
| v73/v70/v71a zero-blend row | 26.753995895 | 0.862119257 | 0.251853049 | tie |
| selected v64/v56 reference | 26.756130219 | 0.862126231 | 0.251691371 | still stronger |

Delta versus selected v64/v56 counter reference:

```text
dPSNR  = -0.002134324
dSSIM  = -0.000006974
dLPIPS = +0.000161678
```

## Candidate Audit

The selected policy remains:

```text
support mode: fit_residual_topk
support added faces: 4096
texture size: 16
fill mode: nearest_observed
selected max_abs_delta_rgb: 0.12
selected blend: 0.0
selected alpha: 0.125
target changed fraction: 0.065630289
```

Policy-val candidate table:

| idx | cap | blend | accepted | alpha | relative gain | SSIM gain | image L1 gain | L1 positive view frac |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.12 | 0.0 | true | 0.125 | 0.026849788 | 0.000196626 | 0.000017083 | 0.916667 |
| 1 | 0.18 | 0.0 | true | 0.125 | 0.026849788 | 0.000196626 | 0.000017083 | 0.916667 |
| 2 | 0.24 | 0.0 | true | 0.125 | 0.026849788 | 0.000196626 | 0.000017083 | 0.916667 |
| 3 | 0.12 | 1.0 | true | 0.125 | 0.025892815 | 0.000190293 | 0.000016563 | 1.000000 |
| 4 | 0.18 | 1.0 | true | 0.125 | 0.025892815 | 0.000190293 | 0.000016563 | 1.000000 |
| 5 | 0.24 | 1.0 | true | 0.125 | 0.025892815 | 0.000190293 | 0.000016563 | 1.000000 |

The cap ladder did not change policy-val scores within each blend group. This indicates the current residual predictions are not materially clipped by the `0.12` cap on `counter`; residual amplitude clipping is not the active bottleneck for this scene.

## Evidence Paths

Persistent small artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v74_delta_cap_ladder_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v74_delta_cap_ladder_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v74_delta_cap_ladder_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v74_delta_cap_ladder_20260624/counter/surface_residual_region_texture_adapter_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v74_delta_cap_ladder_20260624/counter/apply_metrics_counter.log
```

Temporary full render artifact root:

```text
/dev/shm/peilincai_spcarnet_v74_delta_cap_ladder_20260624
```

## Conclusion

v74 closes an important policy-consistency interface, but it does not improve metrics on `counter`. The selected cap is the original conservative cap `0.12`; higher cap candidates have exactly the same policy-val scores under the same blend. This means the current bottleneck is not residual amplitude clipping. The next representation-level upgrade should target residual basis capacity or patch-level surface representation, not another scalar cap/blend sweep.
