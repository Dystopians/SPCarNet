# v78 Target-Footprint Certificate Running Log

Date: 2026-06-24  
Status: `PRE_FIX_COMPLETED_NEGATIVE_NOT_PROMOTED_AND_FIXED_CODE_FORMAL_RERUN_COMPLETED_NEGATIVE_NOT_PROMOTED`

## Purpose

v76 showed that policy-val bin-gain hybrid can select locally positive prior bins, but held-out metrics regressed. v77 tightened the bin certificate and correctly rejected the weak hybrid, but it did not improve the final counter result. v78 tests the next fair generalization certificate:

```text
policy-val-positive bin gain
  + target-view geometry/visibility footprint
  -> allow a prior bin only when it is also visible on held-out target views
```

The target-footprint certificate uses target-view surface addressability only: `face_id`, `barycentric`, optional `barycentric_valid`, and `alpha`. It does not read target GT RGB or target residuals. The intent is to make the local bin-gain hybrid less policy-val-overfit while staying fair for held-out evaluation.

## Implementation

Changed files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
scripts/car_model/summarize_v78_target_certificate_run.py
```

New adapter behavior:

- `build_target_bin_footprint_stats(...)` counts target-view coverage for face/UV bins without target GT.
- `build_policy_val_prior_bin_gain_hybrid_atlas(...)` accepts a target-footprint certificate.
- Policy-val-positive prior bins are kept only if they also pass target-footprint thresholds.
- Audit rows record `target_pixels`, `target_views`, `target_view_fraction`, and `target_footprint_keep`.

New CLI flags:

```text
--enable_target_footprint_bin_certificate
--target_footprint_min_bin_pixels
--target_footprint_min_views
--target_footprint_min_view_fraction
--target_footprint_max_views
```

Static validation already passed before the real runs:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  scripts/car_model/summarize_v78_target_certificate_run.py
```

Helper-level validation also passed: on `counter`, target evidence exposed 30 target views; with `max_views=2`, the footprint counter found `35046` target-covered face/bin keys over `1574` candidate faces and `144628` valid pixels.

## 2026-06-24 Audit Fix

The first v78 runs completed, but a code review found that several audit fields could be misleading even though the rendered metrics were valid:

- target-footprint `target_view_fraction` used only views with covered bins as the denominator, which could inflate the fraction;
- target-support best candidate was selected from all candidates, including policy-regressive candidates that were not eligible after the non-regression guard;
- target-footprint summary only described the selected candidate, so a non-hybrid selected candidate could hide candidate-level footprint evidence;
- W&B did not expose target-footprint certificate counters.

Fixes now in code:

- `target_views_used` is kept as a backward-compatible field but now means examined valid target surface-map views; `views_with_target_coverage` records the smaller coverage count separately;
- policy `score_order` is scoped to candidates eligible after the non-regression guard, with `global_score_order_top` retained only as diagnostic context;
- `target_support_candidate_selection.best_*` now refers to the eligible best candidate, while `global_best_*` is reported separately;
- hybrid-bin profile statistics are computed after final truncation, with pre-trunc counters kept separately;
- the v78 summarizer reports both selected-level and candidate-level target-footprint evidence;
- W&B logs target-footprint requested/selected/candidate counters.

Therefore, the first completed v78 runs below are valid negative metric diagnostics but should be treated as **pre-fix audit evidence**. The fixed-code formal reruns are documented later in this file.

## Experiments

Because `/data` is nearly full (`462M` available at the latest check), the formal reruns write to `/dev/shm`; only small audit artifacts are persisted under `outputs/`, not full checkpoint state dictionaries.

### v78 target-footprint bin certificate

Output root:

```text
/dev/shm/peilincai_spcarnet_v78_target_footprint_bin_certificate_20260624
```

Run directory:

```text
/dev/shm/peilincai_spcarnet_v78_target_footprint_bin_certificate_20260624/counter_v78_targetfootprint_bingain_counter_region_texture_adapter
```

Log:

```text
/dev/shm/peilincai_spcarnet_v78_target_footprint_bin_certificate_20260624/logs/apply_metrics_counter.log
```

W&B:

```text
project = SPCarNet
group = v78_target_footprint_bin_certificate
run = v78_targetfootprint_bingain_counter_20260624
id = l1f349kw
url = https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/l1f349kw
```

Key command:

```bash
WANDB_DIR=/dev/shm/wandb_spcarnet_v78 WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 3 \
  --output_root /dev/shm/peilincai_spcarnet_v78_target_footprint_bin_certificate_20260624 \
  --tag v78_targetfootprint_bingain_counter_region_texture_adapter \
  --texture_size_candidates 16 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_blend_candidates 0,0.5,1.0 \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --max_abs_delta_rgb 0.12 \
  --enable_policy_val_prior_bin_gain_hybrid \
  --prior_bin_gain_hybrid_min_bin_samples 4 \
  --prior_bin_gain_hybrid_min_views 1 \
  --prior_bin_gain_hybrid_min_abs_gain 0.0 \
  --prior_bin_gain_hybrid_min_relative_gain 0.0 \
  --prior_bin_gain_hybrid_min_positive_view_fraction 0.5 \
  --enable_target_footprint_bin_certificate \
  --target_footprint_min_bin_pixels 16 \
  --target_footprint_min_views 2 \
  --target_footprint_min_view_fraction 0.0 \
  --target_footprint_max_views 8 \
  --enable_target_support_candidate_selection \
  --target_support_prerank_top_k 1 \
  --target_support_prerank_max_views 8 \
  --min_target_changed_fraction 0.0 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --wandb_project SPCarNet \
  --wandb_group v78_target_footprint_bin_certificate \
  --wandb_run_name v78_targetfootprint_bingain_counter_20260624 \
  --wandb_mode online \
  --force
```

Final status:

- completed on GPU 3;
- final `results.json`, `per_view.json`, and `surface_residual_region_texture_adapter_audit.json` were written;
- persistent artifacts were copied from `/dev/shm` to `/data` under `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_footprint_bin_certificate_20260624`;
- this is a pre-fix audit run and should not be used as a formal promoted endpoint.

### v78 target-support certificate ablation

Output root:

```text
/dev/shm/peilincai_spcarnet_v78_target_support_cert_20260624
```

Run directory:

```text
/dev/shm/peilincai_spcarnet_v78_target_support_cert_20260624/counter_v78_target_support_cert_counter_region_texture_adapter
```

Log:

```text
/dev/shm/peilincai_spcarnet_v78_target_support_cert_20260624/logs/apply_metrics_counter.log
```

W&B:

```text
project = SPCarNet
group = v78_target_support_certificate_audit
run = v78_target_support_certificate_counter_20260624
```

This ablation enables target-support candidate selection and target-support pre-rank, but not the target-footprint bin certificate or policy-val bin-gain hybrid. It is useful for separating the cost/benefit of target-support selection from the new target-footprint hybrid gate.

Final status:

- completed on GPU 3;
- final `results.json`, `per_view.json`, and `surface_residual_region_texture_adapter_audit.json` were written;
- persistent artifacts were copied from `/dev/shm` to `/data` under `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_support_certificate_audit_20260624`;
- this is an ablation, not a promoted endpoint.

## Results

### Counter Metrics

| row | PSNR | SSIM | LPIPS | decision |
|---|---:|---:|---:|---|
| v78 target-footprint bin certificate | `26.753528595` | `0.862111032` | `0.251881272` | not promoted |
| v78 target-support ablation | `26.753528595` | `0.862111032` | `0.251881331` | not promoted |
| v77 strict bin-gain hybrid | `26.753528595` | `0.862111032` | `0.251881331` | stronger/equal reference for this line |
| v76 policy-val bin-gain hybrid | `26.753532410` | `0.862111092` | `0.251881331` | weak hybrid, not promoted |
| v75 zero-blend local patch | `26.753995895` | `0.862119257` | `0.251853049` | stronger |
| v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | stronger |

### Deltas

| comparison | dPSNR | dSSIM | dLPIPS | strict RGB win |
|---|---:|---:|---:|---:|
| v78 target-footprint vs v64/v56 | `-0.002601624` | `-0.000015199` | `+0.000189901` | `False` |
| v78 target-footprint vs v75 | `-0.000467300` | `-0.000008225` | `+0.000028223` | `False` |
| v78 target-support ablation vs v64/v56 | `-0.002601624` | `-0.000015199` | `+0.000189960` | `False` |
| v78 target-support ablation vs v75 | `-0.000467300` | `-0.000008225` | `+0.000028282` | `False` |

### Selected Policies

| row | accepted | alpha | blend | hybrid | allowed bins | target certificate |
|---|---:|---:|---:|---:|---:|---:|
| v78 target-footprint | `True` | `0.125` | `1.0` | `True` | `1746 / 233306` | selected target-support certificate `True` |
| v78 target-support ablation | `True` | `0.125` | `0.0` | `False` | `0 / 0` | selected target-support certificate `True` |

Interpretation:

- Target-footprint certificate reduced v76-style allowed hybrid bins from `13708` to `1746`, but the selected hybrid still did not generalize beyond v75 or v64/v56.
- The target-support ablation correctly selects the same zero-blend safe line as v77, but that line remains below v75 and the v64/v56 reference.
- The method addition is real and fair, but the effect is not enough for promotion.

## Promotion Criteria

v78 should not become a promoted endpoint unless it satisfies all of the following on `counter` first:

1. Strictly beats v75 zero-blend local patch: `26.753995895 / 0.862119257 / 0.251853049`.
2. Strictly beats the v64/v56 counter reference: `26.756130219 / 0.862126231 / 0.251691371`.
3. Has a clear audit explanation for why target-footprint certificate selected a safer candidate than v76.
4. Does not rely on target GT or held-out metrics for policy selection.
5. Has reasonable runtime cost, or a cheaper fixed variant is derived before any full9 run.

If v78 only matches v75 or remains below v64/v56, it should be documented as a negative diagnostic and not promoted.

## Fixed-Code Formal Reruns

Because the first v78 metrics were produced before the audit-interface fixes above, two fixed-code formal reruns were launched and completed. These runs use the corrected eligible-candidate target-support selection, corrected target-footprint denominator, candidate-level footprint summary, final post-truncation bin counters, and expanded W&B logging.

### v78b target-footprint formal rerun

```text
status = COMPLETED_NOT_PROMOTED
output_root = /dev/shm/peilincai_spcarnet_v78b_target_footprint_formal_20260624
run_dir = /dev/shm/peilincai_spcarnet_v78b_target_footprint_formal_20260624/counter_v78b_targetfootprint_formal_counter_region_texture_adapter
log = /dev/shm/peilincai_spcarnet_v78b_target_footprint_formal_20260624/logs/apply_metrics_counter.log
persistent_summary = outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_footprint_formal_20260624/summary.md
wandb_group = v78b_target_footprint_formal_audit
wandb_run = v78b_targetfootprint_formal_counter_20260624
wandb_id = 7pz9pulx
gpu = 2
```

Main command shape:

```bash
WANDB_DIR=/dev/shm/wandb_spcarnet_v78b WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v78b_target_footprint_formal_20260624 \
  --tag v78b_targetfootprint_formal_counter_region_texture_adapter \
  --enable_policy_val_prior_bin_gain_hybrid \
  --enable_target_footprint_bin_certificate \
  --enable_target_support_candidate_selection \
  --wandb_project SPCarNet \
  --wandb_group v78b_target_footprint_formal_audit \
  --wandb_run_name v78b_targetfootprint_formal_counter_20260624 \
  --wandb_mode online --force
```

### v78b target-support formal ablation

```text
status = COMPLETED_NOT_PROMOTED
output_root = /dev/shm/peilincai_spcarnet_v78b_target_support_formal_20260624
run_dir = /dev/shm/peilincai_spcarnet_v78b_target_support_formal_20260624/counter_v78b_target_support_formal_counter_region_texture_adapter
log = /dev/shm/peilincai_spcarnet_v78b_target_support_formal_20260624/logs/apply_metrics_counter.log
persistent_summary = outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_support_formal_20260624/summary.md
wandb_group = v78b_target_support_formal_audit
wandb_run = v78b_target_support_formal_counter_20260624
wandb_id = fvfj1s4q
gpu = 3
```

Main command shape:

```bash
WANDB_DIR=/dev/shm/wandb_spcarnet_v78b WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter --gpu 3 \
  --output_root /dev/shm/peilincai_spcarnet_v78b_target_support_formal_20260624 \
  --tag v78b_target_support_formal_counter_region_texture_adapter \
  --enable_target_support_candidate_selection \
  --wandb_project SPCarNet \
  --wandb_group v78b_target_support_formal_audit \
  --wandb_run_name v78b_target_support_formal_counter_20260624 \
  --wandb_mode online --force
```

Promotion rule for v78b is unchanged: do not promote unless the formal summary strictly beats both v75 and the v64/v56 counter reference and the corrected certificate fields explain the selected policy. The completed v78b runs do not satisfy this rule.

### v78b formal rerun results

| row | PSNR | SSIM | LPIPS | decision |
|---|---:|---:|---:|---|
| v78b target-footprint formal | `26.753528595` | `0.862111032` | `0.251881272` | not promoted |
| v78b target-support formal ablation | `26.753528595` | `0.862111032` | `0.251881331` | not promoted |
| v75 zero-blend local patch | `26.753995895` | `0.862119257` | `0.251853049` | stronger |
| v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | stronger |

| comparison | dPSNR | dSSIM | dLPIPS | strict RGB win |
|---|---:|---:|---:|---:|
| v78b target-footprint vs v64/v56 | `-0.002601624` | `-0.000015199` | `+0.000189901` | `False` |
| v78b target-footprint vs v75 | `-0.000467300` | `-0.000008225` | `+0.000028223` | `False` |
| v78b target-support vs v64/v56 | `-0.002601624` | `-0.000015199` | `+0.000189960` | `False` |
| v78b target-support vs v75 | `-0.000467300` | `-0.000008225` | `+0.000028282` | `False` |

Corrected audit evidence from the v78b target-footprint run:

- selected policy: `accepted=True`, `alpha=0.125`, `blend=1.0`, `hybrid=True`;
- score order scope: `eligible_after_nonreg_guard`;
- target-support certificate: selected and best certificate both pass;
- target-footprint enabled: `True`;
- allowed hybrid bins: `1746 / 233306` (`0.007483734`);
- selected target covered bins: `323434`;
- selected candidate bins with target footprint: `154767`;
- selected pre-trunc and final allowed bins with target footprint: `1746`;
- target footprint views examined: `8`;
- views with target coverage: `8`.

This confirms the audit-interface fix: v78b no longer inflates target-view fraction by using only covered views as denominator, and it reports candidate-level target-footprint evidence. The negative metric result therefore reflects the method limitation rather than an audit bookkeeping error.

## Next Actions

The completed summaries are persisted here:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_footprint_bin_certificate_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_footprint_bin_certificate_20260624/summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_footprint_bin_certificate_20260624/counter_v78_targetfootprint_bingain_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_footprint_bin_certificate_20260624/counter_v78_targetfootprint_bingain_counter_region_texture_adapter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_footprint_bin_certificate_20260624/counter_v78_targetfootprint_bingain_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_support_certificate_audit_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_support_certificate_audit_20260624/summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_support_certificate_audit_20260624/counter_v78_target_support_cert_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_support_certificate_audit_20260624/counter_v78_target_support_cert_counter_region_texture_adapter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78_target_support_certificate_audit_20260624/counter_v78_target_support_cert_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_footprint_formal_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_footprint_formal_20260624/summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_footprint_formal_20260624/counter_v78b_targetfootprint_formal_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_footprint_formal_20260624/counter_v78b_targetfootprint_formal_counter_region_texture_adapter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_footprint_formal_20260624/counter_v78b_targetfootprint_formal_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_support_formal_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_support_formal_20260624/summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_support_formal_20260624/counter_v78b_target_support_formal_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_support_formal_20260624/counter_v78b_target_support_formal_counter_region_texture_adapter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_target_support_formal_20260624/counter_v78b_target_support_formal_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
```

The exact summarizer command was:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v78_target_certificate_run.py \
  --run_dir /dev/shm/peilincai_spcarnet_v78_target_footprint_bin_certificate_20260624/counter_v78_targetfootprint_bingain_counter_region_texture_adapter \
  --output_json /dev/shm/peilincai_spcarnet_v78_target_footprint_bin_certificate_20260624/summary.json \
  --output_md /dev/shm/peilincai_spcarnet_v78_target_footprint_bin_certificate_20260624/summary.md
```

The same summarizer was also run for the target-support ablation and both fixed-code v78b formal reruns. Because `/data` is full, only small audit/result/log files were copied for v78b; full checkpoint state remained in `/dev/shm` during the run.

## Current Interpretation

v78/v78b is a real method change in the train/eval pipeline and directly addresses the v76/v77 failure analysis. The final fixed-code result is negative: target-footprint certification makes the weak hybrid more selective and its audit accounting is now trustworthy, but the selected candidate is still not good enough to beat v75 or the v64/v56 counter reference. The current headline remains Phase-J, and the best fixed representation-level line remains v64. Future work should stop adding scalar gates around the same low-capacity atlas and instead change the representation capacity or training objective.
