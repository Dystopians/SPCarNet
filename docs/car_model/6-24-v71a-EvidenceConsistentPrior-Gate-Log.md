# v71a Evidence-Consistent Prior Gate Log

Date: 2026-06-24

Status: `COMPLETED_NOT_PROMOTED`

Persistent artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624
```

## Motivation

v69 introduced a count-pyramid multi-scale prior for low-support surface atlas bins, but the fixed strong blend over-smoothed or mis-propagated residuals and regressed below v68/v64 references. v70 added a train-only policy-val blend ladder and correctly selected `blend=0.0`, proving that nonzero prior blending was unsafe under the current policy.

v71a tests a more selective version: keep the count-pyramid prior available, but only let a low-support bin use it when local evidence and coarse prior agree. The goal was to see whether nonzero prior use can become safe if we reject bins with poor direct support, weak prior confidence, inconsistent sign, high variance, or opposing residual direction.

## Implemented Method Change

New adapter/runner flags:

```text
--surface_multiscale_prior_gate_mode evidence_consistent
--surface_multiscale_prior_min_prior_weight 0.05
--surface_multiscale_prior_min_direct_samples 1
--surface_multiscale_prior_min_sign_consistency 0.5
--surface_multiscale_prior_max_mean_variance 0.004
--surface_multiscale_prior_min_cosine 0.0
```

Touched files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

Static validation:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

The adapter and runner help output were also checked to confirm the new CLI args are exposed.

## Execution

Common configuration:

```text
--support_expansion_mode fit_residual_topk
--support_expansion_max_extra_faces 4096
--texture_size_candidates 16
--atlas_empty_bin_fill_mode nearest_observed
--surface_multiscale_prior_mode count_pyramid
--surface_multiscale_prior_block_sizes 2,4,6
--surface_multiscale_prior_min_bin_samples 8
--surface_multiscale_prior_count_tau 32
--surface_multiscale_prior_blend 1.0
--surface_multiscale_prior_blend_candidates 0,1.0
--surface_multiscale_prior_gate_mode evidence_consistent
--surface_multiscale_prior_min_prior_weight 0.05
--surface_multiscale_prior_min_direct_samples 1
--surface_multiscale_prior_min_sign_consistency 0.5
--surface_multiscale_prior_max_mean_variance 0.004
--surface_multiscale_prior_min_cosine 0.0
--view_conditioned_basis_mode normal_camera_linear
--view_conditioned_basis_guard_mode policy_val_nonregressive
--view_conditioned_basis_min_bin_samples 16
--view_conditioned_basis_ridge 0.1
--view_conditioned_basis_ood_mode diag_z
--view_conditioned_basis_ood_max_z 2.5
--view_conditioned_basis_ood_min_std 0.05
--min_policy_val_l1_positive_view_fraction 0.9
--min_target_changed_fraction 0.0
--wandb_project SPCarNet
--wandb_group v71a_evidence_consistent_prior
--wandb_mode online
```

W&B:

| scene | run id | run name |
|---|---|---|
| counter | `zw6b5w63` | `v71a_evidence_consistent_counter_retry1_20260624` |
| kitchen | `fu52k6y1` | `v71a_evidence_consistent_kitchen_retry1_20260624` |

An initial attempt failed with:

```text
ValueError: unsupported surface multiscale prior gate mode: evidence_consistent
```

The bug was fixed by validating `gate_mode in {"none", "evidence_consistent"}` before applying the active-bin gate, and by allowing the no-active-bin path to return cleanly.

## Results

| scene | PSNR | SSIM | LPIPS | selected blend | selected alpha | accepted |
|---|---:|---:|---:|---:|---:|---:|
| counter | `26.753996` | `0.862119` | `0.251853` | `0.0` | `0.125` | yes |
| kitchen | `27.819157` | `0.876533` | `0.199031` | `0.0` | `0.125` | yes |

Comparison:

| scene | v71a vs v70 dPSNR | v71a vs v70 dSSIM | v71a vs v70 dLPIPS | v71a vs selected reference dPSNR | v71a vs selected reference dSSIM | v71a vs selected reference dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| counter | `+0.000000` | `+0.000000000` | `+0.000000000` | `-0.002134` | `-0.000006743` | `+0.000162049` |
| kitchen | `+0.000000` | `+0.000000000` | `+0.000000000` | `-0.003469` | `-0.000004790` | `+0.000181846` |

The selected reference rows are:

```text
counter: 26.756130 / 0.862126 / 0.251691
kitchen: 27.822626 / 0.876538 / 0.198849
```

## Audit

Both scenes selected zero prior blend:

```text
selected_surface_multiscale_prior_blend = 0.0
guard = zero_blend_or_base_face_mean_nonregressive_relative_ssim_l1_cvar_min_view
```

Therefore the selected model has no active prior blending:

| scene | gate mode | blended bins | gate rejected bins | target changed fraction |
|---|---|---:|---:|---:|
| counter | `evidence_consistent` | `0` | `0` | `0.065630289` |
| kitchen | `evidence_consistent` | `0` | `0` | `0.039584677` |

Policy-val evidence for the selected zero-blend rows was positive:

| scene | relative positive view fraction | SSIM gain | SSIM positive fraction | image-L1 gain | image-L1 positive fraction |
|---|---:|---:|---:|---:|---:|
| counter | `1.0` | `+0.000196626` | `1.0` | `+0.000017083` | `0.916667` |
| kitchen | `1.0` | `+0.000178893` | `1.0` | `+0.000019656` | `1.0` |

## Interpretation

v71a is a real method and interface change, but it is not a promoted endpoint. The train-only policy correctly refuses nonzero prior use again, so v71a exactly matches v70 on the tested scenes and remains below the selected v64/v56 references.

The result strengthens the bottleneck diagnosis:

- The problem is not just that v69 used too much prior blend.
- Even with evidence-consistency gates, the current same-face coarse-block prior does not create a policy-val-safe nonzero improvement on `counter` or `kitchen`.
- Continuing to tune scalar thresholds around this prior is unlikely to produce a paper-level breakthrough.
- The next useful direction is representation capacity and target-view support certification, not another blend/gate sweep.

## Runtime Note

The current candidate implementation refits the atlas for every support/texture/fill/prior-blend candidate. With `view_conditioned_basis_guard_mode=policy_val_nonregressive`, each candidate can also trigger a legacy refit for comparison. This made the two-scene v71a run substantially slower than v68/v69/v70.

Before running full9 with similar candidate ladders, add one of:

- cached legacy atlas reuse across prior-blend candidates;
- shared direct atlas fit with cheap prior-only recomputation;
- staged policy: first evaluate zero-blend and one gated prior on a reduced policy-val subset, then run full policy-val only for survivors.

## Artifact Files

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/kitchen/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/kitchen/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/kitchen/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/logs/apply_metrics_counter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/logs/apply_metrics_kitchen.log
```

