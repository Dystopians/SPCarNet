# v85 Target-Footprint Tail-Risk Certificate Log

Date: `2026-06-24`

Status: `COMPLETED_DIAGNOSTIC_NOT_PROMOTED`

## Motivation

The latest representation-level probes exposed a specific failure mode:

- v82b can make a counter-level strict micro-win, but raw hard-triad validation is not stable.
- v83 improves counter PSNR/LPIPS but slightly regresses SSIM.
- The existing target-footprint bin certificate only checks whether a copied face/UV bin is covered by the target split and whether its average policy-val residual error improves.

This leaves a tail-risk gap: a bin can have positive mean policy-val gain while still hurting some policy-val views. The v85 change adds a default-off certificate that rejects target-covered hybrid bins unless their per-view policy-val residual-error gain is tail-safe.

## Method Change

Implemented in:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter flags:

```text
--enable_target_footprint_tail_risk_certificate
--target_footprint_tail_risk_all_bins
--target_footprint_tail_risk_min_positive_view_fraction
--target_footprint_tail_risk_min_min_view_gain
--target_footprint_tail_risk_min_cvar20_view_gain
```

Default behavior is unchanged because the new certificate is disabled unless explicitly requested.

When enabled, the policy-val prior-bin hybrid now records per-bin view gains:

```text
view_gain = baseline_policy_val_residual_error - prior_policy_val_residual_error
```

For each candidate bin it computes:

```text
mean_view_gain
min_view_gain
cvar20_view_gain
positive_view_fraction
```

For target-covered bins, the bin is copied from the prior atlas into the hybrid atlas only if:

```text
positive_view_fraction >= threshold
min_view_gain >= threshold
cvar20_view_gain >= threshold
```

The target-side footprint still uses GT-free geometry/evidence only: face id, barycentric coordinates, alpha, and target coverage. It does not read target RGB GT.

## Verification So Far

Passed:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py --help | \
  rg -n "target_footprint_tail_risk|target_footprint_bin_certificate|policy_val_prior_bin_gain"

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py --help | \
  rg -n "target_footprint_tail_risk|target_footprint_bin_certificate|policy_val_prior_bin_gain"

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  docs/car_model/6-24-SPCarNet-Current-Complete-Method-Experiment-Report-With-Render-Comparisons.zh.md
```

The first adapter help check found an argparse formatting bug caused by `20%` in help text; it was fixed by escaping it as `20%%`, and the help check then passed.

## Subagent Review Fixes

Read-only subagent review found two audit/interface issues after the first implementation:

1. If tail-risk rejected every hybrid bin, the disabled hybrid profile could omit the tail-risk rejection counters.
2. The tail-risk flag could be requested without enabling the prior-bin hybrid feature where the gate is actually applied.

Both were fixed:

- disabled `no_bins_passed_hybrid_gate` profiles now carry `target_footprint_tail_risk_certificate` counts and `top_bins`;
- both the adapter and `run_l1risk_fairnoop_scene.py` now error if `--enable_target_footprint_tail_risk_certificate` is used without `--enable_policy_val_prior_bin_gain_hybrid`;
- compile/help/diff checks were rerun and passed after these fixes.

These fixes improve auditability and CLI correctness. They do not change the core per-bin keep predicate used by the completed counter probe.

## Counter Probe

W&B:

```text
project: SPCarNet
group: v85_target_tailrisk
run: wdahvese
name: v85_target_tailrisk_counter_20260625
```

Output root:

```text
/dev/shm/peilincai_spcarnet_v85_tailrisk_20260625
```

Command:

```text
WANDB_DIR=/dev/shm/wandb_spcarnet_v85_tailrisk WANDB_MODE=online CUDA_VISIBLE_DEVICES=2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v85_tailrisk_20260625 \
  --tag v85_target_tailrisk_counter_region_texture_adapter \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --enable_policy_val_face_alpha_calibration \
  --face_alpha_calibration_max_alpha 0.5 \
  --face_alpha_calibration_min_alpha 0.0 \
  --face_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --face_alpha_calibration_min_face_samples 256 \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_block_sizes 1,2,3 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32.0 \
  --surface_multiscale_prior_blend 1.0 \
  --surface_multiscale_prior_blend_candidates 0,0.5,1.0 \
  --enable_policy_val_prior_bin_gain_hybrid \
  --prior_bin_gain_hybrid_min_bin_samples 4 \
  --prior_bin_gain_hybrid_min_views 1 \
  --prior_bin_gain_hybrid_min_abs_gain 0.0 \
  --prior_bin_gain_hybrid_min_relative_gain 0.0 \
  --prior_bin_gain_hybrid_min_positive_view_fraction 0.5 \
  --enable_target_footprint_bin_certificate \
  --target_footprint_min_bin_pixels 8 \
  --target_footprint_min_views 1 \
  --target_footprint_min_view_fraction 0.0 \
  --enable_target_footprint_tail_risk_certificate \
  --target_footprint_tail_risk_min_positive_view_fraction 1.0 \
  --target_footprint_tail_risk_min_min_view_gain 0.0 \
  --target_footprint_tail_risk_min_cvar20_view_gain 0.0 \
  --view_conditioned_basis_mode none \
  --teacher_distilled_basis_mode none \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.001 \
  --wandb_project SPCarNet \
  --wandb_group v85_target_tailrisk \
  --wandb_run_name v85_target_tailrisk_counter_20260625 \
  --wandb_mode online \
  --force
```

Runtime notes:

- The run started successfully on GPU 2 with low GPU memory pressure.
- It is CPU-heavy during atlas fitting and policy-val evaluation.
- Candidate 1/6 completed and was accepted by policy-val gates with alpha `0.03125`; all later candidates also completed before the final hybrid selector.

Final local result:

- The run completed on GPU 2 with W&B online.
- The selected final candidate was:

```text
hybrid baseline=4/6 support=fit_residual_topk added=4096 faces=5670 texture=32 fill=nearest_observed prior_blend=0 cap=0.12 source=6/6 support=fit_residual_topk added=4096 faces=5670 texture=32 fill=nearest_observed prior_blend=1 cap=0.12
```

- Audit status:

```text
accepted: true
effective_policy: accepted_atlas
selected_alpha: 0.5
changed_fraction: 0.06390131774758576
```

- Policy-val gate evidence:

```text
selected_positive_view_fraction: 1.0
selected_cvar20_view_relative_gain: 0.024758915137043713
selected_min_view_relative_gain: 0.022817611734672175
selected_ssim_gain: 0.0002939055363337199
selected_ssim_positive_view_fraction: 1.0
selected_ssim_min_view_gain: 0.00006026029586791992
selected_image_l1_gain: 0.000026771643509467442
selected_image_l1_positive_view_fraction: 0.9166666666666666
selected_image_l1_min_view_gain: -0.0000008121132850646973
selected_image_l1_cvar20_view_gain: 0.000002679725488026937
```

Final held-out `counter` metrics:

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| v85 target-footprint tail-risk | `26.756134033` | `0.862126231` | `0.251691371` | diagnostic / not promoted |
| v56/v64/v79 anchor | `26.756130219` | `0.862126231` | `0.251691371` | reference |
| v84 selector counter row | `26.756137848` | `0.862126350` | `0.251690656` | stronger counter micro-result |
| delta vs anchor | `+0.000003814` | `+0.000000000` | `+0.000000000` | micro-tie |

This is materially better than the strict SSIM-safe fallback probe, but it is not
a meaningful improvement over the existing `counter` anchor and remains below or
mixed against the v82b/v83 single-scene probes. It is therefore an implementation
and safety-certificate milestone, not a promoted method endpoint.

## Parallel Strict SSIM-Safe Patchmix Probe

A pre-existing parallel probe also completed:

```text
root: /dev/shm/peilincai_spcarnet_v85_ssimsafe_prerank_patchmix_20260624
tag: v85_ssimsafe_prerank_patchmix_facealpha_localpatch_hybrid_tex32_support4096_8192_region_texture_adapter
W&B group: v85_ssimsafe_prerank_patchmix
```

It tested target support pre-rank, `fit_residual_topk_8192`, face-alpha
calibration, local-patch prior, prior-bin gain hybrid, patch-mixture teacher
basis, and strict policy-val SSIM/L1 gates.

All three candidates were rejected by the strict policy gates despite strong
policy-val residual/SSIM/L1 signals, so the effective target policy was a no-op
fallback:

```text
accepted: false
effective_policy: fallback_noop
selected_alpha: 0.0
changed_pixels: 0
changed_fraction: 0.0
```

Final counter test metrics:

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| v85 ssimsafe/prerank/patchmix | `26.749835968` | `0.862049341` | `0.251998007` | not promoted |
| v56/v64/v79 anchor | `26.756130219` | `0.862126231` | `0.251691371` | reference |

This is a useful safety result, not an improvement: the gates prevented test
edits, but the fallback remains below the counter anchor on all three metrics.

Persisted evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/apply_metrics_counter.log
```

## Target-Footprint Tail-Risk Persisted Evidence

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_target_tailrisk_counter_20260625/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_target_tailrisk_counter_20260625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_target_tailrisk_counter_20260625/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_target_tailrisk_counter_20260625/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_target_tailrisk_counter_20260625/logs/apply_metrics_counter.log
```

## Promotion Rule

This probe is not promoted unless it strictly improves the current counter anchor:

```text
v56/v64/v79 counter anchor:
PSNR 26.756130219
SSIM 0.862126231
LPIPS 0.251691371
```

The final v85 tail-risk run is effectively tied with this anchor rather than
strictly improving it. The next required method step is not to promote v85 as-is,
but to make the fallback/selector preserve the strongest existing anchor while
using the tail-risk certificate only as a safety filter for genuinely stronger
candidate edits.

## Relaxed Tail-Risk Diagnostic

Helmholtz subagent analysis found that the strict tail-risk certificate likely
starves the edit: the successful strict run copied only a small fraction of
candidate bins and landed at an anchor-level micro-tie. To test whether this is
a gate-starvation issue rather than a representation-capacity issue, a single
relaxed diagnostic was scheduled:

```text
--target_footprint_tail_risk_min_positive_view_fraction 0.75
--target_footprint_tail_risk_min_min_view_gain -0.0000001
--target_footprint_tail_risk_min_cvar20_view_gain 0.0
```

The first launch created W&B run `xpyo4yj8` but failed before the experiment
properly started. The runner formatted `-0.0000001` as `-1e-07`; the adapter's
argparse path treated that scientific-notation negative value as a missing
argument:

```text
error: argument --target_footprint_tail_risk_min_min_view_gain: expected one argument
```

This was fixed in `scripts/car_model/run_l1risk_fairnoop_scene.py` by forwarding
tail-risk float gates through `fmt_gate_float`, producing fixed decimal strings
such as `-0.000000100000`. Compile and diff checks passed after the patch.

The corrected rerun is active:

```text
W&B run: 4980e5pj
name: v85_tailrisk_relax075_counter_rerun_20260625
root: /dev/shm/peilincai_spcarnet_v85_tailrisk_relax2_20260625
```

Promotion threshold remains unchanged: the relaxed diagnostic must strictly beat
the counter anchor on PSNR, SSIM, and LPIPS before any hard-triad/full9 run.
