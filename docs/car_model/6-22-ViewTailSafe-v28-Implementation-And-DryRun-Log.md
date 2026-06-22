# 6-22 View-Tail-Safe Alpha Shrink v28 Implementation And Dry-Run Log

Date: 2026-06-22

Status: implemented, smoke-tested, dry-run command-verified. Not yet a promoted result.

## Motivation

The v26/v27 Bonsai medium runs exposed two different failure modes:

- v26 hard local-trust was too conservative in the render layer and still failed the selector replay.
- v27 soft local-trust restored nonzero residual transfer, but the train-val view tail remained unsafe.

The key diagnosis is that adaptive alpha was safe only at pooled pixel/bin level. It did not explicitly require the selected alpha map to be safe across policy-validation views. A bin can have nonnegative sampled pixel MSE tail while still creating view-level balanced/LPIPS regressions.

## Method Change

v28 adds **view-tail-safe alpha shrink** to the ELA adaptive alpha calibrator.

After the normal per-bin alpha table is fitted, the calibrator replays a global scale grid on the policy-validation samples:

```text
alpha_final(pixel) = view_tail_scale * alpha_bin(pixel)
```

The selected `view_tail_scale` is chosen from a fixed grid by policy-view evidence only. For each candidate scale, v28 computes per-view mean MSE gain and measures:

- mean view gain;
- worst-tail CVaR view gain;
- negative-view fraction.

The profile accepts the highest-utility safe scale. If no scale satisfies the safety constraints, it falls back to the scale with the best tail evidence; the v28 grid includes `0.0`, so no-op remains available.

This is a real method change in the train/eval pipeline: it changes the render-time residual alpha map used by ELA and is written into the `alpha_calibrator` JSON report.

## Implemented Interfaces

Core implementation:

- `utils/evidence_lumigraph_adapter.py`
  - `AlphaCalibrator.view_tail_scale`;
  - `AlphaCalibrator.view_tail_*` diagnostics in `to_json()`;
  - per-scale `view_tail_candidate_stats` with `scale`, `mean_gain`, `cvar_gain`,
    `negative_fraction`, and `safe`;
  - explicit `view_tail_safe_scale_found` and `view_tail_fallback_used` flags;
  - `alpha_map()` multiplies the per-bin alpha map by `view_tail_scale`;
  - `fit_alpha_calibrator()` now supports policy-view scale selection.

ELA CLI:

- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
  - `--alpha_view_tail_scale_grid`;
  - `--alpha_view_tail_cvar_fraction`;
  - `--alpha_view_tail_min_gain`;
  - `--alpha_view_tail_max_negative_fraction`;
  - W&B scalar logging for view-tail enablement, selected scale, safe/fallback
    status, mean gain, CVaR gain, and negative-view fraction;
  - fail-closed check: `--alpha_region_risk_enable` now requires an existing `--alpha_region_risk_json`.

PhaseK / selector / AutoVisual:

- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
  - forwards `ela_alpha_view_tail_*` to ELA;
  - fails closed if region-risk is enabled but the per-scene JSON is missing.
  - fails fast if a v28 view-tail grid is supplied while fixed Phase-J policy
    replay points to a non-`adaptive_bins` report. This prevents silent no-op
    v28 runs on scenes whose selected Phase-J row uses global alpha.
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
  - forwards `ela_alpha_view_tail_*` to PhaseK;
  - fails closed if `--ela_alpha_region_risk_enable` lacks a valid template/path.
  - selector train-val per-view tail diagnostics now use explicit balanced
    weights, defaulting to `20 * dSSIM - 20 * dLPIPS`, matching the PhaseK gate
    default instead of the older hidden `100/10` formula.
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py`
  - fixed profile `field_region_render_risk_strict_v28`;
  - contract id `field_region_render_risk_strict_v28_view_tail_safe_alpha_shrink`;
  - forwards `--selector_balanced_ssim_weight 20.0` and
    `--selector_balanced_lpips_weight 20.0` into selector commands;
  - profile defaults:

```text
ela_alpha_view_tail_scale_grid = 1.0,0.75,0.5,0.25,0.0
ela_alpha_view_tail_cvar_fraction = 0.25
ela_alpha_view_tail_min_gain = 0.0
ela_alpha_view_tail_max_negative_fraction = 0.50
```

Smoke coverage:

- `scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py`
  - now checks that `fit_alpha_calibrator()` reports `view_tail_enabled=true`
    when a scale grid is supplied;
  - checks that the new safe/fallback flags and per-scale candidate stats are present.

## Subagent Audit Follow-Up

A read-only audit found one important pre-run risk: with
`--ela_policy_source fixed_phasej`, v28 can only be active when the selected
Phase-J report uses `alpha_policy="adaptive_bins"`. If a selected fixed Phase-J
row uses global alpha, the v28 grid would otherwise appear in command manifests
without reaching the adaptive alpha calibrator.

The code now fails fast in that case. For multi-scene v28 validation, either
verify that each target selected Phase-J report is adaptive, or run with a
per-model auto policy that fits adaptive bins for the candidate. After any real
run, the first report check should verify:

```text
alpha_calibrator.view_tail_enabled = true
alpha_calibrator.view_tail_candidate_stats is non-empty
alpha_calibrator.view_tail_safe_scale_found / view_tail_fallback_used are recorded
```

## Verification Commands

Static compilation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  utils/evidence_lumigraph_adapter.py \
  scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

Result: passed.

Whitespace check:

```bash
git diff --check -- \
  utils/evidence_lumigraph_adapter.py \
  scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

Result: passed.

ELA smoke:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

Result:

```text
[ELA smoke] passed
```

CLI visibility:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py --help | rg 'field_region_render_risk_strict_v28|ela_alpha_view_tail'

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py --help | rg 'ela_alpha_view_tail'

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py --help | rg 'alpha_view_tail'
```

Result: v28 profile and all view-tail CLI flags are visible.

Dry-run command manifest:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v28 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 0 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v28_viewtail_20260622 \
  --pipeline_label dryrun_field_region_render_risk_strict_v28_viewtail_20260622 \
  --wandb_mode disabled \
  --dry_run \
  --force
```

Result:

```text
commands: 8
dry_run: true
```

Evidence:

- `/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v28_viewtail_20260622/pipeline_command_manifest.json`
- `/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v28_viewtail_20260622/pipeline_command_manifest.md`
- `/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v28_viewtail_20260622/pipeline_summary.md`

The manifest confirms that plan, candidate-owned refit, and selector commands include:

```text
--ela_alpha_view_tail_scale_grid 1.0,0.75,0.5,0.25,0.0
--ela_alpha_view_tail_cvar_fraction 0.25
--ela_alpha_view_tail_min_gain=0.0
--ela_alpha_view_tail_max_negative_fraction 0.5
--selector_balanced_ssim_weight 20.0
--selector_balanced_lpips_weight 20.0
```

## Current Experimental Status

No v28 medium/long result has been promoted yet. GPU4/GPU5 were still occupied by existing v26/v27 Bonsai jobs when this implementation landed, so the first real v28 run should be launched after those jobs finish or a suitable GPU opens.

Recommended first real run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v28 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu <LOW_OCCUPANCY_GPU> \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu<GPU> \
  --pipeline_label field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu<GPU> \
  --wandb_mode online \
  --force
```

## Acceptance Criteria

v28 should only be promoted if it improves the failure pattern seen in v27:

- `trainval_balanced_delta >= 0`;
- view-tail `balanced_cvar_delta` no longer large negative;
- `lpips_positive_fraction` decreases materially from v27;
- `alpha_calibrator.view_tail_enabled = true`;
- `alpha_calibrator.view_tail_scale` is recorded and justified by policy-view tail stats;
- if region-risk is enabled, `alpha_calibrator.region_risk_enabled = true` and missing JSON is impossible under fail-closed execution.

If v28 collapses to scale `0.0`, it should be treated as a safe rejection/no-op diagnostic, not as an improved method.
