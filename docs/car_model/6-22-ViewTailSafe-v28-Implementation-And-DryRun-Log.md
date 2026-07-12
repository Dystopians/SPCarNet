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

No v28 medium/long result has been promoted yet.

2026-06-22 update: the first real Bonsai medium validation was launched with
W&B online after GPU5 became low-occupancy. An earlier GPU4 scheduler was
cancelled to avoid a duplicate run while v27 was still active on GPU4.

Actual running command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v28 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5 \
  --pipeline_label field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5 \
  --wandb_mode online \
  --force
```

Runtime evidence at launch:

- top-level exec session id: `1468`;
- top-level pipeline process observed: `3461772`;
- PhaseK plan process observed: `3461780`;
- evidence cache process observed: `3461781`;
- GPU5 memory rose from roughly `2444 / 49140 MiB` to roughly
  `15029 / 49140 MiB`;
- output root:
  `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5`;
- logs:
  `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5/logs/plan/bonsai.log`;
  `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5/plan_generation/bonsai/phasek_barycentric_gate.log`.

The observed PhaseK command includes the expected v28 flags:

```text
--ela_alpha_view_tail_scale_grid 1.0,0.75,0.5,0.25,0.0
--ela_alpha_view_tail_cvar_fraction 0.25
--ela_alpha_view_tail_min_gain=0.0
--ela_alpha_view_tail_max_negative_fraction 0.5
--ela_local_trust_mode soft
--ela_local_trust_min_weight 0.02
```

This proves the run is exercising the v28 interface at command level. Promotion
still requires the post-run JSON checks below, especially the
`alpha_calibrator.view_tail_*` fields.

The concurrent read-only subagent audit also checked the fixed Phase-J source
report for Bonsai:

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model/test/ours_26000_phasej_guarded_adaptedge_ela/ela_report.json
alpha_policy = adaptive_bins
```

Therefore the Bonsai v28 run should enter the adaptive alpha calibrator instead
of hitting the fixed-policy fail-fast path. The post-run interpretation remains:

- `alpha_calibrator.view_tail_enabled = true` proves the view-tail module was
  evaluated;
- `alpha_calibrator.view_tail_scale < 1.0` proves actual shrink occurred;
- `alpha_calibrator.view_tail_scale = 1.0` means the mechanism was active but
  selected no shrink, so it must not be advertised as an alpha-shrink win.

Reference launch template for later scenes:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v28 \
  --stages plan,filter,selector \
  --scenes <SCENE> \
  --gpu <LOW_OCCUPANCY_GPU> \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_<SCENE>_medium_gpu<GPU> \
  --pipeline_label field_region_render_risk_strict_v28_viewtail_20260622_<SCENE>_medium_gpu<GPU> \
  --wandb_mode online \
  --force
```

Previous recommended first real run:

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

## Audit Utility

To avoid manually inspecting many ELA reports after each medium/long run, v28 now
has a dedicated read-only audit utility:

```text
scripts/car_model/ecsr_audit_viewtail_alpha_run.py
```

Usage:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_audit_viewtail_alpha_run.py \
  --run_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5 \
  --output_json /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5/viewtail_alpha_audit.json \
  --output_md /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_20260622_bonsai_medium_gpu5/viewtail_alpha_audit.md
```

The script scans every `ela_report.json` and `*decision.json` under a run root,
then summarizes:

- number of reports with `view_tail_enabled=true`;
- number of reports with actual shrink, `view_tail_scale < 1`;
- number of reports stuck at `view_tail_scale == 1`;
- fallback usage;
- missing candidate stats;
- decision acceptance, train-val balanced delta, and rejection reasons.

Verification:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_audit_viewtail_alpha_run.py
```

Result: passed.

A partial audit on the still-running GPU5 run produced zero reports/decisions,
which is expected before the first ELA stage finishes.

## 2026-06-22 Real-Run Interface Fix

The first GPU5 Bonsai medium attempt failed during plan generation before ELA
could run:

```text
RuntimeError: plan failed for bonsai
ValueError: --ela_alpha_region_risk_enable requires a non-empty region-risk JSON path
```

Root cause:

- the AutoVisual plan command inherited `--ela_alpha_region_risk_enable`;
- PhaseK replays Phase-J ELA before the train-render-region objective JSON is
  generated;
- therefore the new fail-closed check correctly rejected a dangling enable flag.

Fix:

- plan generation now does **not** pass `--ela_alpha_region_risk_enable`;
- candidate-owned refit explicitly passes both
  `--ela_alpha_region_risk_enable` and `--ela_alpha_region_risk_json`;
- selector continues to pass `--ela_alpha_region_risk_enable` and
  `--ela_alpha_region_risk_json_template`;
- view-tail args are still passed to plan, candidate-owned refit, and selector.

Verification:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v28 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_v28_viewtail_regionrisk_fix2_20260622 \
  --pipeline_label dryrun_v28_viewtail_regionrisk_fix2_20260622 \
  --wandb_mode disabled \
  --dry_run \
  --force
```

Manifest check:

```text
plan: region_enable=false, json=[], viewtail=true
candidate_owned_refit: region_enable=true,
  json=[.../candidate_owned_render_regions/bonsai/train_render_region_objective_raw_base.json],
  viewtail=true
selector: region_enable=true,
  json_template=[.../candidate_owned_render_regions/{scene}/train_render_region_objective_refit_base.json],
  viewtail=true
```

This keeps fail-closed behavior intact while avoiding a plan-stage dependency
on a not-yet-generated JSON file.

Relaunched fixed real run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v28 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5 \
  --pipeline_label field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5 \
  --wandb_mode online \
  --force
```

Status at relaunch: running in exec session `9975`.

## 2026-06-22 Fix2 Medium Result Snapshot

The fixed Bonsai medium run progressed past the previous interface failure and
produced real PhaseK/ELA evidence. This proves the fail-closed region-risk
interface fix worked, but the method did **not** pass the plan-stage acceptance
gate.

Run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5
```

Plan-stage decision:

```text
plan_generation/decisions/bonsai_decision.json
accepted = false
selected_label = phasej_guarded_adaptedge
decision_reasons =
  psnr_gain_below_0,
  ssim_regression_exceeds_5e-05,
  lpips_regression_exceeds_0.00015,
  balanced_delta_below_0
```

Measured deltas against the fixed Phase-J comparator:

| split | dPSNR | dSSIM | dLPIPS | balanced |
|---|---:|---:|---:|---:|
| train-val | -0.020847 | -0.000283 | +0.000571 | -0.037934 |
| held-out test, report-only | -0.002831 | +0.000139 | +0.000037 | n/a |

View-tail audit:

```text
viewtail_alpha_audit.json
ELA reports = 4
view_tail_enabled = 4
actual shrink reports, scale < 1 = 0
scale == 1 reports = 4
fallback_used = 0
accepted decisions = 0
```

Evidence:

- `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5/plan_generation/phasek_barycentric_gate_summary.md`
- `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5/plan_generation/decisions/bonsai_decision.json`
- `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5/viewtail_alpha_audit.md`

Interpretation:

- v28 is no longer blocked by the command/interface bug;
- the view-tail module is genuinely active in ELA reports;
- however, every report selected `view_tail_scale = 1.0`, so no actual shrink
  occurred;
- the reason is objective mismatch: v28 chooses the scale by policy-view MSE
  gain/tail safety, while the downstream acceptance gate rejects on
  balanced/LPIPS-aware train-val behavior.

Therefore v28 should **not** be promoted. The useful contribution of this run is
diagnostic: the next method change should align the train-only view-tail scale
selection objective with the selector's balanced/LPIPS-aware contract.

At the time of this snapshot, candidate-owned refit is still running. Its final
selector result should be appended before closing the v28 log, but the plan-stage
negative result is already sufficient to reject v28 as a headline method.

## Conditional Next Experiment If v28 Is Rejected

A read-only follow-up audit recommended the next minimal closed-loop experiment
if the fixed Bonsai medium run is rejected by the train-val/render-region gate:
do **not** relax thresholds and do **not** immediately add another ELA risk
module. Instead, run a pre-registered strict full-plan monotone scale ladder
replay.

Rationale:

- v28 changes ELA adaptive alpha through `view_tail_scale`;
- the current fixed profile still uses a single selector strict replay scale
  (`1.0`) for the materialized face-local residual plan;
- if a full-scale physical residual edit is too strong, a smaller monotone scale
  may pass the same train-only gate without changing the gate thresholds;
- this tests a real method question: whether the candidate is unsafe in kind, or
  simply over-amplified in coefficient scale.

Recommended ladder:

```text
selector_strict_replay_scales = 1.0,0.85,0.75,0.6,0.5,0.35,0.2,0.1
```

Use the selector directly rather than overriding the fixed AutoVisual profile,
because fixed profiles intentionally reject profile-field overrides.

Template:

```bash
RUN=/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5
OUT=/data/peilincai/spcarnet_runs/v28_bonsai_strict_scale_ladder_replay_20260622

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes bonsai \
  --gpu 5 \
  --output_root "$OUT" \
  --plan_template "$RUN/filtered_candidate_plans/{scene}/facelocal_visual_candidate_plan_filtered_aggregate_subset.json" \
  --evidence_root "$RUN/surface_evidence" \
  --candidate_prefix v28_bonsai_scale_ladder \
  --phasej_test_method ours_26000_phasej_guarded_adaptedge_ela_field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5 \
  --phasej_trainval_method ours_26000_phasej_trainval_gate_rendercalib_v1_field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5 \
  --selector_strict_replay_scales 1.0,0.85,0.75,0.6,0.5,0.35,0.2,0.1 \
  --no-selector_strict_adaptive_scale_policy \
  --no-selector_strict_fit_plan_alphas \
  --selector_min_trainval_psnr_gain 2e-5 \
  --selector_min_trainval_balanced_delta 5e-5 \
  --selector_tail_min_trainval_balanced_delta 5e-5 \
  --selector_balanced_ssim_weight 20 \
  --selector_balanced_lpips_weight 20 \
  --ela_alpha_holdout_safe_zero \
  --ela_alpha_view_tail_scale_grid 1.0,0.75,0.5,0.25,0.0 \
  --ela_alpha_view_tail_cvar_fraction 0.25 \
  --ela_alpha_view_tail_min_gain=0.0 \
  --ela_alpha_view_tail_max_negative_fraction 0.5 \
  --ela_local_trust_gate \
  --ela_local_trust_mode soft \
  --ela_local_trust_min_weight 0.02 \
  --force
```

Promotion criterion:

- `$OUT/bonsai/coupled_selector_decision.json` has `accepted=true`;
- `selected_trial` is a strict scale-ladder trial, for example `strictfull_s0p75`;
- the selected inner decision has `accepted=true`;
- `trainval_balanced_delta >= 5e-5`;
- held-out test report-only does not show a three-metric near-noop regression.

If all lower scales still fail through `inner_gate_rejected`, the next code-level
change should be a balanced/LPIPS-aware view-tail alpha shrink instead of the
current MSE-only view-tail shrink.

## Final Bonsai Medium Fix2 Selector Snapshot

Run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5
```

Final selector decision:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5/selector/bonsai/coupled_selector_decision.json
```

The selector accepted `strictfull_s1` by trainval only, but the magnitude is
near-noop and the held-out test report-only deltas are slightly negative:

| split | dPSNR | dSSIM | dLPIPS | balanced delta |
|---|---:|---:|---:|---:|
| trainval gate | +0.000330 | +0.0000067 | -0.0000085 | +0.000634 |
| held-out test, report-only | -0.000017 | -0.0000040 | +0.0000104 | -0.000304 |

Decision: **do not promote v28**. The useful outcome is diagnostic:

- the command/interface path is now working end to end;
- view-tail alpha shrink can prevent large unsafe updates;
- MSE-tail safety is not aligned enough with the paper metrics;
- a trainval near-noop acceptance is not a credible paper-level improvement.

This result motivates v29's balanced/LPIPS-aware view-tail objective, but v28
itself should stay as a negative ablation rather than a headline method.
