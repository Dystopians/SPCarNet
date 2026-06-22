# 2026-06-21 Coverage/Dilution Guard and Holdout-Safe Alpha Log

## Status

Status is still `NOT_COMPLETE_SCIENTIFICALLY`.

The current evidence does not support a paper-level claim that SPCarNet
dominates the MeshSplatting-derived baseline. The useful progress in this
checkpoint is narrower and more precise:

- v13 adds a train-only coverage/dilution guard that prevents narrow ROI wins
  from being promoted as full-frame improvements.
- v14 validates a new ELA alpha policy that can zero unsafe holdout bins.
- v15 is a running diagnostic that combines the v13 full20/risk-carrier
  geometry with the v14 holdout-safe ELA policy.

## Why This Was Needed

The bonsai failure mode is now clear. The v12 subset is not a no-op and its
selected carrier is locally strong, but it only changes two train-render
regions. That local gain dilutes to numerical noise in the full-frame
train-val gate.

The rejected carrier has broader coverage and a strong mean ROI gain, but its
train-render tail is unsafe. Therefore the bottleneck is not simply "select a
better scale"; it is the conflict between:

- narrow but tail-safe local repair;
- broader but tail-risky carrier repair;
- post-render ELA that can improve LPIPS while still leaving PSNR/SSIM noise.

## Implemented Method Changes

### v13 coverage/dilution aggregate subset

Files changed:

- `scripts/car_model/ecsr_apply_render_cvar_aggregate_subset.py`
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py`

New aggregate subset stats:

- `unique_views`
- `changed_unique_views`
- `view_coverage_fraction`
- `changed_view_coverage_fraction`
- `total_pixels`
- `changed_pixels`
- `changed_pixel_fraction`
- `area_weighted_core_balanced_delta`
- `dilution_adjusted_core_balanced_delta`

New fixed profile:

```text
profile: field_region_render_risk_strict_v13
contract: field_region_render_risk_strict_v13_coverage_dilution_guarded_aggregate_subset
expected_view_count: 64
min_unique_views: 4
min_changed_unique_views: 4
min_changed_pixel_fraction: 0.05
min_area_weighted_core_balanced_delta: 0.0
min_dilution_adjusted_core_balanced_delta: 1.0e-5
```

The greedy selector was also fixed so coverage shortfalls can accumulate across
carriers. Final selection still has to pass the complete aggregate risk and
coverage checks.

### v14 holdout-safe ELA alpha policy

Files changed:

- `utils/evidence_lumigraph_adapter.py`
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py`

New ELA alpha controls:

```text
--alpha_holdout_safe_zero
--alpha_risk_tail_fraction
--alpha_max_negative_gain_fraction
--alpha_min_tail_gain
```

The alpha calibrator now records tail gain, negative gain fraction, accepted
bins, and risk-zeroed bins. Unsafe bins can be set to alpha 0 instead of being
accepted only because their mean MSE gain is positive.

## Evidence So Far

### v13 offline replay on bonsai

Input:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v11_20260621_bonsai/filtered_candidate_plans/bonsai/facelocal_visual_candidate_plan_filtered.json
```

Objective, matching the v12 aggregate-subset summary:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v11_20260621_bonsai/candidate_owned_render_regions/bonsai/train_render_region_objective_refit_base.json
```

Output:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v13_20260621_bonsai_coverage_dilution_guard_refit/bonsai/aggregate_subset_summary.json
```

Result:

```text
selected_carriers: 0
selected_rows: 0
final reasons: unique_views_below_4, changed_unique_views_below_4
```

Interpretation: v13 correctly refuses the v12 narrow ROI subset. This is a
false-positive guard, not a performance breakthrough.

### v13 full20 risk-carrier alpha-rescue diagnostic

Decision:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v13_20260621_bonsai_alpha_rescue/phasek_full20_riskcarrier_s0p6_clean_s1/decisions/bonsai_decision.json
```

Result:

```text
accepted: false
reason: psnr_gain_below_0
trainval balanced delta: +2.1517276763916016e-05
trainval PSNR delta: -3.814697265625e-06
trainval SSIM delta: -2.384185791015625e-07
trainval LPIPS delta: -1.5050172805786133e-06
```

Interpretation: the broader carrier rescue is close but still not promotable.
The gain is driven by LPIPS and tiny balanced noise; PSNR remains negative.

### Running experiments

v12 selector rerun:

```text
session: 42672
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v12_offline_20260621_bonsai_selector_rerun
GPU: 5
```

v14 holdout-safe alpha Phase-K:

```text
session: 81026
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v14_20260621_bonsai_alpha_holdout_safe_zero/phasek_v12_subset_s0p6_holdout_safe_zero
GPU: 4
```

v15 full20 + holdout-safe ELA diagnostic:

```text
session: 46251
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v13_20260621_bonsai_alpha_rescue/phasek_full20_riskcarrier_s0p6_clean_s1
new methods:
  ours_26000_field_region_render_risk_strict_v15_full20_riskcarrier_s0p6_clean_s1_holdout_safe_zero_phasej_ela
  ours_26000_field_region_render_risk_strict_v15_full20_riskcarrier_s0p6_clean_s1_holdout_safe_zero_trainval_gate
GPU: 1
W&B group: field_region_render_risk_strict_v15_20260621_bonsai_combo_diagnostic
```

## Completed After Launch

### v14 result: failed hard

Decision:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v14_20260621_bonsai_alpha_holdout_safe_zero/phasek_v12_subset_s0p6_holdout_safe_zero/decisions/bonsai_decision.json
```

Result:

```text
accepted: false
reasons:
  psnr_gain_below_0
  ssim_regression_exceeds_5e-05
  lpips_regression_exceeds_0.00015
  balanced_delta_below_0
trainval delta:
  PSNR: -2.531665802001953
  SSIM: -0.025668740272521973
  LPIPS: +0.072795569896698
test report-only delta:
  PSNR: -2.645263671875
  SSIM: -0.030687987804412842
  LPIPS: +0.07866981625556946
```

Interpretation: holdout-safe alpha zeroing is too conservative as a standalone
performance policy. It closes many ELA bins and exposes the poor base candidate
render instead of repairing it.

### v15 result: failed hard

Output metrics:

```text
test:
  LPIPS: 0.25426793098449707
  PSNR: 29.04352378845215
  SSIM: 0.8983045220375061
trainval:
  LPIPS: 0.24315650761127472
  PSNR: 30.070798873901367
  SSIM: 0.908505916595459
```

Compared with the PhaseJ baseline used by the gate:

```text
PhaseJ trainval:
  LPIPS: 0.1703605055809021
  PSNR: 32.602386474609375
  SSIM: 0.934174120426178
```

Interpretation: the full20/risk-carrier geometry plus holdout-safe ELA is not a
viable route. The ELA guard is real (`risk_zeroed_bins=110`, `accepted_bins=15`)
but it removes the correction needed to hide the candidate's base degradation.

### Interface bug found and fixed

The new `ela_alpha_min_tail_gain` argument can default to `-inf`. Passing this
as two argv tokens caused argparse failures like:

```text
meshsplatopt_apply_evidence_lumigraph_adapter.py: error:
argument --alpha_min_tail_gain: expected one argument
```

Fix:

- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py`

All tail-gain arguments that may be negative are now emitted as
`--alpha_min_tail_gain=<value>` or `--ela_alpha_min_tail_gain=<value>`.

Validation:

```text
py_compile: passed
git diff --check: passed
argparse smoke: --alpha_min_tail_gain=-inf parses as -inf
```

The pre-fix v12 selector rerun was terminated because it had already loaded the
old command-generation code. A patched no-force resume was started:

```text
session: 55042
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v12_offline_20260621_bonsai_selector_rerun
GPU: 5
```

## Current Interpretation

The method is cleaner than before, but not yet strong enough. v13 makes the
selector more honest by rejecting narrow ROI wins. v14/v15 show that
holdout-safe ELA zeroing, by itself, is not a viable performance repair.

The next real method step should be per-carrier train-only shrink inside the
aggregate subset policy, emitted as a face-alpha JSON plus strict render-trust
certificate and consumed automatically by the selector. That would turn the
current manual v13 alpha-rescue diagnostic into a fixed, auditable policy.

## Validation Commands

Syntax and whitespace validation already passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_render_cvar_aggregate_subset.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

git diff --check -- \
  scripts/car_model/ecsr_apply_render_cvar_aggregate_subset.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

v13 dry-run manifest:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v13_20260621_dryrun2/pipeline_command_manifest.json
```

The dry-run confirms the aggregate subset command receives the refit train
render-region objective and the new coverage/dilution thresholds.

## 2026-06-21 Update: v14 Fixed Support-Preserving Carrier Alpha Shrink

The first per-carrier shrink replay failed with zero selected carriers because
the shrink model incorrectly reduced `crop_nonzero_pixels` and
`crop_nonzero_fraction`. That mixed two different quantities:

- alpha shrink changes residual amplitude;
- render-region coverage/dilution should describe which train-render regions
  the carrier touches.

Fix:

- `scripts/car_model/ecsr_apply_render_cvar_aggregate_subset.py` now uses a
  support-preserving amplitude shrink. Risk deltas and diff magnitudes shrink
  with alpha, but spatial support is kept for any positive alpha.
- The fixed shrink ladder is now
  `1.0,0.85,0.75,0.6,0.5,0.35,0.2,0.1,0.05,0.035,0.02`.
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py` now automatically
  consumes `render_cvar_aggregate_subset_materialize_alpha_json` from the plan
  and records it in the render-trust certificate.
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py` adds fixed
  profile `field_region_render_risk_strict_v14`, whose contract is
  `field_region_render_risk_strict_v14_coverage_dilution_guarded_per_carrier_alpha_shrink`.

Dry-run validation:

```text
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v14_20260621_dryrun
aggregate subset command: includes --tail_safe_shrink_carriers
selector strict replay scales: 1.0
py_compile: passed
git diff --check: passed
```

Offline bonsai replay using existing v11 train-only evidence:

```text
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v14_20260621_bonsai_carrier_shrink_guard_refit_support_preserve
input carriers / rows: 2 / 20
selected carriers / rows: 2 / 20
alpha json: bonsai/aggregate_subset_materialize_alpha.json
alpha values: 0.035 and 1.0
final train-render stats:
  mean_core_balanced_delta: +0.0304807189
  mean_delta_core_psnr: +0.0186366671
  tail_core_balanced_delta: -0.00001826296
  changed_unique_views: 5
  changed_pixel_fraction: 1.0
  dilution_adjusted_core_balanced_delta: +0.0027501253
```

W&B-online Phase-K validation:

```text
session: 7852
GPU: 1
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v14_20260621_bonsai_carrier_shrink_guard_refit_support_preserve_selector
baseline test method: ours_26000_phasej_guarded_adaptedge_ela_field_region_render_risk_strict_v11_20260621_bonsai
baseline train-val method: ours_26000_phasej_trainval_gate_rendercalib_v1_field_region_render_risk_strict_v11_20260621_bonsai
```

Final v14 selector result:

```text
accepted: false
selected_trial: phasej_fallback
decision reason: balanced_delta_below_0
selector reasons:
  inner_gate_rejected
  selector_psnr_gain_below_2e-05
  selector_balanced_delta_below_5e-05

baseline train-val:
  LPIPS 0.1703605056
  PSNR  32.6023864746
  SSIM  0.9341741204
candidate train-val:
  LPIPS 0.1703606695
  PSNR  32.6023864746
  SSIM  0.9341738224
train-val delta:
  LPIPS +0.0000001639
  PSNR  +0.0000000000
  SSIM  -0.0000002980
train-val balanced delta: -0.0000092387

baseline report-only test:
  LPIPS 0.1725552976
  PSNR  31.8620052338
  SSIM  0.9302796125
candidate report-only test:
  LPIPS 0.1725550890
  PSNR  31.8619937897
  SSIM  0.9302796125
report-only test delta:
  LPIPS -0.0000002086
  PSNR  -0.0000114441
  SSIM  +0.0000000000
report-only test balanced delta: -0.0000072718
```

Interpretation:

- v14 closes an important interface gap: train-only aggregate subset selection
  can now emit per-carrier alpha materialization and the strict selector
  consumes it with an explicit render-trust certificate.
- It does not close the scientific gap. The final promoted output is effectively
  a near-baseline fallback, not a meaningful nonzero improvement over the clean
  MeshSplatting-derived Phase-J baseline.
- The next step should not be full multi-scene expansion of v14. The method
  needs a stronger residual/materialization mechanism that improves full-frame
  train-val metrics before ELA fallback erases the candidate.

Status: `NOT_COMPLETE_SCIENTIFICALLY`.

## 2026-06-21 Update: v15 Selector Result

The W&B-online flowers selector/Phase-K validation finished:

```text
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v15_20260621_flowers_visibility_guard_selector
scene: flowers
trial: strictfull_s1
phase-k accepted: true
coupled selector accepted: false
selected trial: phasej_fallback
selection uses test: false
candidate faces: 14
train-val delta:
  LPIPS: +5.9605e-08
  PSNR:  +3.6240e-05
  SSIM:  +0.0000e+00
train-val balanced delta: +3.5048e-05
selector rejection:
  selector_balanced_delta_below_5e-05
tail rejection signals:
  tail_balanced_delta_below_5e-05
  tail_psnr_negative_fraction_exceeds_0.2
  tail_balanced_negative_fraction_exceeds_0.4
report-only test delta:
  LPIPS: -2.6822e-07
  PSNR:  +1.9073e-06
  SSIM:  -5.9605e-08
```

Interpretation:

- v15 is useful as a claim-safety correction: it blocks spatially diluted
  candidates from being promoted as full-frame wins.
- It is not a performance breakthrough. The best flowers candidate clears the
  inner Phase-K gate, but its full-frame train-val gain is only
  `3.5048e-05` balanced and has unstable tail behavior, so the coupled selector
  correctly falls back.
- The core bottleneck is now clear: local ROI residual repair can look strong
  in cropped regions, but it does not change enough visible full-frame support
  to move global metrics or obvious qualitative comparisons.

Decision:

```text
v15 status: REJECT_AS_PERFORMANCE_METHOD
v15 retained role: full-frame visibility guard / evidence hygiene
next required method change: move upstream to candidate generation or
representation capacity, not selector threshold tuning
```

## 2026-06-21 Update: v15 Full-Frame Visibility Guard

The v14 failure exposed a measurement mismatch. The aggregate subset policy
looked only at ROI/crop evidence plus changed-view coverage. It did not measure
how much of the full train-val image budget was actually affected. On bonsai,
v14 changed only `1484 / 103667264` full-frame train-val pixels
(`1.4315e-05`) while reporting a strong ROI/crop mean. That is too small to
expect visible or full-frame metric movement.

Implemented v15:

```text
profile: field_region_render_risk_strict_v15
contract: field_region_render_risk_strict_v15_full_frame_visibility_guarded_alpha_shrink
new objective rows:
  image_width
  image_height
  frame_pixels
  bbox_pixels
  bbox_frame_fraction
  changed_frame_fraction
new aggregate subset stats:
  full_frame_denominator_pixels
  full_frame_changed_pixel_fraction
  full_frame_visibility_adjusted_delta
fixed v15 thresholds:
  min_full_frame_changed_pixel_fraction: 2.0e-05
  min_full_frame_visibility_adjusted_delta: 5.0e-07
```

Validation:

```text
py_compile: passed
git diff --check: passed
dry-run root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v15_20260621_dryrun2
dry-run contract: field_region_render_risk_strict_v15_full_frame_visibility_guarded_alpha_shrink
bonsai inferred frame pixels: 1619801
```

Offline replay on existing v11 evidence:

```text
bonsai root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v15_20260621_bonsai_visibility_guard_replay2
input carriers / rows: 2 / 20
selected carriers / rows: 0 / 0
final reason: full_frame_changed_pixel_fraction_below_2e-05
best full-frame changed fraction: 1.4315e-05

flowers root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v15_20260621_flowers_visibility_guard_replay2
input carriers / rows: 2 / 14
selected carriers / rows: 2 / 14
full-frame changed fraction: 2.9658e-05
full-frame visibility adjusted delta: 6.8651e-07
train-render tail balanced delta: +1.4512e-05
```

Current interpretation:

- v15 is a real method correction, but it is not yet a performance win.
- It prevents the exact v14 failure mode: a tiny ROI-local edit can no longer
  be promoted as a full-frame visual/metric improvement.
- It keeps flowers because the train-only evidence is more visible and still
  tail-positive; it rejects bonsai because the candidate is too spatially
  diluted.
- The next evidence gate is the active W&B-online flowers selector/Phase-K run.
  If that run does not produce a stable nonzero train-val win, the next method
  change must move earlier to candidate generation/representation capacity,
  not to selector thresholds.

Active validation:

```text
session: 34220
GPU: 1
root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v15_20260621_flowers_visibility_guard_selector
```

Status: `NOT_COMPLETE_SCIENTIFICALLY`.
