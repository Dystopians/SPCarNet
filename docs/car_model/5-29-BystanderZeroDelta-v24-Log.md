# 2026-06-22 v24 Bystander Zero-Delta Objective Log

Status: `IN_PROGRESS_NOT_COMPLETE`.

This log records the first mechanism-level move after v21/v23 failed the
train-val and render-region gates. The goal is to stop treating local
degradation as a post-fit thresholding problem and instead make the fitter
learn a repair that is locally useful while explicitly preserving non-core
regions.

## Why v24 Exists

The v21/v23 evidence shows a repeated failure mode:

```text
local render-region mean can improve;
PSNR can move slightly upward;
LPIPS and SSIM regress on full-frame train-val/test;
render-region tail CVaR remains negative;
selector/gate rejects the candidate.
```

The immediate diagnosis is that the repair model can spend residual capacity
on pixels that are not the actual core evidence region. Mean crop gains are not
enough if the non-core context or outside pixels drift. v24 therefore adds a
train-only bystander zero-delta objective.

## Implemented Method Change

New bottom-level fitter arguments in
`scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`:

```text
--bystander_zero_delta_weight
--bystander_zero_delta_include_context / --no-bystander_zero_delta_include_context
--bystander_zero_delta_min_samples
```

New objective term:

```text
L_bystander = mean_weighted(||predicted_delta||^2 over bystander samples)
```

The bystander set is:

```text
outside render-region samples
+ context samples when --bystander_zero_delta_include_context is enabled
```

The solver now adds:

```text
L_total += bystander_zero_delta_weight * L_bystander
```

The audit now records:

```text
render_region_objective.bystander_samples
render_region_objective.bystander_zero_delta_weight
render_region_objective.final_bystander_zero_delta_loss
```

The interface is threaded through:

```text
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

## Fixed v24 Profile

Profile:

```text
field_region_render_risk_strict_v24
```

Contract:

```text
field_region_render_risk_strict_v24_train_objective_bystander_zero_delta
```

Fixed settings:

```text
delta_render_region_outside_penalty: 0.0
delta_bystander_zero_delta_weight: 0.20
delta_bystander_zero_delta_include_context: true
delta_bystander_zero_delta_min_samples: 64
```

The old outside penalty is set to zero in v24 to avoid silently double-counting
outside samples. The new term covers outside plus context with one auditable
preservation objective.

## Validation So Far

Syntax validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

Result: passed.

Dry-run command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v24 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v24_bystander_zero_delta \
  --pipeline_label dryrun_field_region_render_risk_strict_v24_bystander_zero_delta \
  --wandb_mode online \
  --dry_run \
  --force
```

Dry-run result:

```text
commands: 8
fixed profile: true
resolved_bystander_weight: 0.2
include_context: true
min_samples: 64
```

The dry-run candidate-owned refit command contains:

```text
--delta_render_region_outside_penalty 0.0
--delta_bystander_zero_delta_weight 0.2
--delta_bystander_zero_delta_include_context
--delta_bystander_zero_delta_min_samples 64
```

## v21/v23 Gate Evidence That Motivated v24

v21 selector strictfull_s1:

```text
accepted: false
decision reason: balanced_delta_below_0
trainval balanced delta: -0.0000028610
trainval delta: LPIPS +0.0000018477, PSNR +0.0000495911, SSIM -0.0000007749
report-only test delta: LPIPS +0.0000007153, PSNR +0.0000400543, SSIM -0.0000006557
```

v23 candidate-owned refit:

```text
accepted: false
decision reasons:
  balanced_delta_below_0
  render_region_tail_cvar_below_-2e-05
trainval balanced delta: -0.0006635785
trainval delta: LPIPS +0.0000312477, PSNR +0.0003356934, SSIM -0.0000187159
report-only test balanced delta: +0.0001480579
report-only test delta: LPIPS +0.0000147223, PSNR +0.0006427765, SSIM -0.0000100136
render-region mean core balanced delta: +0.2377047648
render-region tail core balanced CVaR: -0.0879078887
negative core-balanced fraction: 0.2195121951
```

Interpretation: v23 is a better-audited implementation, but not a scientifically
accepted method. v24 must improve train-val balanced delta and render-region
tail behavior, not merely keep PSNR slightly positive.

## Next Required Experiment

Run W&B-online medium validation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v24 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu <available_low_or_mid_occupancy_gpu> \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium \
  --pipeline_label field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium \
  --wandb_mode online \
  --force
```

Gate before expanding to more scenes:

```text
candidate-owned refit accepted or selector promoted;
trainval balanced delta >= 0;
LPIPS/SSIM do not regress beyond strict gate;
render-region tail CVaR >= -2e-05;
audit shows bystander_samples > 0 and final_bystander_zero_delta_loss recorded.
```

If v24 still fails, the next mechanism should be row-level face attribution and
per-view witness constraints, not further threshold tuning.

## 2026-06-22 Medium Run Evidence

Run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium
```

Plan-generation decision on `bonsai` failed the strict gate:

```text
accepted: false
selected_label: phasej_guarded_adaptedge
decision_reasons: balanced_delta_below_0
trainval balanced delta: -0.0002480149269104004
trainval delta: LPIPS +0.0000222474, PSNR +0.0004997253, SSIM -0.0000151396
report-only test delta: LPIPS +0.0000088513, PSNR +0.0005474091, SSIM -0.0000075102
```

The plan-stage audit did confirm that the new bystander objective was active:

```text
accepted: true
policy_pass: true
selected_faces: 5790
accepted_faces: 182
bystander_samples: 83291
core_samples: 2375
context_samples: 28
outside_samples: 83263
final_bystander_zero_delta_loss: 0.0007183143
```

Candidate-owned raw-base render-region evaluation exposed the same core
failure as earlier versions: strong mean local improvement but a bad tail.

```text
mean core balanced delta: +0.2048213503
tail core balanced CVaR: -0.1804324199
negative core-balanced fraction: 0.2619047619
worst core balanced delta: -0.3871169090
```

Candidate-owned refit test metrics did not produce a meaningful win over
PhaseJ:

```text
PhaseJ test:
  LPIPS 0.2512249947, PSNR 29.2167377472, SSIM 0.8995919228
candidate-owned test:
  LPIPS 0.2512391210, PSNR 29.2172718048, SSIM 0.8995839953
delta:
  LPIPS worse by +0.0000141263
  PSNR better by +0.0005340576
  SSIM worse by -0.0000079274
```

Selector `strictfull_s1` finished and was rejected:

```text
strictfull_s1 base test:
  LPIPS 0.2593406141, PSNR 28.8643436432, SSIM 0.8960120678
strictfull_s1 ELA test:
  LPIPS 0.2512256503, PSNR 29.2167434692, SSIM 0.8995916843
PhaseJ test:
  LPIPS 0.2512249947, PSNR 29.2167377472, SSIM 0.8995919228
report-only test delta vs PhaseJ:
  LPIPS +0.0000006557, PSNR +0.0000057220, SSIM -0.0000002384
selector decision:
  accepted: false
  selected_trial: phasej_fallback
  candidate_count: 10
  selected_trainval_balanced_delta: 0.0
strictfull_s1 selector reasons:
  inner_gate_rejected
  selector_psnr_gain_below_2e-05
  selector_balanced_delta_below_5e-05
region-local promotion stats:
  accepted_carriers: 1
  mean_core_balanced_delta: +0.3826133188
  tail_core_balanced_delta: +0.0158917546
  negative_core_balanced_fraction: 0.1176470588
```

Interpretation: v24 is a useful diagnostic but not a successful method. It
keeps the outside/bystander term auditable, but it does not fix the actual
failure mode: local repair can still have negative held-out/tail views. The
appropriate next mechanism is v25 witness-group CVaR, not more v24 threshold
tuning.

Final v24 artifacts:

```text
candidate-owned decision:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium/candidate_owned_refit/decisions/bonsai_decision.json
selector decision:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium/selector/bonsai/coupled_selector_decision.json
selector summary:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium/selector/coupled_selector_summary.json
pipeline summary:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium/pipeline_summary.json
command manifest:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v24_bystander_20260622_bonsai_medium/pipeline_command_manifest.json
```

Pipeline note: the full `plan,filter,selector --force` run completed the
selector outputs, but crashed while writing the manifest because the manifest
hashing helper tried to `stat()` the comma-separated `allowed_face_ids` string
as a path. This was fixed in
`scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py` by catching
`OSError` in `command_record_output_sha256s()`. A selector-only no-force rerun
then successfully wrote the final pipeline summary and manifest.
