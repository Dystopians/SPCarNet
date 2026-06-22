# 5-26 Objective-Aware Bad-Region Alpha Risk v18 Log

Date: 2026-06-22

Status: in progress, not paper-complete.

## Motivation

The previous `field_region_render_risk_strict_v17` run exposed a specific
failure mode: candidate-owned refit had train-render-region evidence, but the
alpha region-risk path treated all region rows as risk candidates. That diluted
the signal because measured-good and measured-bad regions were mixed before the
ELA alpha calibrator made its decision.

The v17 bonsai medium run rejected candidate-owned refit:

```text
run root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v17_20260622_bonsai_medium
decision: accepted=false
reasons: balanced_delta_below_0, render_region_tail_cvar_below_-2e-05
raw objective rows: 41
bad objective rows by core_balanced_delta < 0 / SSIM regression / LPIPS regression: 31
candidate-owned refit test balanced delta: +0.0001622 report-only
candidate-owned refit trainval balanced delta: -0.0001214
```

This is not a satisfactory scientific result. v18 is a targeted diagnostic and
repair attempt: when train-render-region objective evidence exists, only
measured-bad objective rows should drive alpha region-risk suppression.

## Implemented Mechanism

New fixed profile:

```text
profile: field_region_render_risk_strict_v18
contract: field_region_render_risk_strict_v18_objective_aware_bad_region_alpha_risk
```

Core change:

```text
region-risk JSON is no longer consumed as a flat list of boxes only.
If objective_bad_only is enabled, the ELA alpha calibrator only keeps rows that
are bad under objective metrics:
  core_balanced_delta <= threshold, or
  delta_ssim <= threshold, or
  delta_lpips >= threshold.
```

Default v18 thresholds:

```text
ela_alpha_region_risk_objective_bad_only: true
ela_alpha_region_risk_objective_max_balanced_delta: 0.0
ela_alpha_region_risk_objective_max_delta_ssim: 0.0
ela_alpha_region_risk_objective_min_delta_lpips: 0.0
ela_alpha_region_risk_max_negative_fraction: 0.25
```

Files changed:

```text
utils/evidence_lumigraph_adapter.py
scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
scripts/car_model/ecsr_run_facelocal_coupled_selector.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

## Interface Validation

Compilation:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  utils/evidence_lumigraph_adapter.py \
  scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

Dry run:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v18 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v18 \
  --pipeline_label dryrun_field_region_render_risk_strict_v18 \
  --wandb_mode online \
  --dry_run
```

Dry-run manifest checks passed:

```text
profile = field_region_render_risk_strict_v18
profile_contract_id = field_region_render_risk_strict_v18_objective_aware_bad_region_alpha_risk
plan command includes --ela_alpha_region_risk_objective_bad_only
candidate-owned refit command includes --ela_alpha_region_risk_json and --ela_alpha_region_risk_objective_bad_only
selector command includes --ela_alpha_region_risk_json_template and objective-aware thresholds
```

## Active Medium Validation

Command:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v18 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v18_20260622_bonsai_medium \
  --pipeline_label field_region_render_risk_strict_v18_20260622_bonsai_medium \
  --wandb_mode online \
  --force
```

Current run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v18_20260622_bonsai_medium
```

Early plan-stage observation:

```text
plan base test LPIPS/PSNR/SSIM:
0.2593537271 / 28.8647136688 / 0.8960002065

phasej compact-base ELA test LPIPS/PSNR/SSIM:
0.2512249947 / 29.2167377472 / 0.8995919228

plan candidate ELA test LPIPS/PSNR/SSIM:
0.2512328029 / 29.2171821594 / 0.8995860219
```

Important caveat:

```text
The plan stage does not yet have candidate-owned objective JSON, so its ELA
reports correctly show region_risk_enabled=false / objective_bad_only=false.
The real v18 test is candidate-owned refit, where objective JSON is passed into
the ELA wrapper.
```

## Decision Gate

v18 is only considered useful if candidate-owned refit produces all of the
following:

```text
alpha_calibrator.region_risk_enabled = true
alpha_calibrator.region_risk_objective_bad_only = true
region-risk zeroing or alpha selection changes relative to v17
trainval balanced delta >= 0
render-region tail CVaR >= -2e-05
selector accepts or produces a clearly safer promoted candidate
```

If these conditions fail, the next method should be v19:

```text
Move bad-region evidence earlier than ELA.
Use objective-bad train-render-region rows to constrain candidate carrier or
allowed-face selection before candidate-owned refit, so measured-bad carriers
cannot survive merely because alpha post-processing is conservative.
```

This v19 direction would be a stronger method-level policy than v18 and should
be treated as the next serious repair path if v18 remains weak.
