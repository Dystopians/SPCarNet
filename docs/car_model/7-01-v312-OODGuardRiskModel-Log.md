# 2026-07-01 v312 OOD-Guarded Risk Model Log

## Purpose

v311 showed that a learned per-view risk model can be enabled, but it is not
safe across focused target scenes. v312 tests whether a target-blind source
feature OOD guard can prevent harmful source-to-target proxy transfer.

## Implementation

File:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Added:

- source candidate feature entries inside the learned risk-model policy;
- source leave-one-view nearest-neighbor distance distribution;
- `--per_view_risk_model_enable_ood_guard`;
- `--per_view_risk_model_ood_quantile`;
- `--per_view_risk_model_ood_min_samples`;
- target-time OOD rejection with `scene` fallback;
- per-view risk diagnostics including reject reason, OOD distance, and OOD
  threshold.

The mechanism remains target-blind: target GT is not used for selecting
variants.

## Command Pattern

Example:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v312a_ood_guard_risk_model_focused_20260701/<scene> \
  --target_split test \
  --support_source_mode source_split \
  --heldout_stride 4 --heldout_offset 0 \
  --device cuda --k 4 \
  --anchor_alpha 0.25 --learned_scale 0.5 --blend 0.5 \
  --output_variant source_heldout_auto \
  --selector_val_stride 3 --selector_val_offset 0 \
  --enable_per_view_risk_model_policy \
  --per_view_risk_model_reject_variant scene \
  --per_view_risk_model_allow_when_scene_fixed \
  --per_view_risk_model_ridge 0.001 \
  --per_view_risk_model_min_source_psnr_delta -0.05 \
  --per_view_risk_model_min_accept_fraction 0.10 \
  --per_view_risk_model_min_predicted_psnr_delta_vs_scene 0.0 \
  --per_view_risk_model_min_predicted_ssim_delta_vs_scene 0.0 \
  --per_view_risk_model_enable_ood_guard \
  --per_view_risk_model_ood_quantile 0.90 \
  --evidence_max_side 256 --compute_ssim --ssim_max_side 256 \
  --save_example_views 1 --copy_gt --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v312a-risk-ood-<scene>
```

## Results

Machine-readable summary:

```text
docs/car_model/results/v312_ood_guard_risk_model_focused_summary.json
```

Focused scenes:

```text
bicycle, counter, stump, treehill
```

| method | macro PSNR gain | macro SSIM gain | safe scene rate | positive-view fraction | mean min PSNR gain | negative views | OOD rejects |
|---|---:|---:|---:|---:|---:|---:|---:|
| v309 selective KNN | +0.173055 | +0.003173 | 1.00 | 0.887014 | -0.031668 | 9 | 0 |
| v310c tail-risk scene fallback | +0.172930 | +0.003176 | 1.00 | 0.897014 | -0.031668 | 8 | 0 |
| v311c dual-guard risk model | +0.165518 | +0.003099 | 0.50 | 0.905347 | -0.061860 | 7 | 0 |
| v312a OOD-guarded risk model | +0.165518 | +0.003099 | 0.50 | 0.905347 | -0.061860 | 7 | 2 |

Per-scene diagnosis:

- `counter`: OOD guard rejected 2 views, but the final metrics were unchanged
  relative to v311c and remained below the scene-level learned PSNR.
- `stump`: OOD guard rejected 0 views; harmful switches remained.
- `treehill`: OOD guard rejected 0 views; harmful switches remained.
- `bicycle`: safe but below v309/v310c.

## Verdict

v312a is not a main-method candidate. It is a negative diagnostic:

```text
ordinary source-feature OOD distance does not detect the target failures.
```

The failure is more specific than simple OOD shift. The target features can sit
inside the source-heldout feature distribution while the learned risk labels
still transfer incorrectly. The next reliability model must model residual
agreement and geometry/appearance consistency, not just feature distance.

Current best methods remain:

- `v309`: mean-quality frontier;
- `v310c`: tail-balanced frontier.

Current status:

```text
Final status: NOT COMPLETE.
```
