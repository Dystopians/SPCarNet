# v109 Render-Realized Parent Gate Log

Date: 2026-06-25

## Motivation

v108 showed that a coefficient-space descent proxy is not enough. Both `crossfit_risk` and `normal_equation` flowers probes passed identity checks and had negative `joint_descent_objective_delta_mean`, but both were worse than v106 in final render metrics and render-space MSE:

| candidate | dPSNR vs v106 | dSSIM vs v106 | dLPIPS vs v106 | render-space MSE improved views |
|---|---:|---:|---:|---:|
| v108 crossfit-risk | -0.001858 | -0.000162 | +0.000081 | 0 / 22 |
| v108 normal-equation | -0.001305 | -0.000114 | +0.000034 | 0 / 22 |

The next mechanism must therefore protect the validated v106 parent explicitly. v109 changes the certificate target from field proxy descent to render-realized parent preservation.

## Method

New script:

```text
scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py
```

Core rule:

```text
output = parent + mask(candidate, parent) * (candidate - parent)
```

Fairness constraints:

- Calibration may read `calib_split` renders and GT.
- Target/test policy application never reads target/test GT or target/test metrics.
- Target/test GT is only copied into the output method directory for later evaluator use.
- If calibration cannot prove a non-regressive policy, v109 automatically falls back to the parent (`mask = 0`).
- The policy feature is local pooled `abs(candidate - parent)`, not target GT.

This is a feasibility-stage implementation. A stricter paper version should add train-even/train-odd candidate fitting and held-out calibration so the candidate render itself is not scored on the same images used to fit its sidecar.

## Implementation Validation

Commands:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py

git diff --check -- scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py
```

Both passed.

## Planned Feasibility Probe

Target scene: `flowers`.

Prerequisite renders:

- v106 parent train renders for `ours_26000_v106_podmoe_basepreserve_flowers`
- v108 candidate train renders for the candidate being gated
- existing test renders for parent and candidate

Policy:

- calibrate on `train`
- apply on `test`
- evaluate final test output with `evaluate_render_split_metrics.py`
- compare against v106, v108, v104c, and clean

Expected outcome:

- minimum success: v109 rejects a harmful v108 candidate and exactly preserves v106-level metrics;
- stronger success: train-calibrated local mask accepts a subset of candidate pixels and exceeds v106 without target-GT selection.

## Flowers Feasibility Result

Prerequisite train renders were generated for:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/train/ours_26000_v106_podmoe_basepreserve_flowers
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/train/ours_26000_v108_mse_descent_locked_podmoe_ne_flowers
```

Each contains `151` train render PNGs.

Gate command:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 WANDB_MODE=offline TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py \
  --parent_model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model \
  --parent_method_name ours_26000_v106_podmoe_basepreserve_flowers \
  --candidate_model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model \
  --candidate_method_name ours_26000_v108_mse_descent_locked_podmoe_ne_flowers \
  --output_model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model \
  --method_name ours_26000_v109_render_realized_parent_gate_ne_flowers \
  --calib_split train \
  --target_split test \
  --calib_max_views 64 \
  --calib_sampler uniform \
  --threshold_grid 0.0005,0.001,0.002,0.004,0.006,0.008,0.010,0.014,0.020 \
  --softness_grid 0,0.0005,0.001,0.002 \
  --max_blend_grid 0.25,0.50,0.75,1.00 \
  --local_kernels 1,9,25 \
  --min_mean_psnr_gain 0 \
  --max_mean_mse_increase 0 \
  --max_p95_mse_increase 0 \
  --min_p05_score_gain -0.0001 \
  --device cuda
```

Selected policy:

```json
{
  "dilate": 0,
  "frame_threshold": 1000000000.0,
  "kernels": [1, 9, 25],
  "max_blend": 0.0,
  "softness": 0.0,
  "threshold": 1000000000.0
}
```

This is an explicit parent fallback selected from train calibration. Target/test policy application recorded:

```text
no_target_gt_used_for_policy=True
target_views=22
target_mean_mask=0.0
mean_candidate_parent_distance=0.00007136
```

Artifacts:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v109_render_realized_parent_gate_ne_flowers/v109_render_realized_parent_gate_report.json
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v109_render_realized_parent_gate_ne_flowers/v109_render_realized_parent_gate_report.md
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v109_render_realized_parent_gate_ne_flowers/v109_render_realized_parent_gate_calibration_rows.csv
```

Final test metrics:

| method | PSNR | SSIM | LPIPS | relation to v106 |
|---|---:|---:|---:|---|
| v106 parent | 20.077723 | 0.531240 | 0.374393 | baseline |
| v108 normal-equation candidate | 20.076418 | 0.531125 | 0.374427 | worse than v106 |
| v109 parent gate | 20.077723 | 0.531240 | 0.374393 | identical to v106 |

Render-space audit:

| candidate | base | views | improved views | worse views | mean delta MSE | mean abs delta |
|---|---|---:|---:|---:|---:|---:|
| v109 parent gate | v106 parent | 22 | 22 | 0 | +0.00000000 | 0.00000000 |

Audit artifacts:

```text
/dev/shm/peilincai_spcarnet_v109_render_realized_parent_gate_20260625_reports/v109_vs_v106_flowers_render_delta_mse.json
/dev/shm/peilincai_spcarnet_v109_render_realized_parent_gate_20260625_reports/v109_vs_v106_flowers_render_delta_mse.csv
/dev/shm/peilincai_spcarnet_v109_render_realized_parent_gate_20260625_reports/v109_vs_v106_flowers_render_delta_mse.md
```

Interpretation: v109 achieves the minimum safety target on flowers. It does not improve beyond v106, but it prevents v108's negative transfer without using target/test GT for policy selection. This turns a failed candidate into a safe parent-preserving endpoint and gives the next method line a clean paper story: any promoted correction must first beat this parent gate rather than merely pass a proxy field certificate.

## Next Required Work

- Implement stricter train-even/train-odd sidecar fitting so calibration views are held out from candidate fitting.
- Run v109 gate on more hard scenes once candidate train renders exist.
- Search for a genuinely positive candidate under the v109 certificate; fallback-to-parent is safety evidence, not a quality breakthrough.
