# v108 MSE-Descent-Locked POD-MoE Log

Date: 2026-06-25

## Status

v108 is a new method path implemented after the v107b reliability probe showed a consistent regression versus v106. It is not promoted yet. The first full scene probe is running on `flowers`.

Current roots:

```bash
FIELD_ROOT=/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_field
REPORT_ROOT=/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports
```

## Why v108 Exists

v107b cross-fitted POD-MoE is a stricter reliability audit, but the completed four-scene probe is negative versus v106:

| scene | dPSNR v107b-v106 | dSSIM v107b-v106 | dLPIPS v107b-v106 |
|---|---:|---:|---:|
| counter | -0.003185 | -0.000104 | +0.000134 |
| flowers | -0.001806 | -0.000156 | +0.000079 |
| garden | -0.002785 | -0.000116 | +0.000099 |
| bonsai | -0.006840 | -0.000158 | +0.000106 |
| mean | -0.003654 | -0.000134 | +0.000104 |

The likely failure mode is over-suppression of sparse boundary evidence. `boundary_crossfit_gain_mean` remains near zero on completed scenes, while v106's capacity was what produced its small but stable full9 gain.

v108 changes the question from "does each expert look reliable under a gate?" to "does the final joint expert correction have a certified weighted-MSE descent direction?"

## Implemented Method

New method version:

```text
v108_mse_descent_locked_pod_moe
```

New certificate:

```text
joint_two_expert_weighted_normal_equation_box_qp_descent_lock
```

Implementation summary:

- Reuses the v106/v107 POD-MoE representation: v104c-like base plus detail and occlusion-boundary experts.
- Adds a two-expert box-QP descent lock after expert solve.
- Optimizes per-triangle `0 <= lambda_detail, lambda_boundary <= 1` under the accumulated weighted normal-equation proxy.
- Includes zero update as an explicit candidate, so non-descent experts are suppressed.
- Writes the QP scale into `triangle_expert_mse_scale` and additionally into `triangle_expert_descent_scale`.
- `render.py` uses `triangle_expert_descent_scale` first when present, then falls back to legacy `triangle_expert_mse_scale`.
- Runner identity checks now include v108 `method_version`, builder variant, and `expert_mse_certificate`.

Changed files:

- `scripts/car_model/build_v105_evidence_gated_mixture_field.py`
- `render.py`
- `scripts/car_model/run_v105_evidence_gated_mixture_scene.py`
- `scripts/car_model/smoke_test_v108_descent_lock.py`

## Smoke Validation

Commands run in the project environment:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/build_v105_evidence_gated_mixture_field.py \
  render.py \
  scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  scripts/car_model/smoke_test_v108_descent_lock.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_v108_descent_lock.py
```

Smoke output:

```text
v108 descent-lock smoke passed: scales=[[0.0, 1.0], [0.0, 0.0]], gains=[1.0, -0.0], corrected_sse=[0.0, 1.0]
```

Interpretation: the toy bad expert is suppressed, the good expert is retained, and the corrected SSE does not increase.

## First Probe Command

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 WANDB_MODE=offline TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  --scene flowers \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --v102_bank_root /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625 \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --field_root /dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_field \
  --report_root /dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports \
  --v102_report_root /dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/v102_reports \
  --output_method ours_26000_v108_mse_descent_locked_podmoe_flowers \
  --field_variant pod_moe \
  --method_version v108_mse_descent_locked_pod_moe \
  --gate_source crossfit_risk \
  --renderer_scaling 4 \
  --residual_dtype float16 \
  --ridge 0.001 \
  --residual_clip 0.08 \
  --view_std_floor 1e-4 \
  --rank_rtol 1e-7 \
  --condition_max 1e8 \
  --gate_boost 0.5 \
  --view_gate_temperature 0.0 \
  --chunk_pixels 262144 \
  --gpu 3 \
  --force_field --force_render --force_eval
```

Completed result:

| scene | status | PSNR | dPSNR vs v106 | dPSNR vs v104c | dPSNR vs clean | SSIM | dSSIM vs v106 | LPIPS | dLPIPS vs v106 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | passed, identity OK | 20.075865 | -0.001858 | +0.000021 | +0.393608 | 0.531078 | -0.000162 | 0.374474 | +0.000081 |

Artifacts:

```text
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/flowers/flowers_v108_mse_descent_locked_pod_moe_report.json
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/flowers/flowers_v108_mse_descent_locked_pod_moe_report.md
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/v108_crossfit_flowers_vs_v106.json
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/v108_crossfit_flowers_vs_v106.csv
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/v108_crossfit_flowers_vs_v106.md
```

Render-space audit versus v106:

| candidate | base | views | improved views | worse views | mean delta MSE |
|---|---|---:|---:|---:|---:|
| v108 crossfit flowers | v106 flowers | 22 | 0 | 22 | +0.00000403 |

Audit artifacts:

```text
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/v108_crossfit_vs_v106_flowers_render_delta_mse.json
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/v108_crossfit_vs_v106_flowers_render_delta_mse.csv
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_20260625_reports/v108_crossfit_vs_v106_flowers_render_delta_mse.md
```

Interpretation: v108 `crossfit_risk` is a valid but negative probe. It remains better than clean MeshSplatting on `flowers`, but it is worse than v106 on all three reported metrics, and true render-space MSE is worse than v106 on every held-out view. This confirms that the current certificate does not yet close the gap to the final render-realized parent improvement.

## Parallel Mechanism Ablation

Because the v107b diagnosis points to possible cross-fit over-suppression, I launched one mechanism-isolation ablation in parallel. This is not a scene-specific parameter sweep: all numeric settings stay fixed, and only the reliability source changes from `crossfit_risk` to `normal_equation` to separate two hypotheses:

- if `crossfit_risk` fails but `normal_equation` succeeds, the main bottleneck is likely expert suppression;
- if both fail versus v106, the descent certificate is still not closed under the final render-realized correction.

Roots:

```bash
FIELD_ROOT=/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_field
REPORT_ROOT=/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports
```

Command:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 WANDB_MODE=offline TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  --scene flowers \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --v102_bank_root /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625 \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --field_root /dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_field \
  --report_root /dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports \
  --v102_report_root /dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/v102_reports \
  --output_method ours_26000_v108_mse_descent_locked_podmoe_ne_flowers \
  --field_variant pod_moe \
  --method_version v108_mse_descent_locked_pod_moe \
  --gate_source normal_equation \
  --renderer_scaling 4 \
  --residual_dtype float16 \
  --ridge 0.001 \
  --residual_clip 0.08 \
  --view_std_floor 1e-4 \
  --rank_rtol 1e-7 \
  --condition_max 1e8 \
  --gate_boost 0.5 \
  --view_gate_temperature 0.0 \
  --chunk_pixels 262144 \
  --gpu 2 \
  --force_field --force_render --force_eval
```

Completed result:

| scene | status | PSNR | dPSNR vs v106 | dPSNR vs v104c | dPSNR vs clean | SSIM | dSSIM vs v106 | LPIPS | dLPIPS vs v106 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | passed, identity OK | 20.076418 | -0.001305 | +0.000574 | +0.394161 | 0.531125 | -0.000114 | 0.374427 | +0.000034 |

Render-space audit versus v106:

| candidate | base | views | improved views | worse views | mean delta MSE |
|---|---|---:|---:|---:|---:|
| v108 normal-equation flowers | v106 flowers | 22 | 0 | 22 | +0.00000273 |

Artifacts:

```text
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/flowers/flowers_v108_mse_descent_locked_pod_moe_report.json
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/flowers/flowers_v108_mse_descent_locked_pod_moe_report.md
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/v108_ne_flowers_vs_v106.json
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/v108_ne_flowers_vs_v106.csv
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/v108_ne_flowers_vs_v106.md
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/v108_ne_vs_v106_flowers_render_delta_mse.json
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/v108_ne_vs_v106_flowers_render_delta_mse.csv
/dev/shm/peilincai_spcarnet_v108_mse_descent_locked_podmoe_ne_20260625_reports/v108_ne_vs_v106_flowers_render_delta_mse.md
```

Conclusion: normal-equation reliability is slightly better than crossfit-risk, but it still fails against v106 on all three metrics and all held-out views by render-space MSE. This rules out "crossfit over-suppression only" as the explanation. The descent-lock proxy is not sufficient as a paper-level closure because it does not guarantee improvement over the already validated v106 parent.

## v108 Final Decision

v108 is an implemented and validated negative probe, not the next promoted method. The useful lesson is architectural: any next upgrade must preserve v106 as an explicit parent and prove the accepted correction in render space, not only in coefficient-space normal-equation proxy. The next mechanism is therefore v109: a train-calibrated, render-realized parent gate where target/test GT is never used for policy selection.

## Promotion Rule

v108 can only be considered an upgrade if it beats v106 on the same scene without metric tradeoffs and passes identity checks. A single positive `flowers` result is still only a probe. Full promotion requires multi-scene evidence and explicit v108-v106-v107b comparison.
