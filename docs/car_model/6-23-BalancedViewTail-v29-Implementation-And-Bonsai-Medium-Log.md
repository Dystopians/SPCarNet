# 6-23 Balanced View-Tail v29 Implementation And Bonsai Medium Log

Date: 2026-06-23

Status: implemented, compile-checked, smoke-tested, dry-run verified, and Bonsai medium retry completed. Not promoted as the headline method.

## Motivation

v28 added policy-view tail-safe alpha shrink, but the first Bonsai fix2 run exposed an objective mismatch:

- view-tail scale selection was safe under per-view MSE gain;
- all reports selected `view_tail_scale = 1.0`;
- the candidate still failed the honest PhaseK gate under train-val PSNR / SSIM / LPIPS / balanced score;
- plan-stage train-val delta was `-0.020847` PSNR, `-0.000283` SSIM, `+0.000571` LPIPS, with balanced delta `-0.037934`.

This means MSE-safe alpha replay is not sufficient when the selector and paper metrics are PSNR/SSIM/LPIPS based. v29 changes the alpha view-tail objective to match the downstream gate more closely.

## Method Change

v29 keeps the v28 view-tail scale replay:

```text
alpha_final(pixel) = view_tail_scale * alpha_bin(pixel)
```

but changes the scale-selection objective from per-view MSE gain to a train/policy-val render metric:

```text
score = dPSNR + 20 * dSSIM - 20 * dLPIPS
```

The scale grid is still fixed:

```text
1.0, 0.75, 0.5, 0.25, 0.0
```

and test views remain report-only. No held-out test GT is used to choose the scale.

## Implemented Interfaces

Core implementation:

- `utils/evidence_lumigraph_adapter.py`
  - `AlphaCalibrator.view_tail_objective`;
  - `view_tail_ssim_weight`, `view_tail_lpips_weight`;
  - `view_tail_compute_lpips`;
  - `view_tail_metric_max_side`;
  - balanced scale replay over train/policy-val calibration views;
  - per-scale candidate stats: `mean_score`, `mean_psnr_gain`, `mean_ssim_gain`, `mean_lpips_gain`, `mean_mse_gain`, and `lpips_regression_fraction`.

ELA CLI:

- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
  - `--alpha_view_tail_objective {mse,balanced}`;
  - `--alpha_view_tail_ssim_weight`;
  - `--alpha_view_tail_lpips_weight`;
  - `--alpha_view_tail_compute_lpips` / `--no-alpha_view_tail_compute_lpips`;
  - `--alpha_view_tail_metric_max_side`;
  - W&B config/logging includes the new objective and LPIPS flags.

Pipeline:

- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
  - forwards all v29 ELA view-tail objective arguments.
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
  - forwards all v29 ELA view-tail objective arguments.
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py`
  - fixed profile `field_region_render_risk_strict_v29`;
  - contract id `field_region_render_risk_strict_v29_balanced_lpips_view_tail_alpha_shrink`;
  - profile values:

```text
ela_alpha_view_tail_objective = balanced
ela_alpha_view_tail_ssim_weight = 20.0
ela_alpha_view_tail_lpips_weight = 20.0
ela_alpha_view_tail_compute_lpips = true
ela_alpha_view_tail_metric_max_side = 512
```

Audit:

- `scripts/car_model/ecsr_audit_viewtail_alpha_run.py`
  - now reports `view_tail_objective`, LPIPS enablement, selected mean score, selected dPSNR/dSSIM/dLPIPS, and the older MSE/tail diagnostics.

## Verification

Static compile:

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

ELA smoke:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

Result:

```text
[ELA smoke] passed
```

Dry-run:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v29 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 0 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v29_balanced_viewtail_20260622 \
  --pipeline_label dryrun_field_region_render_risk_strict_v29_balanced_viewtail_20260622 \
  --wandb_mode disabled \
  --dry_run \
  --force
```

Result: `commands=8`, `dry_run=true`.

Manifest check confirmed plan, candidate-owned refit, and selector commands include:

```text
--ela_alpha_view_tail_objective balanced
--ela_alpha_view_tail_ssim_weight 20.0
--ela_alpha_view_tail_lpips_weight 20.0
--ela_alpha_view_tail_compute_lpips
--ela_alpha_view_tail_metric_max_side 512
```

## Bonsai Medium Attempts

### GPU1 attempt

Run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_20260623_bonsai_medium_gpu1
```

Command:

```bash
WANDB_MODE=online PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v29 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 1 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_20260623_bonsai_medium_gpu1 \
  --pipeline_label field_region_render_risk_strict_v29_balanced_viewtail_20260623_bonsai_medium_gpu1 \
  --wandb_mode online \
  --force
```

Outcome: failed before reaching ELA. The top pipeline exited with an empty `top_pipeline.nohup.log`; PhaseK log stopped in `ecsr_build_surface_evidence_cache.py` after loading the compact model config. No v29 balanced alpha report was produced. This is treated as a run-environment/evidence-cache failure, not as a method result.

### GPU4 retry

Run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4
```

Command:

```bash
RUN=/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4
mkdir -p "$RUN/logs"
WANDB_MODE=online PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v29 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 4 \
  --output_root "$RUN" \
  --pipeline_label field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4 \
  --wandb_mode online \
  --force \
  > "$RUN/logs/top_pipeline.nohup.log" 2>&1 &
echo $! > "$RUN/pipeline.pid"
```

Initial process inspection confirmed:

- top pipeline PID written to `pipeline.pid`;
- PhaseK plan-generation child started;
- evidence cache child started on GPU4;
- command includes balanced view-tail objective, LPIPS metric computation, W&B online logging, and the fixed v29 contract.

The first GPU4 pipeline launch repeated the same early evidence-cache stop as the GPU1 attempt: the PhaseK log stopped after compact model config loading and did not write an `[exit_code]`. A controlled foreground probe showed that the renderer itself was usable, so the surface evidence cache was generated manually into the same run root:

```bash
RUN=/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  -i images_2 \
  --resolution -1 \
  --eval \
  --iteration 26000 \
  --scene_name bonsai \
  --out_dir "$RUN/surface_evidence" \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --final_method_name ours_26000_phasej_guarded_adaptedge_ela \
  --max_views 12 \
  --view_stride 4 \
  --view_offset 0 \
  --high_error_quantile 0.65 \
  --top_k_faces 8192 \
  --save_view_npz \
  --save_residual_rgb \
  --quiet
```

Artifacts:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/surface_evidence/bonsai/surface_evidence_summary.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/surface_evidence/bonsai/surface_evidence_report.md
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/surface_evidence/bonsai/views/*.npz
```

The pipeline was then relaunched with the same fixed v29 profile and `--force`. PhaseK correctly logged:

```text
[cache] reuse complete surface evidence cache: .../surface_evidence/bonsai
```

Current status: running beyond the evidence-cache step. The active plan stage has entered `ecsr_apply_surface_residual_facelocal_sh1_delta.py`, so the remaining result will test the real v29 balanced view-tail ELA path rather than the cache builder.

## Promotion Criteria

v29 can only replace the current Phase-J headline if all of these are true:

1. medium run completes without infrastructure failure;
2. `ela_report.json` shows `view_tail_objective = balanced`;
3. selected `view_tail_scale` is chosen by train/policy-val balanced objective, not held-out test;
4. train-val gate accepts a real non-noop candidate;
5. held-out test report-only is at least non-regressive and ideally improves PSNR/SSIM/LPIPS;
6. multi-scene replay confirms the effect is not Bonsai-only.

Until then, the paper/PPT-safe endpoint remains Phase-J.

## Immediate Next Checks

```bash
pgrep -af 'field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623|ecsr_build_surface_evidence_cache|meshsplatopt_apply_evidence_lumigraph_adapter' || true

tail -120 \
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/plan_generation/bonsai/phasek_barycentric_gate.log

find \
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4 \
-maxdepth 8 -type f \( -name 'ela_report.json' -o -name '*decision.json' -o -name '*results.json' -o -name '*summary.md' \) | sort
```

## Bonsai Medium Retry: Manual Artifact Repair And Candidate-Owned Refit Result

Run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4
```

After the no-force top pipeline saw existing plan artifacts but missing rendered
`plan_base` products, the missing render/objective artifacts were generated
manually and the fixed `filter,selector` stages were relaunched with W&B online.
This was an orchestration repair, not a method change.

The candidate-region raw-base objective before refit showed a positive mean but
risky tail:

| metric | value |
|---|---:|
| region count | 41 |
| mean core balanced delta | +0.204927 |
| mean core dPSNR | +0.226258 |
| mean core dSSIM | +0.002882 |
| mean core dLPIPS | +0.003948 |
| negative core balanced fraction | 0.243902 |
| tail core balanced CVaR delta | -0.160103 |
| worst core balanced delta | -0.381622 |

The candidate-owned refit path then ran the real v29 balanced/LPIPS-aware ELA
with W&B online:

```text
view_tail_objective = balanced
view_tail_compute_lpips = true
view_tail_ssim_weight = 20
view_tail_lpips_weight = 20
metric_max_side = 512
```

W&B runs observed in the refit stage:

```text
phasej test replay: pzcqfqjx
candidate-owned ELA test: vy0a9w2t
candidate-owned trainval gate: 6tcryqn9
```

Final candidate-owned refit decision:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/candidate_owned_refit/decisions/bonsai_decision.json
```

Decision: **rejected**.

| split | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| trainval gate | -0.019606 | -0.000314 | +0.000646 |
| held-out test, report-only | +0.005390 | +0.000144 | -0.000595 |

Gate reasons:

```text
psnr_gain_below_0
ssim_regression_exceeds_5e-05
lpips_regression_exceeds_0.00015
balanced_delta_below_0
render_region_tail_cvar_below_-2e-05
```

Interpretation:

- v29 does exercise a real method change in the train/eval pipeline;
- the balanced/LPIPS-aware view-tail interface works and is recorded in W&B;
- held-out test has a tiny positive report-only delta, but trainval and tail
  risk reject it, so this is not a publishable improvement;
- this is a useful strict-gate result because it prevents a test-only false
  positive from becoming the headline method.

## Bonsai Medium Retry: Final Selector Result

The remaining `filter,selector` stage finished successfully.

Top-level pipeline result:

```text
output_root = /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4
commands = 7
dry_run = false
```

Final coupled-selector decision:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/selector/bonsai/coupled_selector_decision.json
```

Decision: **accepted** `strictfull_s1` under train-val-only selection.

| selector field | value |
|---|---:|
| selection uses held-out test GT | false |
| selected trial | strictfull_s1 |
| candidate count | 20 |
| selected trainval balanced delta | +0.0006342530 |
| effective held-out test dPSNR, report-only | -0.0000171661 |
| effective held-out test dSSIM, report-only | -0.0000039935 |
| effective held-out test dLPIPS, report-only | +0.0000103563 |

Strict trial decision:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/selector/trials/strictfull_s1/decisions/bonsai_decision.json
```

| split | dPSNR | dSSIM | dLPIPS | balanced delta |
|---|---:|---:|---:|---:|
| trainval gate | +0.000330 | +0.0000067 | -0.0000085 | +0.000634 |
| held-out test, report-only | -0.000017 | -0.0000040 | +0.0000104 | -0.000304 |

Interpretation:

- the final selector path exercised the real v29 balanced/LPIPS-aware ELA path and produced a non-noop candidate;
- the selector did not use held-out test GT for selection;
- the accepted trainval margin is extremely small;
- the held-out report-only delta is also extremely small but negative on all three paper metrics;
- therefore this Bonsai selector result is useful as a pipeline/diagnostic success, but it is **not strong enough to replace Phase-J** as the paper/PPT headline.

The complete v29 evidence chain now says:

1. candidate-owned refit was honestly rejected by trainval/tail risk despite a tiny report-only test positive;
2. the final selector could find a tiny trainval-positive candidate, but held-out report-only moved slightly negative;
3. balanced view-tail selection is correctly wired and logged, but by itself does not yet solve the representation-level bottleneck.
