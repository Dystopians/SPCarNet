# 6-23 Phase-G v30 Triadic Teacher Mask Bonsai Smoke Log

Date: 2026-06-23

Status: implemented, compile-checked, CPU-smoke-tested, Phase-G contract generated, Bonsai GPU smoke completed. Not promoted.

## Motivation

Phase-J remains the current paper/PPT-safe endpoint, but its strongest RGB gain
still comes from render-time ELA. Earlier Phase-G teacher-bake pilots showed
that a topology-frozen checkpoint with a blunt teacher-render loss could not
absorb enough of the ELA teacher: it failed to beat clean-best on the official
pilot scenes and remained far below render-time ELA. v28/v29 then showed that
further ELA alpha/scale tuning gives only noise-scale or negative held-out
movement.

v30 is a small but real representation-baking change. It does not change the
Phase-J headline. It makes the teacher-bake loss more selective so that the
checkpoint only learns pixels where the ELA teacher is demonstrably useful.

## Method Change

The old `teacher_better` mask applied teacher distillation when:

```text
teacher_error(gt) + margin < current_error(gt)
```

That can still waste capacity on pixels where the compact parent is already as
good as the teacher or where the teacher differs from the parent only by
near-zero noise.

The new triadic mask adds parent evidence:

```text
teacher_error(gt) + margin < parent_error(gt)
teacher_error(gt) + margin < current_error(gt)
mean_abs(teacher - parent) >= parent_delta_min
```

The initial mode is:

```text
teacher_better_current_parent_changed
```

This makes the teacher-bake objective closer to the actual Phase-J ELA residual:
learn where ELA improves over the compact parent, ignore regions where ELA is
not a meaningful or currently useful correction, and keep parent rollback as a
separate one-sided guard.

## Implemented Interfaces

Training:

```text
arguments/__init__.py
train.py
```

New optimization argument:

```text
--teacher_render_parent_delta_min
```

New mask modes in `train.py`:

```text
teacher_better_than_parent
teacher_better_parent_changed
teacher_better_current_parent
teacher_better_current_parent_changed
teacher_parent_better_changed
```

Phase-G runner:

```text
scripts/car_model/ecsr_run_phaseg_teacher_bake_recovery.py
```

Strict compact recovery runner:

```text
scripts/car_model/meshsplatopt_run_strict_compact_recovery.py
```

Smoke test:

```text
scripts/car_model/smoke_test_teacher_render_triadic_mask.py
```

## Verification

Compile:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  train.py \
  arguments/__init__.py \
  scripts/car_model/ecsr_run_phaseg_teacher_bake_recovery.py \
  scripts/car_model/meshsplatopt_run_strict_compact_recovery.py \
  scripts/car_model/smoke_test_teacher_render_triadic_mask.py
```

Result: passed.

CPU smoke:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_teacher_render_triadic_mask.py
```

Result:

```text
[teacher triadic smoke] passed
```

The smoke checks four cases:

1. teacher better than parent and current, with enough teacher-parent delta: loss active;
2. teacher not better than parent: no loss;
3. teacher not better than current: no loss;
4. teacher-parent delta below threshold: no loss.

## Phase-G Contract Dry Run

Run root:

```text
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_contract_20260623
```

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phaseg_teacher_bake_recovery.py \
  --scenes bonsai \
  --out_root /data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_contract_20260623 \
  --teacher_method_name ours_26000_phaseg_v30_triadic_train_teacher \
  --final_iteration 26020 \
  --teacher_render_mask_mode teacher_better_current_parent_changed \
  --teacher_render_parent_delta_min 0.01 \
  --teacher_render_error_margin 0.001 \
  --wandb_mode disabled \
  --wandb_group phaseg_v30_triadic_teacher_mask_contract \
  --wandb_prefix phaseg_v30_triadic_contract \
  --gpu -1
```

Key contract outputs:

```text
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_contract_20260623/bonsai/contract/exact_train_command.txt
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_contract_20260623/bonsai/contract/phaseg_scene_summary.json
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_contract_20260623/phaseg_teacher_bake_summary.md
```

The generated train command includes:

```text
--teacher_render_mask_mode teacher_better_current_parent_changed
--teacher_render_error_margin 0.001
--teacher_render_parent_delta_min 0.01
```

## Bonsai GPU Smoke

Run root:

```text
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5
```

Command:

```bash
RUN=/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5
WANDB_MODE=online PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phaseg_teacher_bake_recovery.py \
  --scenes bonsai \
  --out_root "$RUN" \
  --teacher_method_name ours_26000_phaseg_v30_triadic_train_teacher \
  --final_iteration 26080 \
  --teacher_render_mask_mode teacher_better_current_parent_changed \
  --teacher_render_parent_delta_min 0.01 \
  --teacher_render_error_margin 0.001 \
  --teacher_render_lambda 0.05 \
  --parent_render_rollback_lambda 3.0 \
  --wandb_mode online \
  --wandb_group phaseg_v30_triadic_teacher_mask_smoke_20260623 \
  --wandb_prefix phaseg_v30_triadic_smoke \
  --gpu 5 \
  --execute \
  --skip_geometry \
  > "$RUN/logs/phaseg_v30_smoke.log" 2>&1
```

Current expected outputs:

```text
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/logs/phaseg_v30_smoke.log
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/bonsai/recovery_model/phaseg_scene_summary.json
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/phaseg_teacher_bake_summary.md
```

Result: completed.

W&B runs:

```text
train-only ELA teacher: 1ez09i51
topology-frozen bake:  gx7qcyrv
```

The train loop loaded both teacher and parent render caches:

```text
[TeacherRender] loaded 255/255 train renders
[ParentRenderRollback] loaded 255/255 train renders
```

W&B summary for the bake run confirms the new mask was active rather than a
dead option:

| logged field | value |
|---|---:|
| `teacher_render/mask_fraction` | 0.0172509 |
| `loss_components/loss_teacher_render` | 0.005954 |
| `loss_components/loss_teacher_render_pure` | 0.119087 |
| `parent_render_rollback/active_fraction` | 0.013583 |
| `loss_components/loss_parent_render_rollback` | 0.445106 |

Final output artifacts:

```text
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/phaseg_teacher_bake_summary.json
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/phaseg_teacher_bake_summary.md
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/bonsai/recovery_model/phaseg_scene_summary.json
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/bonsai/recovery_model/results.json
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/bonsai/recovery_model/point_cloud/iteration_26080/point_cloud_state_dict.pt
```

Held-out test result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| selected clean `ours_26000` | 28.895233 | 0.896400 | 0.259493 |
| Phase-F source ELA row | 29.784437 | 0.898168 | 0.257401 |
| Phase-F render-time ELA | 30.874977 | 0.917746 | 0.213867 |
| v30 baked checkpoint `ours_26080` | 28.814400 | 0.893765 | 0.263591 |

Delta table:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v30 vs selected clean | -0.080833 | -0.002635 | +0.004098 |
| v30 vs Phase-F source ELA | -0.970037 | -0.004403 | +0.006189 |
| v30 vs Phase-F render-time ELA | -2.060577 | -0.023981 | +0.049723 |

Topology audit:

| checkpoint | triangles | vertices |
|---|---:|---:|
| iteration 26000 | 9,555,533 | 3,295,557 |
| iteration 26080 | 9,555,533 | 3,295,557 |

Topology unchanged: true.

## Acceptance Criteria

This smoke is not allowed to become the headline method. It is useful only if it
proves all of the following:

- train-only teacher render generation completes: **passed**;
- topology-frozen recovery completes without topology update: **passed**;
- final held-out render and metrics are saved: **passed**;
- train logs expose nonzero `teacher_render/mask_fraction` for some views: **passed**;
- candidate is compared against selected clean, compact/source ELA, and
  render-time Phase-F/Phase-J ELA: **passed**;
- if metrics are weak, the result is documented as a failed or diagnostic
  representation-bake attempt: **passed**.

## Current Interpretation

v30 is a better controlled teacher-bake loss, but it is still not enough. The
run proves that the new triadic mask is wired through the real training/eval
pipeline and is active, yet the baked checkpoint is worse than selected clean on
Bonsai. This repeats the key Phase-G lesson in a stricter form: image-level
teacher loss can be made safer, but it still does not give the topology-frozen
checkpoint enough local capacity to absorb the ELA teacher.

Therefore the next implementation should move from image-level teacher loss to
persistent surface-addressed teacher residuals:

```text
teacher_render - compact_parent_render
  -> surface evidence cache with face/barycentric support
  -> face-local SH / low-rank residual basis
  -> Phase-K trainval gate
```
