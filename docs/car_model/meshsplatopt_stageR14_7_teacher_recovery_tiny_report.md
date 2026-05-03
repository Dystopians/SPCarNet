# MeshSplatOpt Stage R14.7 Teacher Recovery Tiny Report

Date: 2026-05-02

## Gate

`PASS`.

R14.7 upgrades the R11 teacher-recovery contract from cache-only infrastructure to a real tiny recovery run on the R14.5 edited parking checkpoint. Training used W&B online, then independent `render.py + metrics.py`, followed by sparse COLMAP geometry evaluation.

## Implementation

Updated:

```bash
scripts/car_model/meshsplatopt_run_teacher_recovery.py
```

New optional behavior:

- `--run_real_tiny` copies the edited model into a recovery output directory;
- reads the source model `cfg_args` and passes `source_path`, image folder, resolution, and `--eval` into `train.py`;
- resumes with `--load_iteration`;
- runs a short recovery window with W&B enabled;
- renders and evaluates the recovered checkpoint independently;
- keeps the existing teacher-cache contract files.

## Command

```bash
WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 \
  /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_run_teacher_recovery.py \
  --model_path outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/model \
  --edit_json outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/edit.json \
  --output_dir outputs/carnet/meshsplatopt/stageR14_7_teacher_recovery_tiny \
  --iterations 20 \
  --load_iteration 200 \
  --run_real_tiny \
  --gpu 0 \
  --python /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  --wandb_project spcarnet_meshprior \
  --wandb_group meshsplatopt_r14_teacher_recovery \
  --wandb_name meshsplatopt_r14_7_parking_fill_recovery_20step_gpu0_retry
```

W&B:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/n05mce4y
```

## Results

| row | iteration | triangles | vertices | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| edited fill candidate | `200` | `64498` | `193494` | `10.949986457824707` | `0.2898596525192261` | `0.6441746354103088` |
| recovered tiny | `220` | `64498` | `193494` | `10.995698928833008` | `0.29370972514152527` | `0.6429890990257263` |

Sparse geometry proxy:

| row | points | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|
| edited fill candidate, iter 200 | `21910` | `0.32417137460470213` | `3.6485552222775537` | `51.68793149935674` |
| recovered tiny, iter 220 | `21911` | `0.325047677579098` | `3.6494193758930376` | `51.93818681106907` |

## Notes

The first recovery attempt initialized W&B but failed before training because `train.py` does not merge model-directory `cfg_args`; it parsed default `source_path` and could not recognize the scene. The recovery script now reads the source model `cfg_args` and passes the dataset arguments explicitly.

This is a tiny recovery functionality test, not a medium-scene method result. It validates that an edited checkpoint can be resumed, W&B-logged, rendered, and evaluated after recovery.

## Decision

`PASS`.

R14 now has checkpoint materialization, render-backed acceptance, and real tiny teacher recovery. Remaining work before medium claims is real edit selection on public scenes and a W&B-logged 2000-iteration comparison against Stage35/PRISM baselines.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_7_teacher_recovery_tiny/teacher_recovery_run_report.json`
- `outputs/carnet/meshsplatopt/stageR14_7_teacher_recovery_tiny/real_tiny_recovery_report.json`
- `outputs/carnet/meshsplatopt/stageR14_7_teacher_recovery_tiny/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_7_teacher_recovery_tiny/recovery_model/geometry_eval_colmap/iter_220_max500.json`
