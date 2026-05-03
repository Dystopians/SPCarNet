# MeshSplatOpt Stage R14.13 Bonsai Post-Edit Recovery Diagnostic Report

Date: 2026-05-02

## Gate

`PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET`.

This stage runs a W&B-logged 200-step recovery diagnostic on `bonsai` after the R14.11 accepted area-outlier edit. It resumes the edited checkpoint from iteration 2000 and trains to iteration 2200.

This is not an equal-budget medium result because no `bonsai` iteration-200 checkpoint was available locally.

## W&B

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/z498br53
```

## Command

```bash
WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online CUDA_VISIBLE_DEVICES=4 \
  /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_run_teacher_recovery.py \
  --model_path outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/gate/candidate_model \
  --edit_json outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/selected_edit.json \
  --output_dir outputs/carnet/meshsplatopt/stageR14_13_bonsai_postedit_recovery_diagnostic \
  --iterations 200 \
  --load_iteration 2000 \
  --run_real_tiny \
  --gpu 4 \
  --python /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  --wandb_project spcarnet_meshprior \
  --wandb_group meshsplatopt_r14_postedit_recovery_diag \
  --wandb_name meshsplatopt_r14_13_bonsai_area_outlier_recovery_200step_gpu4
```

## Results

| row | iteration | triangles | vertices | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline sparse-depth | `2000` | `2487474` | `2478890` | `12.201611518859863` | `0.20731531083583832` | `0.6242585182189941` |
| area-outlier post-edit gate | `2000` | `2487473` | `2478890` | `12.20124340057373` | `0.20730286836624146` | `0.6242548227310181` |
| post-edit recovery diagnostic | `2200` | `2487473` | `2478890` | `13.276382446289062` | `0.24055197834968567` | `0.6113873720169067` |

Sparse COLMAP geometry:

| row | points | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|
| baseline sparse-depth | `18500` | `0.49587362441894434` | `4.907808996255763` | `50.118300749023625` |
| post-edit recovery diagnostic | `18500` | `0.4733479577347401` | `4.762276469029142` | `49.21947049923495` |

## Decision

`PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET`.

The edited `bonsai` checkpoint can be resumed with W&B online, rendered, evaluated, and geometrically checked after recovery. The 2200iter diagnostic improves render metrics and sparse geometry versus the 2000iter baseline, but it uses 200 extra training steps and must not be reported as an equal-budget R14 win.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_13_bonsai_postedit_recovery_diagnostic/teacher_recovery_run_report.json`
- `outputs/carnet/meshsplatopt/stageR14_13_bonsai_postedit_recovery_diagnostic/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_13_bonsai_postedit_recovery_diagnostic/recovery_model/geometry_eval_colmap/iter_2200_max500.json`
