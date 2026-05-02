# MeshPrior Stage 21 Long-Budget Single-Scene Diagnostic Report

Date: 2026-05-02

## Verdict

Gate: `PASS` for aligned single-scene execution, `FAIL` for the Stage17 MeshPrior variant as a long-budget candidate.

All three 7000-iteration runs completed on `parking_phone_tiny` with aligned dataset view, image resolution, checkpoint iteration, render metrics, COLMAP proxy geometry evaluation, topology counts, and W&B records. This is still a single-scene diagnostic because M20 found no second suitable scene.

The important result is negative: the Stage17 MeshPrior-cleaned initialization improves the 2000-iteration diagnostic but collapses by 7000 iterations. It should not be used as the default path for M22 paper evidence. The defensible next direction is topology control / cleanup scheduling on the current branch, not longer training of the Stage17 MeshPrior resume variant.

## Run Setup

Dataset view:

```text
outputs/carnet/meshprior/parking_phone_tiny/dataset_view
```

Budget: `7000` iterations.

GPU: `CUDA_VISIBLE_DEVICES=1`, selected because it was the lightest available GPU at launch.

W&B group:

```text
parking_stage21_long_budget
```

Run roots:

- clean official MeshSplatting: `outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/origin_main_7000iter/model`
- current branch engineering baseline: `outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model`
- Stage17 MeshPrior resume variant: `outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/stage17_meshprior_7000iter/model`

## W&B

| run | W&B |
|---|---|
| clean `origin/main@1a714f3` external log | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/yiwb4d2n |
| current branch 7000 training | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/l5buxl3m |
| Stage17 MeshPrior resume 7000 training | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w3kczubb |

Clean `origin/main` lacks the current `--enable_wandb` integration, so it was externally logged immediately after `render.py + metrics.py` produced `results.json`.

## Independent Render Metrics

These are from `render.py --iteration 7000 --skip_train` followed by `metrics.py`.

| run | PSNR | SSIM | LPIPS | triangles | vertices/proxy | PSNR per 100k tri |
|---|---:|---:|---:|---:|---:|---:|
| clean `origin/main` | 16.134155 | 0.452130 | 0.499124 | 285187 | 517863 | 5.657 |
| current branch | 17.204679 | 0.535045 | 0.450750 | 833775 | 1071408 | 2.064 |
| Stage17 MeshPrior resume | 10.839708 | 0.285366 | 0.662528 | 838883 | 1087793 | 1.292 |

## COLMAP Proxy Geometry

COLMAP proxy geometry is useful for consistency diagnostics, not a ground-truth geometry claim.

| run | depth AbsRel | depth MAE | normal mean angle | points |
|---|---:|---:|---:|---:|
| clean `origin/main` | 0.084499 | 1.617994 | 45.300650 | 21638 |
| current branch | 0.076126 | 1.752241 | 45.561976 | 21814 |
| Stage17 MeshPrior resume | 0.744099 | 5.718173 | 52.580674 | 20985 |

## Training Internal Metrics

These are the training script's own test split metrics at iteration 7000. They should not be mixed with the independent render table.

| run | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|
| clean `origin/main` | 18.132515 | 0.581201 | 0.423590 | 223.256 |
| current branch | 18.141336 | 0.581171 | 0.422990 | 255.037 |
| Stage17 MeshPrior resume | 11.287638 | 0.322786 | 0.659315 | 211.763 |

## Commands

Clean baseline training was run from `/tmp/mesh-splatting-origin-main`:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s /data/peilincai/mesh-splatting/outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m /data/peilincai/mesh-splatting/outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/origin_main_7000iter/model --images images --eval --iterations 7000 --test_iterations 1000 2000 7000 --save_iterations 7000 --checkpoint_iterations 7000 --resolution 4 --scene_name parking_phone_tiny_origin_main_7000iter --wandb_name parking_phone_tiny_origin_main_7000iter
```

Current branch training:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model --images images --eval --iterations 7000 --test_iterations 1000 2000 7000 --save_iterations 7000 --checkpoint_iterations 7000 --resolution 4 --enable_wandb --wandb_project spcarnet_meshprior --wandb_group parking_stage21_long_budget --wandb_name parking_current_branch_gpu1_7000iter --wandb_scalar_log_interval 50 --wandb_disable_fixed_views --scene_name parking_phone_tiny_current_branch_7000iter
```

Stage17 MeshPrior resume preparation and training:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_prepare_parking_recovery_model.py --source_model outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model --copied_checkpoint outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt --output_model outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/stage17_meshprior_7000iter/model --iteration 200
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/stage17_meshprior_7000iter/model --images images --eval --load_iteration 200 --iterations 7000 --test_iterations 1000 2000 7000 --save_iterations 7000 --checkpoint_iterations 7000 --resolution 4 --enable_wandb --wandb_project spcarnet_meshprior --wandb_group parking_stage21_long_budget --wandb_name parking_stage17_meshprior_gpu1_resume_7000iter --wandb_scalar_log_interval 50 --wandb_disable_fixed_views --scene_name parking_phone_tiny_stage17_meshprior_resume_7000iter
```

Each model then ran:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m <MODEL> --images images --eval --iteration 7000 --skip_train --quiet
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m <MODEL>
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m <MODEL> --images images --eval --iteration 7000 --max_points_per_view 500 --output <MODEL>/geometry_eval_colmap/iter_7000.json
```

## Decision

1. Do not continue the Stage17 MeshPrior-cleaned resume variant to longer budgets. Its 7000-iteration result is materially worse than clean and current branch on render and geometry proxy metrics.
2. Keep the clean `origin/main` 7000 run as the current long-budget single-scene baseline.
3. Treat the current branch as the best long-budget implementation baseline on this scene, with the caveat that it uses about `2.92x` the triangles of clean MeshSplatting.
4. Before M22 paper tables, prioritize topology control or a scheduled cleanup ablation for the current branch. The long-budget evidence does not support a positive MeshPrior headline claim yet.

