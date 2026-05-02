# MeshPrior Stage 17 Real Variant Implementation Report

Date: 2026-05-01

## Scope

Stage 17 creates the first real MeshPrior `parking_phone_tiny` scene-training variant. The method applies previously accepted MeshPrior cleanup actions to a copied 200-iteration checkpoint, prepares a normal model directory, and resumes current-branch training from `iteration_200` to `iteration_2000`.

This is no longer only an offline checkpoint evaluation. The MeshPrior-edited checkpoint is used as the initialization for continued scene optimization.

## Files Added

- `scripts/car_model/smoke_test_meshprior_stage17_real_variant.py`
- `docs/car_model/meshprior_stage17_real_variant_design.md`
- `docs/car_model/meshprior_stage17_real_variant_smoke.md`
- `docs/car_model/meshprior_stage17_real_variant_implementation_report.md`

## Inputs

- dataset: `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- source baseline model: `outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model`
- copied MeshPrior cleanup checkpoint: `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt`
- accepted proposal counts:
  - accepted cleanup proposals: `8`
  - rejected no-op proposals: `8`
  - rejected floater proposals: `8`
  - source model edited: `false`

## W&B

Training-time online W&B was used for both resumed-training smoke and the 2000-iteration variant.

- smoke run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/y4432er1`
- 2000-iteration run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vyrun0qo`
- W&B URL file: `outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/wandb_url.txt`

## Commands

Initialization:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_prepare_parking_recovery_model.py --source_model outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model --copied_checkpoint outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt --output_model outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model --iteration 200
```

Smoke training:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_smoke/model --images images --eval --load_iteration 200 --iterations 300 --test_iterations 300 --save_iterations 300 --checkpoint_iterations 300 --resolution 4 --enable_wandb --wandb_project spcarnet_meshprior --wandb_group parking_stage17_real_variant --wandb_name parking_stage17_gpu1_resume_smoke_300iter --wandb_scalar_log_interval 10 --wandb_disable_fixed_views --scene_name parking_phone_tiny_stage17_resume_smoke
```

2000-iteration training:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model --images images --eval --load_iteration 200 --iterations 2000 --test_iterations 1000 2000 --save_iterations 2000 --checkpoint_iterations 2000 --resolution 4 --enable_wandb --wandb_project spcarnet_meshprior --wandb_group parking_stage17_real_variant --wandb_name parking_stage17_gpu1_meshprior_resume_2000iter --wandb_scalar_log_interval 20 --wandb_disable_fixed_views --scene_name parking_phone_tiny_stage17_meshprior_resume_2000iter
```

Post-render metrics:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model --images images --eval --iteration 2000 --skip_train --quiet
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model
```

Geometry proxy:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model --images images --eval --iteration 2000 --max_points_per_view 500 --output outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model/geometry_eval_colmap/iter_2000.json
```

## Results

### Training Internal Metrics

| run | iter | test L1 | test PSNR | test SSIM | test LPIPS | test FPS |
|---|---:|---:|---:|---:|---:|---:|
| Stage17 smoke | 300 | 0.2131838062 | 11.5936053771 | 0.3349873807 | 0.6415096864 | 392.0611966778 |
| Stage17 real variant | 1000 | 0.1739416310 | 13.1176037435 | 0.3794519540 | 0.6071134640 | 343.7609857320 |
| Stage17 real variant | 2000 | 0.1612277984 | 13.4438069308 | 0.3471139595 | 0.6021583963 | 272.8530837309 |

These are training-internal metrics and must not be mixed with `render.py + metrics.py` numbers without labels.

### Post-Render Metrics

| run | iter | PSNR | SSIM | LPIPS | triangles | vertices |
|---|---:|---:|---:|---:|---:|---:|
| origin/main clean candidate | 2000 | 11.0476598740 | 0.2199306488 | 0.6417058110 | 39079 | 58458 |
| current branch engineering | 2000 | 11.5994377136 | 0.2702677548 | 0.6347319484 | 782982 | 820107 |
| Stage17 MeshPrior real variant | 2000 | 13.2782726288 | 0.3039793670 | 0.6076099277 | 777251 | 816498 |

Stage17 improves the post-render metric path versus both existing 2000-iteration baselines, but it does not solve the topology inflation problem.

### COLMAP Geometry Proxy

| run | depth count | depth MAE | depth AbsRel | normal mean angle deg | normal median angle deg |
|---|---:|---:|---:|---:|---:|
| origin/main clean candidate | 19776 | 13.7902993339 | 5.6119052058 | 52.1989385790 | 52.9868341712 |
| current branch engineering | 21911 | 4.4141606252 | 0.4278796566 | 52.5651849634 | 53.6820151423 |
| Stage17 MeshPrior real variant | 21911 | 3.8259249166 | 0.3666914408 | 52.1695839576 | 52.6909967136 |

Stage17 improves sparse depth proxy metrics versus current branch and is slightly better on normal mean angle. This remains a COLMAP proxy, not ground-truth geometry.

### Cleanup Safety

- final cleanup enabled: `false`
- cleanup executed: `false`
- final cleanup pruned: `0`
- triangles: `777251 -> 777251`
- vertices: `816498 -> 816498`

## Interpretation

Stage17 is the first complete real MeshPrior scene-training variant:

- MeshPrior proposes and gates cleanup actions on copied real-scene patches.
- Accepted actions are written into a copied checkpoint.
- The copied checkpoint is resumed as a normal trainable scene model.
- The 2000-iteration run uses online training-time W&B.
- Render and geometry metrics are evaluated with the same scripts as the baselines.

The result is promising on post-render and sparse geometry proxy metrics, but not yet a paper-level claim because topology remains roughly as large as the current-branch engineering baseline. M18 topology-budget comparison is now mandatory before stronger claims.

## Gate

Stage gate: `PASS`.

Claim status: `SOFT`. The real variant is implemented, trainable, W&B-logged, and metric-positive on this scene, but it is not efficiency-normalized and has only one scene.
