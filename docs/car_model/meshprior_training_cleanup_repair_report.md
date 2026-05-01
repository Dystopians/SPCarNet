# MeshPrior Training Cleanup Repair Report

| Field | Value |
|---|---|
| Date | 2026-05-01 |
| Blocking Stage | Before M13 |
| Status | PASS |

## 1. Problem

A 200-iteration wandb training smoke completed, but final cleanup pruned the model from:

```text
5706 triangles -> 15 triangles
17118 vertices -> 45 vertices
```

This happened even though PRISM pruning was disabled.

## 2. Root Cause

`train.py` computed:

```text
cleanup_executed = not (prism_enabled and prism_disable_final_cleanup_prune)
```

For ordinary non-PRISM training, `prism_enabled=false`, so final cleanup executed by default.

## 3. Fix

Final cleanup now executes only when PRISM pruning is enabled and final cleanup is not disabled:

```text
cleanup_executed = prism_enabled and not prism_disable_final_cleanup_prune
```

This keeps ordinary training readout checkpoints from being destructively pruned by a PRISM-specific cleanup path.

## 4. Verification Plan

Run:

```bash
python -m compileall scripts/car_model ss3dm_prior -q
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online python train.py ... --iterations 200 ...
```

Pass criteria:

- wandb run completes;
- final cleanup summary reports `final_cleanup_enabled=false`;
- post-cleanup triangle count equals pre-cleanup triangle count;
- checkpoint and summary are written;
- geometry/render metrics are inspected before deciding whether to continue.

## 5. Verification Results

Repair run:

```text
outputs/carnet/meshprior/wandb_train_m13_repair_no_cleanup/model
```

Wandb:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3swt58x2
```

Training command summary:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py \
  -s outputs/carnet/meshprior/wandb_train_m13_repair_no_cleanup/dataset_view \
  -m outputs/carnet/meshprior/wandb_train_m13_repair_no_cleanup/model \
  --images images --eval --iterations 200 \
  --test_iterations 50 100 200 --save_iterations 200 --checkpoint_iterations 200 \
  --resolution 4 --enable_wandb --wandb_project spcarnet_meshprior \
  --wandb_group meshprior_m13_repair \
  --wandb_name meshprior_m13_video_gpu1_200iter_no_cleanup \
  --wandb_scalar_log_interval 10 --wandb_disable_fixed_views \
  --scene_name meshprior_m13_video_repair
```

Final cleanup summary:

| Metric | Value |
|---|---:|
| `final_cleanup_enabled` | `false` |
| `final_cleanup_pruned` | `0` |
| `pre_prune_triangle_count` | `5706` |
| `post_prune_triangle_count` | `5706` |
| `pre_prune_vertex_count` | `17118` |
| `post_prune_vertex_count` | `17118` |

Render metrics at iteration 200:

| Split | L1 | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|---:|
| test | `0.3613577075302601` | `6.933581471443176` | `0.16371289547532797` | `0.694071426987648` | `334.7374487692397` |
| train | `0.4450825095176697` | `5.236159324645996` | `0.07122536152601243` | `0.7069872736930848` | `336.84369049395616` |

COLMAP sparse geometry at iteration 200:

| Metric | Value |
|---|---:|
| depth count | `500` |
| depth MAE | `0.024122862845250084` |
| depth AbsRel | `0.10470779720655764` |
| delta@1.25 | `0.982` |
| normal count | `500` |
| normal mean angle | `37.51919533010328` |
| normal median angle | `30.924912742808946` |

## 6. Decision

The blocker is fixed.

Do not continue with the old default-cleanup behavior. M13 may proceed only from the repaired behavior where non-PRISM training keeps final cleanup disabled by default.
