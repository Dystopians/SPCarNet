# MeshSplatOpt Stage R14.5 Real Checkpoint Fill Dry-Run Report

Date: 2026-05-02

## Gate

`PASS`.

The constructive `FILL_PATCH` checkpoint copy loads through `render.py`, `metrics.py`, and `evaluate_geometry_colmap.py`.

## Scope

This is a tiny constructive path-validation dry-run. It appends one small patch initialized from nearest-vertex radiance on the existing `parking_phone_tiny` 200-iteration checkpoint. It is not a medium public-scene repair pilot and not a paper-quality result.

## Commands

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_real_checkpoint_fill_dryrun.py

CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -m outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/model \
  --iteration 200 \
  --skip_train

CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py \
  -m outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/model

CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py \
  --model_path outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/model \
  --iteration 200 \
  --max_points_per_view 500 \
  --output outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/model/geometry_eval_colmap/iter_200_max500.json
```

## Checkpoint Edit

| field | value |
|---|---:|
| triangles before | `64497` |
| triangles after | `64498` |
| vertices before | `193491` |
| vertices after | `193494` |
| schema valid | `true` |

## Independent Metrics

| row | triangles | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| baseline 200iter | `64497` | `10.949986457824707` | `0.2898596525192261` | `0.6441746354103088` |
| fill dry-run | `64498` | `10.949986457824707` | `0.2898596525192261` | `0.6441746354103088` |

## Sparse Geometry Proxy

Comparable `--max_points_per_view 500` geometry:

| row | points | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|
| baseline 200iter | `21910` | `0.32417137460470213` | `3.6485552222775537` | `51.68797353552561` |
| fill dry-run | `21910` | `0.32417137460470213` | `3.6485552222775537` | `51.68793149935674` |

## Decision

`PASS`.

Constructive fill materialization can survive independent render and geometry evaluation. The remaining blocker for R14 is real edit selection plus render-backed counterfactual acceptance/recovery, not checkpoint compatibility.
