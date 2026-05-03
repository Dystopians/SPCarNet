# MeshSplatOpt Stage R14.3 Render Eval Dry-Run Report

Date: 2026-05-02

## Gate

`PASS`.

The real checkpoint dry-run copy from R14.2 loads through the normal `render.py`, `metrics.py`, and `evaluate_geometry_colmap.py` paths.

## Scope

This is a path-validation and adapter-safety evaluation only. It deletes one triangle from a 200-iteration parking checkpoint. It is not a MeshSplatOpt method-quality result and not a medium/public-scene pilot.

## GPU

GPU availability was checked before launch. GPU 4 was used:

```bash
CUDA_VISIBLE_DEVICES=4
```

## Commands

```bash
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -m outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model \
  --iteration 200 \
  --skip_train

CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py \
  -m outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model

CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py \
  --model_path outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model \
  --iteration 200 \
  --max_points_per_view 500 \
  --output outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/geometry_eval_colmap/iter_200_max500.json
```

## Independent Render Metrics

| row | triangles | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| baseline 200iter | `64497` | `10.949986457824707` | `0.2898596525192261` | `0.6441746354103088` |
| dry-run delete-one | `64496` | `10.949986457824707` | `0.28985968232154846` | `0.6441748142242432` |

The deltas are effectively neutral, as expected for deleting one triangle.

## Sparse Geometry Proxy

Comparable `--max_points_per_view 500` geometry:

| row | points | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|
| baseline 200iter | `21910` | `0.32417137460470213` | `3.6485552222775537` | `51.68797353552561` |
| dry-run delete-one | `21910` | `0.32417137460470213` | `3.6485552222775537` | `51.68804758349445` |

## Artifacts

- `outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/test/ours_200/`
- `outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/per_view.json`
- `outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/geometry_eval_colmap/iter_200_max500.json`

## Decision

`PASS`.

The checkpoint adapter output is renderable and independently evaluable. The next blocker is no longer basic checkpoint schema/render compatibility; it is implementing safe radiance initialization and recovery for constructive edits such as fill/split, then wiring real counterfactual view selection into R10.
