# MeshSplatOpt Stage R14.2 Real Checkpoint Dry-Run Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R14.2 applies a low-risk dry-run edit to a real existing Mesh Splatting checkpoint copy and creates a normal model directory layout for future render/metrics evaluation.

## Files Added

- `scripts/car_model/meshsplatopt_real_checkpoint_dryrun.py`

## Run

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_real_checkpoint_dryrun.py
```

Input:

```text
outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt
```

Output:

```text
outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/point_cloud/iteration_200/point_cloud_state_dict.pt
```

## Result

- input schema valid: `true`
- output schema valid: `true`
- edit: `DELETE_TRIANGLES`
- triangles: `64497 -> 64496`
- vertices: `193491 -> 193491`

This is a path-validation dry-run only, not a method-quality result.

## Planned Eval Commands

```bash
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py -m outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model --iteration 200 --skip_train
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py --model_path outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model --iteration 200
```

## Decision

`PASS`. Real checkpoint-copy path is valid for delete/snap edits. This still does not unblock certified fill or full R14 repair training.
