# MeshPrior Parking Recovery Model Evaluation Report

Date: 2026-05-01

## Scope

This step prepares an evaluation-ready recovery model directory around the checkpoint-copy cleanup result and runs COLMAP sparse geometry evaluation against the parking phone tiny scene.

The baseline model is not overwritten.

## Inputs

- Source model: `outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model`
- Cleanup checkpoint copy: `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt`
- Dataset view: `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`

## Implementation

Added:

- `scripts/car_model/meshprior_prepare_parking_recovery_model.py`
- `scripts/car_model/smoke_test_meshprior_parking_recovery_model.py`

The recovery model script creates:

- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/cfg_args`
- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/cameras.json`
- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/input.ply`
- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/point_cloud/iteration_200/point_cloud_state_dict.pt`

Recovery model size:

- triangles: `63965`
- vertices: `191895`

## Geometry Evaluation

Command:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup --images images --eval --iteration 200 --max_points_per_view 500 --output outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/geometry_eval_colmap/iter_200.json
```

Output:

- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/geometry_eval_colmap/iter_200.json`

Comparison with the 200-iteration baseline:

| metric | baseline | recovery cleanup | delta |
| --- | ---: | ---: | ---: |
| evaluated views | 54 | 54 | 0 |
| depth count | 21910 | 21910 | 0 |
| depth MAE | 3.6485552223 | 3.6485556683 | +0.0000004460 |
| depth RMSE | 9.4687461707 | 9.4687461711 | +0.0000000004 |
| depth AbsRel | 0.3241713746 | 0.3241717166 | +0.0000003420 |
| normal mean angle | 51.6879735355 | 51.6880043094 | +0.0000307739 |
| normal median angle | 52.2394808006 | 52.2374928512 | -0.0019879494 |

## Verification

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_recovery_model.py
```

Result: PASS.

Additional evaluation:

- recovery model loaded through `Scene(load_iteration=200)`;
- CUDA evaluation ran on GPU 1;
- COLMAP sparse geometry proxy completed successfully.

## Gate

Stage gate: SOFT PASS.

The recovery model is loadable and geometry-proxy stable after copied checkpoint cleanup. The metric deltas are effectively neutral, so this should be treated as a safe structural recovery milestone rather than an improvement claim.

The next step, before any full training run, is a short render-metric evaluation or a small resume run from the recovery model to verify that the copied checkpoint remains trainable and does not degrade photometric metrics.
