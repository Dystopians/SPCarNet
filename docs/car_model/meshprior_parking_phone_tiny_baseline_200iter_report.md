# MeshPrior Parking Phone Tiny Baseline 200-Iter Report

Date: 2026-05-01

## Command

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model --images images --eval --iterations 200 --test_iterations 50 100 200 --save_iterations 200 --checkpoint_iterations 200 --resolution 4 --enable_wandb --wandb_project spcarnet_meshprior --wandb_group parking_phone_tiny_baseline --wandb_name parking_phone_tiny_gpu1_200iter_baseline --wandb_scalar_log_interval 10 --wandb_disable_fixed_views --scene_name parking_phone_tiny_baseline
```

GPU: `1`

Wandb run:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/icjop1fq
```

## Training Result

Output:

```text
outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model
```

Iteration 200 test metrics:

- L1: `0.221637987566215`
- PSNR: `11.576681349012587`
- SSIM: `0.3399546378188663`
- LPIPS: `0.6316130017792737`
- FPS: `374.0412913994465`

Iteration 200 train metrics:

- L1: `0.22219144999980928`
- PSNR: `11.613157081604005`
- SSIM: `0.3674985408782959`
- LPIPS: `0.5895194172859192`
- FPS: `366.0750941379322`

Mesh summary:

- triangles: `64497`
- vertices: `193491`
- rendered triangles: `11048`

Final cleanup:

- `final_cleanup_enabled=false`
- `final_cleanup_pruned=0`
- triangles preserved: `64497 -> 64497`
- vertices preserved: `193491 -> 193491`

## COLMAP Sparse Geometry Eval

Command:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model --images images --eval --iteration 200 --max_points_per_view 500 --output outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/geometry_eval_colmap/iter_200.json
```

Result:

- test views: `54`
- evaluated views: `54`
- depth points: `21910`
- depth MAE: `3.6485552222775537`
- depth RMSE: `9.46874617071305`
- depth AbsRel: `0.32417137460470213`
- depth delta@1.25: `0.4845732542218165`
- normal mean angle: `51.68797353552561`
- normal median angle: `52.23948080060842`

## Interpretation

This is a successful short baseline smoke for the parking scene:

- data view is valid;
- wandb online logging works;
- training completes;
- final cleanup is safe;
- render metrics and sparse geometry metrics are available.

It is not a headline-quality baseline. The next step should be either a longer baseline run or vehicle/ground-aware region mining plus a gated MeshPrior recovery smoke against this baseline.
