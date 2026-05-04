# Final Stage F3 Cross-Scene Clean-to-Compact Plan

Date: 2026-05-04

## Decision

`PASS`.

The plan names exact clean baselines, missing long-baseline commands, output paths, sweep fractions, and the first run to start. Do not launch broad cross-scene compaction until the missing public-scene clean-long baselines are trained or explicitly demoted to matched-screen evidence.

## Current Baseline Audit

| scene | clean baseline present | best clean path | final iter | triangles | Stage35/PRISM baseline | sparse geometry | images/resolution/split | GPU budget | risk |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| `parking_phone_tiny` | yes, long | `outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model` | 22000 | 8,548,242 | PRISM/parking retained rows exist; no single Stage35 row in final registry | yes, iter 22000/30000 | `images`, res 4, eval split | already complete; compact sweeps 4k recovery each | low |
| `mipnerf360/bonsai` | matched-screen only | `outputs/carnet/meshsplatopt/stageR58_02_bonsai_clean_continue_7000to9000/recovery_model` | 9000 | 2,487,474 | `outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model` | yes, iter 9000 | `images_4`, res 4, llff/eval | train clean 9k->22k first; then 2k-4k recovery per fraction | medium |
| `ETH3D/courtyard` | matched-screen only | `outputs/carnet/meshsplatopt/stageR57_02_courtyard_clean_continue_7000to9000/recovery_model` | 9000 | 410,254 | `outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model` | yes, iter 9000 | `images`, res 8, llff/eval | train clean 9k->22k only after bonsai or selector screen | high |
| `mipnerf360/room` | matched-screen only | `outputs/carnet/meshsplatopt/stageR59_03_room_clean_continue_7000to9000` | 9000 | 196,057 | no Stage35 row in final registry | yes, iter 9000 | `images_4`, res 4, llff/eval | optional additional public scene; train clean 9k->22k after selector | medium |
| `mipnerf360/counter` | matched-screen only | `outputs/carnet/meshsplatopt/stageR60_03_counter_clean_continue_7000to9000` | 9000 | 161,465 | no Stage35 row in final registry | yes, iter 9000 | `images_4`, res 4, llff/eval | optional additional public scene; lower priority because R60 is mixed-negative | high |
| `mipnerf360/flowers` | missing locally | none found under `/data/peilincai/mesh_datasets` | n/a | n/a | none | none | n/a | do not schedule | stop for this scene |

## Missing Clean-Long Commands

All training commands must run with:

```bash
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
nvidia-smi
```

Use the lowest-memory available GPU at launch time.

### First Run To Start: Bonsai Clean Long 9k->22k

Bonsai is the first missing clean-long run because R58 is already an all-metric compact-positive matched screen. A long clean baseline is required before making a stronger public-scene claim.

```bash
rsync -a \
  outputs/carnet/meshsplatopt/stageR58_02_bonsai_clean_continue_7000to9000/recovery_model/ \
  outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000/

CUDA_VISIBLE_DEVICES=<low_memory_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000 \
  --images images_4 --resolution 4 --eval \
  --load_iteration 9000 \
  --iterations 22000 \
  --test_iterations 22000 \
  --save_iterations 22000 \
  --checkpoint_iterations 22000 \
  --enable_wandb \
  --wandb_project spcarnet_meshprior \
  --wandb_group finalF3_clean_long \
  --wandb_name finalF3_bonsai_clean_long_9000to22000 \
  --wandb_image_log_interval 1000 \
  --wandb_scalar_log_interval 50

CUDA_VISIBLE_DEVICES=<low_memory_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000 \
  --images images_4 --resolution 4 --eval --iteration 22000 --skip_train

/home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py \
  -m outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000

CUDA_VISIBLE_DEVICES=<low_memory_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000 \
  --images images_4 --eval --iteration 22000 --max_points_per_view 500 \
  --output outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000/geometry_eval_colmap/iter_22000_max500.json
```

### Courtyard Clean Long 9k->22k

```bash
rsync -a \
  outputs/carnet/meshsplatopt/stageR57_02_courtyard_clean_continue_7000to9000/recovery_model/ \
  outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000/

CUDA_VISIBLE_DEVICES=<low_memory_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py \
  -s /data/peilincai/mesh_datasets/eth3d_colmap/courtyard \
  -m outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000 \
  --images images --resolution 8 --eval \
  --load_iteration 9000 \
  --iterations 22000 \
  --test_iterations 22000 \
  --save_iterations 22000 \
  --checkpoint_iterations 22000 \
  --enable_wandb \
  --wandb_project spcarnet_meshprior \
  --wandb_group finalF3_clean_long \
  --wandb_name finalF3_courtyard_clean_long_9000to22000 \
  --wandb_image_log_interval 1000 \
  --wandb_scalar_log_interval 50
```

### Room Clean Long 9k->22k

```bash
rsync -a \
  outputs/carnet/meshsplatopt/stageR59_03_room_clean_continue_7000to9000/ \
  outputs/carnet/meshsplatopt/finalF3_room_clean_long_9000to22000/

CUDA_VISIBLE_DEVICES=<low_memory_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/room \
  -m outputs/carnet/meshsplatopt/finalF3_room_clean_long_9000to22000 \
  --images images_4 --resolution 4 --eval \
  --load_iteration 9000 \
  --iterations 22000 \
  --test_iterations 22000 \
  --save_iterations 22000 \
  --checkpoint_iterations 22000 \
  --enable_wandb \
  --wandb_project spcarnet_meshprior \
  --wandb_group finalF3_clean_long \
  --wandb_name finalF3_room_clean_long_9000to22000 \
  --wandb_image_log_interval 1000 \
  --wandb_scalar_log_interval 50
```

### Counter Clean Long 9k->22k

```bash
rsync -a \
  outputs/carnet/meshsplatopt/stageR60_03_counter_clean_continue_7000to9000/ \
  outputs/carnet/meshsplatopt/finalF3_counter_clean_long_9000to22000/

CUDA_VISIBLE_DEVICES=<low_memory_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/counter \
  -m outputs/carnet/meshsplatopt/finalF3_counter_clean_long_9000to22000 \
  --images images_4 --resolution 4 --eval \
  --load_iteration 9000 \
  --iterations 22000 \
  --test_iterations 22000 \
  --save_iterations 22000 \
  --checkpoint_iterations 22000 \
  --enable_wandb \
  --wandb_project spcarnet_meshprior \
  --wandb_group finalF3_clean_long \
  --wandb_name finalF3_counter_clean_long_9000to22000 \
  --wandb_image_log_interval 1000 \
  --wandb_scalar_log_interval 50
```

## Compact Sweep Definition

Fractions:

```text
50%, 60%, 65%, 70%, 75%, 80%, 90%
```

For each scene and fraction:

1. Apply an evidence-compatible compact edit to the named clean checkpoint.
2. Render/evaluate immediately after compaction.
3. Run strict topology-frozen recovery.
4. Render/evaluate after recovery.
5. Record W&B URL, command path, checkpoint path, independent metrics path, sparse-geometry path, and topology audit.

Output layout:

```text
outputs/carnet/meshsplatopt/finalF3_compact_sweep/{scene}/prune{fraction}/compact_model
outputs/carnet/meshsplatopt/finalF3_compact_sweep/{scene}/prune{fraction}/post_compact_eval
outputs/carnet/meshsplatopt/finalF3_compact_sweep/{scene}/prune{fraction}/recovery_model
```

Area-only temporary command before F4 selector exists:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshprior_apply_topology_control_ablation.py \
  --source_model <clean_model_path> \
  --source_checkpoint <clean_model_path>/point_cloud/iteration_<clean_iter>/point_cloud_state_dict.pt \
  --output_model outputs/carnet/meshsplatopt/finalF3_compact_sweep/<scene>/prune<frac>/compact_model \
  --iteration <clean_iter> \
  --prune_fraction <frac>
```

After F4, replace this with:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_select_compaction_candidates.py \
  --source_model <clean_model_path> \
  --iteration <clean_iter> \
  --mode csef_low_evidence_boundary_protected \
  --target_prune_fraction <frac> \
  --out-dir outputs/carnet/meshsplatopt/finalF3_compact_sweep/<scene>/prune<frac>/selector
```

Recovery command template:

```bash
CUDA_VISIBLE_DEVICES=<low_memory_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py \
  -s <scene_source> \
  -m outputs/carnet/meshsplatopt/finalF3_compact_sweep/<scene>/prune<frac>/recovery_model \
  --images <images> --resolution <resolution> --eval \
  --load_iteration <clean_iter> \
  --iterations <final_iter> \
  --test_iterations <final_iter> \
  --save_iterations <final_iter> \
  --checkpoint_iterations <final_iter> \
  --densify_until_iter <clean_iter> \
  --skip_restricted_delaunay \
  --freeze_topology_updates \
  --enable_wandb \
  --wandb_project spcarnet_meshprior \
  --wandb_group finalF3_compact_sweep \
  --wandb_name finalF3_<scene>_prune<frac>_<clean_iter>to<final_iter> \
  --wandb_image_log_interval 1000 \
  --wandb_scalar_log_interval 50
```

## Launch Order

1. **Start first:** bonsai clean long 9k->22k, because R58 is the strongest public-scene positive and currently lacks a clean-long comparator.
2. Run F4 selector implementation before broad fraction sweeps; area-only already has known negatives on courtyard/counter.
3. After F4 smoke passes, run a one-scene bonsai selector dry-run at fractions `60, 70, 80` from the new bonsai clean 22k checkpoint.
4. If bonsai passes, run the full seven-fraction bonsai sweep.
5. Then run courtyard selector dry-run at `50, 60, 70` because courtyard is the hardest known negative.
6. Use room as the additional public scene if the selector predicts low risk; keep counter as a stress-test negative.

## Gate

`PASS`.

The plan identifies exactly which clean long baselines are present or missing and names the first run: `finalF3_bonsai_clean_long_9000to22000`. Cross-scene compaction should not start before this clean-long baseline exists and F4's non-area selector has passed.
