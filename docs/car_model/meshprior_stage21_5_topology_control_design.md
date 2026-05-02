# MeshPrior Stage 21.5 Topology-Controlled Current-Branch Ablation Design

Date: 2026-05-02

## Goal

Stage 21 showed that the current branch is the best 7000-iteration single-scene row, but it uses about `2.92x` the triangles of clean MeshSplatting. Stage 21.5 tests whether a conservative post-training topology-control pass can keep most of the current-branch quality while reducing triangle count.

## Scope

This stage is a checkpoint-copy ablation. It must not overwrite the source 7000-iteration model.

Source model:

```text
outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model
```

Output root:

```text
outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/
```

## Method

The saved runtime fields `importance_score`, `image_size`, and `pixel_count` are all zero in the current 7000 checkpoint, so the first low-risk ranking signal is world-space triangle area.

Create copied checkpoints that remove the smallest-area triangles and compact unreferenced vertices:

- `prune_25`: remove smallest 25% triangles.
- `prune_50`: remove smallest 50% triangles.
- `prune_66`: remove smallest 66% triangles, approximately matching clean baseline triangle count.

This is not claimed as the final method; it is a diagnostic to determine whether topology inflation is controllable without retraining.

## Evaluation

For each copied model:

- `render.py --iteration 7000 --skip_train`
- `metrics.py`
- `evaluate_geometry_colmap.py`
- topology count from checkpoint
- external W&B summary log

## Gate

`PASS` if at least one ablation reduces triangles by 30% or more while keeping independent render PSNR/SSIM/LPIPS at or above the clean 7000 baseline and without large COLMAP proxy geometry regression.

`SOFT PASS` if the best ablation is a useful quality/topology tradeoff but misses one metric.

`FAIL` if all meaningful pruning levels collapse below clean quality or geometry proxy stability.

