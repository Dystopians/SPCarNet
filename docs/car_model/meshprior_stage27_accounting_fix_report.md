# Stage27 Accounting Fix Report

Date: 2026-05-02

## Summary

M27.0 accounting fix is a `PASS`.

M26 exposed a topology accounting mismatch: W&B `mesh/triangle_count` was logged before the end-of-iteration standard prune/densify block, while final checkpoints and `final_cleanup_summary.json` reflected the post-mutation state. This made W&B runtime topology and checkpoint topology disagree at iterations that trigger standard topology mutation.

The training loop now logs post-topology and final-checkpoint topology explicitly, so future W&B summaries can be compared directly to checkpoint summaries.

## Code Change

File: `train.py`

1. After standard prune/densify mutates topology, the loop now logs:
   - `mesh/triangle_count`
   - `mesh/vertex_count`
   - `mesh/pre_topology_triangle_count`
   - `mesh/pre_topology_vertex_count`
   - `mesh/post_topology_triangle_count`
   - `mesh/post_topology_vertex_count`
   - `prism/standard_topology_mutation`

2. At final checkpoint/final-cleanup bookkeeping, W&B now logs:
   - `mesh/final_checkpoint_triangle_count`
   - `mesh/final_checkpoint_vertex_count`
   - `mesh/triangle_count`
   - `mesh/vertex_count`
   - `prism/final_pre_cleanup_triangle_count`
   - `prism/final_post_cleanup_triangle_count`
   - `prism/final_pre_cleanup_vertex_count`
   - `prism/final_post_cleanup_vertex_count`

## Smoke Run

- output: `outputs/carnet/meshprior/stage27_accounting/eth3d_courtyard_accounting_smoke_520iter/`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/i6lfgt66`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- dataset: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`
- budget: `520` iterations

Command log:

- `outputs/carnet/meshprior/stage27_accounting/eth3d_courtyard_accounting_smoke_520iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage27_accounting/eth3d_courtyard_accounting_smoke_520iter/logs/train.log`

## Verification

Local W&B summary:

```text
mesh/triangle_count 33487
mesh/vertex_count 100461
mesh/final_checkpoint_triangle_count 33487
mesh/final_checkpoint_vertex_count 100461
```

Final cleanup summary:

```text
pre_prune_triangle_count 33487
post_prune_triangle_count 33487
pre_prune_vertex_count 100461
post_prune_vertex_count 100461
```

The accounting paths agree on the final checkpoint topology.

## Decision

`PASS`. Future cross-scene tables should use the new final-checkpoint W&B keys when available. M26 historical rows remain valid as completed experiments, but their topology mismatch should stay documented because those runs were produced before this fix.
