# Final Stage F6 Strict Recovery Runner Design

Date: 2026-05-04

## Purpose

F6 makes topology-frozen compact recovery explicit and reproducible. The runner prevents the old ambiguity where `--skip_restricted_delaunay` skipped only the delayed Delaunay refresh while the standard prune/densify branch could still mutate topology.

## Runner

```text
scripts/car_model/meshsplatopt_run_strict_compact_recovery.py
```

The runner always emits:

```text
recovery_summary.json
topology_audit.json
exact_train_command.txt
wandb_url.txt
render_command.txt
metrics_command.txt
geometry_command.txt
```

## Enforced Training Flags

Every generated recovery command includes:

```text
--densify_until_iter <load_iteration>
--skip_restricted_delaunay
--freeze_topology_updates
--enable_wandb
```

W&B defaults:

```text
WANDB project: spcarnet_meshprior
W&B group: finalF6_strict_recovery
```

## Presets

| preset | sparse depth | topology mutation | role |
| --- | --- | --- | --- |
| `compact_render_only` | off | forbidden | R53/R48 clean-to-compact reproduction |
| `compact_sparse_low_lambda` | low lambda, mixed low-error sampling | forbidden | cross-scene compact recovery with sparse geometry support |
| `compact_sparse_decay` | low lambda plus decay window | forbidden | long-horizon sparse branch |

## Topology Audit

The runner loads the checkpoint at `load_iteration` and `final_iteration`, counts `_triangle_indices` and `triangles_points`, and records whether topology is unchanged. Missing final checkpoints are allowed only with `--allow_missing_final`; otherwise the runner exits non-zero.

## Boundary

The runner can write exact commands without executing them. Use `--execute` only when a low-memory GPU is available and W&B online logging has been confirmed.
