# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_kitchen_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/kitchen/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `35`
- repeats: `3`
- mean elapsed sec: `1.452754`
- mean ms/view: `41.507263`
- mean FPS: `24.092243`
- peak allocated MiB max: `11712.974`
- peak reserved MiB max: `18312.000`
- triangles: `9512393`
- vertices: `2391146`
- checkpoint bytes: `725659423`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.456322 | 41.609201 | 24.033146 | 11712.974 | 18312.000 |
| 2 | 1.451357 | 41.467356 | 24.115355 | 11712.974 | 18312.000 |
| 3 | 1.450583 | 41.445232 | 24.128228 | 11712.974 | 18312.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
