# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_flowers_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `22`
- repeats: `3`
- mean elapsed sec: `0.655206`
- mean ms/view: `29.782082`
- mean FPS: `33.594420`
- peak allocated MiB max: `7389.701`
- peak reserved MiB max: `7452.000`
- triangles: `8509358`
- vertices: `3414899`
- checkpoint bytes: `914527263`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.675658 | 30.711710 | 32.560870 | 7389.701 | 7452.000 |
| 2 | 0.649402 | 29.518257 | 33.877340 | 7389.701 | 7452.000 |
| 3 | 0.640558 | 29.116278 | 34.345049 | 7389.701 | 7452.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
