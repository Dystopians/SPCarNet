# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_counter_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/counter`
- split: `test`
- iteration: `30000`
- views: `30`
- repeats: `3`
- mean elapsed sec: `1.247449`
- mean ms/view: `41.581638`
- mean FPS: `24.049097`
- peak allocated MiB max: `12098.717`
- peak reserved MiB max: `15152.000`
- triangles: `9850919`
- vertices: `2537250`
- checkpoint bytes: `764173855`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.248183 | 41.606111 | 24.034931 | 12098.717 | 15152.000 |
| 2 | 1.245826 | 41.527528 | 24.080412 | 12098.717 | 15152.000 |
| 3 | 1.248338 | 41.611276 | 24.031948 | 12098.717 | 15152.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
