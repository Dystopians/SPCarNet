# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_flowers_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/flowers`
- split: `test`
- iteration: `30000`
- views: `22`
- repeats: `3`
- mean elapsed sec: `0.644251`
- mean ms/view: `29.284125`
- mean FPS: `34.158053`
- peak allocated MiB max: `7691.826`
- peak reserved MiB max: `7742.000`
- triangles: `9649601`
- vertices: `3605171`
- checkpoint bytes: `981469855`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.644856 | 29.311627 | 34.116155 | 7691.826 | 7742.000 |
| 2 | 0.657332 | 29.878749 | 33.468604 | 7691.826 | 7742.000 |
| 3 | 0.630564 | 28.661999 | 34.889401 | 7691.826 | 7742.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
