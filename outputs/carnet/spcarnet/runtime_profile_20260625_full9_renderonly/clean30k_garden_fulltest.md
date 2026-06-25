# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_garden_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/garden`
- split: `test`
- iteration: `30000`
- views: `24`
- repeats: `3`
- mean elapsed sec: `0.772073`
- mean ms/view: `32.169712`
- mean FPS: `31.085143`
- peak allocated MiB max: `8274.193`
- peak reserved MiB max: `13520.000`
- triangles: `11568056`
- vertices: `3414016`
- checkpoint bytes: `987752543`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.771713 | 32.154716 | 31.099637 | 8274.193 | 13520.000 |
| 2 | 0.772291 | 32.178802 | 31.076359 | 8274.193 | 13520.000 |
| 3 | 0.772215 | 32.175619 | 31.079433 | 8274.193 | 13520.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
