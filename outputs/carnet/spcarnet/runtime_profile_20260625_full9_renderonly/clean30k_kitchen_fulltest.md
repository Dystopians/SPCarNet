# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_kitchen_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/kitchen`
- split: `test`
- iteration: `30000`
- views: `35`
- repeats: `3`
- mean elapsed sec: `1.338025`
- mean ms/view: `38.229285`
- mean FPS: `26.158549`
- peak allocated MiB max: `11772.945`
- peak reserved MiB max: `16238.000`
- triangles: `9716239`
- vertices: `2451717`
- checkpoint bytes: `743150879`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.333473 | 38.099221 | 26.247256 | 11772.945 | 16238.000 |
| 2 | 1.347044 | 38.486979 | 25.982814 | 11772.945 | 16238.000 |
| 3 | 1.333558 | 38.101656 | 26.245579 | 11772.945 | 16238.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
