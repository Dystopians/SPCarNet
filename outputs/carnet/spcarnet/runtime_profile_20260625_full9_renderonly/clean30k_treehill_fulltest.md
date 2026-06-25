# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_treehill_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/treehill`
- split: `test`
- iteration: `30000`
- views: `18`
- repeats: `3`
- mean elapsed sec: `0.496234`
- mean ms/view: `27.568567`
- mean FPS: `36.273732`
- peak allocated MiB max: `7109.639`
- peak reserved MiB max: `7442.000`
- triangles: `9527637`
- vertices: `3607676`
- checkpoint bytes: `979063711`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.498681 | 27.704486 | 36.095237 | 7109.639 | 7442.000 |
| 2 | 0.496003 | 27.555727 | 36.290097 | 7109.639 | 7442.000 |
| 3 | 0.494019 | 27.445488 | 36.435862 | 7109.639 | 7442.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
