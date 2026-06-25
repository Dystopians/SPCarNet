# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_bicycle_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/bicycle`
- split: `test`
- iteration: `30000`
- views: `25`
- repeats: `3`
- mean elapsed sec: `0.663363`
- mean ms/view: `26.534529`
- mean FPS: `37.687185`
- peak allocated MiB max: `7586.699`
- peak reserved MiB max: `7640.000`
- triangles: `9422930`
- vertices: `3490855`
- checkpoint bytes: `952251999`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.665056 | 26.602226 | 37.590839 | 7586.699 | 7640.000 |
| 2 | 0.660158 | 26.406337 | 37.869698 | 7586.699 | 7640.000 |
| 3 | 0.664876 | 26.595024 | 37.601019 | 7586.699 | 7640.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
