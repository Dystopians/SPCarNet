# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_stump_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/stump`
- split: `test`
- iteration: `30000`
- views: `16`
- repeats: `3`
- mean elapsed sec: `0.441225`
- mean ms/view: `27.576581`
- mean FPS: `36.266064`
- peak allocated MiB max: `6957.935`
- peak reserved MiB max: `7018.000`
- triangles: `9277087`
- vertices: `3558228`
- checkpoint bytes: `962765407`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.447290 | 27.955628 | 35.770972 | 6957.935 | 7018.000 |
| 2 | 0.437943 | 27.371438 | 36.534434 | 6957.935 | 7018.000 |
| 3 | 0.438443 | 27.402677 | 36.492786 | 6957.935 | 7018.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
