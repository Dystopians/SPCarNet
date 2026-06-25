# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_bonsai_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/bonsai`
- split: `test`
- iteration: `30000`
- views: `37`
- repeats: `3`
- mean elapsed sec: `1.264368`
- mean ms/view: `34.172113`
- mean FPS: `29.263657`
- peak allocated MiB max: `11923.766`
- peak reserved MiB max: `18980.000`
- triangles: `10834182`
- vertices: `3409579`
- checkpoint bytes: `969216671`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.264275 | 34.169589 | 29.265789 | 11923.766 | 18980.000 |
| 2 | 1.262866 | 34.131508 | 29.298441 | 11922.103 | 18966.000 |
| 3 | 1.265964 | 34.215241 | 29.226742 | 11923.766 | 18980.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
