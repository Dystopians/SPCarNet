# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `clean30k_room_fulltest`
- model path: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/room`
- split: `test`
- iteration: `30000`
- views: `39`
- repeats: `3`
- mean elapsed sec: `1.269692`
- mean ms/view: `32.556192`
- mean FPS: `30.716288`
- peak allocated MiB max: `12331.073`
- peak reserved MiB max: `19892.000`
- triangles: `11173063`
- vertices: `2840131`
- checkpoint bytes: `858904543`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.266806 | 32.482195 | 30.786096 | 12331.073 | 19892.000 |
| 2 | 1.268524 | 32.526257 | 30.744392 | 12331.073 | 19892.000 |
| 3 | 1.273745 | 32.660124 | 30.618377 | 12331.073 | 19892.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
