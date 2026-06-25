# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_garden_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `24`
- repeats: `3`
- mean elapsed sec: `0.828922`
- mean ms/view: `34.538410`
- mean FPS: `28.953293`
- peak allocated MiB max: `8170.849`
- peak reserved MiB max: `13508.000`
- triangles: `11166587`
- vertices: `3315236`
- checkpoint bytes: `957570719`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.828971 | 34.540451 | 28.951562 | 8170.849 | 13508.000 |
| 2 | 0.829754 | 34.573104 | 28.924218 | 8170.849 | 13508.000 |
| 3 | 0.828040 | 34.501675 | 28.984100 | 8170.849 | 13508.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
