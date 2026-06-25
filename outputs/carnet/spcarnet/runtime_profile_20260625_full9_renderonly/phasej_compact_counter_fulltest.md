# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_counter_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `30`
- repeats: `3`
- mean elapsed sec: `1.315265`
- mean ms/view: `43.842166`
- mean FPS: `22.809441`
- peak allocated MiB max: `11876.499`
- peak reserved MiB max: `14828.000`
- triangles: `9644247`
- vertices: `2478825`
- checkpoint bytes: `747061087`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.308456 | 43.615196 | 22.927789 | 11876.499 | 14828.000 |
| 2 | 1.316470 | 43.882318 | 22.788222 | 11876.499 | 14828.000 |
| 3 | 1.320870 | 44.028986 | 22.712311 | 11876.499 | 14828.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
