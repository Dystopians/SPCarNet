# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_room_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `39`
- repeats: `3`
- mean elapsed sec: `1.387738`
- mean ms/view: `35.583034`
- mean FPS: `28.103543`
- peak allocated MiB max: `12227.895`
- peak reserved MiB max: `19676.000`
- triangles: `10938652`
- vertices: `2777389`
- checkpoint bytes: `840228127`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.392568 | 35.706876 | 28.005811 | 12227.895 | 19676.000 |
| 2 | 1.388400 | 35.599997 | 28.089890 | 12227.895 | 19676.000 |
| 3 | 1.382247 | 35.442231 | 28.214928 | 12227.895 | 19676.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
