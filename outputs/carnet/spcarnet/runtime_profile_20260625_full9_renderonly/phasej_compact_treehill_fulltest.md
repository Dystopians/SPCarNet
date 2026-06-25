# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_treehill_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `18`
- repeats: `3`
- mean elapsed sec: `0.518417`
- mean ms/view: `28.800940`
- mean FPS: `34.722090`
- peak allocated MiB max: `6808.792`
- peak reserved MiB max: `7156.000`
- triangles: `8402362`
- vertices: `3419320`
- checkpoint bytes: `912878815`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.521509 | 28.972732 | 34.515212 | 6808.792 | 7156.000 |
| 2 | 0.514763 | 28.597969 | 34.967518 | 6808.792 | 7156.000 |
| 3 | 0.518978 | 28.832120 | 34.683540 | 6808.792 | 7156.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
