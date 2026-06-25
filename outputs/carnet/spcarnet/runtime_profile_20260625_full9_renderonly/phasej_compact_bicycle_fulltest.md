# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_bicycle_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `25`
- repeats: `3`
- mean elapsed sec: `0.691921`
- mean ms/view: `27.676855`
- mean FPS: `36.131664`
- peak allocated MiB max: `7308.020`
- peak reserved MiB max: `7360.000`
- triangles: `8309749`
- vertices: `3318902`
- checkpoint bytes: `889769311`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.694855 | 27.794208 | 35.978719 | 7308.020 | 7360.000 |
| 2 | 0.691606 | 27.664237 | 36.147753 | 7308.020 | 7360.000 |
| 3 | 0.689303 | 27.572121 | 36.268519 | 7308.020 | 7360.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
