# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_stump_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/stump/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `16`
- repeats: `3`
- mean elapsed sec: `0.460674`
- mean ms/view: `28.792142`
- mean FPS: `34.732597`
- peak allocated MiB max: `6687.668`
- peak reserved MiB max: `6742.000`
- triangles: `8180134`
- vertices: `3383973`
- checkpoint bytes: `900193311`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.463985 | 28.999052 | 34.483886 | 6687.668 | 6742.000 |
| 2 | 0.459222 | 28.701396 | 34.841511 | 6687.668 | 6742.000 |
| 3 | 0.458816 | 28.675978 | 34.872394 | 6687.668 | 6742.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
