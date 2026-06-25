# Render Runtime Profile

This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.

## Summary

- label: `phasej_compact_bonsai_fulltest`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `37`
- repeats: `3`
- mean elapsed sec: `1.343330`
- mean ms/view: `36.306227`
- mean FPS: `27.543498`
- peak allocated MiB max: `11564.704`
- peak reserved MiB max: `16994.000`
- triangles: `9555533`
- vertices: `3295557`
- checkpoint bytes: `914812447`

## Repeats

| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.342145 | 36.274201 | 27.567803 | 11563.779 | 16994.000 |
| 2 | 1.343454 | 36.309555 | 27.540960 | 11564.704 | 16980.000 |
| 3 | 1.344392 | 36.334926 | 27.521730 | 11563.779 | 16980.000 |

## Scope Note

- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.
- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.
- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.
