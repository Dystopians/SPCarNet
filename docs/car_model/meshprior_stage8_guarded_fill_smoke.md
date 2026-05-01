# MeshPrior Stage 8 Guarded Fill — Smoke Report

| Field | Value |
|---|---|
| Stage | M8 / guarded fill smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Commands

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage8_fill.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_synthetic_damage_benchmark.py --output_dir /tmp/meshprior_stage8_fill_benchmark --damage_types local_hole --methods damaged_input guarded_fill snap_fill
```

## Smoke Output

```text
accepted: true
added_vertex_count: 1
added_face_count: 4
boundary_edge_count_before: 4
boundary_edge_count_after: 0
boundary_edge_delta: 4
component_count_before: 1
component_count_after: 1
component_count_delta: 0
free_space_violation_delta: 0
```

The smoke verified:

- a controlled local hole is detected as one closed boundary loop;
- the guarded patch closes the hole;
- no disconnected component is introduced;
- free-space violation does not increase.

## Synthetic Benchmark Output

Small benchmark:

```text
rows: 3
```

| Method | Boundary edges | Fill accepted | Added faces | Component delta | Free-space violation |
|---|---:|---:|---:|---:|---:|
| `damaged_input` | `4` | `0` | `0` | `0` | `0` |
| `guarded_fill` | `0` | `1` | `4` | `0` | `0` |
| `snap_fill` | `0` | `1` | `4` | `0` | `0` |

## Gate Verdict

`PASS`
