# MeshPrior Stage 7 Conservative Snap — Smoke Report

| Field | Value |
|---|---|
| Stage | M7 / conservative snap smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Commands

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage7_snap.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_synthetic_damage_benchmark.py --output_dir /tmp/meshprior_stage7_synthetic_benchmark --damage_types vertex_noise floater --methods protect_prune_only protect_prune_snap
```

## Sphere Smoke Output

```text
mean_displacement: 0.01666666567325592
max_displacement: 0.019999999552965164
moved_vertex_fraction: 0.8333333333333334
surface_distance_before_mean: 0.10000002384185791
surface_distance_after_mean: 0.08333337306976318
surface_distance_delta_mean: 0.016666650772094727
free_space_violation_delta: 0.0
```

The smoke verified:

- snap reduces distance to the synthetic sphere surface;
- max displacement stays under `0.02`;
- protected vertex displacement is zero;
- free-space violation does not increase.

## Synthetic Benchmark Output

Small benchmark:

```text
rows: 4
```

Key rows:

| Method | Damage | Surface-distance delta | Valid protect recall | Floater recall |
|---|---:|---:|---:|---:|
| `protect_prune_only` | `vertex_noise` | `0.0` | `0.9166666666666666` | `0.0` |
| `protect_prune_snap` | `vertex_noise` | `0.01073157787322998` | `0.9166666666666666` | `0.0` |
| `protect_prune_only` | `floater` | `0.0` | `1.0` | `1.0` |
| `protect_prune_snap` | `floater` | `0.0` | `1.0` | `1.0` |

The initial benchmark trial with `snap_max_disp=0.02` failed the preservation tolerance, so benchmark snap was tightened to `0.005`.

## Gate Verdict

`PASS`
