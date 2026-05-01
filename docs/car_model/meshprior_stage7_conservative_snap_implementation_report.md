# MeshPrior Stage 7 Conservative Snap — Implementation Report

| Field | Value |
|---|---|
| Stage | M7 / conservative snap |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage7_conservative_snap_design.md` |

## 1. Files Added or Updated

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/snap.py` | Conservative vertex snap proposal utilities, risk evaluation, and acceptance gate. |
| `scripts/car_model/meshprior_make_snap_proposals.py` | CLI that writes snap proposal arrays and a risk summary. |
| `scripts/car_model/smoke_test_meshprior_stage7_snap.py` | Synthetic sphere-field smoke test. |
| `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py` | Adds `protect_prune_snap` benchmark method beside `protect_prune_only`. |
| `docs/car_model/meshprior_stage7_conservative_snap_design.md` | Stage design. |

## 2. Implementation Summary

M7 introduces snap proposals as a bounded geometry-movement step. It does not directly mutate scene meshes by default.

Implemented functions:

- `compute_field_gradient(decoder, z, points)`;
- `propose_vertex_snap(...)`;
- `apply_snap_proposal(mesh, proposal)`;
- `evaluate_snap_risk(...)`;
- `accept_snap_proposal(...)`.

The snap direction follows the negative gradient of:

```text
(sigmoid(field(x; z)) - iso_level)^2
```

Every displacement is clipped to `max_disp`. Boundary vertices are fixed by default, protected vertices are fixed, high uncertainty disables movement, and high observed-support vertices are fixed when observed support is supplied.

## 3. Benchmark Integration

The Stage-6 synthetic benchmark now accepts:

```text
--methods protect_prune_only protect_prune_snap
--snap_max_disp 0.005
```

The benchmark uses:

- analytic box occupancy for snap direction;
- analytic box surface support for protect/prune scoring.

An initial `snap_max_disp=0.02` trial improved vertex-noise surface distance but reduced valid-surface protect recall from `0.9167` to `0.8333`, exceeding the 5 percent preservation gate. The benchmark default was therefore tightened to `0.005`.

## 4. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage7_snap.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_synthetic_damage_benchmark.py --output_dir /tmp/meshprior_stage7_synthetic_benchmark --damage_types vertex_noise floater --methods protect_prune_only protect_prune_snap
```

Results:

- M7 sphere smoke: PASS.
- M6 benchmark regression: PASS, `rows=8`, floater recall `1.0`.
- M7 small benchmark: PASS, `rows=4`.
- `vertex_noise` snap surface-distance improvement: `0.01073157787322998`.
- `vertex_noise` valid-surface protect recall remained `0.9166666666666666`.
- `floater` prune recall remained `1.0`.

## 5. Stage Gate

| Gate | Result |
|---|---|
| Snap improves synthetic distance-to-surface | PASS |
| Boundary/protected vertices stay fixed in smoke | PASS |
| Displacement is clipped to max displacement | PASS |
| Free-space violation does not increase in smoke/benchmark | PASS |
| Synthetic benchmark compares `protect_prune_only` and `protect_prune_snap` | PASS |
| Visible preservation drop is not above 5 percent after final gate | PASS |

Decision: `PASS`. The next allowed stage is M8 guarded patch/fill proposals.
