# MeshPrior Stage 8 Guarded Fill — Implementation Report

| Field | Value |
|---|---|
| Stage | M8 / guarded fill |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage8_guarded_fill_design.md` |

## 1. Files Added or Updated

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/fill.py` | Boundary-loop detection, guarded fill proposal construction, local field sampling, and risk evaluation. |
| `scripts/car_model/meshprior_make_fill_proposals.py` | CLI that writes fill proposal arrays and a summary JSON. |
| `scripts/car_model/smoke_test_meshprior_stage8_fill.py` | Controlled synthetic local-hole smoke test. |
| `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py` | Adds `damaged_input`, `guarded_fill`, and `snap_fill` benchmark methods. |
| `docs/car_model/meshprior_stage8_guarded_fill_design.md` | Stage design. |

## 2. Implementation Summary

M8 implements guarded local fill proposals. The implementation intentionally stays proposal-only.

Implemented functions:

- `find_boundary_loops(mesh)`;
- `score_hole_candidates(mesh, boundary_loops, region_evidence)`;
- `extract_local_field_patch(decoder, z, local_bbox, resolution)`;
- `clip_patch_to_hole_boundary(...)`;
- `build_fill_proposal(...)`;
- `evaluate_fill_risk(...)`.

The first patch generator uses a boundary fan cap. It adds one centroid vertex and one triangle per boundary-loop edge. This closes controlled holes without introducing far-away disconnected geometry.

## 3. Evidence Gates

A fill proposal is rejected when:

- the loop is not closed;
- the loop has fewer than three vertices;
- decoder support is below `min_support`;
- uncertainty is high.

Risk metrics include:

- boundary edge count before/after;
- connected component count before/after;
- added vertex/face counts;
- free-space violation delta when a free-space function is supplied.

## 4. Benchmark Integration

The synthetic benchmark now supports:

```text
--methods damaged_input guarded_fill snap_fill
```

For M8, the controlled benchmark uses `local_hole`.

`snap_fill` is included for interface completeness. In the current local-hole box case, snap moves no vertices because boundary vertices are fixed by default, so `snap_fill` and `guarded_fill` behave identically.

## 5. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage8_fill.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_synthetic_damage_benchmark.py --output_dir /tmp/meshprior_stage8_fill_benchmark --damage_types local_hole --methods damaged_input guarded_fill snap_fill
```

Results:

- M8 smoke: PASS.
- M6 benchmark regression: PASS, `rows=8`, floater recall `1.0`.
- M8 local-hole benchmark: PASS, `rows=3`.

Smoke metrics:

```text
added_vertex_count: 1
added_face_count: 4
boundary_edge_count_before: 4
boundary_edge_count_after: 0
component_count_delta: 0
free_space_violation_delta: 0
```

Small benchmark:

- `damaged_input`: `boundary_edge_count=4`, `hole_boundary_score=0.23529411764705882`.
- `guarded_fill`: `boundary_edge_count=0`, `fill_accepted=1`, `fill_boundary_edge_delta=4`.
- `snap_fill`: `boundary_edge_count=0`, `fill_accepted=1`, `fill_boundary_edge_delta=4`.

## 6. Known Limitations

- The first fill patch is a fan cap, not a full local marching-cubes surface.
- It closes controlled synthetic holes but is not yet approved for scene-level hidden-side completion.
- Face count is not intended to match the original mesh triangulation.

## 7. Stage Gate

| Gate | Result |
|---|---|
| Controlled synthetic hole closes | PASS |
| Fill does not create disconnected floaters | PASS |
| Free-space violation does not increase | PASS |
| Benchmark compares damaged input, guarded fill, and snap+fill | PASS |
| Regression does not break M6 benchmark | PASS |

Decision: `PASS`. The next allowed stage is M9 scene evidence gates and rollback.
