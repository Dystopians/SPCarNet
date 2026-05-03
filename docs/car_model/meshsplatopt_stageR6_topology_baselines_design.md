# MeshSplatOpt Stage R6 Topology Baselines Design

Date: 2026-05-02

## Goal

Build topology-reduction baselines so MeshSplatOpt repair claims cannot hide behind weak pruning comparisons.

## Baselines

Implemented in R6:

- `prism_score_topk_delete`: deletes lowest score faces; area is used as a deterministic proxy if no PRISM score is supplied.
- `random_same_count_delete`: deletes a random same-count face subset.
- `low_visibility_delete`: deletes low proxy-visibility faces, currently small-area faces.
- `boundary_protected_delete`: deletes interior faces first and avoids boundary faces.
- `qem_style_edge_collapse`: greedy shortest-edge collapse approximation.
- `planar_face_merge`: conservative coplanar adjacent face removal approximation.
- `external_simplification`: records missing optional dependency when trimesh/pymeshlab simplification is unavailable.

## Budgets

Supported target face fractions:

- `0.90`
- `0.75`
- `0.50`
- `0.25`

## Outputs

- `topology_baseline_runs.json`
- `topology_baseline_table.csv`
- `topology_baseline_report.md`

## Gate

`PASS` requires synthetic execution of delete, boundary-protected delete, and at least one collapse/merge-style baseline with valid meshes and target triangle counts within tolerance.
