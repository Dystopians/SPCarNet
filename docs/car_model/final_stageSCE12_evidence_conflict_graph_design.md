# Final Stage SCE12 Evidence Conflict Graph Design

Date: 2026-05-06

Decision: `SCE12_IMPLEMENTED_PENDING_REAL_AUDIT`

## Goal

SCE12 turns sparse sentinel rollback from a single loss term into a paper-facing Evidence Conflict Graph. The graph links views, COLMAP sparse points, rendered pixel samples, approximate mesh clusters, certificates, and edit actions.

## Graph Semantics

Node types:

- `view_node`
- `sparse_point_node`
- `pixel_sample_node`
- `mesh_cluster_node`
- `certificate_node`
- `edit_action_node`

Edges:

- view observes sparse point
- sparse point projects to pixel sample
- pixel sample is approximately explained by a mesh cluster
- certificate constrains pixel/sparse point
- edit action targets cluster

## Conflict Score

For each correspondence:

```text
depth_conflict = max(0, candidate_abs_error - parent_abs_error - margin_abs)
absrel_conflict = max(0, candidate_absrel - parent_absrel - margin_rel)
render_gain = max(0, parent_rgb_residual - candidate_rgb_residual)
certificate_pressure =
  depth_conflict
  + 5 * absrel_conflict
  + 2 * candidate_invalid
  + 0.25 * render_gain if render gain conflicts with depth
```

Train/calibration ECGs may drive policy. Test ECGs are audit-only and must not be used for training sentinel selection.

## Outputs

- `evidence_conflict_graph.json`
- `evidence_conflict_graph.npz`
- `ecg_cluster_summary.csv`
- `ecg_report.md`

