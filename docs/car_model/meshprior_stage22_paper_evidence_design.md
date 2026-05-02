# MeshPrior Stage 22 Paper Evidence Design

Date: 2026-05-02

## Goal

Collect the scattered MeshPrior evidence into one reproducible paper-evidence package without hiding missing experiments or mixing metric classes.

## Inputs

- M13 object/synthetic/scene evidence.
- M17-M18 2000-iteration MeshPrior and topology-budget diagnostics.
- M21 7000-iteration clean/current/Stage17 diagnostic.
- M21.5 topology-control ablation.
- Parking patch proposal gate and rollback reports.

## Metric Classes

- `object_prior`: SP-CarNet posterior quality on object-level validation data.
- `synthetic_damage`: inference-time synthetic repair diagnostics.
- `scene_render_geometry_topology`: render metrics, COLMAP proxy geometry, topology, W&B, and claim role.
- `proposal_gate_rollback`: accept/reject/rollback safety evidence.
- `failure_cases`: negative rows that must remain visible.
- `missing_rows`: required but unavailable paper rows.

## Gate

`PASS` if all available evidence is regenerated from local artifacts, metric classes are separated, and missing rows remain visible.

`SOFT PASS` if the package is reproducible but important rows remain `MISSING`.

`FAIL` if required scene rows are missing or headline tables can silently omit failures.
