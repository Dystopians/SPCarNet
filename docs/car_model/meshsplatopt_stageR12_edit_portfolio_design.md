# MeshSplatOpt Stage R12 Edit Portfolio Design

Date: 2026-05-02

## Goal

Choose among delete, collapse, snap, split, fill, and appearance recovery under budget and safety constraints.

## State Machine

States:

1. `GEOMETRY_ACQUISITION`
2. `DEFECT_MINING`
3. `LOW_RISK_CLEANUP`
4. `SNAP_REPAIR`
5. `GIANT_VOID_REPAIR`
6. `OBJECT_PRIOR_REPAIR`
7. `APPEARANCE_RECOVERY`
8. `TOPOLOGY_RETENTION`
9. `VALIDATION_ROLLBACK`
10. `FINAL_AUDIT`

## Portfolio Score

Candidates are scored by expected CSEF debt reduction per topology/render cost, with penalties for free-space risk, uncertainty, and prior-only flags.

## Gate

`PASS` requires the synthetic state machine to execute at least three edit classes and produce an auditable trace while rejecting a bad prior-only fill in normal mode.
