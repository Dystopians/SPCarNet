# Final Stage SCE13 Certificate-Carrying Edit Planner Design

Date: 2026-05-06

Decision: `SCE13_IMPLEMENTED_PENDING_REAL_PILOT`

## Goal

SCE13 converts Evidence Conflict Graph clusters into local mesh-surgery actions that carry explicit certificates. This makes SCE more than a rollback regularizer: it becomes a decision system over rollback, appearance-only repair, snap, split, fill, delete, and reject.

## Actions

- `ROLLBACK_ONLY`
- `APPEARANCE_ONLY`
- `SNAP_LOCAL`
- `SPLIT_ALLOCATE`
- `FILL_PATCH_LOCAL`
- `DELETE_OR_COLLAPSE`
- `REJECT_UNOBSERVED`

## Rules

- Sparse-depth certificate violations choose rollback or snap.
- Supported render debt chooses split/fill.
- Low-support redundant clusters choose delete/collapse.
- High prior-only risk chooses reject.
- Appearance changes are allowed when render can improve while geometry stays frozen.

Every plan row includes required certificates, expected risk, topology-touching status, headline eligibility, and recommended recovery flags.

