# MeshPrior Scene Application Loop Design

| Field | Value |
|---|---|
| Stage | Pre-M16 / real-scene application bridge |
| Date | 2026-05-01 |
| Status | DESIGN |

## Motivation

M14 concluded `MORE_SCENE_EVIDENCE_REQUIRED`. M15 showed retrieval-deformation should remain a baseline, not a pivot. The next blocking capability is a safe scene application loop: accepted proposals must be applied to a mesh copy with rollback, then recovery/evaluation can run.

## Safety Contract

- Never overwrite the source scene model.
- Apply only proposals accepted by a gate report.
- Save rollback NPZ before each accepted proposal.
- Write a manifest with initial/final mesh stats.
- Generate recovery/evaluation commands for human review before full training.

## Scope

This bridge operates on NPZ mesh proposal artifacts from the current MeshPrior pipeline. It does not yet patch mesh-splatting checkpoints or PRISM internals. That is intentional: the first milestone is a reproducible, auditable applied mesh copy.

## Outputs

- `application_manifest.json`
- `application_report.md`
- `applied_mesh.npz`
- per-proposal rollback files
- optional `recovery_commands.sh`

## Gate

The bridge passes when the synthetic M10/M11 proposal pipeline can produce an accepted proposal, apply it to a copy, save rollback, and emit a recovery command plan.
