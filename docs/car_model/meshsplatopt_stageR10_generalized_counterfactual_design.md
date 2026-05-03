# MeshSplatOpt Stage R10 Generalized Counterfactual Design

Date: 2026-05-02

## Goal

Generalize PRISM-style validation from delete-only masks to arbitrary reversible edits.

## R10 Scope

R10 implements a synthetic/mesh-level gate contract:

1. snapshot state;
2. apply edit temporarily;
3. evaluate topology integrity;
4. evaluate free-space/risk metadata;
5. evaluate edit-specific safety rules;
6. accept or rollback;
7. write an auditable gate report.

Real render, sparse depth, normal, and changed-pixel integrations are represented in the report schema but not faked when no renderable model exists.

## Edit-Specific Rules

- `DELETE_TRIANGLES`: reject if metadata says supported surface would be deleted.
- `SNAP_VERTICES`: reject if free-space risk is too high.
- `FILL_PATCH`: accept only with topology validity, low free-space risk, and no unsupported prior-only headline flag.
- Other edits require topology validity and low risk.

## Gate

`PASS` requires:

1. good fill accepted on synthetic hole;
2. bad floater insertion rejected;
3. snap through free space rejected;
4. delete supported surface rejected;
5. rollback exactly restores state on reject.
