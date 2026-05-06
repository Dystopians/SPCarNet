# Final Stage SCE9 Local Surgery Report

Date: 2026-05-06

Decision: `SCE9_SYNTHETIC_PASS_REAL_COURTYARD_ROLLBACK_ONLY`

## Synthetic Gate

The smoke test covers:

- dented supported plane -> `SNAP_VERTICES`
- large triangle with depth variation -> `SPLIT_TRIANGLES`
- supported hole -> `FILL_PATCH`
- unknown unobserved void -> `REJECT`
- appearance ghost -> `APPEARANCE_RESET`

Synthetic accepted edits reduce sentinel error in the smoke model.

## Real Courtyard Diagnostic

The current courtyard train/test ECGs identify sparse-depth certificate conflicts but do not provide support for snap/split/fill/delete. The safe action is rollback/protect. This is a useful certificate outcome: the system refuses unsupported topology surgery.

