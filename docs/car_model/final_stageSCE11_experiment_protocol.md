# Final Stage SCE11 Experiment Protocol

Date: 2026-05-06

## Split Rules

- Train/calibration sentinels may drive loss and policy.
- Test sentinels and test ECG are audit-only.
- Every cache must record `no_test_leakage=true` before use in training.

## Evaluation

- `render.py`
- `metrics.py`
- `evaluate_geometry_colmap.py`
- SCE analyzer/gate/ECG for diagnostic evidence.

## Logging

- W&B online for medium/long runs.
- Store exact command, recovery summary, topology audit, and report docs per stage.

