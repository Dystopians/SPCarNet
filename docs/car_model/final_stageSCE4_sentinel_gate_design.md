# Final Stage SCE4 - Sentinel Parent-Pareto Gate Design

Date: 2026-05-06

## Goal

SCE4 adds a fast parent-vs-candidate gate on a SCE2 sentinel cache. It evaluates sparse-depth non-regression before accepting or launching expensive recovery/evaluation runs.

## Checks

The gate reports:

1. Candidate mean sentinel AbsRel <= parent mean sentinel AbsRel + tolerance.
2. Candidate mean sentinel Depth MAE <= parent mean sentinel Depth MAE + tolerance.
3. Worst-view regression count <= threshold.
4. Top cluster regression <= threshold unless below cluster-size threshold.

Test-split gates are report-only. Train/calibration gates may be used as pre-run diagnostics.

## Outputs

- `sentinel_parent_pareto_gate.json`
- `sentinel_per_view_summary.csv`
- `sentinel_cluster_summary.csv`
- `sentinel_gate_report.md`

