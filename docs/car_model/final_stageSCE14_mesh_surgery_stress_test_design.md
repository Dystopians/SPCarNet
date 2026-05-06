# Final Stage SCE14 Mesh Surgery Stress Test Design

Date: 2026-05-06

Decision: `SCE14_SYNTHETIC_BENCHMARK_IMPLEMENTED`

## Goal

SCE14 creates a controlled mesh-surgery stress-test benchmark so the method is evaluated as a repair system, not only as a compact-recovery metric table.

## Defect Families

- `FLOATER_INSERTION`
- `SUPPORTED_SURFACE_DELETE`
- `DENT_DEFORM`
- `ROUGH_SURFACE_NOISE`
- `BOUNDARY_HOLE`
- `GROUND_VOID`
- `APPEARANCE_GHOST`
- `OVERCOMPACT_CLUSTER`

Each synthetic defect records touched vertices/faces, topology deltas, reversibility, and required certificates.

## Split Discipline

The synthetic manifest stores `split` and `no_test_leakage`. Train/calibration manifests can drive policy; test manifests are for final audit.

## Interfaces

- `ss3dm_prior/meshsplatopt/stress_test_defects.py`
- `scripts/car_model/meshsplatopt_make_stress_test_defects.py`
- `scripts/car_model/meshsplatopt_run_stress_test_suite.py`
- `scripts/car_model/meshsplatopt_collect_stress_test_results.py`
- `scripts/car_model/smoke_test_stageSCE14_stress_test_defects.py`

## Gate

The synthetic gate passes when all eight defect generators are reversible and `sce_certificate_planner` repairs at least five defect families without false repair on unsupported voids.

