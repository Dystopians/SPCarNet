# MeshSplatOpt Stage R15 full-budget sweep design

Date: 2026-05-03

## Purpose

R15 turns the strongest validated result, R53/R55 clean-to-compact recovery, into a reproducible full-budget interface instead of a hand-run experiment. The interface writes explicit train, render, metrics, and sparse-geometry commands with W&B enabled.

## Implemented interface

- `scripts/car_model/meshsplatopt_run_full_budget_sweep.py`
- output manifest: `outputs/carnet/meshsplatopt/full_budget_sweep/full_budget_jobs.json`
- optional shell runner: `outputs/carnet/meshsplatopt/full_budget_sweep/run_full_budget_jobs.sh`

Each job records:

- scene and method;
- source dataset path;
- compacted model path;
- recovery output path;
- load and final iterations;
- W&B project/group/name;
- strict topology freeze flags;
- independent render, metrics, and geometry commands.

## Current jobs

The current manifest includes the validated parking jobs:

| job | method | role |
|---|---|---|
| `R53_full_parking_prune70_22000to26000` | clean-to-compact prune70 | headline quality-dominating row |
| `R55_full_parking_prune65_22000to26000` | clean-to-compact prune65 | LPIPS/normal Pareto row |

## Gate status

`PARTIAL_PASS_INTERFACE_READY`. The interface is in place and the parking rows are independently validated. The remaining R15 requirement is to add at least two more geometry-observable scenes with clean-long checkpoints and matching compaction/recovery jobs.

