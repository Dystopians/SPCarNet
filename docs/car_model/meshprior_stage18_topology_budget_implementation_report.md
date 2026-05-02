# MeshPrior Stage 18 Topology Budget Implementation Report

Date: 2026-05-01

## Scope

Stage 18 adds a reproducible topology-budget comparison for the three 2000-iteration `parking_phone_tiny` runs:

- clean Mesh Splatting candidate: `origin_main_2000iter`
- current branch engineering baseline: `current_branch_2000iter`
- real MeshPrior variant: `stage17_meshprior_2000iter`

The purpose is to prevent quality-only claims from hiding triangle-count inflation.

## Files Added

- `scripts/car_model/meshprior_collect_topology_budget_comparison.py`
- `scripts/car_model/smoke_test_meshprior_topology_budget_comparison.py`
- `docs/car_model/meshprior_stage18_topology_budget_design.md`
- `docs/car_model/meshprior_stage18_topology_budget_implementation_report.md`

## Outputs

- `outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison/topology_budget_comparison.json`
- `outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison/topology_budget_comparison.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison/topology_budget_comparison.md`

## Commands

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_collect_topology_budget_comparison.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_topology_budget_comparison.py
```

Result: `PASS`.

## Main Table

| label | triangles | vertices | PSNR | SSIM | LPIPS | PSNR/100k tri | AbsRel | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| origin_main_2000iter | 39079 | 58458 | 11.047660 | 0.219931 | 0.641706 | 28.270068 | 5.611905 | 271.313 |
| current_branch_2000iter | 782982 | 820107 | 11.599438 | 0.270268 | 0.634732 | 1.481444 | 0.427880 | 257.567 |
| stage17_meshprior_2000iter | 777251 | 816498 | 13.278273 | 0.303979 | 0.607610 | 1.708364 | 0.366691 | 272.853 |

## Interpretation

Stage17 improves post-render PSNR, SSIM, LPIPS, and sparse COLMAP depth proxy versus the current branch baseline:

- PSNR delta versus current: `+1.6788349152`
- SSIM delta versus current: `+0.0337116122`
- LPIPS delta versus current: `-0.0271220207`
- depth AbsRel delta versus current: `-0.0611882158`

But Stage17 has `777251` triangles, while the clean candidate has `39079` triangles. This is a `19.889x` triangle ratio versus clean.

The collector decision is therefore:

`QUALITY_GAIN_NOT_TOPOLOGY_NORMALIZED`

This preserves the positive Stage17 result while blocking a paper-level claim until topology-budget control or efficiency-normalized reporting is complete.

## Known Limitation

Training FPS is read from documented training-internal summaries, not from a standalone machine-readable training summary. The training script should eventually emit a JSON eval summary so future collectors do not need documented constants.

## Gate

Stage gate: `PASS`.

Claim gate: `BLOCKED_BY_TOPOLOGY_BUDGET`.

Next required stage: M19 baseline audit or a topology-control extension before any stronger paper claim.
