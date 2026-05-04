# MeshSplatOpt Stage R17 paper package report

Date: 2026-05-03

## Purpose

R17 packages the current project state into paper-facing documents and makes the go/no-go decision explicit.

## Implemented interface

- `scripts/car_model/meshsplatopt_make_neurips_package.py`
- manifest: `outputs/carnet/meshsplatopt/neurips_package/manifest.json`

Generated files:

- `docs/car_model/reports/meshsplatopt_neurips_manuscript_skeleton.md`
- `docs/car_model/reports/meshsplatopt_neurips_method.md`
- `docs/car_model/reports/meshsplatopt_neurips_experiments.md`
- `docs/car_model/reports/meshsplatopt_neurips_ablation.md`
- `docs/car_model/reports/meshsplatopt_neurips_related_work.md`
- `docs/car_model/reports/meshsplatopt_neurips_reviewer_risk_checklist.md`
- `docs/car_model/reports/meshsplatopt_neurips_final_go_no_go.md`

## Current go/no-go

`WORKSHOP_OR_ARXIV_UNTIL_R15_R16_COMPLETE`.

R53/R55 are strong enough to justify continued full-budget work and materially repair the clean-baseline failure. They are not sufficient by themselves for a NeurIPS main-conference claim because the original prompt requires multi-scene full-budget validation, systematic ablations, and stronger real bidirectional edit evidence.

## Upgrade criteria

Upgrade to `GO_NEURIPS` only if:

1. R53-like clean-to-compact dominance replicates on at least 2/3 scenes, or another CSEF edit route produces stronger cross-scene gains.
2. R16 proves at least three core components are empirically necessary.
3. Giant-hole repair has synthetic plus at least one real or realistic non-hallucinated example.
4. Independent metrics remain separated from training metrics.

