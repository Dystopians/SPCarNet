# Final Stage SCE16 Reviewer-Killer Ablation Plan

Date: 2026-05-06

Decision: `SCE16_COLLECTOR_IMPLEMENTED_INITIAL_COURTYARD_TABLE`

## Goal

SCE16 directly targets the strongest alternative explanations:

- global sparse-depth loss is enough;
- dense render-depth anchor is enough;
- vertex/freeze anchors are enough;
- LPIPS or RGB recovery is doing all the work;
- sentinel targeting is just parameter search;
- delete-only CSEF or QEM explains the gains.

## Implemented Collectors

- `scripts/car_model/meshsplatopt_collect_reviewer_killer_ablations.py`
- `scripts/car_model/meshsplatopt_make_ablation_latex_tables.py`

## Current Evidence

The initial courtyard table uses existing validated artifacts. It already shows that the best SCE7 row beats F82 on RGB, LPIPS, AbsRel, and normal, while still missing strict all-metric parent-Pareto only on Depth MAE by `0.001787`. Negative controls from SCE6/SCE7 show hard/far, top-k, stronger sparse loss, and over-continuation do not close the gap.

## Required Final Table

A full reviewer-facing table still needs every row below populated with matched-horizon artifacts:

- `no_sentinel`
- `global_sparse_only`
- `global_render_depth_anchor`
- `vertex_anchor`
- `freeze_geometry`
- `sentinel_all_points`
- `sentinel_conflict_only`
- `no_parent_one_sided`
- `no_train_test_separation` marked invalid
- `delete_only_csef`
- `qem_or_decimation`
- `lpips_heavy`

No invalid or test-leaking row may be used for a headline claim.

