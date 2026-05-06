# Final Stage SCE0 - State Audit for SCE-Repair

Date: 2026-05-06

Decision: `PROCEED_TO_SCE1`

## Scope

This stage locks the current source of truth before the SCE-Repair line. It does not change training code and does not launch new training runs.

The audit prioritizes the newest documents and code paths:

- `README.md` / `README.zh.md`
- `docs/car_model/final_stageF90_F96_lessons_and_bottleneck_report.md`
- `docs/car_model/final_stageF85_F89_repair_progress_report.md`
- `docs/car_model/final_stageF82_policy_v5_robustness_report.md`
- `docs/car_model/final_stageF75_adaptive_policy_reflection_report.md`
- `docs/car_model/final_stageF47_F48_csef_family_all_metric_repair_report.md`
- `docs/car_model/SPCarNet_research_log.md`
- strict recovery, COLMAP geometry proxy, counterfactual gate, scoring, adaptive policy, compact selector, and the sparse-depth / teacher / checkpoint-anchor sections in `train.py`

## Required Preflight

Commands run from repository root:

```bash
git status --short
python --version
python -m compileall scripts/car_model ss3dm_prior utils -q
git log --oneline -20
```

Observed state:

- Python: `Python 3.13.2`
- `compileall`: pass, no reported errors.
- Worktree non-clean only because of user/current-session artifacts:
  - `?? docs/Finalized_prompts_v2_topconf.md`
  - ` ? submodules/effrdel`
  - ` ? submodules/simple-knn`
- Latest commits include:
  - `5390382 Add F90-F96 bottleneck lessons report`
  - `3b76e52 Add checkpoint render geometry anchor`
  - `8b51963 Expose recovery optimizer learning rates`
  - `03aa7d4 Add checkpoint geometry anchor recovery loss`
  - `86ead62 Add F85-F89 repair progress report`
  - `3ab4278 Add teacher render repair audit gate`
  - `c3841c2 Add F82 qualitative gallery audit`
  - `bdceeb9 Add two-seed fixed adaptive policy validation`

## Accepted Current Baseline

The accepted current baseline remains **F82 fixed adaptive policy v5**.

F82 is the safer paper-facing fixed-policy evidence because it gives `8 / 8` all-metric clean-long wins over bonsai / courtyard / room / counter across two seeds, with topology unchanged and without per-scene retuning. The v5 repair reduced the small-scene budget from the more aggressive F79/F80 setting:

- bonsai: 25.00% removed
- courtyard: 72.00% removed
- room: 15.25% removed
- counter: 15.25% removed

All F82 rows use strict topology freeze, sparse-depth lambda `0.001`, LPIPS lambda `0.00025`, W&B online logging, and `22000 -> 26000` recovery.

## Strongest Rejected Repair Candidate

The strongest rejected repair candidate is **F95 courtyard render-space geometry anchor recovery**.

F95 improves the visible/render side relative to F82:

| metric | F82 parent | F95 candidate | candidate direction |
|---|---:|---:|---|
| PSNR | 12.198611 | 12.276576 | better |
| SSIM | 0.308649 | 0.315319 | better |
| LPIPS | 0.566687 | 0.565402 | better |
| per-view PSNR | 5 common views | 0 negative views | better |
| Normal angle | 40.215702 | 40.167017 | better |
| AbsRel | 0.301884 | 0.303441 | worse |
| Depth MAE | 3.339873 | 3.378707 | worse |

F95 is therefore rejected by the strict parent-Pareto gate even though it is currently the best visual / per-view / normal repair candidate.

## Exact Remaining Bottleneck

The exact blocker is **courtyard sparse depth / AbsRel parent-Pareto failure**.

The remaining F95 gap against F82 is small but decisive:

- AbsRel regresses from `0.3018837` to `0.3034414`
- Depth MAE regresses from `3.3398725` to `3.3787072`

This is not a general RGB failure and not primarily a per-view robustness failure anymore. The failure is gate-specific: a small subset of sparse COLMAP correspondence measurements can regress even while mean render quality, per-view PSNR samples, LPIPS, and normal angle improve.

## Why Global Lambda Sweeps Are Low Priority

F85-F96 show that more global sweeps are unlikely to solve the current blocker efficiently:

- Conservative budget changes trade render and geometry but do not dominate F82.
- Longer continuation can improve selected averages while creating per-view or geometry regressions.
- Teacher-render recovery improves RGB/per-view metrics but can drift sparse depth.
- Vertex checkpoint anchoring does not align with the sparse gate; strong vertex anchoring harms LPIPS/per-view/normal.
- Freezing vertices alone shifts the tradeoff but does not solve all metrics.
- Increasing dense render-depth anchor globally in F96 did not repair sparse depth and regressed normal relative to F95.

The likely mismatch is between dense render-space `surf_depth`, sparse COLMAP correspondence depth, and RGB teacher-induced geometry movement. The next step must be per-correspondence sparse-depth analysis before more full recovery runs.

## Code Hooks Already Available

Existing code already provides the raw ingredients needed for SCE-Repair:

- `utils/prism_geometry_proxy.py`
  - `collect_view_sparse_depth_correspondences`
  - `build_geometry_proxy_context`
  - `evaluate_view_sparse_geometry_proxy`
  - shared `GeometryProxyConfig`
- Renderer output:
  - train/eval code consumes render package `surf_depth`
  - render package also exposes `rend_normal` where normal comparisons are needed
- `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py`
  - strict topology-frozen recovery wrapper
  - writes exact train/render/metrics/geometry commands
  - always includes `--freeze_topology_updates --skip_restricted_delaunay`
  - enables W&B for recovery contracts
- `train.py`
  - sparse COLMAP depth loss is opt-in through `--enable_sparse_colmap_depth_loss`
  - teacher-render recovery loss is opt-in through `--enable_teacher_render_loss`
  - checkpoint vertex anchor is opt-in through `--enable_checkpoint_geometry_anchor`
  - checkpoint render depth/normal anchor is opt-in through `--enable_checkpoint_render_geometry_anchor`
- `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py`
  - exposes optimizer LR overrides: `lr_triangles_points_init`, `feature_lr`, `weight_lr`
- `utils/prism_counterfactual.py`, `utils/prism_scoring.py`, `utils/prism_adaptive_policy.py`, and `ss3dm_prior/meshsplatopt/compact_selector.py`
  - provide existing counterfactual gate, CSEF scoring, adaptive policy state, and compact selector infrastructure.

## Missing Code Hooks

The SCE line still needs four missing hooks:

1. **Per-correspondence parent-vs-candidate regression analyzer**  
   Required to compare F82 and F95 on the exact sparse correspondence distribution, not only aggregate JSON metrics.

2. **Sentinel cache builder**  
   Required to freeze train/calibration sparse correspondences without test leakage and expose high-risk sentinel points to training.

3. **One-sided parent rollback sparse-depth loss**  
   Required to penalize only current-vs-parent regressions at sentinel correspondences, without pulling all depth toward the parent or penalizing improvements.

4. **Sentinel-aware recovery policy and gate**  
   Required to decide when F95-style visual repair needs targeted rollback, and to reject candidates before expensive full evaluations when sentinel depth regresses.

## SCE0 Gate

SCE0 passes.

The immediate next step is **per-correspondence sparse-depth analysis before more full recovery runs**. Specifically, SCE1 must build and smoke-test the parent-vs-candidate sparse-depth regression analyzer, then run it on courtyard F82 vs F95 if local artifacts are present.

