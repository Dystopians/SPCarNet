# Stage SCE21 Tail-Risk Sentinel Design

Date: 2026-05-06

Decision: `CTR_SCE_MECHANISM_IMPLEMENTED`

## Mechanism

SCE21 introduces **Conditional Tail-Risk Sentinel Envelope (CTR-SCE)**.

The earlier SCE rollback loss optimized a weighted mean over train/calibration sentinels. That was too weak for the actual failure: a small number of held-out sparse correspondences could dominate Depth MAE while the average train sentinel loss looked acceptable.

CTR-SCE changes the optimization target:

```text
loss = CVaR_tail( SmoothL1( ReLU(error_current - error_parent - margin) ) )
```

where the tail can be computed over individual sentinels or local clusters. The loss remains one-sided: improvements over the parent are not penalized.

## Literature Link

The design combines four research ideas:

- CVaR tail-risk optimization from Rockafellar and Uryasev, "Optimization of conditional value-at-risk" (Journal of Risk, 2000): https://doi.org/10.21314/JOR.2000.038
- Conformal risk control: https://arxiv.org/abs/2208.02814
- Influence-style debugging of local training evidence: https://proceedings.mlr.press/v70/koh17a.html
- Sparse SfM / COLMAP evidence as a geometry certificate, consistent with COLMAP and depth-supervised radiance-field practice: https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html and https://www.cs.cmu.edu/~dsnerf/

In MeshSplatOpt terms, the key claim is:

> Geometry recovery should minimize certificate tail risk, not mean sparse-depth loss.

## New Interfaces

Added opt-in rollback fields:

- `--sparse_depth_parent_rollback_aggregation {mean,cvar,cluster_cvar}`
- `--sparse_depth_parent_rollback_cvar_fraction`
- `--sparse_depth_parent_rollback_cvar_min_points`
- `--sparse_depth_parent_rollback_pixel_radius`
- `--sparse_depth_parent_rollback_patch_reduce {center,max_violation,mean_violation}`

Default behavior is unchanged:

```text
aggregation = mean
pixel_radius = 0
patch_reduce = center
```

## Local Envelope

CTR-SCE can sample a small pixel neighborhood around each sentinel and aggregate the worst one-sided violation. This does not use test split information. It makes the train sentinel a small local certificate rather than a single brittle pixel.

This matters for out-of-trajectory views: a held-out sparse correspondence may not land at the exact same pixel as any training sentinel, but it can share a local surface/ray neighborhood.

## Why This Is Not Another Parameter Sweep

The change is structural:

- old objective: mean one-sided parent rollback over selected points
- new objective: CVaR / cluster-CVaR over the worst active sparse-depth certificate violations
- optional local pixel envelope makes each sentinel robust to small projection shifts

The mechanism targets the observed failure mode directly: a small tail of sparse-depth errors defeating the aggregate gate.

## Smoke Test

`scripts/car_model/smoke_test_stageSCE21_tail_risk_rollback.py`

The smoke test verifies:

- CVaR loss is larger than mean loss when one sentinel is a tail failure
- cluster-CVaR selects the worst cluster
- local pixel envelope can increase the detected max violation
