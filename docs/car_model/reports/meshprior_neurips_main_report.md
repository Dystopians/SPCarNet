# MeshPrior Evaluation Matrix Report

This report separates object-prior quality, synthetic repair, scene metrics, and safety ablations. Missing experiments are retained as `MISSING` rather than dropped.

## Matrix Status

{
  "total": 11,
  "available": 7,
  "missing": 4
}

## Table 1 — Object Prior Quality

| method | output_type | recon_chamfer_l1 | hidden_chamfer_l1 | visible_preservation_error | zero_corruption_chamfer | free_space_violation | mesh_extraction_success | inference_time | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0.7 residual baseline | points |  |  |  |  |  |  |  | MISSING |
| v0.8.2 point-flow baseline | points | 0.1231232387131279 | 0.1548924239225758 | 0.13506823825170694 | None | 0.030602088341346152 | None | 15.578299045562744 | AVAILABLE |
| SP-CarNet Stage 3 posterior encoder | mesh field | 0.0663909994752951 | 0.0990753869336207 | 0.06268131060218349 | 0.06664571945456046 | 0.033534966626213594 | 1.0 | None | AVAILABLE |
| SP-CarNet Stage 4 MAP refinement | mesh field |  |  |  |  |  |  |  | MISSING |
| SP-CarNet Stage 5 oracle K=8 analysis | oracle mesh field |  |  |  |  |  |  |  | MISSING |

## Table 2 — Synthetic Mesh Repair

| method | damage_type | hole_closure | floater_prune_precision | floater_prune_recall | valid_surface_protect_recall | visible_preservation | free_space_violation | triangle_count_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| protect/prune proposals | None | None | None | None | None | None | None | None | MISSING |
| none | vertex_noise | None | None | None | 0.8333333333333334 | -0.08333333333333326 | 0.0 | None | AVAILABLE |
| surface_support_v1 | vertex_noise | None | None | None | 0.9166666666666666 | 0.0 | 0.0 | None | AVAILABLE |
| protect/prune + snap + fill | None | 4.0 | None | None | None | None | 0.0 | 4.0 | AVAILABLE |
| none | vertex_noise | None | None | None | 0.8333333333333334 | -0.08333333333333326 | 0.0 | None | AVAILABLE |
| surface_support_v1 | vertex_noise | None | None | None | 0.9166666666666666 | 0.0 | 0.0 | None | AVAILABLE |

## Table 3 — Scene Mesh Optimization

| method | scene | checkpoint_iteration | psnr | ssim | lpips | colmap_absrel | sparse_depth_mae | normal_mean_angle | triangle_count | controlled_fps | car_roi_hole_floater_metrics | accepted_proposals | rejected_proposals | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene baseline, 200-iter no-cleanup smoke | video_colmap_smoke | iteration_200 | 6.933581471443176 | 0.16371289547532797 | 0.694071426987648 | 0.10470779720655764 | 0.024122862845250084 | 37.51919533010328 | 5706 | 334.7374487692397 | None | None | None | AVAILABLE |
| scene baseline + MeshPrior proposals | synthetic_local_hole | dry_run_synthetic | None | None | None | None | None | None | None | None | None | 1 | 0 | AVAILABLE |

## Table 4 — Safety Ablation

| row | expected_status | available_evidence | risk |
| --- | --- | --- | --- |
| direct insert, no gate | not approved | not run | hallucination/free-space |
| prior score only | diagnostic | object prior metrics | object confidence without scene evidence |
| prior + free-space gate | partial | dry-run free-space deltas | no render gate |
| prior + geometry gate | available | M9/M11 dry-run gate | dry-run only |
| prior + render gate | missing | not connected | render degradation unknown |
| full gated method | partial | proposal gate + cleanup-repaired training smoke | real scene integration incomplete |

## Failure Cases

See `outputs/carnet/meshprior/reports/failure_cases.md`.
