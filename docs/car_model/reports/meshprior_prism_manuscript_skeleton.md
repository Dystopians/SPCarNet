# PRISM MeshPrior Manuscript Skeleton

Date: 2026-05-02

## Working Title

PRISM: Auditable Topology Control for Mesh Splatting with Scene-Evidence Gates

## Abstract Draft

Mesh-splatting methods can reconstruct visually plausible scenes but often carry redundant or unstable mesh topology. We study a conservative topology-control layer, PRISM, that proposes local mesh edits during optimization and accepts them only through scene-evidence gates, rollback, and retained-topology audits. The method treats learned or hand-designed priors as proposal mechanisms rather than unconditional geometry writers: every topology edit must survive render, sparse-geometry, calibration-view, and recovery-window checks. On a parking-phone scene, PRISM reduces topology in a long-budget optimization row while preserving independent render quality. On Mip-NeRF 360 `bonsai`, the retained relaxed refresh variant reduces final topology from the Stage33 reference while improving independent PSNR, SSIM, and LPIPS. On ETH3D `courtyard`, it reduces topology and improves PSNR/SSIM among selected rows, with an explicit LPIPS tradeoff. These results support PRISM as an auditable topology-control framework, not as a universal image-quality optimizer.

## Method Summary

PRISM adds a topology proposal and validation controller to mesh-splatting optimization.

Inputs:

- calibrated images and COLMAP-style sparse geometry;
- the current mesh-splatting state;
- per-triangle render, geometry, orientation, uncertainty, and protection scores;
- optional learned prior/proposal metadata from the earlier MeshPrior pipeline.

Outputs:

- accepted or rejected topology edits;
- rollback checkpoints;
- final topology audit records;
- independent render metrics and paper-facing evidence tables.

Core mechanism:

1. Collect per-triangle statistics during training.
2. Rank candidate triangles for conservative pruning.
3. Evaluate candidate edits using counterfactual calibration views.
4. Commit only if the candidate passes local gates.
5. Run recovery-window validation and rollback failed edits.
6. For post-commit no-candidate cases, allow a default-off relaxed refresh path with a retained commit cap and strict proxy gate.
7. Write final cleanup and retained-topology audit metadata.

The key design principle is: prior proposes, scene evidence disposes.

## Experimental Setup

Datasets:

- `parking_phone_tiny`: local phone-captured parking scene with COLMAP reconstruction.
- Mip-NeRF 360 `bonsai`: public COLMAP-compatible scene.
- ETH3D `courtyard`: public COLMAP-compatible scene.

Metrics:

- paper-facing image metrics: independent `render.py + metrics.py` PSNR, SSIM, LPIPS;
- topology: final checkpoint triangle count from final-cleanup summaries;
- audit: active PRISM commits, relaxed commit attempts, validation rollbacks, cap-reached metadata;
- training-time metrics are diagnostic only and are not substituted for independent render metrics.

## Main Result Table

Source: `outputs/carnet/meshprior/stage38_paper_assets/final_paper_table.md`

| scene | row | topology | PSNR | SSIM | LPIPS | audit note |
|---|---|---:|---:|---:|---:|---|
| parking_phone_tiny | late_prism_freeze_after_first_commit | 254491 | 17.314823 | 0.559230 | 0.442099 | long-budget parking topology-retention evidence |
| mipnerf360_bonsai | diverse_calib_measured_rank_cap512 | 633787 | 12.199921 | 0.276533 | 0.612583 | Stage33 reference |
| mipnerf360_bonsai | retained_relaxed_cap1_strict_gate | 633275 | 12.267367 | 0.277617 | 0.611939 | 1 active relaxed edit; 4 validation rollbacks recorded |
| eth3d_courtyard | measured_rank_cap512 | 102404 | 15.138977 | 0.484960 | 0.579188 | selected earlier row |
| eth3d_courtyard | retained_relaxed_cap1_strict_gate | 101913 | 15.383161 | 0.508091 | 0.584694 | 1 active relaxed edit; cap reached later |

## Claim Boundaries

Supported:

- PRISM is an auditable topology-control layer for mesh-splatting optimization.
- Retained relaxed refresh improves `bonsai` independent PSNR/SSIM/LPIPS while reducing topology versus the Stage33 reference.
- Retained relaxed refresh improves `courtyard` topology, PSNR, and SSIM among selected rows.
- Validation rollback and retained-topology audits are necessary: several relaxed commits can be rolled back before one survives.

Not supported:

- universal image-quality dominance;
- radar-only or scan-only reconstruction;
- Tanks and Temples geometry claims from the current mirror without true sparse COLMAP tracks;
- replacing independent metrics with training-time metrics.

## Limitations

- `courtyard` LPIPS is worse for M35 than selected M32/M33 rows.
- Current public-scene runs are 2000-iteration medium runs, not full-budget public-scene evidence.
- The current Tanks mirror is trainable but not geometry-observable enough for sparse-track claims.
- PRISM is conservative by design and can under-prune when the retained relaxed cap is set to one.
- The earlier object-prior modules are not the current strongest narrative; the strongest evidence is scene-evidence-gated topology control.

## Figure Plan

Use the generated render-vs-GT panels:

- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/parking_m24_2_retention_7000.png`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/bonsai_m35_retained_relaxed.png`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/courtyard_m35_retained_relaxed.png`

Captions are in:

- `outputs/carnet/meshprior/stage38_paper_assets/figure_captions.md`

## Next Decision

Do not start another full-budget public-scene run until the final table identifies a concrete missing row. If that changes, run exactly one full-budget Stage35 public-scene experiment first, with W&B enabled and GPU availability checked before launch.

