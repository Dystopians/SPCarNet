# Stage37 Visual Failure Package Report

Date: 2026-05-02

## Status

`PASS`.

Stage37 packages the current best visual evidence, failure cases, and paper-safe claim wording. It does not run new training.

## Code

New packager:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_package_visual_failures.py --output_dir outputs/carnet/meshprior/stage37_visual_failure_package
```

Generated artifacts:

- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_failure_package.json`
- `outputs/carnet/meshprior/stage37_visual_failure_package/failure_case_table.csv`
- `outputs/carnet/meshprior/stage37_visual_failure_package/failure_case_table.md`
- `outputs/carnet/meshprior/stage37_visual_failure_package/paper_claim_wording.md`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/parking_m24_2_retention_7000.png`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/bonsai_m35_retained_relaxed.png`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/courtyard_m35_retained_relaxed.png`

## Failure Cases

| failure type | concrete artifact | decision |
|---|---|---|
| post-commit no-candidate | `docs/car_model/meshprior_stage34_post_commit_refresh_report.md` | Use relaxed post-commit discovery only behind retained-edit controls. |
| validation rollback | `outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model/prism_debug/relaxed_retained_topology_audit.json` | Report active retained commits separately from total relaxed attempts. |
| relaxed cap reached | `outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model/prism_round_checkpoints/iter_001683_candidate_meta.json` | Treat cap-1 as the safe default row; sweep higher caps later. |
| metric-path mismatch | `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.csv` | Use independent `render.py + metrics.py` values in paper tables. |
| dataset geometry observability | `docs/car_model/meshprior_stage25_multidataset_validation_report.md` | Use Mip-NeRF 360 and ETH3D for geometry-observable claims until Tanks tracks are rebuilt. |
| perceptual metric tradeoff | `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.md` | State that Stage35 is not universal dominance on every image metric. |

## Paper-Safe Claim

Safe claim:

PRISM is an auditable topology-control layer for mesh-splatting optimization. On the selected public COLMAP scenes, the retained relaxed refresh variant reduces final mesh topology while preserving or improving key independent render metrics on `bonsai` and improving topology/PSNR/SSIM on `courtyard`.

Do not claim:

Do not claim universal image-quality dominance: `courtyard` LPIPS is worse than selected earlier rows, and Tanks geometry evidence is not yet paper-grade without real sparse tracks.

## Full-Budget Decision

Do not start full-budget public-scene training yet. The current blocker is not raw training availability; it is paper packaging quality. The highest-value next step is to turn the Stage36/Stage37 tables and panels into final figure/table assets and then decide whether a single full-budget Stage35 public-scene run is worth the GPU time.

Gate: `PASS`. Visual/failure artifacts exist for parking, `bonsai`, and `courtyard`; every failure case links to a concrete local artifact; the next training decision is explicit.

