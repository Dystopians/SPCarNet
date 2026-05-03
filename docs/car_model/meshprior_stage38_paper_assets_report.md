# Stage38 Paper Assets Report

Date: 2026-05-02

## Status

`PASS`.

Stage38 converts the M36/M37 evidence package into paper-draft assets: a final selected-row table, figure captions, limitations, and an explicit full-budget public-scene training decision.

## Code

New asset builder:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_make_paper_assets.py --metric_table outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.csv --output_dir outputs/carnet/meshprior/stage38_paper_assets
```

Generated artifacts:

- `outputs/carnet/meshprior/stage38_paper_assets/paper_assets_package.json`
- `outputs/carnet/meshprior/stage38_paper_assets/final_paper_table.md`
- `outputs/carnet/meshprior/stage38_paper_assets/figure_captions.md`
- `outputs/carnet/meshprior/stage38_paper_assets/limitations.md`

## Final Selected Table

All image metrics are independent `render.py + metrics.py` values.

| scene | row | topology | PSNR | SSIM | LPIPS | audit note |
|---|---|---:|---:|---:|---:|---|
| parking_phone_tiny | late_prism_freeze_after_first_commit | 254491 | 17.314823 | 0.559230 | 0.442099 | long-budget parking topology-retention evidence |
| mipnerf360_bonsai | diverse_calib_measured_rank_cap512 | 633787 | 12.199921 | 0.276533 | 0.612583 | baseline/earlier row |
| mipnerf360_bonsai | retained_relaxed_cap1_strict_gate | 633275 | 12.267367 | 0.277617 | 0.611939 | 1 active relaxed edit; 4 validation rollbacks recorded |
| eth3d_courtyard | measured_rank_cap512 | 102404 | 15.138977 | 0.484960 | 0.579188 | baseline/earlier row |
| eth3d_courtyard | retained_relaxed_cap1_strict_gate | 101913 | 15.383161 | 0.508091 | 0.584694 | 1 active relaxed edit; cap reached later |

## Limitations

- The evidence supports PRISM as an auditable topology-control layer, not a universal image-quality optimizer.
- Independent `render.py + metrics.py` values are the paper-facing metrics; training-time eval values are diagnostic.
- M35 improves all selected independent metrics on `bonsai`, but `courtyard` LPIPS is worse than selected M32/M33 rows.
- The available Tanks and Temples mirror is not geometry-observable enough for sparse-track geometry claims.
- A full-budget Stage35 public-scene run should be launched only if it fills a concrete paper-table gap.

## Full-Budget Decision

`NO_GO_FOR_NOW`.

Reason: the immediate blocker is paper-asset clarity, not missing short-run evidence. One full-budget Stage35 public-scene run should be revisited only after the final table identifies a specific missing row. If revisited, W&B must be enabled and GPU availability must be checked before training.

Gate: `PASS`.

