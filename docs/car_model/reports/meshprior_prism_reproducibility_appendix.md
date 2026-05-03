# PRISM MeshPrior Reproducibility Appendix

Date: 2026-05-02

## Repository State

- branch: `clean-submit`
- remote target: `spcarnet/main`
- current method reports:
  - `docs/car_model/meshprior_stage35_retained_refresh_report.md`
  - `docs/car_model/meshprior_stage36_metric_reconciliation_report.md`
  - `docs/car_model/meshprior_stage37_visual_failure_package_report.md`
  - `docs/car_model/meshprior_stage38_paper_assets_report.md`

## Datasets

- Parking phone scene: local parking-phone COLMAP/image scene used in prior reports.
- Mip-NeRF 360 `bonsai`: `/data/peilincai/mesh_datasets/mipnerf360/bonsai`
- ETH3D `courtyard`: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`

## Main W&B Runs

- M24.2 parking topology retention: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vsv2bs79`
- M33 `bonsai` diverse calibration reference: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kg5htc8u`
- M35 `bonsai` retained relaxed refresh: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rszvl7gn`
- M32 `courtyard` measured-rank reference: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fb7jfcaj`
- M35 `courtyard` retained relaxed refresh: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/u2s15ok0`

## Rebuild Evidence Tables

Metric reconciliation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_collect_metric_reconciliation.py --output_dir outputs/carnet/meshprior/stage36_metric_reconciliation
```

Visual/failure package:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_package_visual_failures.py --output_dir outputs/carnet/meshprior/stage37_visual_failure_package
```

Paper assets:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_make_paper_assets.py --metric_table outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.csv --output_dir outputs/carnet/meshprior/stage38_paper_assets
```

## Metric Policy

Paper tables use independent `render.py + metrics.py` values stored in each model's `results.json`. Training-time evaluation values are collected only as diagnostics and must not replace independent metrics.

## Key Local Artifacts

- M35 `bonsai` model: `outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model`
- M35 `courtyard` model: `outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- M36 evidence table: `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.md`
- M37 failure cases: `outputs/carnet/meshprior/stage37_visual_failure_package/failure_case_table.md`
- M38 final paper table: `outputs/carnet/meshprior/stage38_paper_assets/final_paper_table.md`

## Validation Commands

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/meshprior_collect_metric_reconciliation.py scripts/car_model/meshprior_package_visual_failures.py scripts/car_model/meshprior_make_paper_assets.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
git diff --check
```

## Optional Future Full-Budget Run Policy

Do not launch a full-budget public-scene Stage35 run without a concrete missing table row. If one is needed:

1. Run `nvidia-smi`.
2. Choose a light GPU.
3. Set `WANDB_MODE=online` and `WANDB_PROJECT=spcarnet_meshprior`.
4. Use `--enable_wandb`, a unique `--wandb_name`, and save render/metrics logs.
5. Run independent `render.py + metrics.py`.
6. Rebuild M36-M38 tables.

