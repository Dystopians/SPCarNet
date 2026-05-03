# Stage36 Metric Reconciliation Report

Date: 2026-05-02

## Status

`PASS`.

Stage36 turns the M24-M35 experimental record into a reproducible paper-facing evidence table. It does not run new training. It reads local artifacts and keeps training-time evaluation metrics separate from independent `render.py + metrics.py` metrics.

## Code

New collector:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_collect_metric_reconciliation.py --output_dir outputs/carnet/meshprior/stage36_metric_reconciliation
```

Generated artifacts:

- `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_report.json`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.csv`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.md`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/visual_panels/bonsai_m35_retained_relaxed.png`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/visual_panels/courtyard_m35_retained_relaxed.png`

## Evidence Table

| label | scene | method | triangles | PSNR | SSIM | LPIPS | active PRISM commits | relaxed active/rolled back |
|---|---|---|---:|---:|---:|---:|---:|---:|
| parking_m24_2_retention_7000 | parking_phone_tiny | late_prism_freeze_after_first_commit | 254491 | 17.314823 | 0.559230 | 0.442099 | 2 | 0/0 |
| bonsai_m26_sparse_depth_baseline | mipnerf360_bonsai | sparse_depth_baseline | 2487474 | 12.201612 | 0.207315 | 0.624259 | 0 | 0/0 |
| bonsai_m29_cap512 | mipnerf360_bonsai | candidate_cap512_adaptive | 633787 | 12.185925 | 0.276379 | 0.612921 | 1 | 0/0 |
| bonsai_m33_diverse_calib | mipnerf360_bonsai | diverse_calib_measured_rank_cap512 | 633787 | 12.199921 | 0.276533 | 0.612583 | 1 | 0/0 |
| bonsai_m34_relaxed_v3 | mipnerf360_bonsai | post_commit_relaxed_score_v3 | 631739 | 12.201998 | 0.275728 | 0.612961 | 6 | 0/0 |
| bonsai_m35_retained_relaxed | mipnerf360_bonsai | retained_relaxed_cap1_strict_gate | 633275 | 12.267367 | 0.277617 | 0.611939 | 6 | 1/4 |
| courtyard_m26_sparse_depth_baseline | eth3d_courtyard | sparse_depth_baseline | 410254 | 14.946162 | 0.438775 | 0.592443 | 0 | 0/0 |
| courtyard_m32_measured_rank | eth3d_courtyard | measured_rank_cap512 | 102404 | 15.138977 | 0.484960 | 0.579188 | 1 | 0/0 |
| courtyard_m33_diverse_calib | eth3d_courtyard | diverse_calib_measured_rank_cap512 | 102407 | 15.073723 | 0.484009 | 0.578974 | 1 | 0/0 |
| courtyard_m35_retained_relaxed | eth3d_courtyard | retained_relaxed_cap1_strict_gate | 101913 | 15.383161 | 0.508091 | 0.584694 | 2 | 1/0 |

## Interpretation

Stage35 is the current best `bonsai` retained-edit row: it improves PSNR, SSIM, and LPIPS over Stage33 while lowering final topology. It also fixes the Stage34 ambiguity by recording which relaxed commits were validation-rolled back and which one survived.

On ETH3D `courtyard`, Stage35 has the best PSNR and SSIM among the selected rows and the lowest topology in this table. LPIPS is worse than Stage32/33, so the paper claim should state scene-dependent perceptual tradeoffs rather than universal metric dominance.

## Metric Policy

The headline table uses only independent `render.py + metrics.py` values. The collector also stores training-time test/train metrics in CSV/JSON, but those are diagnostic fields and must not replace independent metrics in paper tables.

## Failure Taxonomy

- no-candidate after topology sync: recent protection can mask every survivor after a commit
- validation rollback: a relaxed commit can pass the counterfactual proxy yet fail recovery-window validation
- retained relaxed cap reached: M35 intentionally blocks additional relaxed fallback once one active relaxed edit survives
- metric-path mismatch: training eval metrics and independent render metrics are different rows
- dataset geometry observability: Mip-NeRF 360 and ETH3D converted scenes support COLMAP proxy geometry; the current Tanks mirror does not support sparse-track geometry claims

## Verification

Completed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_collect_metric_reconciliation.py --output_dir outputs/carnet/meshprior/stage36_metric_reconciliation
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m json.tool outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_report.json
```

Gate: `PASS`. The table is reproducible from local artifacts and does not mix metric paths.

