# MeshPrior Stage27 Schedule Ablation Report

Date: 2026-05-02

## Gate

`SOFT PASS`

M27 fixed topology accounting and completed two cross-scene schedule ablations with online W&B, independent `render.py + metrics.py`, final-checkpoint topology accounting, PRISM candidate metadata, and PRISM validation artifacts. The best schedule improves ETH3D `courtyard` strongly, but it does not yet produce stable topology reduction on Mip-NeRF 360 `bonsai`, so this is not a final paper-strength schedule.

## Goal

M26 showed that M24.2 PRISM transfers mechanically to public COLMAP-style scenes, but direct 2000-iteration W&B topology reduction was too small. M27 tested whether stronger direct PRISM pressure can reduce final checkpoint topology without breaking independent render quality or geometry-observable validation.

## Accounting Fix

Before schedule tuning, `train.py` was patched so W&B topology counts are logged after standard prune/densify topology mutation and again at final checkpoint save. The 520-iteration ETH3D smoke confirmed:

- W&B `mesh/triangle_count`: `33487`
- W&B `mesh/final_checkpoint_triangle_count`: `33487`
- final cleanup `post_prune_triangle_count`: `33487`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/i6lfgt66`

Detailed report: `docs/car_model/meshprior_stage27_accounting_fix_report.md`.

## Data And Commands

Scenes:

- Mip-NeRF 360 `bonsai`: `/data/peilincai/mesh_datasets/mipnerf360/bonsai`, images `images_4`, resolution `4`
- ETH3D `courtyard`: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`, images `images`, resolution `8`

All valid runs used:

- `WANDB_MODE=online`
- `--enable_wandb --wandb_project spcarnet_meshprior --wandb_group m27_schedule_ablation`
- sparse COLMAP depth loss
- PRISM counterfactual gate
- `--prism_freeze_densification_after_first_commit`
- independent evaluation with `render.py` and `metrics.py`

Command logs:

- `outputs/carnet/meshprior/stage27_schedule_ablation/mipnerf360_bonsai_ratio0p01_geom1200_v2_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage27_schedule_ablation/eth3d_courtyard_ratio0p01_geom1200_v3_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage27_schedule_ablation/mipnerf360_bonsai_ratio0p02_geom1400_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage27_schedule_ablation/eth3d_courtyard_ratio0p02_geom1400_2000iter/logs/train_command.txt`

## W&B Runs

- `bonsai` ratio `0.01`, geometry acquisition until `1200`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mlftnbt5`
- `courtyard` ratio `0.01`, geometry acquisition until `1200`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qvrnsj2v`
- `bonsai` ratio `0.02`, geometry acquisition until `1400`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/27vl4jnt`
- `courtyard` ratio `0.02`, geometry acquisition until `1400`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ffp07dua`

Invalid / diagnostic attempts:

- Initial `ratio0p01_geom1200` used obsolete CLI flags and is not a valid training run.
- `eth3d_courtyard_ratio0p01_geom1200_v2_2000iter` stopped early and is not used as a completed row.

## Results

Training metrics are the in-training test split. Independent metrics are from post-training `render.py + metrics.py`.

| scene | schedule | final triangles | final vertices | PRISM decisions | validation | train PSNR / SSIM / LPIPS | independent PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|---|---|---|
| `bonsai` | M26 PRISM reference | `1350319` | n/a | `1` commit, `3` rollback, `2` no-candidate | `4/4` observable, `2/4` pass | `23.6994 / 0.8004 / 0.2386` | `12.1712 / 0.2378 / 0.6182` |
| `bonsai` | `0.01`, geom `1200` | `1350762` | `1633075` | `1` commit, `0` rollback, `61` no-candidate | `4/4` observable, `2/4` pass | `23.7041 / 0.8012 / 0.2395` | `12.2429 / 0.2362 / 0.6182` |
| `bonsai` | `0.02`, geom `1400` | `1357128` | `1641840` | `0` commit, `6` rollback, `0` no-candidate | `3/3` observable, `2/3` pass | `24.1369 / 0.8293 / 0.2035` | `12.3005 / 0.2408 / 0.6194` |
| `courtyard` | M26 PRISM reference | `217058` | n/a | `3` commits, `0` rollback, `4` no-candidate | `5/5` observable, `3/5` pass | `19.6932 / 0.6518 / 0.4284` | `15.0614 / 0.4734 / 0.5838` |
| `courtyard` | `0.01`, geom `1200` | `410350` | `444916` | `0` commit, `6` rollback, `20` no-candidate | `3/3` observable, `1/3` pass | `19.6799 / 0.6543 / 0.4234` | `14.6853 / 0.4354 / 0.5910` |
| `courtyard` | `0.02`, geom `1400` | `100858` | `168210` | `1` commit, `0` rollback, `41` no-candidate | `4/4` observable, `3/4` pass | `19.5602 / 0.6503 / 0.4380` | `15.0739 / 0.4857 / 0.5794` |

Aligned sparse-depth baseline references from M26:

- `bonsai` baseline final independent metrics: PSNR `12.2016`, SSIM `0.2073`, LPIPS `0.6243`; W&B triangles `1357104`
- `courtyard` baseline final independent metrics: PSNR `14.9462`, SSIM `0.4388`, LPIPS `0.5924`; W&B triangles `220339`

## Interpretation

The `ratio0p02_geom1400` row is the useful M27 finding.

On ETH3D `courtyard`, it accepts one PRISM edit at iteration `1501`, reducing `102916 -> 100858` triangles, then freezes densification and keeps the final checkpoint at `100858` triangles. Relative to the aligned sparse-depth baseline, this is about `54.2%` fewer final W&B/final-checkpoint triangles, while independent metrics improve from `14.9462 / 0.4388 / 0.5924` to `15.0739 / 0.4857 / 0.5794`.

On Mip-NeRF 360 `bonsai`, the same stronger setting does not reduce topology. It generates candidates at iterations `1501` through `1506`, but the counterfactual gate rolls back all six. Independent PSNR and SSIM improve versus M26 PRISM and the sparse-depth baseline, but LPIPS is slightly worse than M26 PRISM and final triangles stay near the baseline. This means the gate is protecting render quality correctly, but the candidate generator is not finding useful removable topology for `bonsai` under this budget and schedule.

The earlier `ratio0p01_geom1200` schedule is not the answer. It starts the candidate phase too early: before the standard 1500-iteration topology transition, candidate pools are often empty; after the transition, `bonsai` can commit only a tiny edit and `courtyard` rejects the candidates.

## Failure Mode

M27 isolates the main cross-scene failure:

1. Topology pressure is scene- and schedule-sensitive.
2. Earlier candidate windows can produce many `no_candidates` events before standard topology growth.
3. Stronger pressure at `1400` works on ETH3D but rolls back on `bonsai`.
4. The controller currently optimizes a local candidate edit, not a learned cross-scene policy for when and where topology is redundant.

This is a method-design issue, not a logging issue. Accounting is now consistent.

## Decision

M27 is a `SOFT PASS`.

It satisfies the execution requirements: online W&B, two public geometry-observable scenes, schedule ablations, independent metrics, validation artifacts, and clear failure analysis. It does not satisfy the full `PASS` gate because meaningful topology reduction is not achieved on both scenes.

## Next Step

Do not launch a broad full-budget public-scene sweep with the current static schedule. The next technical step should be M28: make PRISM schedule selection adaptive instead of fixed. The most direct path is:

1. add candidate-window diagnostics that summarize removable-topology mass before attempting a round;
2. allow the controller to wait until candidate mass is nonzero and validation is stable;
3. compare fixed `ratio0p02_geom1400` against adaptive trigger on `bonsai` and `courtyard`;
4. only then scale to 7000-iteration public-scene sweeps.

