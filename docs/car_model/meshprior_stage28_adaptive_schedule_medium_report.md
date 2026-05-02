# MeshPrior Stage28 Adaptive Schedule Medium Report

Date: 2026-05-02

## Gate

`SOFT PASS`

The adaptive candidate retry implementation is correct and useful, and the medium public-scene ablation used online W&B, final-checkpoint accounting, independent `render.py + metrics.py`, PRISM counterfactual metadata, and validation artifacts. It preserves the strong ETH3D `courtyard` result from M27, but it does not solve the Mip-NeRF 360 `bonsai` topology failure. Therefore Stage28 is a mechanism and diagnosis pass, not a final cross-scene method pass.

## Goal

M27 found that a fixed `ratio0p02_geom1400` schedule strongly reduces ETH3D `courtyard` topology, while `bonsai` rolls back all candidate edits. M28 tests whether rollback-driven candidate-ratio decay can recover a useful lower-pressure edit on `bonsai` without hurting `courtyard`.

## Implementation Summary

Stage28 adds opt-in rollback-driven candidate retry:

- base candidate ratio: `--prism_candidate_prune_ratio`
- adaptive enable: `--prism_adaptive_candidate_retry_on_rollback`
- ratio decay: `--prism_adaptive_candidate_ratio_decay`
- floor: `--prism_adaptive_candidate_min_ratio`
- retry budget: `--prism_adaptive_candidate_max_rollback_retries`

Default behavior remains unchanged. The smoke report confirms the retry sequence and metadata:

- report: `docs/car_model/meshprior_stage28_adaptive_schedule_smoke_report.md`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1kmwbu8g`
- smoke sequence: `0.04 -> 0.02 -> 0.01`

## Data And Commands

Scenes:

- Mip-NeRF 360 `bonsai`: `/data/peilincai/mesh_datasets/mipnerf360/bonsai`, images `images_4`, resolution `4`
- ETH3D `courtyard`: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`, images `images`, resolution `8`

Both medium runs used:

- `WANDB_MODE=online`
- `--enable_wandb --wandb_project spcarnet_meshprior --wandb_group m28_adaptive_schedule`
- sparse COLMAP depth loss
- PRISM validation and counterfactual gate
- `--prism_freeze_densification_after_first_commit`
- fixed M27 base schedule `ratio0p02_geom1400`
- adaptive retry: decay `0.5`, min ratio `0.005`, max rollback retries `2`

Command logs:

- `outputs/carnet/meshprior/stage28_adaptive_schedule/mipnerf360_bonsai_adaptive_ratio0p02_geom1400_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage28_adaptive_schedule/eth3d_courtyard_adaptive_ratio0p02_geom1400_2000iter/logs/train_command.txt`

## W&B Runs

- `bonsai` adaptive ratio `0.02 -> 0.01 -> 0.005`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/38p6bgw4`
- `courtyard` adaptive schedule, no rollback path needed: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/piadupsm`

## Results

Training metrics are the in-training test split. Independent metrics are from post-training `render.py + metrics.py`.

| scene | schedule | final triangles | final vertices | PRISM decisions | validation | train PSNR / SSIM / LPIPS | independent PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|---|---|---|
| `bonsai` | M27 fixed `0.02`, geom `1400` | `1357128` | `1641840` | `0` commit, `6` rollback | `3/3` observable, `2/3` pass | `24.1369 / 0.8293 / 0.2035` | `12.3005 / 0.2408 / 0.6194` |
| `bonsai` | M28 adaptive `0.02 -> 0.01 -> 0.005` | `1357119` | `1641759` | `0` commit, `8` rollback | `3/3` observable, `2/3` pass | `24.1378 / 0.8285 / 0.2043` | `12.3054 / 0.2410 / 0.6196` |
| `courtyard` | M27 fixed `0.02`, geom `1400` | `100858` | `168210` | `1` commit, `41` no-candidate | `4/4` observable, `3/4` pass | `19.5602 / 0.6503 / 0.4380` | `15.0739 / 0.4857 / 0.5794` |
| `courtyard` | M28 adaptive `0.02` | `100858` | `168239` | `1` commit, `41` no-candidate | `4/4` observable, `3/4` pass | `19.5691 / 0.6500 / 0.4375` | `15.0919 / 0.4844 / 0.5778` |

Aligned sparse-depth baseline references from M26:

- `bonsai` baseline independent metrics: PSNR `12.2016`, SSIM `0.2073`, LPIPS `0.6243`; W&B triangles `1357104`
- `courtyard` baseline independent metrics: PSNR `14.9462`, SSIM `0.4388`, LPIPS `0.5924`; W&B triangles `220339`

## PRISM Candidate Evidence

`bonsai`:

| iteration | ratio | pool | selected | gate result | counterfactual PSNR delta | changed pixels | adaptive retries |
|---:|---:|---:|---:|---|---:|---:|---:|
| `1501` | `0.02` | `28725` | `12685` | reject | `-1.5283` | `0.0724` | `1` |
| `1511` | `0.01` | `4797` | `4797` | reject | `-0.5623` | `0.0560` | `2` |
| `1521` | `0.005` | `4884` | `3171` | reject | `-0.2593` | `0.0282` | `2` |
| `1522-1526` | `0.005` | `4884` | `3171` | reject | about `-0.259` | about `0.028` | `2` |

`courtyard`:

- iteration `1501`: `0.02` ratio, pool `58701`, selected `2058`, counterfactual accepted, final checkpoint remains `100858` triangles.
- iterations `1592-1992`: `41` no-candidate retries after densification freeze, with topology preserved.

## Interpretation

Adaptive retry works as a control mechanism. On `bonsai`, the active ratio decays from `0.02` to `0.01` to `0.005`, the candidate selected count drops from `12685` to `4797` to `3171`, and W&B records the final active ratio and retry count. This turns the M27 failure from "2% candidates are too aggressive" into a clearer diagnosis: even the current 0.5% global candidate set is too visually/geometrically risky under the present candidate ranking and gate.

On `courtyard`, adaptive scheduling does not hurt the good M27 behavior. The first `0.02` candidate edit still commits, final topology stays at `100858` triangles, and independent PSNR/LPIPS are slightly better than M27 fixed while SSIM is slightly lower.

The important negative result is that schedule decay alone is not enough for cross-scene robustness. `bonsai` needs more granular candidate selection, not just a smaller global ratio.

## Decision

M28 medium public-scene ablation is a `SOFT PASS`.

It satisfies the execution requirements and gives a stronger diagnosis, but it fails the intended full method gate because `bonsai` topology is essentially unchanged (`1357128 -> 1357119`, only `9` triangles fewer than M27 fixed).

## Next Step

The next prompt should target candidate selection rather than schedule timing alone. The direct M29 direction is:

1. add a per-round cap such as `--prism_candidate_max_count_per_round` so tiny ratios cannot still select thousands of triangles on dense scenes;
2. add microbatch/counterfactual attribution so a large candidate set can be split into smaller candidate batches and only locally safe batches are committed;
3. log candidate pool, selected count, accepted count, rejected count, and per-batch gate deltas to W&B;
4. rerun the same `bonsai` / `courtyard` medium comparison before any full-budget sweep.

