# MeshPrior Stage29 Candidate Cap Sweep Report

Date: 2026-05-02

## Gate

`PASS` for the M29 cap-sweep diagnostic.

The sweep identifies the useful candidate-count range on Mip-NeRF 360 `bonsai`: caps `256` and `512` commit, while cap `1024` rolls back all attempts. Cap `512` is the best current topology-quality Pareto row among the tested values.

## Goal

After cap512 made `bonsai` accept a PRISM edit but introduced a PSNR tradeoff, run a small cap sweep to determine whether the useful threshold is lower or higher.

## Runs

All runs used online W&B, M28 adaptive schedule, `ratio0p02_geom1400`, and independent `render.py + metrics.py`.

- cap256 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mzglj2qw`
- cap512 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ck157wtl`
- cap1024 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j5v0debo`

Command logs:

- `outputs/carnet/meshprior/stage29_candidate_selection/mipnerf360_bonsai_cap256_adaptive_ratio0p02_geom1400_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage29_candidate_selection/mipnerf360_bonsai_cap512_adaptive_ratio0p02_geom1400_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage29_candidate_selection/mipnerf360_bonsai_cap1024_adaptive_ratio0p02_geom1400_2000iter/logs/train_command.txt`

## Results

| cap | final triangles | final vertices | PRISM decisions | train PSNR / SSIM / LPIPS | independent PSNR / SSIM / LPIPS |
|---:|---:|---:|---|---|---|
| `none` / M28 | `1357119` | `1641759` | `0` commit, `8` rollback | `24.1378 / 0.8285 / 0.2043` | `12.3054 / 0.2410 / 0.6196` |
| `256` | `634043` | `1047330` | `1` commit, `41` no-candidate | `24.1060 / 0.8275 / 0.2048` | `12.1430 / 0.2753 / 0.6134` |
| `512` | `633787` | `1047113` | `1` commit, `41` no-candidate | `24.0933 / 0.8269 / 0.2056` | `12.1859 / 0.2764 / 0.6129` |
| `1024` | `1357128` | `1641841` | `0` commit, `8` rollback | `24.1338 / 0.8294 / 0.2019` | `12.2882 / 0.2398 / 0.6211` |

M26 sparse-depth baseline reference:

- final W&B triangles: `1357104`
- independent PSNR `12.2016`, SSIM `0.2073`, LPIPS `0.6243`

## Candidate-Gate Evidence

Cap256:

- iteration `1501`, selected `256`, accepted
- counterfactual deltas: PSNR `-0.0049`, MAE `+0.000032`, AbsRel `+0.000041`, changed-pixel ratio `0.00133`
- topology: `634299 -> 634043`

Cap512:

- iteration `1501`, selected `512`, accepted
- counterfactual deltas: PSNR `-0.0140`, MAE `+0.000072`, AbsRel `+0.000115`, changed-pixel ratio `0.00251`
- topology: `634299 -> 633787`

Cap1024:

- iteration `1501`, selected `1024`, rejected
- counterfactual deltas: PSNR `-0.0388`, MAE `+0.000198`, AbsRel `+0.000209`, changed-pixel ratio `0.00616`
- later adaptive retries at cap1024 also reject; final topology remains near M28/no-cap.

## Interpretation

The useful cap range for `bonsai` is below `1024` candidates under the current gate. The threshold is sharp: `512` candidates pass with a changed-pixel ratio `0.00251`, while `1024` candidates exceed the changed-pixel threshold with `0.00616` and roll back.

Cap512 is better than cap256 on all independent metrics while keeping essentially the same large topology reduction. It is the current best M29 row for `bonsai`.

The remaining issue is not whether capping works. It does. The issue is how to recover PSNR while keeping the topology reduction. The likely next method step is microbatch candidate gating, because cap1024 contains useful candidates but the whole 1024 batch is too risky as a single edit.

## Decision

M29 cap sweep is a `PASS` diagnostic.

Best current row:

- `bonsai` cap512: final `633787` triangles, independent PSNR `12.1859`, SSIM `0.2764`, LPIPS `0.6129`

This is a strong Pareto point, not a final headline default.

## Next Step

Implement microbatch candidate gating:

1. select a larger candidate pool, e.g. cap `1024`;
2. split it into microbatches of `128` or `256`;
3. run counterfactual gates per microbatch;
4. commit only accepted microbatches;
5. compare against cap512 on `bonsai` and M28 on `courtyard`.

