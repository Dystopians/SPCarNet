# MeshPrior Stage29 Candidate Cap Medium Report

Date: 2026-05-02

## Gate

`SOFT PASS`

M29 candidate capping solves the immediate M28 `bonsai` rollback failure: a cap of `512` makes `bonsai` accept a candidate edit and cuts final topology by more than half versus M28. The result is not a full pass because PSNR drops on `bonsai`, and ETH3D `courtyard` no longer preserves the exact M27/M28 best final topology.

## Goal

Test whether limiting candidate prune edits to a small fixed count can make dense public scenes accept PRISM topology edits without relying only on global candidate-ratio decay.

## Data And Commands

Scenes:

- Mip-NeRF 360 `bonsai`: `/data/peilincai/mesh_datasets/mipnerf360/bonsai`, images `images_4`, resolution `4`
- ETH3D `courtyard`: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`, images `images`, resolution `8`

Both runs used:

- `WANDB_MODE=online`
- `--enable_wandb --wandb_project spcarnet_meshprior --wandb_group m29_candidate_selection`
- M28 adaptive schedule: base ratio `0.02`, decay `0.5`, min ratio `0.005`, max rollback retries `2`
- `--prism_candidate_max_count_per_round 512`
- `--prism_freeze_densification_after_first_commit`
- independent `render.py + metrics.py`

Command logs:

- `outputs/carnet/meshprior/stage29_candidate_selection/mipnerf360_bonsai_cap512_adaptive_ratio0p02_geom1400_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage29_candidate_selection/eth3d_courtyard_cap512_adaptive_ratio0p02_geom1400_2000iter/logs/train_command.txt`

## W&B Runs

- `bonsai` cap512: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ck157wtl`
- `courtyard` cap512: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1ey4qzbd`

## Results

Training metrics are the in-training test split. Independent metrics are from post-training `render.py + metrics.py`.

| scene | schedule | final triangles | final vertices | PRISM decisions | train PSNR / SSIM / LPIPS | independent PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|---|---|
| `bonsai` | M28 adaptive no cap | `1357119` | `1641759` | `0` commit, `8` rollback | `24.1378 / 0.8285 / 0.2043` | `12.3054 / 0.2410 / 0.6196` |
| `bonsai` | M29 adaptive cap512 | `633787` | `1047113` | `1` commit, `41` no-candidate | `24.0933 / 0.8269 / 0.2056` | `12.1859 / 0.2764 / 0.6129` |
| `courtyard` | M28 adaptive no cap | `100858` | `168239` | `1` commit, `41` no-candidate | `19.5691 / 0.6500 / 0.4375` | `15.0919 / 0.4844 / 0.5778` |
| `courtyard` | M29 adaptive cap512 | `102916` | `169890` | `1` commit, `41` no-candidate | `19.5330 / 0.6470 / 0.4408` | `15.0344 / 0.4812 / 0.5804` |

Aligned sparse-depth baseline references from M26:

- `bonsai`: independent PSNR `12.2016`, SSIM `0.2073`, LPIPS `0.6243`; W&B triangles `1357104`
- `courtyard`: independent PSNR `14.9462`, SSIM `0.4388`, LPIPS `0.5924`; W&B triangles `220339`

## Candidate Evidence

`bonsai`:

- iteration `1501`
- ratio target: `12685`
- candidate pool: `28536`
- cap target: `512`
- selected: `512`
- counterfactual gate: accepted
- counterfactual delta: PSNR `-0.0140`, MAE `+0.000072`, AbsRel `+0.000115`, changed-pixel ratio `0.00251`
- topology: `634299 -> 633787`, final checkpoint `633787`

`courtyard`:

- iteration `1501`
- ratio target: `2058`
- candidate pool: `58707`
- cap target: `512`
- selected: `512`
- counterfactual gate: accepted
- counterfactual delta: PSNR `-0.0004`, MAE `+0.000006`, AbsRel `+0.000019`, changed-pixel ratio `0.000324`
- immediate topology: `102916 -> 102404`
- final checkpoint topology: `102916`

## Interpretation

This is the first public-scene run where the hard `bonsai` scene accepts an integrated PRISM topology edit. That is a real method improvement over M27 and M28. The cap changes the failure mode from "all `bonsai` candidates roll back" to "a small local edit passes the gate and strongly lowers final topology."

The tradeoff is visible. On `bonsai`, final triangles drop from `1357119` to `633787`, SSIM improves from `0.2410` to `0.2764`, and LPIPS improves from `0.6196` to `0.6129`; however independent PSNR drops from `12.3054` to `12.1859`, slightly below the M26 sparse-depth baseline PSNR. This is likely acceptable as a Pareto row, not as the single headline method setting.

On `courtyard`, cap512 is still better than the sparse-depth baseline, but it is worse than M27/M28 no-cap adaptive. The immediate accepted edit prunes `512` triangles, but final topology returns to `102916`; this suggests a topology-retention or recovery-window issue after small accepted edits. It is not an accounting mismatch: final cleanup and W&B final checkpoint counts agree.

## Decision

M29 medium ablation is a `SOFT PASS`.

It confirms the correct next direction, but the current cap value is not the final schedule. The best paper story now has a real topology-quality Pareto point for `bonsai`, plus a clear remaining issue: preserve or improve render PSNR while keeping the large topology reduction.

## Next Step

Do not jump directly to full-budget training with cap512 as the default. The next M29 substep should be a small cap sweep:

1. `bonsai`: compare cap `256`, `512`, and `1024` at the same 2000-iteration budget.
2. `courtyard`: rerun the best `bonsai` cap and check whether topology retention remains stable.
3. If cap sweep cannot improve the tradeoff, implement microbatch candidate gating so selected candidates are split into smaller counterfactual-tested batches and only accepted batches are committed.
4. Add a topology-retention diagnostic for the `courtyard` case where immediate post-commit topology is `102404` but final checkpoint returns to `102916`.

