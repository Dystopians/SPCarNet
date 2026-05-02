# Stage32 PRISM Measured Candidate-Impact Ranking Report

Date: 2026-05-02

## Status

`SOFT PASS / diagnostic PASS`.

Stage32 adds a default-off measured candidate-impact selector. The implementation is stable, W&B-logged, and validated by parking smoke plus public-scene medium runs. It is not promoted as the default because it improves ETH3D `courtyard` but does not match Stage29 cap512 on Mip-NeRF 360 `bonsai`.

## Code Changes

- `arguments/__init__.py`
  - Added:
    - `--prism_candidate_measured_impact_rank`
    - `--prism_candidate_measured_pool_multiplier`
    - `--prism_candidate_measured_group_size`
    - `--prism_candidate_measured_max_groups`
- `train.py`
  - When enabled, selects a larger candidate pool, splits it into deterministic groups, evaluates each group through the existing counterfactual calibration path, writes per-group JSON, then chooses the final cap-limited candidate set by measured impact score.
  - Added W&B/TensorBoard/round metadata:
    - `last_candidate_measured_rank_enabled`
    - `last_candidate_measured_group_count`
    - `last_candidate_measured_accepted_count`
    - `last_candidate_measured_selected_count`
    - `last_candidate_measured_best_score`

## Validation

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile train.py arguments/__init__.py utils/prism_counterfactual.py
git diff --check
```

GPU choice:

- parking smoke and Mip-NeRF 360 `bonsai`: GPU1
- ETH3D `courtyard`: GPU5

## Runs

### Parking Smoke

- Output: `outputs/carnet/meshprior/stage32_measured_candidate_rank/parking_measured_rank_cap512_smoke_140iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xg4fsvd8`
- Result: iter `91` committed `64497 -> 63985`.
- Measured-rank metadata:
  - iter `81`: `8` groups, `0` accepted, rollback.
  - iter `86`: `6` groups, `1` accepted, rollback.
  - iter `91`: `3` groups, `3` accepted, final selected `512`, commit.

### Mip-NeRF 360 `bonsai`

- Output: `outputs/carnet/meshprior/stage32_measured_candidate_rank/mipnerf360_bonsai_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/56l3tz23`
- Commit: iter `1501`, `634299 -> 633787`, selected `512`, `8/8` measured groups accepted.
- Training internal final test:
  - PSNR `24.0828`
  - SSIM `0.8275`
  - LPIPS `0.2056`
- Independent `render.py + metrics.py`:
  - PSNR `12.1742`
  - SSIM `0.2758`
  - LPIPS `0.6137`

### ETH3D `courtyard`

- Output: `outputs/carnet/meshprior/stage32_measured_candidate_rank/eth3d_courtyard_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fb7jfcaj`
- Commit: iter `1501`, `102916 -> 102404`, selected `512`, `8/8` measured groups accepted.
- Training internal final test:
  - PSNR `19.5740`
  - SSIM `0.6497`
  - LPIPS `0.4382`
- Independent `render.py + metrics.py`:
  - PSNR `15.1390`
  - SSIM `0.4850`
  - LPIPS `0.5792`

### `bonsai` Measured + Quality Diagnostic

- Output: `outputs/carnet/meshprior/stage32_measured_candidate_rank/mipnerf360_bonsai_measured_quality_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xooe27um`
- Commit: iter `1501`, final `633787` triangles.
- Independent `render.py + metrics.py`:
  - PSNR `12.1708`
  - SSIM `0.2760`
  - LPIPS `0.6133`

## Comparison

Stage29 cap512 remains the conservative default:

- `bonsai` M29 cap512: PSNR `12.1859`, SSIM `0.2764`, LPIPS `0.6129`, triangles `633787`
- `bonsai` M32 measured: PSNR `12.1742`, SSIM `0.2758`, LPIPS `0.6137`, triangles `633787`
- `bonsai` M32 measured+quality: PSNR `12.1708`, SSIM `0.2760`, LPIPS `0.6133`, triangles `633787`

Stage32 improves `courtyard`:

- `courtyard` M29 cap512: PSNR `15.0344`, SSIM `0.4812`, LPIPS `0.5804`, triangles `102916`
- `courtyard` M31 quality-rank: PSNR `15.0732`, SSIM `0.4837`, LPIPS `0.5788`, triangles `102404`
- `courtyard` M32 measured: PSNR `15.1390`, SSIM `0.4850`, LPIPS `0.5792`, triangles `102404`

## Decision

Stage32 is a diagnostic success but not a method promotion.

Measured group ranking is useful infrastructure and gives the best `courtyard` PSNR/SSIM so far at the same low topology. However, the `bonsai` degradation means the measured score is not robust enough as a default policy. The failure mode is informative: all measured groups can pass the local counterfactual gate while still producing a worse independent test-set Pareto row. This means the next step should improve calibration-set representativeness and candidate diversity, not simply add more local group scoring.

## Gate

`SOFT PASS / diagnostic PASS`.

Keep Stage29 cap512 as the conservative default. Keep Stage31 and Stage32 flags opt-in.
