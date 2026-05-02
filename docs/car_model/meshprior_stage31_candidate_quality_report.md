# Stage31 PRISM Candidate-Quality Ranking Report

Date: 2026-05-02

## Status

`SOFT PASS / diagnostic PASS`.

Stage31 adds a default-off candidate-quality ranking path for PRISM candidate pruning. The implementation is stable, logged to W&B, and validated by smoke plus two public-scene medium runs. It is not promoted as the default because it does not robustly beat Stage29 cap512 on both public scenes.

## Code Changes

- `arguments/__init__.py`
  - Added default-off candidate-ranking flags:
    - `--prism_candidate_quality_rank`
    - `--prism_candidate_quality_prune_weight`
    - `--prism_candidate_quality_render_penalty`
    - `--prism_candidate_quality_geometry_penalty`
    - `--prism_candidate_quality_orientation_penalty`
    - `--prism_candidate_quality_utility_penalty`
    - `--prism_candidate_quality_uncertainty_penalty`
- `utils/prism_counterfactual.py`
  - `select_prism_candidate_ids` can now select by an optional rank score tensor instead of raw prune score.
- `train.py`
  - Computes the opt-in rank score:

```text
rank = prune_weight * prune_score
     - render_penalty * render_keep
     - geometry_penalty * geometry_keep
     - orientation_penalty * orientation_keep
     - utility_penalty * utility
     - uncertainty_penalty * uncertainty
```

  - Sorts selected candidates by the quality rank when enabled.
  - Logs candidate-quality means to TensorBoard/W&B and PRISM round metadata.

## Validation

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile train.py arguments/__init__.py utils/prism_counterfactual.py
git diff --check
```

Selector sanity check:

- raw prune-score top5 selected `{5, 6, 7, 8, 9}`
- rank-score flipped top5 selected `{0, 1, 2, 3, 4}`

GPU choice:

- render/metrics used GPU1; before post-eval it had about `4046 / 49140 MiB` allocated and `0%` GPU utilization.

## Runs

### Parking Smoke

- Output: `outputs/carnet/meshprior/stage31_candidate_quality/parking_quality_rank_cap512_smoke_140iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ucqyou26`
- Result: accepted at iter `91`, committed `64497 -> 63985`.
- Candidate-quality metadata at commit:
  - quality score mean `0.3549446`
  - prune score mean `0.6299592`
  - render keep mean `0.0`
  - geometry keep mean `0.0`
  - orientation keep mean `1.0`
  - utility mean `0.1000583`
  - uncertainty mean `6.91e-13`

### Mip-NeRF 360 `bonsai`

- Output: `outputs/carnet/meshprior/stage31_candidate_quality/mipnerf360_bonsai_quality_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/22r3et7s`
- Commit: iter `1501`, `634299 -> 633787`, selected `512`.
- Training internal final test:
  - PSNR `24.1192`
  - SSIM `0.8278`
  - LPIPS `0.2035`
- Independent `render.py + metrics.py`:
  - PSNR `12.1891`
  - SSIM `0.2756`
  - LPIPS `0.6136`
- Comparison to M29 cap512:
  - M29 cap512: PSNR `12.1859`, SSIM `0.2764`, LPIPS `0.6129`, triangles `633787`
  - M31 quality-rank: PSNR `12.1891`, SSIM `0.2756`, LPIPS `0.6136`, triangles `633787`

### ETH3D `courtyard`

- Output: `outputs/carnet/meshprior/stage31_candidate_quality/eth3d_courtyard_quality_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xt4a2cn0`
- Commit: iter `1501`, `102916 -> 102404`, selected `512`.
- Training internal final test:
  - PSNR `19.5364`
  - SSIM `0.6489`
  - LPIPS `0.4413`
- Independent `render.py + metrics.py`:
  - PSNR `15.0732`
  - SSIM `0.4837`
  - LPIPS `0.5788`
- Comparison to M29 cap512:
  - M29 cap512: PSNR `15.0344`, SSIM `0.4812`, LPIPS `0.5804`, triangles `102916`
  - M31 quality-rank: PSNR `15.0732`, SSIM `0.4837`, LPIPS `0.5788`, triangles `102404`

## Decision

Stage31 is a diagnostic success but not a method promotion.

The candidate-quality score clearly changes the chosen set and improves `courtyard` on independent metrics while keeping lower topology. On `bonsai`, it only gives a tiny PSNR gain and regresses SSIM/LPIPS, with identical final topology to M29 cap512. That fails the M31 promotion gate, which required matching or improving cap512 independent metrics on both public scenes.

The next method step should not be another hand-weighted proxy. The likely bottleneck is that proxy ranking is still too indirect. M32 should use a larger candidate pool, evaluate candidate subsets on calibration views with the existing counterfactual machinery, then rank or choose the final cap512 set from measured impact rather than only from local tensors.

## Gate

`SOFT PASS / diagnostic PASS`.

Keep Stage29 cap512 as the conservative default. Keep Stage31 behind `--prism_candidate_quality_rank`.
