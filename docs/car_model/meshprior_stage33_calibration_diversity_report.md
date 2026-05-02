# Stage33 PRISM Calibration Diversity Report

Date: 2026-05-02

## Summary

Stage33 adds default-off view-diverse PRISM calibration diagnostics. The counterfactual gate can now seed calibration from evenly spaced held-out/test cameras and evenly spaced train cameras before adding hard train views, records the chosen view manifest, and writes per-view counterfactual deltas.

This is a `SOFT PASS / diagnostic PASS`. It improves the `bonsai` measured-rank result enough to exceed Stage29 cap512 on independent metrics at the same final topology, and it keeps `courtyard` better than Stage29 cap512. It does not replace Stage32 for `courtyard`, because Stage32 still has the best `courtyard` independent PSNR/SSIM row.

## Code Changes

- `arguments/__init__.py`
  - Added default-off calibration-diversity flags:
    - `--prism_calib_diverse_views`
    - `--prism_calib_diverse_test_views`
    - `--prism_calib_diverse_train_views`
- `utils/prism_counterfactual.py`
  - Added evenly spaced calibration view selection.
  - Added calibration-view manifest construction.
  - Added per-view counterfactual deltas to debug JSON.
- `train.py`
  - Plumbed the new flags into PRISM state and `CalibrationConfig`.
  - Writes `prism_debug/calibration_views.json` when `--prism_save_debug_json` is enabled.

All new behavior is opt-in; defaults are unchanged.

## Runs

### Parking Smoke

- Output: `outputs/carnet/meshprior/stage33_calibration_diversity/parking_diverse_calib_measured_rank_cap512_smoke_140iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ms95810g`
- GPU: `1`
- Key flags: measured candidate-impact rank, cap512, diverse calibration, `4` diverse test views, `4` diverse train views, `--prism_save_debug_json`.
- Result: all candidate attempts rolled back under strict smoke thresholds; final topology stayed `64497` triangles.
- Interpretation: this is useful. Diverse views caught small per-view degradations that the less diverse smoke setup did not expose.

Calibration manifest:

- `10` views total: `4` diverse test, `4` diverse train, `2` hard train.
- Worst visible per-view deltas in the last gate included `images_00196` (`delta_psnr -0.0063`) and `images_00058` (`delta_psnr -0.0063`).

### Mip-NeRF 360 `bonsai`

- Dataset: `/data/peilincai/mesh_datasets/mipnerf360/bonsai`
- Output: `outputs/carnet/meshprior/stage33_calibration_diversity/mipnerf360_bonsai_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kg5htc8u`
- GPU: `1`
- Calibration manifest: `24` views total, including `12` diverse test, `4` diverse train, and `8` hard train.
- PRISM decision: iter `1501` committed one cap512 measured-rank edit, `634299 -> 633787`; later candidate checks selected `0`.
- Final W&B/final-checkpoint topology: `633787` triangles.
- Independent `render.py + metrics.py`: PSNR `12.1999207`, SSIM `0.2765326`, LPIPS `0.6125830`.

Comparison:

| row | final triangles | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Stage29 cap512 | `633787` | `12.1859255` | `0.2763788` | `0.6129207` |
| Stage32 measured rank | `633787` | `12.1742411` | `0.2758206` | `0.6137036` |
| Stage33 diverse calibration | `633787` | `12.1999207` | `0.2765326` | `0.6125830` |

Stage33 exceeds Stage29 cap512 on all three independent render metrics with equal topology on `bonsai`.

### ETH3D `courtyard`

- Dataset: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`
- Output: `outputs/carnet/meshprior/stage33_calibration_diversity/eth3d_courtyard_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w9c0b65f`
- GPU: `1`
- Calibration manifest: `17` views total, including `5` diverse test, `4` diverse train, and `8` hard train.
- PRISM decision: iter `1501` committed one cap512 measured-rank edit, `102919 -> 102407`; later candidate checks selected `0`.
- Final W&B/final-checkpoint topology: `102407` triangles.
- Independent `render.py + metrics.py`: PSNR `15.0737228`, SSIM `0.4840090`, LPIPS `0.5789739`.

Comparison:

| row | final triangles | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Stage29 cap512 | `102916` | `15.0344` | `0.4812` | `0.5804` |
| Stage32 measured rank | `102404` | `15.1390` | `0.4850` | `0.5792` |
| Stage33 diverse calibration | `102407` | `15.0737228` | `0.4840090` | `0.5789739` |

Stage33 is better than Stage29 cap512 on `courtyard`, but worse than Stage32 measured rank on PSNR/SSIM.

## Decision

`SOFT PASS / diagnostic PASS`.

Stage33 should stay opt-in. It validates the hypothesis that calibration-set representativeness matters: `bonsai` improves over both Stage29 and Stage32 without losing topology, and the parking smoke shows that diverse calibration catches view-local regressions. However, it is not a universal promotion because `courtyard` regresses from Stage32 measured rank on PSNR/SSIM.

Current practical default:

- Use Stage29 cap512 as the conservative topology-quality baseline.
- Use Stage32 measured rank when optimizing `courtyard`-like scenes.
- Use Stage33 diverse calibration for diagnosis, safety checks, and `bonsai`-like scenes where local calibration was previously misleading.

## Next Step

The next bottleneck is no longer only candidate ranking. The candidate pool often becomes empty after one cap512 edit, which means the controller lacks a second-stage search mechanism. M34 should add a controlled post-commit candidate refresh / relaxed low-risk candidate discovery path, then test whether a second useful edit can be found without hurting the Stage33 `bonsai` metric gains.
