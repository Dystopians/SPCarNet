# Stage30 PRISM Microbatch Candidate Gate Report

Date: 2026-05-02

## Status

`SOFT PASS / diagnostic PASS`.

Stage30 adds opt-in microbatch counterfactual gating for PRISM candidate pruning. The mechanism works and is auditable: a large selected candidate set can be split into smaller batches, each batch can be tested cumulatively, and only accepted batches are committed. The current `1024 x 256` policy is not promoted as the default because it does not beat the Stage29 cap512 Pareto row on independent metrics.

## Code Changes

- `arguments/__init__.py`
  - added `--prism_candidate_microbatch_gate`
  - added `--prism_candidate_microbatch_size`
  - added `--prism_candidate_microbatch_max_batches`
- `train.py`
  - added microbatch gate config plumbing,
  - preserves default behavior when microbatch gating is disabled,
  - orders candidate IDs by prune score before microbatch splitting,
  - writes per-microbatch counterfactual JSON files,
  - writes microbatch counts into PRISM round metadata,
  - logs microbatch counters to W&B/TensorBoard.

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile train.py arguments/__init__.py utils/prism_counterfactual.py
git diff --check
```

Both passed.

## Smoke

Output:

- `outputs/carnet/meshprior/stage30_microbatch_gate/parking_microbatch_gate_smoke_1024x256_140iter/model`

W&B:

- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dioe1cz1`

Key metadata:

- iter `81`: selected `1024`, microbatches `4`, accepted `0`, rejected `4`, rollback, topology `64497 -> 64497`.
- iter `86`: selected `1024`, microbatches `4`, accepted `0`, rejected `4`, rollback, topology `64497 -> 64497`.
- iter `91`: selected `644`, microbatches `3`, accepted `3`, rejected `0`, commit, topology `64497 -> 63853`.

Gate: `PASS`. The new path emits per-microbatch JSON, W&B counters, round metadata, and a valid final checkpoint.

## Medium Ablation

All medium runs used online W&B, sparse COLMAP depth loss, adaptive retry, freeze-after-first-commit, and independent `render.py + metrics.py`.

### Mip-NeRF 360 `bonsai`

Output:

- `outputs/carnet/meshprior/stage30_microbatch_gate/mipnerf360_bonsai_microbatch1024x256_adaptive_ratio0p02_geom1400_2000iter/model`

W&B:

- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mfvhexjb`

PRISM decision:

- iter `1501`: pool `28766`, target `12685`, cap `1024`, selected `1024`.
- microbatches: `4`.
- accepted: `3`.
- rejected: `1`.
- accepted triangles: `768`.
- topology: `634299 -> 633531`.

Gate deltas by cumulative microbatch:

| cumulative candidates | accept | delta PSNR | changed pixels |
|---:|---|---:|---:|
| 256 | yes | `-0.00901` | `0.00161` |
| 512 | yes | `-0.01400` | `0.00278` |
| 768 | yes | `-0.02681` | `0.00402` |
| 1024 | no | `-0.04051` | `0.00535` |

Metrics:

| row | final triangles | internal test PSNR | independent PSNR | independent SSIM | independent LPIPS |
|---|---:|---:|---:|---:|---:|
| M29 cap512 | `633787` | `24.0933` | `12.1859` | `0.2764` | `0.6129` |
| M30 microbatch1024x256 | `633531` | `24.1108` | `12.1423` | `0.2770` | `0.6136` |

Interpretation: microbatch recovers useful partial edits from the failed cap1024 row and deletes `256` more triangles than cap512, but independent PSNR and LPIPS are worse than cap512. This is a mechanism win, not a new best row.

### ETH3D `courtyard`

Output:

- `outputs/carnet/meshprior/stage30_microbatch_gate/eth3d_courtyard_microbatch1024x256_adaptive_ratio0p02_geom1400_2000iter/model`

W&B:

- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ha9qi1ih`

PRISM decision:

- iter `1501`: pool `58541`, target `2058`, cap `1024`, selected `1024`.
- microbatches: `4`.
- accepted: `4`.
- rejected: `0`.
- accepted triangles: `1024`.
- topology: `102919 -> 101895`.

Metrics:

| row | final triangles | internal test PSNR | independent PSNR | independent SSIM | independent LPIPS |
|---|---:|---:|---:|---:|---:|
| M28 adaptive | `100858` | n/a | `15.0919` | `0.4844` | `0.5778` |
| M29 cap512 | `102916` | `19.5330` | `15.0344` | `0.4812` | `0.5804` |
| M30 microbatch1024x256 | `101895` | `19.6066` | `15.0635` | `0.4828` | `0.5802` |

Interpretation: microbatch improves over M29 cap512 on both topology and independent metrics, but it still does not beat the M28/M27 best `courtyard` row.

## Decision

Stage30 should stay opt-in. It is useful when diagnosing an over-large candidate set because it tells us how much of the set the counterfactual gate can tolerate. It should not replace the current Stage29 cap512 row as the default schedule.

Recommended next step:

1. Keep cap512 as the current conservative default for dense public-scene PRISM ablations.
2. Use microbatch JSON as a diagnostic to learn a better candidate scoring rule. The `bonsai` 768-accepted row shows the gate tolerates cumulative PSNR loss beyond cap512 but this does not translate to better independent quality.
3. Add candidate-quality calibration: prefer candidates that are low render-impact across calibration views, not only high prune score. A practical M31 target is to rank candidates by a blended score that includes counterfactual/pixel-impact proxies before the cap or microbatch split.

## Gate

`SOFT PASS / diagnostic PASS`.

The implementation is stable and verified with smoke plus two public-scene medium runs. The method does not yet improve the strongest cross-scene Pareto row, so the paper-facing schedule remains unresolved.
