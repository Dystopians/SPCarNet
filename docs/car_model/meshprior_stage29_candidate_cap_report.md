# MeshPrior Stage29 Candidate Cap Report

Date: 2026-05-02

## Gate

`PASS` for implementation smoke.

Stage29 adds an opt-in per-round cap for PRISM candidate pruning. Default behavior is unchanged because the cap defaults to `0`, meaning disabled. The parking smoke confirms that the cap is applied after the ratio target is computed and before counterfactual mutation, and that metadata/W&B record the ratio target, cap-limited target, and selected count.

## Motivation

M28 showed that adaptive ratio decay is not enough on dense scenes. On Mip-NeRF 360 `bonsai`, the ratio decayed to `0.005`, but the global candidate selector still chose `3171` triangles and the counterfactual gate rejected the edit. M29 therefore reduces candidate edit granularity directly.

## Code Changes

Files:

- `arguments/__init__.py`
- `utils/prism_counterfactual.py`
- `train.py`

New flag:

- `--prism_candidate_max_count_per_round`

Behavior:

1. `0` preserves legacy behavior.
2. Positive values cap candidate-prune selection after the ratio-derived target count is computed.
3. The cap applies only to `candidate` PRISM pruning, not dead-triangle pruning.
4. Metadata and W&B include:
   - `candidate_target_count`
   - `candidate_cap_count`
   - `candidate_selected_count`
   - `prism/last_candidate_target_count`
   - `prism/last_candidate_cap_count`
   - `prism/last_candidate_selected_count`

## Verification

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile train.py arguments/__init__.py utils/prism_counterfactual.py
```

Selector check:

- uncapped `20%` request on `100` toy triangles selects `20`
- capped request with `candidate_max_count=7` selects `7`
- capped set is a subset of the uncapped top-ranked set

Smoke GPU:

- GPU: `1`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rgvzhx6k`
- Output: `outputs/carnet/meshprior/stage29_candidate_selection/parking_candidate_cap_smoke_256_140iter/model`
- Command log: `outputs/carnet/meshprior/stage29_candidate_selection/parking_candidate_cap_smoke_256_140iter/logs/train_command.txt`

## Smoke Result

The smoke uses base ratio `0.04`, adaptive retry, strict counterfactual thresholds, and `--prism_candidate_max_count_per_round 256`.

| iteration | ratio | pool | ratio target | cap target | selected | gate | triangles |
|---:|---:|---:|---:|---:|---:|---|---|
| `81` | `0.04` | `64497` | `2579` | `256` | `256` | rollback | `64497 -> 64497` |
| `86` | `0.02` | `64497` | `1289` | `256` | `256` | rollback | `64497 -> 64497` |
| `91` | `0.01` | `64497` | `644` | `256` | `256` | commit | `64497 -> 64241` |

Counterfactual metadata:

- iter `81`: `256` candidates, rejected; PSNR delta `-0.3444`, changed-pixel ratio `0.3496`
- iter `86`: `256` candidates, rejected; PSNR delta `-0.3522`, changed-pixel ratio `0.3525`
- iter `91`: `256` candidates, accepted; PSNR delta near zero, changed-pixel ratio `0.0`

Final accounting:

- final triangles: `64241`
- final vertices: `193491`
- final cleanup executed: `false`
- W&B `mesh/final_checkpoint_triangle_count`: `64241`
- W&B `prism/last_candidate_target_count`: `644`
- W&B `prism/last_candidate_cap_count`: `256`
- W&B `prism/last_candidate_selected_count`: `256`

## Decision

M29 implementation smoke is a `PASS`.

The cap is active, auditable, default-neutral, and able to turn the short parking smoke from repeated large rejected edits into a small accepted candidate edit. This is not yet a cross-scene quality result.

## Next Step

Run a medium public-scene ablation against M28:

1. Mip-NeRF 360 `bonsai`, M28 adaptive schedule plus `--prism_candidate_max_count_per_round 512` or `1024`.
2. ETH3D `courtyard`, same settings to verify the strong topology result is preserved.
3. Online W&B, independent `render.py + metrics.py`, PRISM validation, counterfactual JSON, and final-checkpoint accounting are required.

