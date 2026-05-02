# MeshPrior Stage28 Adaptive Schedule Smoke Report

Date: 2026-05-02

## Gate

`PASS` for implementation smoke.

This stage adds an opt-in adaptive PRISM candidate retry path. It does not change default training behavior. The smoke confirms that rollback can reduce the candidate prune ratio and retry without consuming the candidate round immediately, while W&B and candidate metadata record the active ratio and candidate counts.

## Motivation

M27 showed that fixed PRISM schedules are not cross-scene robust:

- ETH3D `courtyard` worked well with `ratio0p02_geom1400`.
- Mip-NeRF 360 `bonsai` rolled back all six 2% candidate edits.

The immediate design change is to stop treating a rollback at one prune ratio as a final schedule decision. Instead, the controller can retry with a smaller candidate ratio before consuming the effective candidate round.

## Code Changes

Files:

- `arguments/__init__.py`
- `train.py`

New opt-in flags:

- `--prism_adaptive_candidate_retry_on_rollback`
- `--prism_adaptive_candidate_ratio_decay`
- `--prism_adaptive_candidate_min_ratio`
- `--prism_adaptive_candidate_max_rollback_retries`

New metadata / W&B fields:

- `prism/adaptive_candidate_prune_ratio`
- `prism/adaptive_candidate_rollback_retries`
- `prism/last_candidate_pool_count`
- `prism/last_candidate_target_count`
- `prism/last_candidate_selected_count`
- per-round JSON fields: `candidate_prune_ratio`, `candidate_pool_count`, `candidate_target_count`, `candidate_selected_count`

Behavior:

1. Default behavior is unchanged because adaptive retry is disabled by default.
2. When enabled, a candidate rollback reduces the active candidate ratio by `ratio_decay`, clamps it at `min_ratio`, and schedules a retry after `prism_no_candidate_retry_iters`.
3. After `max_rollback_retries`, the controller consumes the candidate round normally.
4. A successful candidate commit or non-rollback candidate result resets the adaptive ratio to the configured base ratio.

## Verification

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile train.py arguments/__init__.py utils/prism_pipeline.py
```

Smoke GPU:

- GPU: `1`
- W&B run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1kmwbu8g`
- Output: `outputs/carnet/meshprior/stage28_adaptive_schedule/parking_adaptive_retry_smoke_v5_140iter/model`

Important command choices:

- dataset: `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- iterations: `140`
- base candidate ratio: `0.04`
- adaptive decay: `0.5`
- min ratio: `0.005`
- max rollback retries: `2`
- `--prism_recent_age_iters 0` for this short smoke so fresh triangles are not all protected by age.

## Smoke Result

Candidate metadata:

| iteration | ratio | pool | selected | rollback | no candidates | adaptive retries |
|---:|---:|---:|---:|---:|---:|---:|
| `81` | `0.04` | `30632` | `2579` | `1` | `0` | `1` |
| `86` | `0.02` | `33723` | `1289` | `1` | `0` | `2` |
| `91` | `0.01` | `35621` | `644` | `1` | `0` | `2` |
| `92` | `0.01` | `35621` | `644` | `1` | `0` | `2` |

Final checkpoint / accounting:

- final triangles: `64497`
- final vertices: `193491`
- final cleanup executed: `false`
- W&B `mesh/final_checkpoint_triangle_count`: `64497`
- final cleanup `post_prune_triangle_count`: `64497`

Training smoke metrics:

- test PSNR: `10.876985`
- test SSIM: `0.315548`
- test LPIPS: `0.640531`

The strict gate intentionally forced rollbacks. The key result is not quality; it is that rollback-driven ratio decay is observable and auditable.

## Diagnostics

Earlier smokes also found that short candidate-path tests can be misleading if `prism_recent_age_iters` remains at the default. With recent-age protection enabled, all fresh short-smoke triangles are protected and candidate pools stay at zero. For real 2000-iteration public-scene runs this is less of a problem, but smoke commands must set `--prism_recent_age_iters 0` when the goal is to exercise candidate rollback logic early.

## Decision

M28 implementation smoke is a `PASS`.

The code now supports adaptive candidate-ratio retry and logs the evidence needed to compare it against fixed M27 schedules. This does not yet prove cross-scene improvement. The next M28 work should run medium public-scene ablations:

1. `bonsai` adaptive retry starting from `0.02`, decay `0.5`, min `0.005`.
2. `courtyard` same adaptive retry to verify the strong M27 ETH3D result is preserved.
3. independent `render.py + metrics.py` for both.
4. gate against M27 `ratio0p02_geom1400`.

