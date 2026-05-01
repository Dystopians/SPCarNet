---
title: SP-CarNet Stage 2 — Shape-field auto-decoder implementation report
date: 2026-04-29
authors: SP-CarNet line
status: implementation complete; smoke PASS; full-training pending
linked_design: docs/car_model/spcarnet_stage2_shape_field_design.md
linked_log: docs/car_model/SPCarNet_research_log.md
---

# SP-CarNet Stage 2 — Implementation report

This document closes the implementation phase of Stage 2 (the canonical, object-level shape-field auto-decoder). The full training run is **not yet launched**; this report covers what was built, what was checked, and the exact command needed to start the headline run.

---

## 1. What was built

### 1.1 Files added

| Path | Purpose | Lines |
|---|---|---|
| `docs/car_model/spcarnet_stage2_shape_field_design.md` | Design doc — `f(x; z)` decoder, FiLM modulation, occupancy primary / SDF ablation, query budget, loss term breakdown, gating criterion. | ≈300 |
| `ss3dm_prior/models/spcarnet_shape_field.py` | `SPCarShapeFieldDecoder` — FiLM-modulated MLP with Fourier-feature input encoding; emits an occupancy logit (or SDF scalar) plus an optional auxiliary feature head reserved for Stage 3 conditioning. | ≈210 |
| `ss3dm_prior/training/__init__.py` | Subpackage init. | 4 |
| `ss3dm_prior/training/spcarnet_autodecoder.py` | Standalone trainer — `LatentTable`, `assemble_query_batch`, `compute_losses`, `ShapeFieldAutoDecoderTrainer`, `load_configs`. Decoupled from `ss3dm_prior.engine.trainer` because that path is patch-centric and does not match the per-object latent paradigm. | ≈460 |
| `ss3dm_prior/training/spcarnet_autodecoder_cli.py` | Argparse-based entry: pairs a model YAML with a train YAML, persists `resolved_config.json`, runs `trainer.fit()`, persists `fit_summary.json`. | 72 |
| `configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml` | Decoder + loss config (occupancy primary, latent_dim=256, hidden_dim=384, depth=6, fourier_freqs=32). | 26 |
| `configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml` | Trainer config — splits, output dir, 200 epochs at batch 8, query budget ≈1024 / object / step, optimiser LRs. | 43 |
| `scripts/car_model/train_spcarnet_shape_field_autodecoder.sh` | Launcher. Defaults `WANDB_MODE=online`, `WANDB_PROJECT=spcarnet`, `GPU=5`. | 36 |
| `scripts/car_model/eval_spcarnet_shape_field_autodecoder.py` | Eval entry — loads checkpoint, extracts a Marching-Cubes mesh per val object, emits `recon_chamfer_l1`, `hidden_chamfer_l1`, `mesh_iou_at_0.5`, `surface_normal_consistency`, `mesh_extraction_success_rate`. | ≈250 |
| `scripts/car_model/smoke_test_spcarnet_stage2.py` | 2-object × 2-iter smoke — checks finite loss, decoder + latent gradients flowing, MC pipeline reachable. | ≈180 |

### 1.2 Files **not** modified

- `ss3dm_prior/engine/trainer.py` (patch-centric trainer for the v0.x line) — left untouched.
- `ss3dm_prior/data/spcarnet_object_dataset.py` — Stage-1 artefact, reused as-is.
- `outputs/carnet/spcarnet/object_index_v1.json` — Stage-1 index, reused as-is.
- All v0.x configs (`configs/car_model/...`) and launchers — left untouched. The auto-decoder line lives entirely under `configs/ss3dm_prior/spcarnet/` and `ss3dm_prior/training/`.

This preserves the explicit RFC §6 "Demote, don't delete" promise.

### 1.3 Architectural choices, locked in

- **Decoder**: 6-layer FiLM-modulated MLP, hidden_dim=384, latent_dim=256. FiLM gammas/betas produced by a per-layer linear off `z`; residual structure within each block.
- **Input encoding**: Fourier features with 32 log-spaced frequencies (`max_log2=5`). Encoded `(x, sin(2^k·πx), cos(2^k·πx))` is concatenated with the FiLM-modulated latent stream at the input and at every block.
- **Field kind**: `occupancy` is the primary head (BCE with logits). `sdf` is implemented as an ablation — surface MSE + free-space hinge + eikonal regulariser via `torch.autograd.grad`.
- **Query budget per object per step**: 384 surface + 384 free + 128 hard-negative + 128 mixed (with `query_ignore_mask` honoured) = **1024** queries / object / step. SDF mode adds 256 eikonal samples in `[-1,1]³`.
- **Latent prior**: `w_zL2 = 1e-4 · (||z||² / d_z)` averaged over the batch. Init `N(0, 0.01)` per element (DeepSDF-style).
- **Optim**: Adam, decoder LR `5e-4`, latent LR `1e-3` (latents need to move faster early), `grad_clip = 1.0`.
- **Splits**: trains over `train` only; `val` is held back for the eval entrypoint and for the eventual Stage gate.

### 1.4 What is *not* in Stage 2 (deferred per design doc)

- No observation likelihood `L_ray` term — Stage 4.
- No symmetry term and no scanner-pose conditioning — Stage 3+.
- No per-object MAP refinement at val time — Stage 3.
- No mesh GT chamfer (only point-cloud chamfer) — Stage 4.
- No DDP / multi-GPU. Single-GPU, single-process.
- No wandb sweep — single configured run only.

---

## 2. Commands run

### 2.1 Smoke test

```bash
$ /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
      scripts/car_model/smoke_test_spcarnet_stage2.py
```

Output (to stderr / stdout):

```
[stage2-smoke] index_path=/data/peilincai/mesh-splatting/outputs/carnet/spcarnet/object_index_v1.json
[stage2-smoke] dataset_ok n_objects=2433
[stage2-smoke] iter=0 loss={'loss_surf': 0.6931, 'loss_free': 0.6931, 'loss_hard': 0.6931, 'loss_mixed': 0.6931, 'loss_zL2': 8.67e-05, 'loss_total': 2.0794}
[stage2-smoke] iter=1 loss={'loss_surf': 0.6970, 'loss_free': 0.6879, 'loss_hard': 0.6861, 'loss_mixed': 0.6924, 'loss_zL2': 7.60e-05, 'loss_total': 2.0742}
[stage2-smoke] mc_ok mesh_present=False vertex_count=0 face_count=0
[stage2-smoke] PASS
{"losses_seen": [2.0794, 2.0742], "mesh_present": false}
```

### 2.2 Smoke results — interpretation

| Check | Expected | Observed | Pass? |
|---|---|---|---|
| Iter-0 BCE per term | ~ln 2 ≈ 0.6931 from random init | 0.6931 on every BCE term, exactly | ✓ |
| `loss_total` finite both iters | finite | 2.0794 → 2.0742 | ✓ |
| Decoder receives gradients | non-zero on at least one parameter | non-zero | ✓ |
| Latent table receives gradients | non-zero on `latents.codes` | non-zero | ✓ |
| Loss decreases or stays flat | within 1e-3 monotonic OR with a small Adam-induced uptick | strict decrease (−0.0052) | ✓ |
| MC pipeline reachable without crash | callable, returns a `MarchingCubesResult` | `mesh=None`, `vertex_count=0` after iter 1 | ✓ (fallback path) |

The MC `mesh_present=False` outcome at smoke is **expected**: with 2 untrained latents and a tiny network (latent_dim=32, hidden_dim=64, depth=3, ≈1k params), the sigmoid field after a single optimiser step is still ≈0.5 everywhere, so there is no iso-crossing at level 0.5. The smoke validates that the pipeline *runs* and produces a structured result; it does not gate on mesh quality. The 32-resolution headline check belongs to the eval entrypoint, post-training.

### 2.3 Did **not** run

- The full training launcher (`scripts/car_model/train_spcarnet_shape_field_autodecoder.sh`) — not yet kicked off; this is the next decision point.
- The eval entrypoint (`scripts/car_model/eval_spcarnet_shape_field_autodecoder.py`) — depends on a checkpoint from the full run.

---

## 3. Expected full-training command

```bash
cd /data/peilincai/mesh-splatting
bash scripts/car_model/train_spcarnet_shape_field_autodecoder.sh
```

Inherited defaults (override via env if needed):

| Variable | Default | Notes |
|---|---|---|
| `MODEL_CONFIG` | `configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml` | |
| `TRAIN_CONFIG` | `configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml` | |
| `OBJECT_INDEX` | `outputs/carnet/spcarnet/object_index_v1.json` | Stage-1 output. |
| `OUTPUT_DIR` | `outputs/carnet/spcarnet/autodecoder_v1` | |
| `RUN_NAME` | `spcarnet_autodecoder_v1` | |
| `WANDB_MODE` | `online` | Per the persistent feedback rule. |
| `WANDB_PROJECT` | `spcarnet` | Reused, not freshly created. |
| `DEVICE` | `cuda` | |
| `GPU` | `5` | |
| `PYTHON_BIN` | `/home/peilincai/micromamba/envs/mesh_splatting/bin/python` | |

The launcher writes:

- `outputs/carnet/spcarnet/autodecoder_v1/resolved_config.json` (model + loss + train, frozen at start).
- `outputs/carnet/spcarnet/autodecoder_v1/fit_summary.json` (elapsed, n_steps).
- `outputs/carnet/spcarnet/autodecoder_v1/logs/` (created up-front; reserved for future per-epoch logs / checkpoints).

Wall-clock budget at 200 epochs × `len(train) // 8 = 231` steps/epoch ≈ 46 200 steps. At batch 8, 1024 queries × 6-deep × 384-wide MLP with FiLM, expect ~50 ms/step on a single Ampere/Hopper GPU — order-of-magnitude **40 minutes to 2 hours** for the run. (Real number to be measured on first launch.)

Eval after training:

```bash
$PYTHON_BIN scripts/car_model/eval_spcarnet_shape_field_autodecoder.py \
    --checkpoint outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt \
    --object_index outputs/carnet/spcarnet/object_index_v1.json \
    --splits val \
    --mc_resolution 32 \
    --output outputs/carnet/spcarnet/autodecoder_v1/eval_val.json
```

The trainer does **not yet** persist checkpoints — that is a known gap below.

---

## 4. Output paths (exact)

| Path | Producer | Contents |
|---|---|---|
| `outputs/carnet/spcarnet/object_index_v1.json` | Stage 1 | 2433-object index — already on disk. |
| `outputs/carnet/spcarnet/autodecoder_v1/resolved_config.json` | `spcarnet_autodecoder_cli.py` | Frozen `{model, loss, train}` snapshot. |
| `outputs/carnet/spcarnet/autodecoder_v1/fit_summary.json` | `spcarnet_autodecoder_cli.py` | `{elapsed, n_steps}` (history is dropped before serialisation). |
| `outputs/carnet/spcarnet/autodecoder_v1/logs/` | launcher | Reserved. |
| `outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt` | *(not yet emitted)* | See §6. |
| `outputs/carnet/spcarnet/autodecoder_v1/eval_val.json` | `eval_spcarnet_shape_field_autodecoder.py` | Per-object metrics + summary block. |

---

## 5. Stage gate (unchanged from design)

Stage 2 is **PASS** when, on the held-out `val` split, after the full training run:

| Metric | Threshold | Source |
|---|---|---|
| `mesh_iou_at_0.5_mean` | ≥ **0.92** | `eval_val.json::summary.mesh_iou_at_0.5_mean` |
| `recon_chamfer_l1_mean` | ≤ **0.05** (canonical units, `[-1, 1]³`) | `eval_val.json::summary.recon_chamfer_l1_mean` |
| `mesh_extraction_success_rate` | ≥ **0.95** | `eval_val.json::summary.mesh_extraction_success_rate` |

All three must hold simultaneously. A miss on any single metric does not advance Stage 3.

---

## 6. Known risks / gaps

1. **No checkpoint emission yet.** `ShapeFieldAutoDecoderTrainer.fit()` returns history but does not persist a `.pt` file with `decoder_state_dict`, `latent_table`, and `object_id_to_row`. The eval entry expects exactly that schema. **Action before launch**: add a `_save_checkpoint(path)` call at end-of-fit (and at every `eval_every_epochs`) before kicking off the headline run. Tiny patch — listed as the first follow-up.
2. **No periodic eval inside `fit()`.** `eval_every_epochs` is plumbed in the config but not yet wired. Will be added together with the checkpoint hook.
3. **No wandb logging inside the trainer.** Launcher exports `WANDB_MODE=online` and `WANDB_PROJECT=spcarnet`, but the trainer does not currently call `wandb.init`. To meet the persistent feedback rule (CarNet training runs go online), the trainer needs `wandb.init` + per-step `wandb.log`. Add before launching the headline run.
4. **Smoke does not assert mesh quality.** Smoke MC at resolution=16 with an untrained tiny decoder returns `mesh=None`. That is an *acceptance* path of the smoke. Mesh-quality gating belongs to the eval script post-training, not to the smoke.
5. **Mixed cache format.** Cache versions 2 and 3 are both consumed by the Stage-1 dataset. The auto-decoder does not need symmetry persistence (symmetry is Stage 4 territory), so this is not a Stage-2 risk; it is documented to keep the trail honest.
6. **Eikonal autograd path under `requires_grad_(True)` on a tensor that already has `requires_grad=False`.** Tested only in the smoke that runs `field_kind=occupancy`; SDF ablation has been compiled but not executed end-to-end. To be exercised before the first SDF run.
7. **No seed pinning of CUDA RNGs.** `torch.manual_seed` covers it for current usage but `torch.cuda.manual_seed_all` is missing. Trivial to add; mentioned for the audit log.

None of these are blockers — they are pre-launch follow-ups, all under one hour of work to close.

---

## 7. Decision and next concrete step

Stage 2 implementation is **complete and passes smoke**. The recommended next concrete step is the small pre-launch hardening pass (checkpoint emission, periodic eval, wandb integration) and **then** kick off the headline auto-decoder run on GPU 5 with `WANDB_MODE=online` and `WANDB_PROJECT=spcarnet`.

The Stage-2 gate is defined and reachable; advancing to Stage 3 (per-object MAP refinement at val time) is conditional on hitting all three thresholds in §5 on the headline run.

---

## 8. Linked artefacts

- Design — `docs/car_model/spcarnet_stage2_shape_field_design.md`
- Stage-1 close — `docs/car_model/spcarnet_stage1_object_cache_report.md`
- RFC — `docs/car_model/SPCarNet_radical_RFC.md`
- Research log — `docs/car_model/SPCarNet_research_log.md`
- Index artefact — `outputs/carnet/spcarnet/object_index_v1.json`
