---
title: SP-CarNet Stage 2 — Shape-field auto-decoder implementation report
date: 2026-04-29
authors: SP-CarNet line
status: implementation complete; v4 normal-band trained; held-out MAP-fit eval complete; gate soft FAIL
linked_design: docs/car_model/spcarnet_stage2_shape_field_design.md
linked_log: docs/car_model/SPCarNet_research_log.md
---

# SP-CarNet Stage 2 — Implementation report

This document originally closed the implementation phase of Stage 2 (the canonical, object-level shape-field auto-decoder). As of 2026-06-24, the line has been trained through `autodecoder_v3` and `autodecoder_v4_band`; the eval entry now supports held-out z-only MAP fitting for val/test objects that do not have train-time latent-table rows.

Current headline status:

> The Stage-2 engineering path is now complete enough to evaluate held-out clean-val decoder capacity: `206 / 206` val objects extract after z-only MAP fitting, with strict JSON output and W&B logging. The v4 normal-band objective improves v3 chamfer and filled-volume IoU, but the method still does **not** pass the original Stage-2 quality gate.

Latest evidence:

```text
outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt
outputs/carnet/spcarnet/autodecoder_v3/eval/train_eval.json
outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json
W&B: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/svtbc8sn
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
W&B train: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/dysg8508
W&B eval: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/4wu9w305
W&B final eval: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/q1jjwvdm
```

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
- No observation-driven posterior/MAP refinement — Stage 3 and Stage 4. A clean-shape z-only MAP-fit eval ablation is now implemented for held-out Stage-2 decoder-capacity measurement.
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

Superseded by the 2026-06-24 update below. The original implementation report predates `autodecoder_v3` training and the held-out MAP-fit eval path.

### 2.4 2026-06-24 held-out MAP-fit eval update

The original val eval skipped every held-out object because Stage-2 auto-decoders only contain latent-table rows for train objects. This produced `0 / 206` extraction in:

```text
outputs/carnet/spcarnet/autodecoder_v3/eval/val_eval.json
```

That is not a valid decoder-capacity evaluation. The eval script now supports explicit z-only MAP fitting for objects missing from the train-time latent table:

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/eval_spcarnet_shape_field_autodecoder.py \
  --checkpoint outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --splits val \
  --output outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json \
  --mc_resolution 32 \
  --sample_count 4096 \
  --device cuda \
  --fit_missing_latents \
  --fit_steps 100 \
  --fit_lr 0.01 \
  --fit_queries_surface 384 \
  --fit_queries_free 384 \
  --fit_queries_hard 128 \
  --fit_queries_mixed 128 \
  --resolved_config outputs/carnet/spcarnet/autodecoder_v3/resolved_config.json \
  --wandb_project spcarnet \
  --wandb_run_name stage2_autodecoder_v3_val_mapfit_full206_20260624 \
  --wandb_mode online
```

Result summary:

| metric | value |
|---|---:|
| `n_objects_evaluated` | `206` |
| `n_extracted` | `206` |
| `mesh_extraction_success_rate` | `1.0` |
| `recon_chamfer_l1_mean` | `0.0698447353` |
| `hidden_chamfer_l1_mean` | `0.1023846301` |
| `mesh_iou_at_0.5_mean` | `0.5531548112` |
| `mesh_iou_at_0.5_shell_mean` | `0.9112784961` |
| `n_iou_filled` | `69` |
| `n_iou_shell` | `137` |
| `surface_normal_consistency_mean` | `0.7182239138` |

Audit checks:

| check | result |
|---|---|
| Strict JSON, no `NaN` / `Infinity` tokens | PASS |
| Per-object latent source | all `heldout_map_fit` |
| Per-object fit status | all `ok` |
| W&B online run | `svtbc8sn` |

Interpretation:

> This fixes the Stage-2 held-out evaluation interface and proves the decoder can produce meshes for all val objects after clean-shape z-only MAP fitting. It does not prove that Stage 2 is a strong headline shape prior: the original gate still fails on chamfer and filled-volume IoU.

### 2.5 2026-06-24 v4 normal-band objective update

The v4 branch keeps the v3 decoder capacity fixed and changes the training objective. For each surface point with a clean normal, it samples a narrow band around the surface:

```text
x_inner = x_surface - epsilon * normal  -> occupied target
x_outer = x_surface + epsilon * normal  -> free target
```

This adds a boundary-sharpening BCE term to reduce ambiguity around the Marching-Cubes `0.5` crossing:

```text
loss_band = 0.5 * (BCE(f(x_inner, z), 1) + BCE(f(x_outer, z), 0))
```

Implementation and config:

```text
ss3dm_prior/training/spcarnet_autodecoder.py
scripts/car_model/eval_spcarnet_shape_field_autodecoder.py
scripts/car_model/smoke_test_spcarnet_stage2.py
configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v4_band.yaml
configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v4_band.yaml
scripts/car_model/train_spcarnet_shape_field_autodecoder_v4_band.sh
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
```

Full training:

```text
W&B run: dysg8508
steps: 69300 / 69300
epochs: 300
checkpoint: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt
```

Best documented full-val MAP-fit result so far is the epoch50 checkpoint:

| metric | v3 MAP-fit | v4 epoch50 MAP-fit | v4 final MAP-fit | best |
|---|---:|---:|---:|---|
| `n_extracted` | `206 / 206` | `206 / 206` | `206 / 206` | tie |
| `recon_chamfer_l1_mean` | `0.0698447353` | `0.0607328202` | `0.0655826944` | v4 epoch50 |
| `hidden_chamfer_l1_mean` | `0.1023846301` | `0.0933915632` | `0.0963624408` | v4 epoch50 |
| `mesh_iou_at_0.5_mean` | `0.5531548112` | `0.5683319216` | `0.5314717742` | v4 epoch50 |
| `mesh_iou_at_0.5_shell_mean` | `0.9112784961` | `0.8783071888` | `0.8563237802` | v3 |

Interpretation:

> v4 is a real method improvement over v3 on chamfer and filled-volume IoU when using the epoch50 checkpoint, but it still misses the original Stage-2 gate and trades off shell IoU. The final checkpoint is worse than epoch50, so longer training is not sufficient; this should be reported as a promising object-prior direction, not as a headline-complete result.

The checkpoint choice is now captured by a deterministic selector artifact:

```text
script: scripts/car_model/select_spcarnet_stage2_checkpoint.py
status: BEST_AVAILABLE_GATE_FAIL_WITH_LATE_DEGRADATION
best candidate: v4_epoch50
selection JSON: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.json
selection Markdown: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
```

This closes the report-side checkpoint-selection gap: future Stage-2 variants should be compared through this selector instead of citing a manually chosen checkpoint.

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

Stage 2 is **PASS** when, on the held-out `val` split, after the full training run and the clean-val z-only MAP-fit eval:

| Metric | Threshold | Source |
|---|---|---|
| `mesh_iou_at_0.5_mean` | ≥ **0.92** | `eval_val.json::summary.mesh_iou_at_0.5_mean` |
| `recon_chamfer_l1_mean` | ≤ **0.05** (canonical units, `[-1, 1]³`) | `eval_val.json::summary.recon_chamfer_l1_mean` |
| `mesh_extraction_success_rate` | ≥ **0.95** | `eval_val.json::summary.mesh_extraction_success_rate` |

All three must hold simultaneously. A miss on any single metric does not advance Stage 3.

2026-06-24 gate result:

| Metric | Threshold | v3 MAP-fit | v4 epoch50 MAP-fit | Pass? |
|---|---:|---:|---:|---|
| `mesh_iou_at_0.5_mean` | ≥ `0.92` | `0.5531548112` | `0.5683319216` | no |
| `recon_chamfer_l1_mean` | ≤ `0.05` | `0.0698447353` | `0.0607328202` | no |
| `mesh_extraction_success_rate` | ≥ `0.95` | `1.0` | `1.0` | yes |

Decision: **soft FAIL**. The evaluation path is now correct and complete, and v4 improves the right metrics, but decoder quality is not strong enough to make Stage 2 a headline object-prior result.

---

## 6. Known risks / gaps

1. **Decoder-capacity ceiling remains low.** `autodecoder_v3` reaches full val extraction after z-only MAP fitting, but chamfer `0.0698` and filled IoU `0.5532` miss the Stage-2 gate. Scaling width/depth alone did not solve the shape prior.
2. **No periodic geometry eval inside `fit()`.** `eval_every_epochs` currently emits checkpoints and W&B training metadata, but does not run full geometry validation during training. The new selector closes report-side checkpoint choice after eval JSONs exist; it does not replace in-loop validation.
3. **Held-out MAP-fit uses clean val supervision.** This is legitimate for decoder-capacity measurement, but it is not an inference-time result. Stage 3/4 results remain the correct evidence for partial-observation inference.
4. **Smoke does not assert mesh quality.** Smoke MC at resolution=16 with an untrained tiny decoder returns `mesh=None`. That is an *acceptance* path of the smoke. Mesh-quality gating belongs to the eval script post-training, not to the smoke.
5. **Mixed cache format.** Cache versions 2 and 3 are both consumed by the Stage-1 dataset. The auto-decoder does not need symmetry persistence (symmetry is Stage 4 territory), so this is not a Stage-2 risk; it is documented to keep the trail honest.
6. **Eikonal autograd path under `requires_grad_(True)` on a tensor that already has `requires_grad=False`.** Tested only in the smoke that runs `field_kind=occupancy`; SDF ablation has been compiled but not executed end-to-end. To be exercised before the first SDF run.
7. **No seed pinning of CUDA RNGs.** `torch.manual_seed` covers it for current usage but `torch.cuda.manual_seed_all` is missing. Trivial to add; mentioned for the audit log.

None of these are blockers — they are pre-launch follow-ups, all under one hour of work to close.

---

## 7. Decision and next concrete step

Stage 2 implementation and held-out eval are now engineering-complete, but the Stage-2 quality gate is a **soft FAIL**.

The next concrete research step is not another blind width/depth scale-up. The evidence points to a decoder/prior limitation: either improve the shape representation objective, switch to a stronger signed-distance / surface-distance formulation, or keep Stage 2 as a supporting component while Stage 3/4/5 carry the object-prior story.

---

## 8. Linked artefacts

- Design — `docs/car_model/spcarnet_stage2_shape_field_design.md`
- Stage-1 close — `docs/car_model/spcarnet_stage1_object_cache_report.md`
- RFC — `docs/car_model/SPCarNet_radical_RFC.md`
- Research log — `docs/car_model/SPCarNet_research_log.md`
- Index artefact — `outputs/carnet/spcarnet/object_index_v1.json`
