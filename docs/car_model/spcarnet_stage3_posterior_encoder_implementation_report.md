---
title: SP-CarNet Stage 3 — Posterior Encoder Implementation Report
date: 2026-04-29
authors: SP-CarNet research line
status: implementation complete; smoke + integration smoke PASS; full training pending
linked_design: docs/car_model/spcarnet_stage3_posterior_encoder_design.md
linked_log: docs/car_model/SPCarNet_research_log.md
---

# SP-CarNet Stage 3 — Implementation report

Closes the implementation phase of Stage 3. Smoke and trainer-integration smoke against the real Stage-2 checkpoint both pass. The headline Stage-3 training run is **not yet launched** — the user has the launch command in §3 below and the relevant gates in §6 of the design.

---

## 1. Files added

| Path | Purpose | LoC |
|---|---|---|
| `docs/car_model/spcarnet_stage3_posterior_encoder_design.md` | Design doc — freeze/finetune decision, posterior parameterisation, latent-regression supervision, loss weighting, multi-modal handling, file plan, gate. | ≈420 |
| `ss3dm_prior/models/spcarnet_posterior.py` | `SPCarPosteriorEncoder` — PointNet tokeniser → cross-attention/self-attention stack over learnable latent queries → `(μ, logvar)` heads. `SPCarPosteriorCompletionModel` wraps it around a (frozen) `SPCarShapeFieldDecoder`. Optional scanner-pose / symmetry conditioning adapter. | ≈300 |
| `ss3dm_prior/training/spcarnet_posterior.py` | Trainer — config dataclasses, query assembly, KL + free-bits + latent regression + BCE loss assembly, step / fit, checkpoint emission, periodic save, wandb integration. | ≈400 |
| `ss3dm_prior/training/spcarnet_posterior_cli.py` | Argparse-based CLI; pairs a model YAML with a train YAML, persists `resolved_config.json`, runs `trainer.fit()`, persists `fit_summary.json`. | 75 |
| `configs/ss3dm_prior/spcarnet/model_spcarnet_posterior_encoder.yaml` | Encoder + loss config. References Stage-2 checkpoint at `outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt`. | 35 |
| `configs/ss3dm_prior/spcarnet/train_spcarnet_posterior_encoder.yaml` | 150 epochs, batch 16, 1024 queries / object / step, encoder LR 3e-4, weight decay 1e-4, KL warmup 10 ep, save every 10 ep. | 35 |
| `scripts/car_model/train_spcarnet_posterior_encoder.sh` | Launcher with `WANDB_MODE=online`, `WANDB_PROJECT=spcarnet`, `GPU=5`. Honors `DECODER_CHECKPOINT` env override (defaults to `autodecoder_v1/checkpoint_last.pt`). | 35 |
| `scripts/car_model/eval_spcarnet_posterior_encoder.py` | Eval entrypoint emitting all 8 required metrics + baseline comparison block. | ≈350 |
| `scripts/car_model/smoke_test_spcarnet_stage3.py` | 5-step smoke (forward / sampling / decode / backward / freeze). | ≈140 |
| `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md` | This report. | — |

## 2. Files **not** modified

- `ss3dm_prior/training/spcarnet_autodecoder.py` (Stage 2 trainer) — not touched.
- `ss3dm_prior/models/spcarnet_shape_field.py` (Stage 2 decoder) — not touched.
- `ss3dm_prior/data/spcarnet_object_dataset.py` (Stage 1 dataset) — not touched.
- `ss3dm_prior/engine/trainer.py` (CarNet v0.x trainer) — not touched.
- v0.x configs and launchers — not touched.
- `outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt` — **read-only input to Stage 3**.

The Stage-2 trainer / dataset / decoder still run untouched (verified by re-running the Stage-2 smoke after Stage-3 changes; PASS unchanged).

---

## 3. Architectural choices, locked in

| Decision | Value | Source |
|---|---|---|
| Decoder posture | **frozen by default**; tail-finetune (last 2 FiLM blocks + field head) ablation behind `decoder_finetune_enabled` | design §1.1 |
| Posterior parameterisation | **variational** (Gaussian, reparameterised); deterministic switch retained for collapse diagnosis | design §1.2 |
| `z` supervision | direct L2 on `μ` against frozen Stage-2 latent table; mask zero on non-train objects | design §1.3 |
| Loss combination | weighted sum, `w_z` warmup `2 → 10` over 10 ep, `w_kl` warmup `0 → 1e-3` over 10 ep, free-bits 0.1 nats/dim | design §1.4 |
| Multi-modality | unimodal Gaussian for Stage 3; multi-hypothesis is Stage 5 | design §1.5 |
| Encoder | PointNet tokeniser + 4 cross-attn / 2 self-attn over 32 learnable queries, `feature_dim=256` | design §2 |
| Conditioning | scanner-pose / symmetry adapter wired but inactive (zero-bound) on the current cache | design §2.3 |
| Param budget (encoder + frozen decoder) | ≈ 5.3 M (encoder) + 3.5 M (decoder, frozen) ≈ 8.8 M | design §2.4 |

---

## 4. Smoke results

### 4.1 Standalone smoke (`scripts/car_model/smoke_test_spcarnet_stage3.py`)

Run on CPU; tiny model (latent_dim=32, feature_dim=48, 2 cross-attn / 1 self-attn / 8 queries / heads=4 / ffn=96, decoder hidden=64 depth=3 fourier=8).

```
[stage3-smoke] encoder_forward_ok z_mean.shape=(2, 32) logvar.mean=-9.2103
[stage3-smoke] sampling_ok pairwise_delta=0.011081
[stage3-smoke] decode_ok logits.shape=(2, 64) sigmoid.mean=0.5000
[stage3-smoke] backward_ok encoder_grad=True decoder_grad=False loss=4.6246 l_z=0.3102 l_kl=136.3302 l_surf=0.6931
[stage3-smoke] PASS
```

| Check | Expected | Observed | Pass? |
|---|---|---|---|
| Forward `(B, d_z)` shape | `(2, 32)` | `(2, 32)` | ✓ |
| Initial logvar bias | `≈ log(0.01²) = −9.21` | `−9.2103` | ✓ |
| Two reparameterised samples differ | `> 1e-6` | `1.1e-2` | ✓ |
| Decoded logits shape + finiteness | `(2, 64)`, finite | `(2, 64)`, finite | ✓ |
| Init field is uniform 0.5 | `sigmoid.mean ≈ 0.5` | `0.5000` | ✓ |
| Encoder gradients flow | non-zero | True | ✓ |
| Decoder gradients **don't** flow | zero | False (no grad) | ✓ |
| `l_surf` at init | `≈ ln 2 = 0.6931` | `0.6931` | ✓ |

`l_kl=136` at init is expected — `μ` is small but `logvar` is `−9.21`, so `exp(logvar) − 1 − logvar ≈ −1 − (−9.21) = 8.21` per dim, summed over 32 dims gives ≈ 130. The free-bits floor will let the trainer collapse this naturally as `μ` and `logvar` learn to balance.

### 4.2 Trainer-integration smoke (real Stage-2 checkpoint, 3 steps)

```
WANDB_MODE=online WANDB_PROJECT=spcarnet \
python -m ss3dm_prior.training.spcarnet_posterior_cli \
    --model_config configs/ss3dm_prior/spcarnet/model_spcarnet_posterior_encoder.yaml \
    --train_config configs/ss3dm_prior/spcarnet/train_spcarnet_posterior_encoder.yaml \
    --decoder_checkpoint outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt \
    --output_dir /tmp/spcarnet_stage3_smoke_run \
    --run_name stage3_integration_smoke \
    --max_steps 3
```

- Trainer initialised against the real Stage-2 checkpoint (latent_table loaded; object_id_to_row mapped). No shape mismatch; latent_dim sanity check passed.
- 3 training steps completed in 2.18 s on GPU 5.
- wandb run `9kehaimo` synced to project `spcarnet`.
- Outputs: `resolved_config.json`, `fit_summary.json`, `checkpoint_last.pt`, all written.

Both smokes confirm: **Stage 3 implementation is launchable**.

---

## 5. Headline training command

```bash
cd /data/peilincai/mesh-splatting
bash scripts/car_model/train_spcarnet_posterior_encoder.sh
```

Inherited defaults (override via env if needed):

| Variable | Default | Notes |
|---|---|---|
| `MODEL_CONFIG` | `configs/ss3dm_prior/spcarnet/model_spcarnet_posterior_encoder.yaml` | |
| `TRAIN_CONFIG` | `configs/ss3dm_prior/spcarnet/train_spcarnet_posterior_encoder.yaml` | |
| `OBJECT_INDEX` | `outputs/carnet/spcarnet/object_index_v1.json` | Stage-1 output. |
| `DECODER_CHECKPOINT` | `outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt` | Override to `autodecoder_v2/checkpoint_last.pt` after the v2 retrain finishes. |
| `OUTPUT_DIR` | `outputs/carnet/spcarnet/posterior_encoder_v1` | |
| `RUN_NAME` | `spcarnet_posterior_encoder_v1` | |
| `WANDB_MODE` | `online` | Persistent rule. |
| `WANDB_PROJECT` | `spcarnet` | Reused across all SP-CarNet runs. |
| `DEVICE` | `cuda` | |
| `GPU` | `5` | The Stage-2 v2 retrain is currently on GPU 5; this launcher will collide unless GPU is overridden or v2 finishes first. |
| `PYTHON_BIN` | `/home/peilincai/micromamba/envs/mesh_splatting/bin/python` | |

**GPU collision warning**: the parallel Stage-2 v2 retrain (started by the sub-agent) is currently on GPU 5. Wait for it to finish (~40 min remaining) or set `GPU=4` on the Stage-3 launcher. Both runs use a wide enough memory margin that they could co-locate, but the user has not given that explicit permission.

Wall-clock budget at 150 epochs × `len(train) // 16 = 115` steps/epoch ≈ 17 250 steps. At ~0.7 s/step (cross-attention is more expensive than the FiLM-only Stage-2 path), expect **3-4 hours** for the full run.

---

## 6. Eval command

```bash
$PYTHON_BIN scripts/car_model/eval_spcarnet_posterior_encoder.py \
    --checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
    --object_index outputs/carnet/spcarnet/object_index_v1.json \
    --splits val \
    --mc_resolution 32 \
    --use_glb_iou \
    --baseline_stage2 outputs/carnet/spcarnet/autodecoder_v1/eval_train_64.json \
    --output outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json
```

Reported metrics (per the design §4):

| Metric | What it measures | Stage-3 gate |
|---|---|---|
| `recon_chamfer_l1_mean` | mesh-sampled vs `clean_points_object` | **≤ 0.10** (matches v0.7) |
| `hidden_chamfer_l1_mean` | mesh-sampled vs `hidden_clean_points` | reported, not gated |
| `visible_preservation_error_mean` | nearest-mesh distance per `partial_observed_points` | reported |
| `free_space_violation_rate_mean` | fraction of `free_query_points` with `σ(f) > 0.5` | **strictly better than v0.7** |
| `mesh_iou_at_0.5_mean` | filled-volume IoU (GLB-derived GT when available; sparse-point shell fallback otherwise) | reported, not gated |
| `zero_corruption_recon_chamfer_l1_mean` | `recon_chamfer_l1` with `clean_points` fed to the encoder instead of `partial_observed_points` | reported (upper bound on encoder fit) |
| `latent_retrieval_error_mean` | `‖z_pred − z_target‖₂` for train-split objects (diagnostic only) | reported |
| `mesh_extraction_success_rate` | fraction with `mesh != None and len(faces) > 0` | reported |

The Stage-3 **pass condition** (RFC §7) is **`recon_chamfer_l1_mean ≤ 0.10` and `free_space_violation_rate_mean` strictly better than v0.7**, both on `val`.

### 6.1 Baseline comparison

The eval CLI accepts three baseline JSONs to embed into the report:

| Baseline | File | Notes |
|---|---|---|
| `--baseline_v07` | path to v0.7 residual eval JSON if available | Quoted as 0.10 chamfer floor in `carnet_v0_6_to_v0_8_2_report.md` if no checkpoint is run side-by-side. |
| `--baseline_v082` | path to v0.8.2 point-flow eval JSON if available | Quoted as 0.12 ceiling. |
| `--baseline_stage2` | `outputs/carnet/spcarnet/autodecoder_v1/eval_train_64.json` | Stage-2 train-reconstruction with the **true** latent. The amortised encoder upper-bounds itself by this number; the gap is the amortisation cost. |

If a baseline is not provided, the report quotes the table number from `docs/car_model/carnet_v0_6_to_v0_8_2_report.md` and labels it as "report-only".

---

## 7. Expected failure modes & how to diagnose them

### 7.1 Posterior collapse (KL → 0, σ → 0)

**Symptom**: `posterior/logvar_mean` (logged under `train/posterior/logvar_mean`) drops below `−6` and stays.

**Diagnosis**:
- Inspect `train/loss_kl` curve in wandb. If it crashes to 0 within the first 10 epochs (i.e. before `kl_warmup_epochs` completes), the warmup is *too aggressive*; lower `w_kl` or extend `kl_warmup_epochs`.
- If `loss_kl` collapses *after* warmup, free-bits is too low; raise `free_bits_per_dim` to 0.2.
- If `recon_chamfer_l1` is good but the encoder behaves deterministically (e.g. `(z_pred[i] − z_pred[j])` for distinct corruptions of the same object is < 1e-3), the deterministic posterior is preferable; switch `posterior_kind: deterministic` and drop the KL term.

### 7.2 Posterior collapse to the manifold mean (mode averaging)

**Symptom**: every output mesh looks like the mean car. `recon_chamfer_l1` plateaus at ~0.06–0.08 (Stage-2-mean territory).

**Diagnosis**:
- Compare `latent_retrieval_error` on train: if it's small (< 1.0 in 256-D), the encoder is actually retrieving the right z. Otherwise, it has converged to a single z regardless of input.
- If retrieval error is small on train but `recon_chamfer_l1` is high on val, the encoder *can* do amortised inference but val objects are out of the trained manifold — try the decoder-finetune ablation.
- If retrieval error is high everywhere, raise `w_z_warmup` from 2 to 5 and rerun.

### 7.3 Reconstruction-from-clean is poor

**Symptom**: `zero_corruption_recon_chamfer_l1_mean` is also bad (≥ 0.07).

**Interpretation**: the encoder cannot fit even a perfect observation. The bottleneck is upstream: either the encoder architecture is undersized, or the Stage-2 decoder is too narrow. Run the "v2" Stage-2 retrain (in flight as of writing) and re-pair Stage 3 against it.

### 7.4 Distinguishing "real completion" from "memorised retrieval"

This is the user's explicit anti-leakage criterion. Two diagnostics, both built into the eval script:

1. **`latent_retrieval_error_mean` on `val` — should be `nan`** (val objects have no Stage-2 latent target). If a non-NaN value appears, either `object_id_to_row` was mis-keyed or the Stage-2 latent table secretly includes val objects (it does not — verified by `len(stage2_latent_table) == n_train`).
2. **`recon_chamfer_l1` on `val` vs `train`** — if val is dramatically worse than train (gap ≥ 2×), the encoder has memorised train latents. The corrective is to:
   - Increase encoder dropout (currently 0.1 → 0.2).
   - Reduce `w_z` (currently 10 → 5).
   - Add a held-out regulariser: random subsampling of train queries during Stage 3 to avoid overfitting the per-object surface set.
3. **Out-of-distribution probe (deferred to Stage 7)**: feed the encoder a Semantic-KITTI cropped car or a noised perturbation of a training mesh and inspect the resulting mesh. If output is the *exact* nearest-train-object, the encoder has degenerated to retrieval.

---

## 8. Output paths (exact)

| Path | Producer | Contents |
|---|---|---|
| `outputs/carnet/spcarnet/object_index_v1.json` | Stage 1 | 2433-object index. |
| `outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt` | Stage 2 (v1) | Read-only Stage-3 input — decoder weights + Stage-2 latent table. |
| `outputs/carnet/spcarnet/autodecoder_v2/checkpoint_last.pt` | Stage 2 (v2, in-flight) | Optional Stage-3 input once v2 finishes. |
| `outputs/carnet/spcarnet/posterior_encoder_v1/resolved_config.json` | `spcarnet_posterior_cli.py` | Frozen `{stage, model, loss, train}` snapshot. |
| `outputs/carnet/spcarnet/posterior_encoder_v1/fit_summary.json` | `spcarnet_posterior_cli.py` | `{elapsed, n_steps, checkpoint_path}`. |
| `outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt` | trainer | `encoder_state_dict + decoder_state_dict + stage2_latent_table + stage2_object_id_to_row + state + model_cfg`. |
| `outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json` | eval script | `{summary, per_object, baselines, args}`. |
| `outputs/carnet/spcarnet/posterior_encoder_v1/wandb/` | wandb | `spcarnet_posterior_encoder_v1` run. |

---

## 9. Constraint compliance audit

| Constraint | Status |
|---|---|
| No validation/test clean shapes for training-time latent supervision | ✓ — `L_z` is masked to entries present in Stage-2 `object_id_to_row`, which is train-only by Stage-2 construction. |
| No self-retrieval leakage | ✓ — Stage 3 builds no retrieval index. The latent table is consulted only by `object_id` key, only at training time, and only for train-split objects. |
| Existing CarNet_v0 training still runs | ✓ — no shared file modified. Stage-2 trainer untouched. v0.x configs untouched. |
| Stage-2 checkpoint preserved | ✓ — `outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt` is read-only input. The sub-agent's parallel v2 retrain writes to `autodecoder_v2/`; v1 remains intact. |
| wandb online by default | ✓ — launcher exports `WANDB_MODE=online` and `WANDB_PROJECT=spcarnet`; trainer initialises wandb in `__init__` and finishes the run on `fit()` exit. |

---

## 10. Decision and next concrete step

Stage 3 implementation is **complete and passes both smokes** (standalone + integration). The recommended next concrete step is to launch the headline run on a free GPU once the parallel Stage-2 v2 retrain finishes (or on GPU 4 immediately if the user authorises).

If the headline run hits the Stage-3 gate (`recon_chamfer_l1 ≤ 0.10` on val with strictly-better free-space violation than v0.7), Stage 4 (test-time MAP refinement) is the next stage. If it misses by ≤ 0.005 chamfer, run the decoder-finetune ablation. If it misses by more, the RFC §7 specifies a Stage-6 pivot to retrieval-deformation.

---

## 11. Linked artefacts

- Design — `docs/car_model/spcarnet_stage3_posterior_encoder_design.md`
- RFC — `docs/car_model/SPCarNet_radical_RFC.md` (§3.4–§3.7, §6 EN-Q row, §7 Stage-3 gate)
- Stage-2 close — `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md`
- Stage-1 close — `docs/car_model/spcarnet_stage1_object_cache_report.md`
- Research log — `docs/car_model/SPCarNet_research_log.md`
