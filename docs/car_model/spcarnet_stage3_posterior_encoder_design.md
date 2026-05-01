# SP-CarNet Stage 3 — Amortised Posterior Encoder `q(z | O)` (Design)

| Field | Value |
|---|---|
| Stage | 3 / 7 (per `SPCarNet_radical_RFC.md` §9) |
| Status | DESIGN (precedes implementation) |
| Date | 2026-04-29 |
| Predecessor | Stage 2 (`spcarnet_stage2_shape_field_implementation_report.md`) |
| Successor | Stage 4 (test-time MAP refinement) |

---

## 0. Purpose & Gate

Stage 3 replaces the per-object free latent table from Stage 2 with an **amortised posterior** `q(z | O)`: a learnable encoder that maps a partial observation `O = (P_obs, …)` to a distribution over latent shape codes consumed by the **frozen Stage-2 decoder**. This is the headline SP-CarNet variant `EN-Q` from RFC §6.

Stage-3 gate (per RFC §7):

> **Pass**: on `val`, the encoder + frozen decoder achieves `recon_chamfer_l1 (sampled from extracted mesh) ≤ 0.10` (matches v0.7) **and** `free_space_violation_rate` strictly better than v0.7's, within 150 epochs.
>
> **Fail**: trigger Stage-6 pivot to retrieval-deformation (`RT-DEF`).

Stage 3 is the first stage that consumes a **corrupted/partial observation**. It is also the first stage whose loss surface is over a *posterior*, not a deterministic mapping — which is what the RFC §1 diagnosis identified as the missing ingredient.

---

## 1. Four key decisions

The user's task spec asks four things to be settled *before* code is written. They are settled here.

### 1.1 Freeze vs finetune the shape-field decoder

**Decision**: **freeze the Stage-2 decoder by default**; an optional warm-tail finetune is gated behind a config switch (`decoder_finetune.enabled`, default `false`).

Reasoning:

- **Latent-regression target stability.** The strong supervision in Stage 3 is `L_z = ||z_pred − z_target||²` against the Stage-2-trained per-object latent. If we finetune the decoder concurrently, the latent table is no longer the optimum of the Stage-2 objective: the decoder shifts, the optima shift, and the regression target becomes a moving signal. The encoder learns to chase decoder noise.
- **Identifiability.** A frozen `f(·; z)` defines a unique shape-manifold mapping `z → mesh`. Training the encoder under a frozen decoder is exactly the amortised inference problem the RFC §3.4 specifies. Joint training would also subsume Stage 2; the staged design is meant to *prevent* that conflation.
- **Compute.** Frozen decoder eliminates one optimiser (the decoder optimiser) and removes the gradient flow through `f` from the encoder's update graph. Roughly 30 % step-time saving.

Optional warm-tail finetune (off by default) unlocks `decoder.blocks[-2:]` and the output head at LR `1e-5` after `decoder_finetune.warmup_epochs = 50` epochs. Wraps the latent regression loss in a stop-gradient on `z_target` so the regression target stays anchored even if the decoder drifts. Used as an ablation if the Stage-3 gate is missed by ≤ 0.005 chamfer.

### 1.2 Posterior parameterisation

**Decision**: **variational posterior `q(z | O) = N(μ(O), σ(O)²)`** by default, with reparameterised sampling and a Gaussian prior `p(z) = N(0, I)`. A `posterior_kind: deterministic` config switch is provided as a fallback for diagnosing posterior collapse.

Reasoning:

- **RFC §3.4** explicitly specifies a reparameterised Gaussian posterior.
- **Multi-hypothesis (Stage 5)** requires a posterior we can sample from. A deterministic `z = μ(O)` would force Stage 5 to inject sampling noise post-hoc, which is poorly motivated.
- **Free-bits robustness.** Posterior collapse (σ → 0, KL → 0) is a real risk on small per-object KL terms. Free-bits (`min_kl_per_dim`) clamps the KL contribution from below, preserving information capacity.

Initialization: the σ branch is initialised to predict `log σ² = log 0.01²` (i.e. `-9.21`) so early steps behave nearly deterministically. KL warmup linearly ramps from `β = 0` to `β = 1e-3` over the first 10 epochs (matching the RFC §10 risk mitigation). Free-bits at `0.1 nats/dim`.

### 1.3 How to supervise `z` using auto-decoder latent codes

**Decision**: **direct L2 regression of `z_pred` (or `μ` in the variational case) onto the Stage-2-trained latent `z_target`** for every training-split object.

Mechanics:

- Stage-2 checkpoint persists `latent_table: Tensor (N_train, 256)` and `object_id_to_row: dict[str, int]`. Stage 3 loads both at trainer init, freezes the table, and looks up `z_target` per training object.
- Loss: `L_z = ||z_pred − sg(z_target)||² / d_z`. The `sg(·)` (stop-gradient) is redundant because the table is frozen, but is kept in the code as a safety guard for the decoder-finetune ablation.
- For variational posterior: the regression is on `μ`, not on a sample. The encoder's σ branch is regularised by KL only.
- **Validation/test objects** have no `z_target` (the Stage-2 table covers `train` only). `L_z` is masked to zero for non-train objects at training time. This is structurally consistent with the RFC's "no leak from val/test" constraint, and it is also what the user's prompt mandates.

Why latent regression as the strong supervision (rather than reconstruction-only):

- A 256-D L2 loss on a known target is a much stronger gradient signal than 1024 BCEs through a frozen 6-layer FiLM decoder. Empirical tests in DeepSDF / Occupancy Networks all confirm: amortised encoders trained against pre-fit latents converge an order of magnitude faster than pure observation-likelihood training.
- The reconstruction terms (occupancy / free-space, see §1.4) act as a **consistency check**: they ensure `z_pred` is not just numerically close to `z_target`, but that the *decoded shape* under `z_pred` agrees with the visible evidence. Without `L_z`, the encoder has too many ways to land at a low-occupancy-loss `z` that is far from the trained manifold.

### 1.4 How to combine latent regression with occupancy / free-space losses

**Decision**: weighted sum, with `L_z` dominant early and reconstruction terms ramping in.

```
L_total = w_z   * L_z                                     # latent regression (sg target)
        + w_kl  * KL(q(z|O) || N(0,I))                    # variational only
        + w_surf * BCE(σ(f(P_obs;   z_pred)), 1)          # observed-surface consistency
        + w_free * BCE(σ(f(Q_free;  z_pred)), 0)          # free-space evidence
        + w_hard * BCE(σ(f(Q_hard;  z_pred)), 0)          # hard negative
        + w_mix  * BCE_with_ignore(...)                    # combined occupancy supervision
        + w_vc   * chamfer(visible_clean_points, sample(mesh(z_pred)))  # visible chamfer (off by default)
        + w_hc   * chamfer(hidden_clean_points,  sample(mesh(z_pred)))  # hidden chamfer (off by default during training)
```

Default weights:

| Term | Weight | Note |
|---|---|---|
| `w_z` | 10.0 | Strong; will be ramped from `w_z_warmup=2.0` to `w_z=10.0` over 10 epochs to give reconstruction terms time to engage. |
| `w_kl` | 1e-3 (after warmup) | KL warmup from 0 to 1e-3 over 10 epochs. |
| `w_surf` | 1.0 | Observed surface consistency. |
| `w_free` | 1.0 | Free-space evidence. |
| `w_hard` | 0.5 | Hard negatives. |
| `w_mix` | 0.5 | Combined occupancy queries with ignore mask. |
| `w_vc` | 0.0 (default) | Optional; mesh-sampled chamfer is differentiable only via soft Marching-Cubes, which is too slow at training time. Reserved for short fine-tunes. |
| `w_hc` | 0.0 (default) | Same. |
| `free_bits_per_dim` | 0.1 nats | Anti-collapse. |

The reconstruction BCE terms reuse the Stage-2 query infrastructure unchanged: `surface_query_points` ∪ `clean_points`, `free_query_points`, `free_space_query_hard_negatives`, `query_points_all` with `query_ignore_mask`. **The encoder's input is `partial_observed_points`** — that is the observation `O`. The supervision is the *same* BCE objective Stage 2 uses, but with `z_pred` substituted for the per-object `z_target`.

Crucially: the encoder never sees `clean_points` or `query_points_all` at inference time. Those are *targets*, not inputs.

### 1.5 How to handle multi-modal ambiguity

**Decision**: Stage 3 uses a **unimodal Gaussian posterior** with KL warmup + free-bits. Multi-hypothesis sampling is **explicitly Stage 5's responsibility**.

Reasoning:

- A unimodal Gaussian is the simplest posterior consistent with the variational ELBO. It will smear over modes, producing a posterior mean somewhere between the modes. This is a known limitation of VAEs.
- The RFC §3.7 commits to multi-hypothesis sampling as a **separate stage** (Stage 5). Conflating Stage-3 mode coverage with Stage-5 multi-hypothesis architecture would muddy the Stage-3 gate.
- For Stage 3 specifically: a `multi_sample_train.enabled` config switch (default `false`) is provided. When on, the encoder draws `K=4` samples per training step, decodes each, and adds a **diversity-aware loss**: `L_div = −w_div · mean_pairwise_chamfer(sampled meshes)`. This is the simplest defence against posterior collapse if KL warmup + free-bits prove insufficient. It is **not** the multi-hypothesis Stage-5 mechanism (no reranking, no eval-time K).
- For Stage 3 eval, the headline metric is computed with `z_pred = μ(O)` — i.e. the posterior mean. A `--mc_samples=K` flag in the eval CLI re-decodes K samples from the posterior and reports `best_of_k_chamfer` as a diagnostic, *not* a headline number.

How we'll diagnose mode collapse if it happens:
- Watch `train/posterior_logvar_mean` — if it drops below `−6` and stays, free-bits has lost.
- Watch `val/recon_chamfer_l1` — if it stays high while `train/L_z` is small, the encoder has memorised train latents but the manifold doesn't generalise; investigate via `latent_retrieval_error` (defined in §3 below).

---

## 2. Architecture (`SPCarPosteriorEncoder`)

The RFC §3.4 says "reuse the v11 cross-attention backbone". The cross-attention v10/v11 modules are `~600 LoC` patch-centric files tightly coupled to residual / point-flow heads (see `cross_attention_patch_prior_v10.py:39-`, `latent_flow_patch_prior_v11.py:39-`). Subclassing them and disabling all the auxiliary heads is more brittle than building a leaner Stage-3-specific encoder over the **shared low-level building blocks** (`PointNetEncoder`, `CrossAttentionBlock`, `SelfAttentionBlock`).

This *is* "reusing the backbone" in the sense that matters: every attention primitive comes from `ss3dm_prior/models/attention_blocks.py` and `ss3dm_prior/models/pointnet.py`. The patch-centric coupling does not.

### 2.1 Forward sketch

```
input:
  P_obs : (B, N_obs, 3)         partial observation in canonical [-1, 1]^3
  N_obs : (B, N_obs, 3)         optional partial normals (zeros if absent)

# 1. point tokenization
tok = PointNetEncoder(in_dim=6 if normals else 3, feature_dim=F)(cat([P_obs, N_obs]))
                                                  -> (B, N_obs, F=256)
# 2. learnable latent queries
Q = nn.Parameter((1, num_latent_queries=32, F))
Q = Q.expand(B, -1, -1)                            -> (B, 32, F)

# 3. cross-attention: latent queries attend to point tokens
for layer in 0..num_xattn_layers-1=3:
    Q = CrossAttentionBlock(Q, tok)                # 8 heads, ffn 1024
    Q = SelfAttentionBlock(Q)                      # for layer in alternating cadence

# 4. global pooled summary
g = mean(Q, dim=1)                                  -> (B, F)

# 5. heads
mu      = Linear(F -> d_z=256)(g)                   -> (B, d_z)
logvar  = Linear(F -> d_z=256)(g)                   -> (B, d_z)
                                                     # initialised to log(0.01^2) bias
```

### 2.2 Reparameterisation

```
if posterior_kind == "variational":
    std = exp(0.5 * logvar)
    z   = mu + std * torch.randn_like(std)
elif posterior_kind == "deterministic":
    z = mu
    logvar = None
```

### 2.3 Auxiliary scanner-pose / symmetry conditioning (deferred)

The user's prompt lists scanner-pose / ray metadata and symmetry/retrieval context as inputs. The current cache (Stage 1 audit) shows scanner pose is not persisted and symmetry persistence is partial. Stage 3 therefore:

- **Reads `scanner_pose` from the dataset if present**, otherwise zeros it out and sets a flag bit on the encoder input. Wired but inactive on the current cache.
- **Reads `symmetry_plane_normal/offset/confidence` if present**, otherwise zeros + flag bit.
- These conditioning channels enter through a small projection head concatenated to the global summary `g` before `μ`/`logvar` heads.

This is a cheap insurance: when the LiDAR / symmetry pipelines persist these fields (Stage 4+, RFC §3.5 / §3.7), no encoder rewrite is needed.

### 2.4 Parameter budget

| Module | Params |
|---|---|
| PointNet tokenizer | ~150 K |
| 32 latent queries (parameter) | 32 × 256 = 8 K |
| 4 × cross-attn + ffn (1024) | ~4 M |
| 2 × self-attn among queries | ~1 M |
| μ / logvar heads | 2 × 256 × 256 ≈ 130 K |
| Optional scanner/symmetry adapter | ~10 K |
| **Total** | **~5.3 M** |

The decoder (frozen) adds ~3.5 M. Total runtime model: ~8.8 M params. Comparable to v0.7 / v0.8.2.

---

## 3. `SPCarPosteriorCompletionModel` (encoder + frozen decoder wrapper)

```
class SPCarPosteriorCompletionModel(nn.Module):
    encoder : SPCarPosteriorEncoder
    decoder : SPCarShapeFieldDecoder         # loaded from Stage-2 checkpoint, optionally frozen
    decoder_finetune_enabled : bool

    def forward(self, observation: dict, query_points: dict | None) -> dict:
        z_mean, z_logvar, z = self.encoder(observation)
        out = {"z_mean": z_mean, "z_logvar": z_logvar, "z": z}
        if query_points is not None:
            out["surf_logits"] = self.decoder(query_points["surface"], z)
            out["free_logits"] = self.decoder(query_points["free"], z)
            out["hard_logits"] = self.decoder(query_points["hard"], z)
            out["mixed_logits"] = self.decoder(query_points["mixed"], z)
        return out
```

`decoder.requires_grad_(False)` at init unless `decoder_finetune.enabled`. In the latter case, only the last 2 FiLM blocks + output head get `requires_grad_(True)` — and only after `decoder_finetune.warmup_epochs`.

For mesh extraction at eval time:
```
def occupancy_fn(query):
    with torch.no_grad():
        return torch.sigmoid(self.decoder(query.unsqueeze(0), z_mean.unsqueeze(0))).squeeze(0)
```

Same Marching-Cubes path as Stage 2.

---

## 4. Required eval metrics

The user's prompt lists eight metrics. Mapping them to implementation:

| Metric | Definition | Source |
|---|---|---|
| `recon_chamfer_l1` | mean bidirectional L1 chamfer between sampled-mesh points (4 096 sampled from extracted mesh) and `clean_points_object` | `extract_patch_mesh` → `mesh.sample(K)` → `cdist`; same path as Stage 2 eval |
| `hidden_chamfer_l1` | same but against `hidden_clean_points` | dataset `hidden_clean_points` |
| `visible_preservation_error` | mean L1 distance from each `partial_observed_points` to its nearest mesh-sampled point | `cdist(partial_obs, mesh_samples).min(dim=1).mean()` |
| `free_space_violation_rate` | fraction of `free_query_points` where `σ(f(q; z_pred)) > 0.5` | decoder forward + threshold; complement of "free-space accuracy" |
| `mesh_iou_at_0.5` | IoU between filled GT mesh volume (from GLB) and decoder volume (32³ or 64³) — uses the **fixed metric from sub-task B**, not the broken sparse-point voxelisation | `_voxelise_gt_mesh` (introduced by sub-task B) + `_decode_volume` |
| `zero_corruption_recon_chamfer_l1` | `recon_chamfer_l1` evaluated when the encoder is fed `clean_points` instead of `partial_observed_points` (zero corruption) | re-run encoder with `clean_points` as `O` |
| `latent_retrieval_error` | mean L2 distance from `z_pred` to `z_target` for objects whose Stage-2 latent is available; tells us whether the encoder has *memorised* the latent or actually *generalises* | only defined for train-split objects in eval (val/test have no target); we report it as a diagnostic-only value, not a leak |
| `mesh_extraction_success_rate` | fraction of objects with `result.mesh is not None and len(faces) > 0` | inherited from Stage 2 eval |

`latent_retrieval_error` is gated to train-split-only on the eval side. Reporting it for val *would* require Stage-2 latents for val, which by construction don't exist. We report it for `train` only as a diagnostic ("does the encoder converge to the right latent on objects it can be supervised against?"); we report `recon_chamfer_l1` and friends for val as the actual generalisation evidence.

### 4.1 Comparisons against baselines

The eval report should include side-by-side comparisons against:

- **v0.7 residual baseline** (`docs/car_model/carnet_v0_6_to_v0_8_2_report.md`): val_recon_chamfer_l1 ≈ 0.10, free-space violation unspecified
- **v0.8.2 point-flow baseline** (same report): val_recon_chamfer_l1 ≈ 0.12, pf_loss ≈ 0.4
- **Stage-2 auto-decoder train-reconstruction** (`spcarnet_stage2_shape_field_implementation_report.md` + the `eval_train_64.json` from the headline run): chamfer ≈ 0.066, mesh_iou ≈ 0.49 (broken metric), mesh_extraction_success_rate = 1.0

If the user supplies checkpoint paths for v0.7 / v0.8.2, the eval CLI runs them on the same val subset to produce true side-by-side numbers. Otherwise the report quotes the report numbers.

---

## 5. Smoke-test contract

`scripts/car_model/smoke_test_spcarnet_stage3.py` must:

1. Build / reuse the Stage-1 object index (`outputs/carnet/spcarnet/object_index_v1.json`).
2. Build a tiny encoder (`hidden_dim=64`, `num_xattn_layers=2`, `num_latent_queries=8`, `latent_dim=32`) and load a tiny matching decoder (`hidden_dim=64`, `depth=3`, `latent_dim=32`).
3. Forward a `B=2` batch through the encoder; verify `z_mean.shape == (2, 32)`, `z_logvar.shape == (2, 32)`, `z.shape == (2, 32)`, and that `z_logvar` is finite.
4. Sample `K=2` latent candidates via reparameterisation; verify they are distinct.
5. Decode occupancy on a small `(2, 64, 3)` query grid; verify shape `(2, 64)` and finiteness.
6. Run a backward pass through `L_total`; verify `encoder.parameters()` receive non-zero gradients on at least one parameter; verify `decoder.parameters()` receive **zero** gradients (frozen).
7. Print `[stage3-smoke] PASS`.

---

## 6. File plan

| File | Role |
|---|---|
| `ss3dm_prior/models/spcarnet_posterior.py` | `SPCarPosteriorEncoder`, `SPCarPosteriorCompletionModel`, sampling helpers. |
| `ss3dm_prior/training/spcarnet_posterior.py` | Trainer, dataclasses, loss assembly, checkpoint emission, periodic eval. |
| `ss3dm_prior/training/spcarnet_posterior_cli.py` | CLI wrapper with `--max_steps` / model+train YAML pair. |
| `configs/ss3dm_prior/spcarnet/model_spcarnet_posterior_encoder.yaml` | Encoder + loss config. |
| `configs/ss3dm_prior/spcarnet/train_spcarnet_posterior_encoder.yaml` | Trainer config (epochs / queries / LR / KL warmup / paths). |
| `scripts/car_model/train_spcarnet_posterior_encoder.sh` | Launcher. `WANDB_MODE=online`, `WANDB_PROJECT=spcarnet` (per persistent rule). |
| `scripts/car_model/eval_spcarnet_posterior_encoder.py` | Eval entrypoint emitting all §4 metrics + comparison table. |
| `scripts/car_model/smoke_test_spcarnet_stage3.py` | §5 contract. |
| `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md` | Closing report. |

No existing file is modified. CarNet v0.x configs, the patch-centric trainer, the Stage-2 trainer, the Stage-2 dataset wrapper, and the v11 cross-attention modules remain untouched. Stage-2 checkpoints under `outputs/carnet/spcarnet/autodecoder_v1/` are read-only inputs.

---

## 7. Out of scope (deferred to later stages)

- **MAP refinement of `z` at eval time** — Stage 4. Stage 3 reports `recon_chamfer_l1` with the amortised mean only.
- **Multi-hypothesis K∈{1,4,8} reranking** — Stage 5.
- **Ray-cast likelihood `L_ray`** — Stage 4 (cache does not yet expose ray endpoints; runtime sampler is a Stage-4 deliverable).
- **Symmetry-assisted variant `EN-Q-SYM`** — Stage 7-aux (RFC §6).
- **Retrieval-deformation alternative `RT-DEF`** — Stage 6, conditional on Stage-3 gate failure.
- **Distribution-shift transfer** (Semantic-KITTI, ShapeNet cars) — Stage 7.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Posterior collapse (σ → 0, KL → 0). | KL warmup from 0 → 1e-3 over 10 ep; free-bits at 0.1 nats/dim; periodic logging of `posterior_logvar_mean`. |
| Latent regression dominates → encoder memorises training z's, reconstruction terms have no influence. | `w_z` warmup from 2 → 10 over 10 ep; reconstruction terms have non-trivial weight from step 1; reported `latent_retrieval_error` on val (well-defined as `||z_pred − μ̄_train||` if needed) flags memorisation. |
| Frozen decoder is a bad fit for the encoder's output distribution → no `z_pred` in the trained manifold reconstructs the partial observation well. | Decoder finetune ablation (§1.1) is wired, gated, and easy to flip on; if Stage 3 misses by ≤ 0.005 chamfer, run the ablation. |
| `partial_observed_points` is a weak observation (only 768 points, all visible). | Stage-3 eval explicitly reports `zero_corruption_recon_chamfer_l1` to upper-bound encoder fit; if even with full clean points we miss the gate, the issue is the encoder or the decoder ceiling, not the corruption. |
| Mode collapse on hard occlusions (cars half-hidden by parking neighbours). | Diversity-aware multi-sample training (`multi_sample_train.enabled`) gated; Stage 5 adds proper multi-hypothesis. |
| Train/test latent leakage. | Stage-2 latent table only contains `train` rows; Stage 3 reads `object_id_to_row` and explicitly masks `L_z` for non-train objects. Verified in the smoke. |
| Eval-time IoU dependency on the broken sparse-point voxelisation. | Sub-task B (parallel) replaces the voxelisation with a GLB-derived filled volume. Stage-3 eval consumes the fixed function. |

---

## 9. Open questions (for the implementation report)

1. **Does latent regression alone suffice?** If `w_surf = w_free = 0`, does `recon_chamfer_l1` still beat 0.10? This bounds how much the reconstruction terms contribute vs the regression. Will be measured as an ablation row.
2. **How tight is the amortised gap?** `recon_chamfer_l1` (Stage 3 amortised) − `recon_chamfer_l1` (Stage 2 train-reconstruction with the *true* latent) is the amortisation gap; if it's > 0.02 we know Stage 4 MAP refinement has a useful amount to recover.
3. **Does posterior breadth correlate with occlusion?** Plot `posterior_logvar_mean` against the visible-fraction (computed from `visible_clean_points / clean_points`). If yes, the encoder is correctly uncertainty-aware.

---

## 10. Constraint compliance

- **No validation/test clean shapes for training-time latent supervision** — `L_z` masked to train-only by construction; Stage-2 latents only exist for train.
- **No self-retrieval leakage** — Stage 3 does not build a retrieval table at all. Stage-2 latent table is consulted only by `object_id`-key, and only training-split keys are used during training.
- **CarNet v0.x training still runs** — no shared file is modified. The Stage-2 trainer runs untouched and the Stage-2 checkpoint is read-only.

_End of design._
