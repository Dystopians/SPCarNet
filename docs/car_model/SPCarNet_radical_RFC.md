# SPCarNet — Radical RFC: Object-Centric Shape Posterior Framework

| Field | Value |
|---|---|
| Status | **PROPOSED** (awaiting go/no-go) |
| Date | 2026-04-29 |
| Supersedes | CarNet v0.x point-space line (v0.6 → v0.8.2) |
| New research line | **SPCarNet** (Shape-Posterior CarNet) |
| Submission target | NeurIPS (unchanged) |
| Primary benchmark | MeshFleet whole-car (`meshfleet_car_cache_v5`) |
| Decision horizon | 4 weeks of full investment; then go/no-go vs reverting to v0.7 residual line |

---

## 0. TL;DR

The CarNet line has hit a hard ceiling at **val_recon_chamfer_l1 ≈ 0.10**. Two paradigms have been tried and exhausted:

1. **v0.7 residual decoder** (`recon = corrupted + δ`) collapses to a fixed ~0.04 low-pass smoother. It is **not learning to repair** — it is learning to neutrally smooth any input. 23 % of points are pushed *further* from ground truth.
2. **v0.8/v0.8.2 point-space flow matching** (per-point MLP, Gaussian → clean) plateaus at 0.12 with `pf_loss ≈ 0.4`. Velocity regression never converges. Mode is structurally weaker than residual.

These are not bugs to be fixed by reweighting losses, adding EdgeConv, or extending Euler steps. They are symptoms of a **single deeper error**:

> **We have been treating mesh repair as per-point regression / per-point generation over an unordered point set. The actual problem is posterior inference over a category-structured *shape field*, conditioned on partial point + ray + free-space evidence.**

This RFC proposes **SPCarNet**: a fundamentally different framework where:
- the model is an **implicit shape-field decoder** `f(x; z)` (occupancy or SDF) parameterised by a per-object latent code `z`,
- repair is **amortised posterior inference** `q(z | observation)`,
- supervision uses the **full Bayesian likelihood**: surface points, free-space queries, and ray-cast occlusion evidence,
- the **mesh** (extracted via Marching Cubes from `f`) becomes the primary output artefact; sampled point clouds become a downstream sample.

A staged 7-step roadmap with explicit kill criteria follows.

---

## 1. Current Failure Diagnosis

_All numbers cite `docs/car_model/carnet_v0_6_to_v0_8_2_report.md` (2026-04-29) and `docs/car_model/carnet_v0_8_diagnosis.md`._

### 1.1 v0.7 residual decoder is **not** shape completion

The residual head computes `recon = corrupted + δ(z, corrupted)`. Under chamfer supervision the optimisation surface has a deep, broad **identity-copy basin**: the trivial solution `δ = 0` is locally near-optimal because chamfer is small whenever `recon` and `clean` are pointwise close, and `corrupted` is *already* pointwise close to `clean` for most points.

Concrete evidence:

- **Per-corruption sweep (v0.7 ckpt)**: 6 of 7 single-corruption profiles (point_dropout, normal_noise, local_hole_mask, outlier_cluster, density_imbalance, zero) yield **negative gain**. Only `gaussian_jitter` produces positive gain — exactly the case where smoothing happens to coincide with denoising.
- **Per-point ratio probe** (50 patches × 2048 pts): median `||recon − clean|| / ||corrupted − clean|| = 0.74`. **23.1 % of points** have ratio ≥ 1.0 — the model pushed them *further* from ground truth.
- **Floor decomposition**: 0.04 (smoothing floor) + 0.02 (jitter residual) + 0.04 (structural error) ≈ 0.10. This is architectural, not capacity-limited.

The model is not performing repair; it is performing a fixed-magnitude isotropic blur. Repair would be **conditional on what is wrong with the input** — the residual decoder cannot express that conditionality because it has no representation of "where the ground truth surface is" independent of the input.

### 1.2 Clean → clean degradation indicates smoothing collapse

Sanity check: feed the model a perfectly clean point cloud (`chamfer_before = 0.0`). A correct repairer should output the same cloud (`chamfer_after ≈ 0.0`). v0.7 produces:

| Profile | chamfer_before | chamfer_after |
|---|---|---|
| `zero` (no corruption) | 0.0000 | **0.0399** |
| `only_normal_noise` (xyz untouched) | 0.0000 | **0.0398** |

The residual head **introduces a 0.04 chamfer error onto perfectly clean input**. This is the unique signature of **smoothing collapse**: the decoder has converged to a fixed displacement function applied to every input, rather than a conditional repair. No reweighting, no curriculum, and no symmetry/RAG side-head can rescue this — the **decoder has lost the identity map** because there was no incentive to preserve it; chamfer over a fixed clean dataset accommodates a small fixed perturbation.

### 1.3 Point-space flow matching from noise is a poor fit for unordered point sets

v0.8/v0.8.2 replaces the residual decoder with a per-point conditional flow `x_t = (1−t)·ε + t·x_clean`, `v_target = x_clean − ε`, decoded by a per-point MLP (with EdgeConv k-NN aggregation in v0.8.1+).

This was supposed to eliminate the identity-copy basin. It does — at the cost of introducing **two new structural problems**:

1. **No point-correspondence prior**. From `ε ~ N(0,I)`, point `i` of `recon` should land on *which* surface point of `x_clean`? The supervision (chamfer) is permutation-invariant, but the generative process is not — index `i` of `ε` and index `i` of `x_clean` are coupled through the linear interpolant. The decoder must therefore learn an *arbitrary* permutation that happens to be self-consistent across timesteps. With 2048 unordered points and per-point MLPs, this is a hopelessly under-constrained inverse problem. **Empirical signature**: `pf_loss` plateaus at ~0.4 (the variance floor of the unconstrained random matching) and never drops to the < 0.15 region that would indicate true convergence.

2. **Velocity regression on i.i.d. noise input has no spatial inductive bias**. EdgeConv k=16 in `x_t` space (v0.8.1, v0.8.2) was meant to fix this by giving each point local neighbours — but at small `t` the `x_t` cloud is essentially Gaussian noise, so the k-NN graph is *meaningless*. Only at `t ≈ 1` does the graph become informative, but by then the velocity is already nearly zero. EdgeConv contributes noise-conditioned regularisation, not structural prior.

Result: **FM ceiling ≈ 0.12** (v0.8.2 best `val_recon_chamfer_l1 = 0.1200 @ ep76`), strictly worse than the 0.10 floor of the (broken) residual decoder. The mode itself is the wrong tool.

### 1.4 Symmetry / RAG / EdgeConv / K-step extensions cannot move the ceiling

Each of these has been tried and / or analysed:

| Extension | Status | Effect on `val_recon_chamfer_l1` |
|---|---|---|
| `symmetry_consistency_loss` (v0.6+) | enabled | none isolable |
| Retrieval anchors (v0.6+) | enabled | none isolable |
| EdgeConv k=16 (v0.8.1, v0.8.2) | enabled | ≈ +0.008 (noise) |
| K = 4 → 8 Euler steps (v0.8.1, v0.8.2) | enabled | ≈ 0 |
| Corrupted warm-start (v0.8.1) | enabled then reverted | **−0.097** (catastrophic; SNR-collapse in FM target) |

**None of these address the failure mode of §1.1–§1.3**, because none of them change the representational target. Symmetry and RAG add inductive bias to a decoder that is still asked to produce 2048 unordered output points; EdgeConv and K refine the integration of a generative path that is itself misformulated. They are second-order modulations of a first-order architectural error.

A useful thought experiment: even with **ground-truth symmetry and retrieval at zero-error**, a per-point MLP from Gaussian noise to 2048 surface points still has no canonical mechanism to express a *category prior* (cars are 4-wheeled, have a roof, have bilateral structure) — it has to re-derive that prior every forward pass for every point. This is not a problem one can attack with another head.

---

## 2. New Central Hypothesis

> **Mesh repair is object-centric posterior inference over a shape field.**
>
> Given a corrupted observation `O = (P_obs, R_rays, F_free)` — partial points, sensor rays, free-space evidence — and a category-shape prior `p(z)`, the task is to compute a posterior over complete shape fields:
>
> `p(f | O) ∝ p(O | f) · p(f)`
>
> where `f : ℝ³ → ℝ` is an implicit shape function (occupancy or signed distance), and `p(f) = ∫ p(f | z) p(z) dz` is induced by a category-conditioned latent decoder `f(x; z)`.

This re-frames every component:

| Old framing (CarNet v0.x) | New framing (SPCarNet) |
|---|---|
| Predict 2048 output points | Predict a shape field `f(x; z)` over ℝ³ |
| Supervise with chamfer over points | Supervise with `p(O | f)`: surface likelihood + free-space + ray |
| Loss surface has identity / smoothing basins | Loss surface is dominated by likelihood — identity is *not* a critical point |
| Single output | Posterior — admits multi-hypothesis sampling and reranking |
| Mesh is a post-hoc Marching-Cubes derivative | Mesh is the primary artefact; points are samples from `f` |
| Symmetry / RAG / occupancy as side heads | Symmetry / RAG / free-space as **likelihood terms** in the posterior |

The hypothesis explicitly predicts:

1. The clean → clean degradation **vanishes**: a posterior conditioned on clean points has its mode at the data manifold by construction, not at a smoothed neighbourhood.
2. Free-space and ray evidence (currently a side head) becomes a **first-class likelihood** that disambiguates the half-occluded LiDAR case — exactly where v0.7 fails worst.
3. Multi-hypothesis sampling stops being decorative — it is *required* by the framing, because the posterior over `z` is genuinely multimodal under heavy occlusion.

---

## 3. Proposed New Architecture (SPCarNet-A)

### 3.1 Canonical object frame

All shapes are placed in a **canonical object frame** before encoding/decoding: front-axis +x, up +z, scale normalised so the bounding sphere has radius 1. This is non-negotiable: implicit fields without a canonical frame learn rotations and scales as part of `z`, wasting capacity. Per-patch frames (current v0.x behaviour) are *replaced* by per-object frames at this layer (a per-patch local-frame transform is still allowed as a downstream adapter).

Existing `local_frame` transforms in `ss3dm_prior` will be **audited and re-purposed** in Stage 1.

### 3.2 Latent shape code `z`

`z ∈ ℝ^{d_z}` (target `d_z = 256`), a per-object latent that fully parameterises the implicit field. Two acquisition modes:

- **Auto-decoder mode (Stage 2)**: `z` is a free per-shape parameter, jointly optimised with `f` à la DeepSDF. No encoder. Used to upper-bound decoder capacity.
- **Amortised mode (Stage 3+)**: `z = μ(O) + σ(O) · ε` from an encoder `q(z | O)`, learned end-to-end with `f`.

### 3.3 Implicit shape-field decoder `f(x; z)`

`f : ℝ³ × ℝ^{d_z} → ℝ`, implemented as a positional-encoded MLP with weight-modulation by `z` (FiLM or hyper-network style). Output is **occupancy logits** in the primary variant (`σ(f) ∈ [0,1]`); a signed-distance variant with eikonal regularisation is a Stage-2 ablation.

Decoder size target: 2–4 M params. The decoder **is not** the place to spend parameters — the encoder `q` is.

### 3.4 Amortised posterior encoder `q(z | O)`

Encoder consumes the full corrupted observation:
- partial points `P_obs ∈ ℝ^{N×3}`,
- partial normals (where available),
- visible / hidden mask,
- (when LiDAR-realistic): scanner pose and per-ray endpoint.

Architecture: reuse the v11 cross-attention backbone (we have already validated `occupancy_iou_visible > 0.9` in v0.8.2 — the encoder already learns an excellent latent). The **head replaces the residual / point-flow decoder** with a Gaussian posterior `(μ, log σ²)` on `z`.

### 3.5 Differentiable observation / ray / free-space consistency objective

Training loss is the negative log posterior of the observation under `f(·; z)`, with three explicit terms:

1. **Surface likelihood** (visible points lie on the level set):
   `L_surf = Σ_{p ∈ P_obs} BCE(σ(f(p; z)), 1)` — visible points should be high-occupancy.
2. **Free-space evidence** (queries between sensor and visible surface are unoccupied):
   `L_free = Σ_{q ∈ Q_free} BCE(σ(f(q; z)), 0)` — uses the existing `free_query_points` / `query_labels_all` from the cache.
3. **Ray-cast evidence** (each ray's first level-set crossing matches the visible-point depth — only when LiDAR-realistic corruption is active):
   `L_ray = Σ_{r ∈ R} | depth_first_crossing(f, r) − depth_obs(r) |`.

A **clean-shape regulariser** (KL on `q` to a Gaussian prior + reconstruction of clean queries when the clean teacher is available) anchors the posterior during training.

The chamfer-style point loss is *not removed* but is **demoted to a sanity metric** — see §5.

### 3.6 Optional test-time MAP refinement of `z`

At inference, given `(O, z₀ = q(z | O))`, optionally optimise `z` for a small number of steps (e.g. 30 steps of Adam) on:

`z* = argmin_z [ L_surf(z) + λ_free L_free(z) + λ_prior · ||z||² ]`

This converts the amortised posterior point estimate into a MAP estimate. Use cases:
- Hard occlusions where `q(z|O)` is broad.
- Out-of-distribution car shapes (the LiDAR / Semantic-KITTI transfer benchmark).

Kill criterion in §7: must improve hidden chamfer **without** increasing free-space violation rate.

### 3.7 Multi-hypothesis sampling and reranking

Sample `K` codes `z_k ~ q(z | O)`, decode each to a mesh, score under the full posterior:

`score(z_k) = log p(O | f(·; z_k)) + log p(z_k) + λ_sym · sym_consistency + λ_rag · rag_consistency`

Return the top-1 by score, plus the K-best as auxiliary outputs for benchmarking. K ∈ {1, 4, 8}.

This **subsumes** the current `stochastic_eval_k_list` infrastructure in `eval.py`.

### 3.8 Mesh-primary output, point cloud as derivative

Final output:
1. **Mesh** via Marching Cubes on `f(x; z)` over a 64³ → 128³ grid (sparse evaluation near zero level set). Already implemented in `ss3dm_prior/mesh/marching_cubes.py`.
2. **Point cloud** sampled from the mesh (Poisson disk; 2048 points + normals from triangle face normals). This is a **downstream sample**, not the model output — solving the chamfer / mode-averaging problem at its root.

Primary metrics become `mesh_iou_at_0.5`, `surface_normal_consistency`, `watertight_fraction`. `recon_chamfer_l1` (sampled from the mesh) is reported but no longer drives optimisation.

---

## 4. Existing Infrastructure to **Reuse**

The CarNet codebase has invested heavily in components that are **fully reusable** under the new framing:

| Component | Path | New role under SPCarNet |
|---|---|---|
| LiDAR-realistic corruption (beam, incidence, range, ring) | `ss3dm_prior/data/corruptions.py` | **Provides ray evidence** — every dropped point becomes a free-space ray. Effectively becomes a richer training signal, not just a harder corruption. |
| Symmetry head + targets | `ss3dm_prior/models/symmetry_head.py`, `ss3dm_prior/data/symmetry_targets.py` | Two new uses: (a) **symmetry prior on `z`** (learned axis is consumed by `q`), (b) **likelihood term** in §3.7 reranking. |
| Retrieval anchor bank | `ss3dm_prior/retrieval/anchor_bank.py` | (a) **Posterior init** — nearest anchor `z` becomes `z₀` for MAP refinement. (b) Backbone for the **retrieval-deformation baseline** (§6). |
| Occupancy / free-space / surface query supervision | cache format v3 (`query_points_all`, `query_labels_all`, `query_ignore_mask`) | **Directly becomes** `L_surf` and `L_free`. Already persisted in `meshfleet_car_cache_v5`. **No cache rebuild needed.** |
| Marching-Cubes extractor | `ss3dm_prior/mesh/marching_cubes.py` | **Primary output path** instead of post-hoc artefact. |
| v11 cross-attention encoder | `ss3dm_prior/models/cross_attention_hybrid_v10.py` (and v11) | Becomes the encoder `q(z | O)` — head is replaced, body is reused unchanged. |
| Trainer skeleton (curriculum, wandb logging, EMA, AMP fixes) | `ss3dm_prior/engine/trainer.py` | Reused. SPCarNet introduces a new model_type (`v13_shape_posterior`); curriculum gates and metric logging carry over. |
| `_cdist_fp32_safe`, `local_frame`, `_compute_local_frame` AMP fixes | per `CarNet_v0_update_log.md` 2026-04-17 | Reused — the dtype hazards are independent of the head choice. |
| Diagnose / probe tooling (`diagnose_carnet`, `analyze_probe`) | `ss3dm_prior/tools/` | Extended with mesh-IoU and ray-evidence variants but the harness is reused. |
| Visualisation panels (`epoch_NNN/*.png`) | trainer-driven | Extended with `mesh_triptych` (input cloud / extracted mesh / GT mesh) but the cadence and storage layout carry over. |
| Wandb cleanup (consolidated metric logging from this session) | `ss3dm_prior/engine/trainer.py` | Reused — the new metric scheme already accommodates non-point outputs. |

**Net new infrastructure**: ≈ 1500 LoC (decoder, posterior head, ray-likelihood, MAP refiner, mesh-IoU eval). All other plumbing is in place.

---

## 5. Existing Infrastructure to **Demote**

| Component | Old role | New role |
|---|---|---|
| Residual decoder (`use_residual_reconstruction=true`, v0.7 path) | Primary decoder | **Baseline only** — kept as `BL-RES` in §6, retired from the active research line. |
| Point-space flow matching head (v12, v0.8/v0.8.2) | Replacement decoder | **Negative / auxiliary baseline** (`BL-PF` in §6). Retained only if the diagnostic at the end of Stage 3 shows complementarity to SPCarNet on a specific corruption mode. Otherwise removed in Stage 5. |
| `recon_chamfer_l1` over point output | Primary objective | **Secondary metric** — reported but does not drive training. Computed on points sampled from the extracted mesh, not on a direct point head. |
| `point_flow_matching_loss` | Primary FM objective | **Removed** unless the FM head is retained as auxiliary. |
| `nearest_neighbor_l1`, `reverse_nearest_neighbor_l1` | Secondary chamfer-style losses | **Removed** in SPCarNet variants — they encode point-correspondence assumptions that don't apply to a field-based decoder. |
| `recon_normal_loss` over point head | Normal supervision | Kept, but applied to **mesh face normals** at sampled surface points. |

The intent is clear: **the point cloud is no longer a model output**. It is a sample drawn from the predicted shape field for the purpose of producing comparable benchmark numbers and panels. Any loss that requires a fixed point-cloud output is structurally incompatible.

---

## 6. New Experiment Matrix

All variants run on `meshfleet_car_cache_v5` (no cache rebuild required), 150 epochs, identical optimiser/EMA settings, online wandb under project `carnet_v0_2`. Run names prefixed `spcarnet_<id>_<variant>`.

| ID | Variant | Encoder | Decoder | `z` source | Test-time refinement | Key config switch | Purpose |
|---|---|---|---|---|---|---|---|
| **BL-RES** | v0.7 residual baseline | v11 cross-attn | residual `δ` head | n/a | none | `model_carnet_v0_7.yaml` (unchanged) | Reference: 0.10 chamfer floor, 0.04 smoothing collapse |
| **BL-PF** | v0.8.2 point-flow baseline | v11 cross-attn | per-point FM (K=8, EdgeConv k=16) | n/a | none | `model_carnet_v0_8_2.yaml` (unchanged) | Reference: 0.12 ceiling, pf-loss plateau |
| **BL-OCC** | Occupancy-only baseline | v11 cross-attn | occupancy field only (no `z`) | direct from latent tokens | none | shared encoder, occupancy head used as decoder | Tests whether the *encoder* is a sufficient repair backbone before introducing `z`. |
| **AD-SF** | Object shape-field auto-decoder | **none** | implicit field `f(x; z)` | per-shape free latent (DeepSDF-style) | n/a | `model_spcarnet_ad.yaml` | **Decoder capacity ceiling** — upper-bounds what `f(·; z)` can express on clean data. |
| **EN-Q** | Amortised posterior encoder | v11 cross-attn → `(μ, log σ²)` head | shared decoder from AD-SF | `q(z | O)` | none | `model_spcarnet_en.yaml` | Headline SPCarNet variant. |
| **EN-Q-MAP** | EN-Q + MAP refinement | as EN-Q | as EN-Q | `q(z | O)` then MAP | 30-step Adam on `z` | `eval_spcarnet_en_map.yaml` (eval-only knob) | Tests whether amortised gap is closeable. |
| **EN-Q-MH** | EN-Q + multi-hypothesis | as EN-Q | as EN-Q | `K` samples from `q(z|O)` | per-sample optional MAP | `K ∈ {1, 4, 8}` | Tests posterior multimodality utility. |
| **RT-DEF** | Retrieval-deformation baseline | retrieval encoder (existing) | TPS / local deformation field | nearest anchor `z` | TPS optimisation | `model_spcarnet_rtdef.yaml` | **Kill-fallback**: if this beats EN-Q, pivot the line. |
| **EN-Q-SYM** | Symmetry-assisted posterior | as EN-Q + symmetry head | as EN-Q | `q(z | O ∪ reflect(O))` | optional MAP | `use_symmetry_head: true`, evidence doubling on σ > τ | Tests whether structured priors compound. |

### 6.1 Headline comparisons

- **BL-RES vs EN-Q**: does amortised posterior over a shape field beat residual smoothing on `recon_chamfer_l1` (mesh-sampled)? Threshold for success: ≤ 0.085 (≥ 15 % relative improvement on the v0.7 floor).
- **BL-PF vs EN-Q**: does the new framing dominate point-flow on every corruption type? Threshold: every per-corruption profile in the §1.1 sweep moves into positive `gain`.
- **EN-Q vs EN-Q-MAP**: does test-time refinement close the amortised gap? Threshold: ≥ 0.005 chamfer improvement *and* equal-or-better free-space violation rate (so we know it's not overfitting).
- **EN-Q vs RT-DEF**: does the neural posterior beat memorised + deformed retrieval? If RT-DEF wins, the Stage-6 pivot triggers.

### 6.2 Transfer benchmarks (post-headline)

- Semantic-KITTI cars (LiDAR realism transfer).
- ShapeNetCore cars (distribution shift).
- Whole parking-lot COLMAP scene (qualitative).

---

## 7. Kill Criteria

Each is a **hard threshold**, not a heuristic. Crossing it ends the corresponding sub-line and triggers a documented failure analysis (§8).

| Idea | Kill criterion |
|---|---|
| **AD-SF** (Stage 2) | If the auto-decoder cannot reach `mesh_iou_at_0.5 ≥ 0.92` and `recon_chamfer_l1 (sampled) ≤ 0.05` on **clean validation reconstructions** within 100 epochs, the implicit decoder family is too weak. **Stop the entire SPCarNet line.** Pivot to RT-DEF or revert to the v0.7 residual line. |
| **EN-Q** (Stage 3) | If `val_recon_chamfer_l1 (sampled)` does not beat **0.10** within 150 epochs (i.e. fails to improve over v0.7), and free-space violation rate fails to beat v0.7's by ≥ 20 %, the amortised posterior route is not delivering. Trigger Stage-6 pivot to RT-DEF. |
| **EN-Q-MAP** (Stage 4) | If MAP refinement improves **observed-consistency** (matching visible points) but degrades **hidden chamfer** by > 5 % or **free_space_violation_rate** by > 10 %, classify as overfitting to the visible side; revert to the no-refinement variant. |
| **EN-Q-MH** (Stage 5) | If `K = 8` does not beat `K = 1` by ≥ 0.005 chamfer **and** does not double the diversity-aware top-3 score, the posterior is too peaked / encoder is over-confident. Drop multi-hypothesis from the headline; keep as ablation. |
| **RT-DEF** (Stage 6) | If retrieval-only (no deformation) already beats EN-Q on the headline number, **pivot the entire research line to retrieval-deformation** and recast SPCarNet as the secondary contribution. |
| **EN-Q-SYM** (Stage 7-aux) | If symmetry-assisted variant fails to improve **hidden chamfer** by ≥ 0.005 on objects with high predicted symmetry σ > 0.6, the symmetry prior does not compose with the posterior — drop from the headline. |
| **Whole SPCarNet line** | If after **4 weeks of full investment** no variant beats v0.7's `val_recon_chamfer_l1 = 0.1015`, abandon SPCarNet, write a comprehensive failure analysis, and revert to the residual line with a different attack (e.g. adversarial corruption against the smoothing collapse, or geometry-token VQ-VAE). |

The kill criteria are **scheduled as gates, not optional checks**. Each stage cannot proceed past its gate without the preceding stage's metrics in the research log.

---

## 8. Required Documentation Policy

For every major change in this research line, the following five artefacts **must** be produced. Missing artefacts block the merge.

1. **Design doc** — `docs/car_model/SPCarNet_<stage>_<topic>_design.md`. Written **before** any code changes. Includes: motivation, hypothesis, math (loss form, decoder architecture), expected metrics, kill criterion link.
2. **Implementation report** — appended to the design doc or separate `_impl.md`. Includes: list of code touched (file:line), unit-level smoke results, parameter count, forward/backward time on a representative batch.
3. **Smoke-test report** — `docs/car_model/SPCarNet_<stage>_smoke.md`. Single forward + backward + optimiser step on a tiny synthetic batch. Confirms gradients reach all expected parameters; no NaNs; no obvious dtype issues.
4. **Failure analysis** — `docs/car_model/SPCarNet_<stage>_failure.md`. Required **whenever a metric regresses** vs the immediate predecessor or vs a baseline in the matrix. Must include: hypothesised cause, supporting evidence, decision (continue / pivot / kill).
5. **Research log entry** — one paragraph appended to `docs/car_model/SPCarNet_research_log.md`. Date-stamped, links back to all four artefacts above. The log is the single source of truth for "what was tried and how it went" — it replaces the per-version `carnet_progress_report_*` documents.

The research log file is created in **Stage 1** as part of the audit deliverable. The first entry documents the audit itself.

---

## 9. Minimal Implementation Roadmap

Each stage has a **gate** (the §7 kill criterion) and a **deliverable** (a doc + code merge). Stages run **sequentially** unless the kill gate forces a branch.

### Stage 1 — Object-centric data / canonicalization audit (D0–D2)

**Goal**: confirm that `meshfleet_car_cache_v5` already supports object-frame, normalised, ray-aware training without a cache rebuild.

Tasks:
- Inventory per-shape canonical frame, body-axis convention, scale normalisation. Verify that cars are consistently aligned (front +x, up +z) or document the residual misalignment magnitude.
- Confirm `query_points_all`, `query_labels_all`, `query_ignore_mask` integrity (cache format v3, already present).
- Verify per-corruption masks expose the **dropped points** so they can be reused as free-space rays in LiDAR-realistic mode.
- Decide: do we need a per-object frame transform or is per-patch local-frame sufficient? (Probably per-object for SPCarNet — patches are sub-regions of one object.)

Deliverable: `docs/car_model/SPCarNet_stage1_data_audit.md` + first entry in `SPCarNet_research_log.md`.

**Gate**: cache audit shows no blocking gap. If a cache rebuild is required, **stop and re-plan**: cache rebuild is 12+ h and must be its own stage.

---

### Stage 2 — Shape-field auto-decoder (D3–D7)

**Goal**: prove the decoder family `f(x; z)` can represent clean MeshFleet cars.

Tasks:
- Implement `ss3dm_prior/models/shape_field_decoder.py` — modulated MLP, occupancy primary, SDF as ablation.
- Auto-decoder training: per-shape `z` jointly optimised; surface + free-space queries from the existing cache.
- Evaluation: extract mesh via existing Marching-Cubes module, report `mesh_iou_at_0.5`, `surface_normal_consistency`, `chamfer_l1` (mesh-sampled).
- Smoke: 1 shape, 100 steps, produces a recognisable car mesh.

Deliverable: `SPCarNet_stage2_autodecoder_design.md`, `_impl.md`, `_smoke.md`.

**Gate (§7)**: `mesh_iou_at_0.5 ≥ 0.92` and `chamfer_l1 ≤ 0.05` on clean val. If failed → **stop the whole line.**

---

### Stage 3 — Amortised posterior encoder (D8–D14)

**Goal**: replace per-shape `z` with `q(z | O)` and train end-to-end on corrupted observations.

Tasks:
- Implement `q(z | O)` head on top of the v11 cross-attention encoder. Reparameterised Gaussian.
- Implement the three likelihood terms (`L_surf`, `L_free`, optional `L_ray`).
- KL prior, clean-shape regulariser.
- New `model_type: v13_shape_posterior`. New configs `configs/ss3dm_prior/spcarnet_v0_en/`.
- Train EN-Q for 150 ep on `meshfleet_car_cache_v5`.

Deliverable: `SPCarNet_stage3_posterior_design.md`, `_impl.md`, `_smoke.md`, **headline metric report**.

**Gate (§7)**: EN-Q `val_recon_chamfer_l1 (sampled)` ≤ 0.10 (matches v0.7) within 150 ep, **and** free-space violation strictly better than v0.7. If failed → trigger Stage-6 pivot.

---

### Stage 4 — Ray / free-space MAP refinement (D15–D17)

**Goal**: close the amortised gap on hard cases.

Tasks:
- Implement `ss3dm_prior/eval/map_refine.py` — gradient descent on `z` given `O`, with the three likelihood terms.
- Eval-only — no training change.
- Sweep refinement steps ∈ {0, 10, 30, 100}, step size ∈ {1e-3, 1e-2}.
- Report per-corruption-type breakdown of the improvement.

Deliverable: `SPCarNet_stage4_map_design.md`, `_impl.md`, `_smoke.md`.

**Gate (§7)**: ≥ 0.005 chamfer improvement and equal-or-better free-space rate. If overfits to visible side, drop. (Stage 5 still runs without MAP.)

---

### Stage 5 — Multi-hypothesis sampling and reranking (D18–D20)

**Goal**: convert posterior breadth into a usable signal.

Tasks:
- Sample `K ∈ {1, 4, 8}` codes from `q(z|O)`, decode K meshes, score under full posterior + symmetry + retrieval terms.
- Reuse the existing rerank infrastructure from `eval.py` (`stochastic_eval_k_list`, `stochastic_rerank_weights`).
- Report `best_of_k_hidden_chamfer`, `reranked_hidden_chamfer`, and `sample_diversity`.

Deliverable: `SPCarNet_stage5_mh_design.md`, `_impl.md`, `_smoke.md`.

**Gate (§7)**: K=8 beats K=1 by ≥ 0.005 chamfer **and** doubles top-3 diversity. Otherwise drop multi-hypothesis from the headline.

---

### Stage 6 — Retrieval-deformation alternative (D21–D27, **conditional**)

**Goal**: only built if EN-Q (Stage 3) fails its gate or RT-DEF beats EN-Q in Stage 5 ablations.

Tasks:
- Implement `ss3dm_prior/models/retrieval_deformation.py` — TPS / FFD over the nearest anchor's mesh, fitted to the partial observation.
- Compare against the residual decoder and EN-Q on the headline benchmark.
- If RT-DEF wins, **rewrite the paper story** around retrieval-deformation with SPCarNet (or v0.7) as ablations.

Deliverable: `SPCarNet_stage6_rtdef_design.md` + paper-story-pivot decision memo.

---

### Stage 7 — NeurIPS-grade benchmark / report generator (D28–D30)

**Goal**: make the headline numbers paper-ready.

Tasks:
- Generate the full §6 experiment matrix as a single auto-generated report: comparison tables, per-corruption breakdowns, qualitative panels (corrupted input / extracted mesh / GT mesh) per benchmark.
- Implement transfer evaluation harness for Semantic-KITTI cars.
- Write `docs/car_model/SPCarNet_paper_results.md` — the final results document.
- Aggregate ablations: K-sweep, symmetry σ-threshold, anchor count, MAP iterations, corruption leave-one-out.

Deliverable: paper-ready figures (`outputs/spcarnet/figures/*.pdf`) and tables (`*.tex` / `*.md`).

---

## 10. Risks and Open Questions

| Risk | Mitigation |
|---|---|
| Implicit-field training is slow (per-step grid query). | Sparse query sampling near the level set; pre-computed query-point cache per shape (already in v3 cache). |
| `q(z|O)` posterior collapse (KL term too strong). | β-VAE schedule; KL warmup over 10 epochs (already in CarNet trainer scaffolding). |
| Ray evidence requires LiDAR corruption pipeline integration; the synthetic corruption pipeline doesn't expose rays. | Stage 1 audit explicitly flags this; LiDAR-realistic corruption is wired to expose ray endpoints already (per `CarNet_v0_update_log.md` 2026-04-17, Phase 3). |
| Mesh-IoU metric requires `scikit-image`; not all eval boxes have it. | Marching-Cubes module already has graceful fallback (returns mesh=None, retains volume IoU). Stage 1 audits the eval boxes. |
| 4-week window may be too tight for all 7 stages. | Stages 6 and 7 are conditional / final. Stages 1–5 are the critical path; if Stage 3 lands by D14, stages 4–5 fit in the remaining 16 days. |
| Reviewers may consider "implicit shape repair" derivative of DeepSDF / OccNet / ConvONet. | The headline novelty is **observation-conditioned posterior with ray + free-space + symmetry + retrieval evidence**, not the implicit decoder itself. Position the paper as posterior inference, not as another implicit-field architecture. |

---

## 11. Decision Surface

The proposed line **commits** to:
1. Treating mesh repair as posterior inference, not regression.
2. Replacing the point-cloud output with a shape-field output; mesh becomes primary.
3. Reusing every encoder, query-cache, corruption, retrieval, and Marching-Cubes component already in the repository — **no green-field rewrite**.
4. Demoting (not deleting) the v0.7 residual and v0.8.x point-flow paths to baselines.
5. Hard kill criteria with scheduled gates.

The proposed line **does not** commit to:
1. A specific implicit-field parameterisation (occupancy vs SDF — Stage 2 ablation).
2. Specific KL schedule / decoder width (Stage 2/3 hyperparameter scan).
3. The Semantic-KITTI transfer story landing within the 4-week window — Stage 7 is contingent on Stage 5 success.

---

## Appendix A — Symbol Glossary

- `O` — corrupted observation `(P_obs, R_rays, F_free)`.
- `z` — latent shape code (target dim 256).
- `f(x; z)` — implicit shape field (occupancy or SDF).
- `q(z | O)` — amortised posterior encoder (Gaussian).
- `p(z)` — Gaussian prior (or learned via the auto-decoder pre-train).
- `L_surf, L_free, L_ray` — three observation likelihood terms.
- `BL-*, AD-*, EN-*, RT-*` — experiment matrix IDs from §6.

## Appendix B — Document Index Required for This Line

| File | Stage | Status |
|---|---|---|
| `docs/car_model/SPCarNet_radical_RFC.md` | this RFC | **proposed** |
| `docs/car_model/SPCarNet_research_log.md` | Stage 1 onwards | not yet created |
| `docs/car_model/SPCarNet_stage1_data_audit.md` | Stage 1 | not yet created |
| `docs/car_model/SPCarNet_stage2_autodecoder_{design,impl,smoke}.md` | Stage 2 | not yet created |
| `docs/car_model/SPCarNet_stage3_posterior_{design,impl,smoke}.md` | Stage 3 | not yet created |
| `docs/car_model/SPCarNet_stage4_map_{design,impl,smoke}.md` | Stage 4 | not yet created |
| `docs/car_model/SPCarNet_stage5_mh_{design,impl,smoke}.md` | Stage 5 | not yet created |
| `docs/car_model/SPCarNet_stage6_rtdef_{design,impl,smoke}.md` | Stage 6 | conditional |
| `docs/car_model/SPCarNet_paper_results.md` | Stage 7 | not yet created |

---

_End of RFC._
