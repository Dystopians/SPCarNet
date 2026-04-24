# CarNet_v0 — Plan & Lineage

This document names the current research line **CarNet** and registers its
first formal version **CarNet_v0**. The scope is deliberately reframed: the
deliverable is a **general corrupted-mesh repair model**, with particular
emphasis on **LiDAR-scanned meshes**. Cars (MeshFleet_TRELLIS) remain the
primary experimental subject but are no longer the scope.

Submission target: **NeurIPS**.

---

## 0. Version Lineage

All prior car-model work is listed here so that CarNet_v0 sits on an explicit
inheritance chain instead of appearing as a fresh branch.

| Tag | Status | Params | Key idea | Artifacts |
|-----|--------|--------|----------|-----------|
| `train_v8_car_novis` | archived | ~8.4M | First whole-car baseline. Observed conditioning on, visibility-aware metrics off. | `outputs/ss3dm_prior_car/train_v8_car_novis/`, formal eval in `eval_v8_car_novis/` |
| `train_v3_car_novis` | archived | ~8.4M | Intermediate car run, preceded the trio. | `outputs/ss3dm_prior_car/train_v3_car_novis/` |
| `v3_5_baseline` / `v3_5_occ` / `v3_5_occ_vq` | archived | ~17.2M | Trio scale-up. Introduced (intended) occupancy supervision and VQ prototype codebook. Diagnosed the three dominant bugs in the pipeline. | `configs/ss3dm_prior/{model,train}_v3_5_car_*.yaml`, `outputs/ss3dm_prior_car/v3_5_ablations/`, report `outputs/ss3dm_prior_car/v3_5_experiment_report.md` |
| `v4_baseline` / `v4_occ` / `v4_occ_vq` | superseded | ~17.2M | Re-run of the v3.5 trio after the cache+trainer bug-fix pass (visibility split, occupancy/free-space/unknown query supervision, VQ collapse mitigation, data-driven metric filter, wandb redundancy cleanup). Same configs as v3.5 but on a corrected cache. | `configs/ss3dm_prior/{model,train}_v3_5_car_*.yaml` (reused), `scripts/car_model/train_meshfleet_car_v4_trio.sh`, cache `meshfleet_car_cache_v4/`, report `outputs/ss3dm_prior_car/v4_experiment_report.md`, focus note `docs/car_model/v4_focus.md` |
| **`CarNet_v0`** | **active** | TBD (baseline ~30M) | First version under the rebranded research line. Combines A1 (conditional latent flow matching), A2 (learned symmetry prior), A3 (retrieval-augmented decoder), plus LiDAR-realistic corruption. Retires the v3.5/v4 deterministic single-sample paradigm. | This document. Configs to be added under `configs/ss3dm_prior/carnet_v0/`. |

The trio ablation slots (**baseline / occ / occ_vq**) continue as sub-variants
where applicable, but CarNet_v0 introduces its own ablation matrix (see §6).

### Key fixes inherited from the v3.5 → v4 pass

These are already landed on `clean-submit` and CarNet_v0 builds on them:

1. **Cache query supervision**: `build_car_mesh_patch_cache` now emits
   `visible_clean_points` / `hidden_clean_points` (cosine-threshold visibility
   split) and `surface/free/unknown_query_points` with `query_points_all +
   query_labels_all + query_ignore_mask`. Bumps cache format version `1 → 2`.
2. **VQ collapse mitigation**: `codebook_size 128→64`, `vq_commitment 0.1→0.05`,
   `prototype_diversity 0.02→0.01`, `vq_start_epoch 2→12`.
3. **Trainer metric filter**: removed the name-based whole-car hard-drop of
   visibility/occupancy metrics; filter is now data-driven (NaN + weight ≤ 0)
   so new-cache runs naturally surface real numbers.
4. **Wandb cleanup**: dropped triplicate loss keys (`epoch/{split}_loss_raw/*`
   and `train_step/loss_raw/*` removed in favour of canonical `{split}/X_loss`
   plus a single weighted-contribution view). Added
   `epoch_visualization_interval_epochs` knob (default 5) to throttle panel
   uploads. Silenced `score_spearman` / matplotlib warning spam.
5. **Retrieval filter**: whole-car runs drop `retrieval_top5_*` and
   `retrieval_top1_self_aligned` (saturate to 1.0 since each car is its own
   sequence).

---

## 1. CarNet_v0 Scope Statement

CarNet is not an SS3DM sub-project and is not restricted to cars. The
positioning is:

> A conditional generative prior for corrupted 3D geometry, trained on
> synthetic and realistic corruption streams, transferable across object
> classes and scanner modalities (camera multi-view, LiDAR, synthetic).

Car meshes are the **primary benchmark** because (a) MeshFleet provides 1616
clean, normalized meshes, (b) cars have strong latent structure (bilateral
symmetry, part composition) that makes representation learning tractable,
(c) prior work is sparse on whole-object repair at this scale.

The **secondary target** is LiDAR-scanned meshes (e.g. Semantic-KITTI,
KITTI-360 cars; optionally nuScenes) where the corruption distribution is
physically grounded (beam occlusion, incidence-angle dropout) rather than
synthetic. A credible NeurIPS submission must show non-trivial transfer to
at least one real-LiDAR benchmark.

**Downstream scenario that drives the architectural choice**: the model must
eventually operate on **large parking-lot COLMAP scenes** (many cars per
scene, mixed partial-observation quality). This motivates the patch-centric
architecture (D1 decided): a patch index tiles any scene regardless of scale,
each patch is denoised independently, and the SS3DM-style cross-patch
evaluation applies. Whole-object token models would have to be re-engineered
for scene-scale input.

---

## 2. Paper Story (one paragraph)

CarNet_v0 proposes **SR-AFM: Symmetric Retrieval-Augmented Flow Matching for
Corrupted-Mesh Repair**. Given a corrupted observation (point cloud + sparse
visible subset) of an object, CarNet_v0 generates a clean geometry completion
by (i) performing **conditional latent flow matching** over a learned shape
manifold, (ii) enforcing a **learned soft symmetry prior** that detects an
object's dominant symmetry plane and penalises asymmetric completions when
confidence is high, and (iii) **retrieving top-K clean anchors** from an
external memory bank and cross-attending to them during decoding. At inference
K candidate completions are sampled via Euler ODE integration of the velocity
field and reranked by a consensus score combining observed consistency,
free-space safety, and symmetry consistency.

The three contributions are independently ablatable and together produce a
model that (a) generates plausible completions rather than regressing to a
mean blob (flow matching), (b) exploits object-category structure without
hard-coding it (learned symmetry), and (c) scales with the training
distribution via RAG rather than sheer parameter count.

---

## 3. Prior-Art Gap

| Axis | Existing work | CarNet_v0 position |
|------|---------------|--------------------|
| Deterministic regression (PoinTr, AdaPoinTr, SnowflakeNet) | strong but single-mode; fails on ambiguous completions | generative, multi-candidate |
| Point-cloud diffusion (LION, PVD, Point-E) | strong priors but no conditioning on partial observation + free-space constraints; no symmetry prior | observation-conditional flow; free-space aware rerank |
| Symmetry priors (Mirror3D, EPN) | hard-coded or SO(3)-equivariant; expensive | lightweight learned head; class-agnostic soft constraint |
| Retrieval-augmented 3D (few papers — Text2Shape, ShapeCrafter, some retrieval-then-refine) | retrieval is coarse and one-shot; not fused into decoder | top-K anchors as cross-attention context, jointly trained |
| LiDAR mesh repair (EfficientLPS-mesh, KISS-ICP artifacts) | heuristic mesh cleanup | learned prior with realistic corruption |

**The headline claim**: first unified pipeline that combines observation-
conditional flow matching, learned symmetry, and 3D RAG, with explicit
evaluation under LiDAR-realistic corruption.

---

## 4. Architecture (CarNet_v0)

Reuses (and extends) the `ss3dm_prior` model zoo. Starting point:
`LatentFlowHybridPatchPriorV11`, which already inherits
`CrossAttentionHybridPatchPriorV10`.

### 4.1 Encoder stack (inherited from v10/v11, already implemented)

- Multi-stream PointNet encoders for `corrupted / observed / visible / hidden
  / clean-teacher` inputs.
- Cross-attention core: 48 learnable latent queries × 4 cross-attention layers
  × 2 self-attention refinement layers. Keys/values are pooled encoder tokens.
- Optional VQ codebook (size 64 per v3.5→v4 fix).

### 4.2 Flow-matching head (A1 — mostly implemented in v11)

- `LatentFlowResidualHead`: time-conditioned velocity MLP in latent space.
- Training schedule: linear interpolation `z_t = (1−t) · ε + t · z_target`
  with t ∼ U(0,1). Target velocity = `z_target − ε`.
- Conditioning vector: concat of `(fused_latent, corrupted_summary,
  observed_summary, visible_summary, hidden_summary)` projected through
  `flow_condition_proj`.
- Loss: MSE velocity matching (`latent_flow_matching_loss`, weight TBD
  but non-zero for CarNet_v0).

**CarNet_v0 additions** (to implement):

- Replace fixed-step Euler integrator with adaptive Heun or Dopri5 (torchdiffeq).
- Classifier-free guidance: 10% dropout of observation conditioning during
  training; guided sampling at inference with scale γ ∈ [1, 3].
- Variance-reducing noise schedule (logit-normal or cosine) instead of uniform
  `t ∼ U(0,1)`.

### 4.3 Symmetry head (A2 — new)

- `SymmetryHead`: takes `fused_latent` (post cross-attention, pre-VQ) →
  predicts `(n ∈ S², d ∈ ℝ, σ ∈ [0,1])` where `(n,d)` parameterise a reflection
  plane in the patch's canonical frame and σ is symmetry confidence.
- During training supervise σ weakly: compute ground-truth symmetry score of
  `clean_points` by fitting best reflection plane (closed-form SVD + Chamfer
  residual) and use it as a soft target for σ.
- Loss contribution:
  ```
  L_sym = σ · Chamfer(recon_points, reflect(recon_points, n, d))
        + λ_plane · ||n − n_target||₂² + |d − d_target|   (when σ_target > τ)
        + λ_σ · BCE(σ, σ_target)
  ```
- At inference, if σ > τ (e.g. 0.5), include mirrored `observed_points` as
  additional context tokens — free 2× effective sensor coverage for symmetric
  objects.
- Class-agnostic: on asymmetric scenes σ → 0 and the loss vanishes. (**D2**)

### 4.4 Retrieval-augmented decoder (A3 — new)

**D3 locked**: CarNet_v0 anchor bank is built from MeshFleet training splits
only. A mixed ShapeNetCore + MeshFleet bank is deferred to CarNet_v0.1. The
infrastructure below is designed bank-agnostic so the switch is a config
change only.


- **Anchor bank construction**: at the end of each epoch (or periodically),
  forward all training samples in eval mode, store `(clean_retrieval_embedding,
  clean_latent_tokens_top_K)`. Persist to disk as FAISS index
  (`IndexFlatIP`) + `.npz` of anchor latents. ~1600 cars × 256-d
  embedding + 48 × 512 latent tokens ≈ 150 MB.
- **Inference-time retrieval**: compute corrupted sample's
  `retrieval_embedding` → FAISS top-K (K=3 by default) → pull K sets of
  anchor latent tokens.
- **Decoder integration**: anchor tokens concatenated into the cross-attention
  KV pool alongside encoder tokens. Learnable type-embedding distinguishes
  `{corrupted, observed, visible, hidden, query, anchor}` tokens.
- **Training**: 70% of steps use the bank snapshot from previous epoch
  (stale but stable); 30% use an in-batch shortcut (anchors drawn from other
  batch members' clean embeddings) to avoid a chicken-and-egg start. Both
  modes share one head so the model sees anchors from the beginning.
- **Retrieval regularisation**: retain `retrieval_align_loss` but add an
  auxiliary loss: reconstructed anchor-conditional output should not stray
  from clean geometry further than the non-anchor-conditional output
  (monotonicity: anchors never hurt).

### 4.5 Best-of-K sampling and reranking (inherited from v11 `eval.py`)

Already implemented in `eval.py` for the `is_stochastic_v11` branch:

- Sample K ∈ {1, 4, 8} latents via ODE.
- Per-candidate metrics: hidden Chamfer, free-space violation rate,
  observed-consistency, prototype similarity.
- Reranking: weighted sum `score = w_obs · obs_cons − w_fs · free_violations
  + w_sym · sym_consistency + w_proto · proto_align`. **CarNet_v0 adds
  symmetry term** and exposes weights in config.

### 4.6 Dual output: point cloud + Marching-Cubes mesh (D4)

CarNet_v0 produces two output artifacts per patch:

1. **Point cloud head** (primary, inherited from v10/v11):
   `recon_points ∈ ℝ^{N×3}` + `recon_normals ∈ ℝ^{N×3}`, supervised by
   Chamfer L1 and normal cosine.
2. **Marching-Cubes mesh** (auxiliary, new):
   - Query the existing occupancy head on a regular grid inside the patch
     bounding ball (default 64³ resolution, sparse-evaluated where the
     nearest-surface distance from the point-cloud head is below a margin).
   - Run Marching Cubes on the predicted occupancy field → triangle mesh.
   - At whole-scene inference (parking-lot COLMAP), adjacent patches are
     merged by overlap-weighted averaging of occupancy values before MC is
     run on the merged field, giving a seam-free mesh.
   - Eval-time only; training cost unchanged. Training remains point-cloud
     + occupancy-BCE + free-space; the mesh is a post-hoc artefact of the
     occupancy field.

This means we can report **`mesh_iou_at_0.5`** and **`surface_normal_consistency`**
(mesh ↔ GT mesh) as first-class metrics alongside Chamfer, and downstream
consumers get a watertight mesh rather than only a sampled point cloud.

---

## 5. LiDAR-Realistic Corruption (new corruption module)

The current `corruptions.py` has six modes, all position-jitter + random
dropout, none of them physically grounded. CarNet_v0 adds a second corruption
pipeline specifically for LiDAR realism.

### 5.1 New corruption types (to implement in `corruptions.py`)

- **`beam_occlusion`**: given a simulated scanner pose `p_s`, remove any
  clean point whose ray from `p_s` hits the mesh elsewhere first (i.e. the
  point is self-occluded). Produces realistic missing-surface patterns.
- **`incidence_angle_dropout`**: keep each point with probability
  `cos(θ_incidence)^k`, modelling the fact that grazing-angle returns are
  weak. k ∈ [2, 4].
- **`range_dependent_noise`**: per-point gaussian σ scales with
  `||p − p_s||²` (inverse-square return intensity).
- **`azimuthal_ring_sparsity`**: pre-compute per-point azimuth from scanner
  and drop points in narrow `Δφ` bands to simulate column-wise beam spacing.

Selection between classical synthetic corruptions and LiDAR-realistic
corruptions is config-driven: `corruptions.pipeline: [synthetic | lidar |
mixed]`.

### 5.2 Scanner pose sampling

- Default: sample `p_s` on a hemisphere of radius 3 × mesh radius above the
  ground plane; jitter slightly.
- For asymmetric cases (toppled or occluded vehicles), sample 2 scanner poses
  to mimic multi-scan accumulation.

### 5.3 Why this matters

- Under uniformly random dropout, the model learns to fill isotropic holes —
  a task of limited practical value.
- Under beam-occlusion, the model must reason about **half-visible objects**
  (the side facing the scanner is dense, the far side is empty). This is
  exactly the real-world LiDAR failure mode and is where our symmetry and
  retrieval priors should shine.

---

## 6. Experiment Matrix

### 6.1 Main comparison (MeshFleet_TRELLIS_RECONSTRUCTED_v4)

| Variant | A1 flow | A2 sym | A3 RAG | Corruption | Expected primary gain |
|---------|:-------:|:------:|:------:|------------|-----------------------|
| `carnet_v0_det` | ✗ | ✗ | ✗ | synthetic | matches v4 baseline (sanity) |
| `carnet_v0_flow` | ✓ | ✗ | ✗ | synthetic | `best_of_k_hidden_completion` ≫ det |
| `carnet_v0_flow_sym` | ✓ | ✓ | ✗ | synthetic | lower hidden Chamfer on occluded side |
| `carnet_v0_flow_rag` | ✓ | ✗ | ✓ | synthetic | better tail (rare car types) |
| `carnet_v0_full` | ✓ | ✓ | ✓ | synthetic | main headline result |
| `carnet_v0_full_lidar` | ✓ | ✓ | ✓ | LiDAR-sim | generalization story |

### 6.2 Transfer evaluation (D5: both zero-shot and light-finetune)

For each transfer target below we report **two numbers**: zero-shot (checkpoint
trained on MeshFleet only, evaluated directly) and light-finetune (10–20% of
the target's train split, ≤ 5 epochs). Zero-shot defends the generality
claim; light-finetune is the headline number.

1. **Semantic-KITTI cars** (primary LiDAR benchmark, ~30k car instances after
   per-instance mesh extraction).
2. **ShapeNetCore cars** (distribution shift; classical clean-to-corrupted).
3. (Stretch) **nuScenes LiDAR cars** or a small pedestrian/cyclist subset —
   proves the "general mesh repair" claim beyond cars.
4. (Stretch) **Large parking-lot COLMAP scene** — qualitative whole-scene
   reconstruction, consistent with the D1 motivation.

### 6.3 Ablations beyond the matrix

- K sweep: K ∈ {1, 2, 4, 8, 16}. Diminishing-returns curve.
- Guidance scale γ ∈ {1, 1.5, 2, 3}. Quality vs diversity tradeoff.
- Symmetry confidence threshold τ.
- Anchor count per query K_anchor ∈ {0, 1, 3, 5}.
- Corruption-type leave-one-out (which corruption mode contributes most).

### 6.4 Metrics

Primary (point-cloud head):
- `recon_chamfer_l1`
- `best_of_k_hidden_completion_chamfer`
- `reranked_hidden_completion_chamfer`
- `free_space_violation_rate`
- `recon_normal_cosine`
- `symmetry_consistency` (new: Chamfer between recon and its reflection,
  weighted by predicted σ)

Primary (mesh head — enabled by D4):
- `mesh_iou_at_0.5` (occupancy > 0.5 vs GT volume)
- `surface_normal_consistency` (normals of MC-extracted mesh vs GT mesh)
- `mesh_watertight_fraction` (fraction of patches producing a closed mesh
  after seam-merging)

Secondary:
- `sample_diversity` (mean pairwise Chamfer among K candidates)
- `retrieval_top1_nonself` (bank health sanity)

---

## 7. Work Plan and Phasing

Status key: **DONE** = implemented and smoke-tested; **PENDING** = not yet
touched. A detailed chronological record of changes lives in
`docs/car_model/CarNet_v0_update_log.md`.

### Phase 0 — Stabilise v4 trio · **DONE (superseded)**

- Cache format v3 rebuild landed. Symmetry targets now persisted alongside
  the visibility split and occupancy queries. The v4 trio is subsumed by
  `carnet_v0_det` (see §6.1) rather than run separately.

### Phase 1 — CarNet_v0 flow baseline · **DONE**

- Configs: `configs/ss3dm_prior/carnet_v0/model_carnet_v0_flow.yaml`,
  `train_carnet_v0_flow.yaml`. Model type `v11_latent_flow_hybrid`;
  `latent_dim=512`, `recon_point_count=2048`, `stochastic_flow_steps=16`,
  `stochastic_eval_k_list=[1,4,8]`.
- Trainer curriculum extended with `flow_matching_start_epoch` (default
  `main_start_epoch`, set to 6 in CarNet_v0 configs).
- Single-variant launcher: `scripts/car_model/train_carnet_v0_flow.sh`.

### Phase 2 — Learned symmetry (A2) · **DONE**

- `ss3dm_prior/data/symmetry_targets.py`: closed-form PCA+Chamfer
  estimator of the dominant reflection plane + soft confidence σ.
- `ss3dm_prior/models/symmetry_head.py`: lightweight MLP head producing
  (n, d, σ) from the fused pre-VQ latent; integrates into v11 behind an
  opt-in `use_symmetry_head` flag.
- `ss3dm_prior/losses.py::symmetry_consistency_loss`: three components
  (self-symmetry Chamfer, plane regression, confidence BCE) with a loss
  key `symmetry_consistency_loss`.
- Cache builder writes `symmetry_plane_normal`, `symmetry_plane_offset`,
  `symmetry_target_confidence`, `symmetry_chamfer_residual` into every
  NPZ. **Cache format bumped 2 → 3.**
- Trainer curriculum adds `symmetry_start_epoch` (default 4 in CarNet_v0).

### Phase 3 — LiDAR-realistic corruption · **DONE**

- New corruption types in `corruptions.py`:
  `beam_occlusion`, `incidence_angle_dropout`, `range_dependent_noise`,
  `azimuthal_ring_sparsity`, plus hemispherical scanner pose sampling.
- Configured in YAML via a top-level `corruptions.lidar` block; classical
  synthetic types remain side-by-side so mixed pipelines are just two
  config blocks.
- Used by `carnet_v0_full_lidar` variant.

### Phase 3b — Marching-Cubes mesh extractor (D4) · **DONE**

- New module `ss3dm_prior/mesh/marching_cubes.py` with
  `extract_patch_mesh()`, `save_patch_mesh()`, `mesh_iou_at_iso()`,
  `surface_normal_consistency()`.
- Graceful fallback when `scikit-image` is missing (returns mesh=None and
  a warning); IoU / volume metrics remain available.
- Eval-time only; training code untouched. Integration into `eval.py` is
  a follow-up — callers can already import the module directly.

### Phase 4 — Retrieval augmentation (A3) · **DONE (in-batch variant)**

- `ss3dm_prior/retrieval/anchor_bank.py`: FAISS `IndexFlatIP` +
  NumPy-cosine fallback. Source-agnostic API so a future
  ShapeNetCore+MeshFleet mixed bank (D3 roadmap / CarNet_v0.1) plugs in
  via a different record iterable.
- v11 gains `use_retrieval_anchors`, `num_retrieval_anchors`,
  anchor-token projection, and a learnable type embedding.
- Training uses an **in-batch shortcut** (cyclic-shifted `latent_seed`)
  so the RAG pathway trains from epoch 0 without a pre-built bank.
- Persistent end-of-epoch bank rebuild is deferred (see §9 kill criteria);
  the in-batch shortcut is the stronger simplification for the first
  NeurIPS cut.

### Phase 5 — Transfer eval & paper experiments (~2 weeks, D5) · **PENDING**

- Semantic-KITTI per-instance car mesh extraction pipeline.
- ShapeNetCore-cars evaluation preprocessing.
- **Zero-shot pass** on each transfer target (no weight updates).
- **Light-finetune pass** on each transfer target (≤ 5 epochs on 10–20% of
  the target train split).
- Full ablation matrix of §6.1.
- Stretch: parking-lot COLMAP qualitative demo (D1 motivation payoff).
- Writing phase.

---

## 8. Design Decisions (all locked)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| **D1** | Architecture granularity. | **Patch-centric.** | Future target scenarios include large parking-lot COLMAP scenes (many cars per scene); the patch index tiles natively to arbitrary scene scale. Whole-object token budgets would have to be re-engineered. |
| **D2** | Symmetry prior form. | **Generic learned reflection plane.** | Keeps the "general mesh repair" positioning. Under D1 a patch often covers only part of an object (a door, a roof segment); a learned confidence σ correctly down-weights the loss on asymmetric or partial patches, whereas a hard bilateral-Y assumption would inject wrong structural bias. |
| **D3** | Anchor-bank composition. | **MeshFleet-only for CarNet_v0;** mixed (ShapeNetCore + MeshFleet) deferred to CarNet_v0.1. | Controls scope — one new subsystem per version. Mixed bank stays on the roadmap but doesn't gate v0 submission. |
| **D4** | Output format. | **Dual:** point-cloud head (primary) **+ Marching-Cubes mesh** extracted from the occupancy head (auxiliary). | IoU becomes reportable, downstream mesh-consumers are unblocked, and the occupancy head already exists — marginal cost is a per-patch MC pass at eval time plus a seam-merging utility for whole-scene outputs. |
| **D5** | LiDAR transfer evaluation. | **Both zero-shot and light-finetune.** | Zero-shot defends the generality claim; light-finetune gives a stronger headline number on real-LiDAR benchmarks. Extra ~1 week compute accepted. |

---

## 9. Risks and Kill Criteria

- **A1 (flow matching) doesn't beat deterministic by epoch 20**: if
  `best_of_4_hidden_chamfer` is within 3% of deterministic, the flow story
  is dead. Fall back to diffusion (VP-SDE) instead of flow matching — similar
  code path, slightly more inductive bias.
- **A2 (symmetry) confidence collapses to 0 or 1 everywhere**: predicted σ
  is useless. Mitigation: warm start from closed-form SVD plane; add an
  entropy regulariser on σ.
- **A3 (RAG) top-1 retrieval ≠ similar car**: bank quality is the bottleneck.
  Mitigation: supervised contrastive auxiliary loss on
  `(clean_embedding, clean_embedding)` pairs of visually similar cars
  (defined by PointNet classifier).
- **LiDAR transfer fails outright**: the gap between corrupted-synthetic and
  real-LiDAR is too wide. Mitigation: include a small real-LiDAR fine-tune
  dataset (~few hundred samples) in the main training stream.

---

## 10. Non-Goals for CarNet_v0

- Real-time inference (K-sample ODE integration is not fast).
- Texture / appearance repair (geometry only).
- Classification / segmentation (we're not building a generic 3D backbone).
- Deformable / non-rigid objects (clothes, organics).

---

## 11. Pointers

- Current trainer: `ss3dm_prior/engine/trainer.py`
- Current model zoo: `ss3dm_prior/models/`
- Flow-matching prior stub: `ss3dm_prior/models/latent_flow_patch_prior_v11.py`
- Cross-attention decoder: `ss3dm_prior/models/cross_attention_patch_prior_v10.py`
- Best-of-K eval (already implemented): `ss3dm_prior/eval.py`, the
  `is_stochastic_v11` branch (lines ~390–724).
- Car cache builder: `ss3dm_prior/tools/build_car_mesh_patch_cache.py`
- Prior reports:
  - `outputs/ss3dm_prior_car/v3_5_experiment_report.md`
  - `outputs/ss3dm_prior_car/v4_experiment_report.md`
  - `docs/car_model/v4_focus.md`
