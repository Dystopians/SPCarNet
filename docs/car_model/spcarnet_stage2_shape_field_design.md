# SP-CarNet Stage 2 — Shape-Field Auto-Decoder (Design)

| Field | Value |
|---|---|
| Stage | 2 / 7 (per `SPCarNet_radical_RFC.md` §9) |
| Status | DESIGN (precedes implementation) |
| Date | 2026-04-29 |
| Predecessor | Stage 1 (`spcarnet_stage1_object_cache_report.md`) |
| Successor | Stage 3 (amortised posterior encoder) |

---

## 0. Purpose & Gate

Stage 2 implements the **decoder-capacity ceiling** for SP-CarNet: a per-object auto-decoded implicit shape field `f(x; z_i)` trained jointly with per-object latent codes `z_i ∈ ℝ^{d_z}`. The output is an occupancy (or signed-distance) value at any query point in canonical object coordinates, and a Marching-Cubes mesh extracted on demand.

Stage-2 gate (per RFC §7):
> **Pass**: clean validation reconstructions reach `mesh_iou_at_0.5 ≥ 0.92` and `recon_chamfer_l1 (mesh-sampled) ≤ 0.05`.
> **Fail**: stop the entire SP-CarNet line; revert to retrieval-deformation or to v0.7.

Stage 2 **does not** see corrupted input. The encoder side comes in Stage 3. The whole purpose of Stage 2 is to upper-bound what the decoder family can express on clean shapes; if that upper bound is itself worse than v0.7's 0.10 floor, no downstream amortised path will rescue it.

---

## 1. Occupancy vs SDF

**Decision**: occupancy primary; SDF as a Stage-2 ablation switch.

Reasons:

1. **Existing supervision is occupancy-shaped.** The cache stores `query_points_all` with `query_labels_all ∈ {0, 1}` and an `query_ignore_mask`. `surface_query_labels = 1` and `free_query_labels = 0` are the unambiguous ingredients of a binary BCE objective. Using SDF would require either re-discretising those labels into a margin-free banded SDF (lossy and slow) or sampling a fresh distance field at every step.
2. **Marching-Cubes is iso-level-agnostic but binary-friendly.** The existing `extract_patch_mesh()` API (`ss3dm_prior/mesh/marching_cubes.py`) consumes occupancy probabilities and thresholds at `iso_level=0.5`. Occupancy plugs in directly.
3. **No eikonal regulariser overhead.** SDF requires `||∇f|| ≈ 1`; computing per-step gradient norms doubles the forward cost. Occupancy has no equivalent constraint.

The `model_type: spcarnet_shape_field_autodecoder.field_kind` config switch admits `"occupancy"` (default) and `"sdf"` (experimental). The SDF variant adds an eikonal term and changes the iso-level convention to `0.0` — a 5-line difference in the trainer; the RFC keeps SDF as a Stage-2 ablation, not a parallel headline.

---

## 2. Latent code size & layout

| Parameter | Value | Justification |
|---|---|---|
| `latent_dim` (`d_z`) | **256** | RFC §3.2 target. DeepSDF ShapeNet uses 256–512; cars are a single category with strong shared structure, so 256 should suffice for 2 433 instances. |
| Init | `Z[i] ~ N(0, 0.01²)` per element | Standard DeepSDF init; small enough that `f` initially produces near-mean predictions while the decoder establishes a working surface. |
| Storage | one `nn.Parameter` of shape `(N_train, 256)` indexed by `object_id` → row | Train-only; val/test latents are MAP-fit at eval time (Stage 2 ablation). |
| Reg | `λ_z · ||z_i||² / d_z` per object per batch (default `λ_z = 1e-4`) | DeepSDF default scaled by `d_z` so the regulariser is dimension-invariant. |

A single hash table mapping `object_id → row index` is persisted alongside the checkpoint so eval scripts can re-load the latent for a given car deterministically.

---

## 3. Decoder architecture (`SPCarShapeFieldDecoder`)

```
input:
  x : (B, Q, 3)  query points in canonical object coords
  z : (B, d_z)   per-object latent

embed_x = sin/cos Fourier features (32 frequencies, log-linear) -> (B, Q, 192)
proj_x  = Linear(192 -> 384)                                     -> (B, Q, 384)

z_film  = MLP(d_z -> 6 * 2 * 384)                                 -> (γ_l, β_l) for 6 layers

for layer l in 0..5:
    h = ReLU(LayerNorm(proj_x))
    h = γ_l * h + β_l                                             # FiLM
    h = Linear(384 -> 384)(h)
    proj_x = proj_x + h                                           # residual

out = Linear(384 -> 1)(proj_x)                                    # logits / SDF
```

- Width 384, depth 6 residual blocks, FiLM modulation by `z` per layer.
- Total params: ~3.5 M (small by design — capacity goes into latents, not weights).
- Returns `(B, Q)` logits when `field_kind="occupancy"`, `(B, Q)` raw when `field_kind="sdf"`.
- Optional second head (`feature_head`) returns `(B, Q, F)` features for Stage 3 posterior conditioning. `F = 64`. Disabled by default during Stage 2.

---

## 4. Query sampling strategy

Each object's NPZ already provides four query streams. Stage 2 samples from these directly — no new query generation code is added.

| Stream | Source field | Per-object size | Label |
|---|---|---|---|
| Surface (positives, dense) | `clean_points (2048, 3)` | 2048 | 1 (occupied) |
| Surface queries (positives, additional) | `surface_query_points (512, 3)` | 512 | 1 |
| Free queries (negatives) | `free_query_points (512, 3)` | 512 | 0 |
| Hard-negative free queries | `free_space_query_hard_negatives (128, 3)` | 128 | 0 |
| Combined occupancy supervision | `query_points_all (1280, 3)` + `query_labels_all` | 1 280 | mixed (with ignore mask) |

Per training step we sample **`Q_per_obj = 1024`** queries per object using stratified sampling:
- 384 from surface set (`clean_points` ∪ `surface_query_points`)
- 384 from free set (`free_query_points`)
- 128 from hard negatives (oversampled)
- 128 from `query_points_all` (excluding ignore-masked entries)

Sampling is per-step (random subset), so over an epoch each query is seen multiple times stochastically. Memory: `B=8 × Q_per_obj=1024 × 3 = ~24K floats` per batch — trivial.

Eikonal sampling (SDF mode only): an additional 256 random `x ~ U([-1,1]³)` per object for `||∇f||` regulariser.

---

## 5. Losses

The total objective is a sum of weighted BCE terms with latent regularisation:

```
L = w_surf  * BCE(σ(f(x_surf;  z)), 1)
  + w_free  * BCE(σ(f(x_free;  z)), 0)
  + w_hard  * BCE(σ(f(x_hard;  z)), 0)
  + w_mixed * BCE_with_ignore(σ(f(x_qall; z)), label_qall, ignore_qall)
  + w_zL2   * ||z||² / d_z
  + (sdf only) w_eik * (||∇f||₂ - 1)²
  + (optional) w_normal * cos_loss( ∇f(x_surf), n_surf )
```

Default weights:

| Term | Weight |
|---|---|
| `w_surf` | 1.0 |
| `w_free` | 1.0 |
| `w_hard` | 0.5 |
| `w_mixed` | 0.5 (uses ignore mask) |
| `w_zL2` | 1e-4 |
| `w_eik` (SDF only) | 0.1 |
| `w_normal` (optional) | 0.0 (off in baseline) |

`pos_weight` for BCE: 1.0 — positives and negatives are stratified equally per-step, so no class imbalance correction is needed.

Latent regulariser is **per-step over the batch only** (DeepSDF style, not a global term over all latents). Equivalent to a Gaussian prior on `z`.

---

## 6. Marching-Cubes eval path

The existing `ss3dm_prior/mesh/marching_cubes.py` is reused unmodified.

```python
def occupancy_fn(query: torch.Tensor) -> torch.Tensor:
    # query: (Q, 3) in canonical coordinates
    z = latent_for_object_id(object_id)             # (1, d_z)
    logits = decoder(query.unsqueeze(0), z)          # (1, Q)
    return torch.sigmoid(logits).squeeze(0)          # (Q,)

result = extract_patch_mesh(
    occupancy_fn=occupancy_fn,
    device=device,
    patch_radius=1.0,
    resolution=32,            # smoke; 64 / 128 for full eval
    iso_level=0.5,
)
```

Smoke uses `resolution=32` (32 768 voxel queries per object — fast). Full eval uses 64³ (262 K queries — ~1 s/object on GPU). Stage-7 mesh-IoU benchmark uses 128³ (~2 M queries — ~5 s/object).

Mesh-extraction success criteria:
- `result.mesh is not None` and `len(result.faces) > 0` for **at least 95 %** of validation objects (`mesh_extraction_success_rate ≥ 0.95`).
- `result.watertight` for ≥ 80 % is a soft target; not a Stage-2 gate but logged.

Sampled-point chamfer: 4 096 points sampled uniformly from the extracted mesh via `trimesh.sample.sample_surface(mesh, 4096)` → bidirectional L1 chamfer against `clean_points (2048, 3)` and against `hidden_clean_points` separately.

---

## 7. How this differs from the existing auxiliary occupancy head

The CarNet v11 backbone has an `OccupancyHead` that:
1. Takes the **encoder's latent tokens computed from the corrupted observation** (a 32 × 384 cross-attention output).
2. Predicts occupancy at query points, supervised by the same `query_points_all` + `query_labels_all`.
3. Reports `occupancy_iou_visible` as a side-metric (≈ 0.91 on v0.8.2).
4. Has zero contact with the primary point-cloud loss; its gradient flows back to the encoder but does *not* shape the decoder's outputs.

SP-CarNet's `SPCarShapeFieldDecoder` differs on every axis:

| Axis | v11 OccupancyHead | SP-CarNet Stage 2 decoder |
|---|---|---|
| Conditioning | Encoder output of corrupted obs | **Free-trained per-object latent `z_i`** (no encoder) |
| Trained jointly with | A point head + many auxiliary heads | **Itself only** (Stage 2) — clean ceiling experiment |
| Role at eval | Side-metric | **Primary output** — Marching-Cubes mesh and sampled point cloud |
| Loss family | BCE only | BCE + latent L2 (+ optional eikonal / normal) |
| What it tests | Whether encoder produces a useful latent for occupancy | **Whether the decoder family `f(x; z)` can represent MeshFleet cars at all** |

Stage 3 will reuse SP-CarNet's *decoder* with a freshly-trained encoder `q(z | O)`; the v11 occupancy head will then be retired from the Stage 3 path (kept in CarNet-v0.x for the `BL-RES` and `BL-PF` baselines).

---

## 8. Why this is not just another point decoder

1. **Output is a continuous field, not a finite point set.** No fixed-size output array is part of the model. The mesh is a side-effect of the field.
2. **Loss is per-query BCE, not chamfer.** Chamfer's permutation symmetry is what allowed v0.7's smoothing collapse and v0.8's underspecified point assignment. BCE has no permutation symmetry over outputs — every query has a *fixed* target label and a *fixed* spatial location. There is no degenerate solution analogous to "translate every output point by a small vector".
3. **Identity is the global optimum, not a side-basin.** A perfectly-trained `f(x; z_i)` has `f(x; z_i) > 0.5 ⇔ x ∈ clean_object_i`. The clean-on-clean degradation that v0.7 exhibits cannot occur — there is no input cloud that the model could "smooth", because there is no input cloud at all in Stage 2. The only inputs are query coordinates and a learned latent.
4. **Mesh-primary output dissolves the chamfer-mode-averaging issue.** A 64³ Marching-Cubes mesh that is locally accurate everywhere produces a chamfer that is bounded by the grid pitch (`2 × patch_radius / resolution = 0.0625` at 32³, 0.031 at 64³, 0.016 at 128³). This is *categorically smaller* than v0.7's 0.04 smoothing floor.
5. **Latent codes provide an explicit shape manifold for Stage 3.** A trained Stage-2 decoder gives Stage 3 a target distribution over `z` to imitate. Without Stage 2 we would be training a posterior `q(z | O)` with no guarantee that the decoder can express the data distribution at all.

---

## 9. Files to be added (this stage)

| File | Role |
|---|---|
| `ss3dm_prior/models/spcarnet_shape_field.py` | `SPCarShapeFieldDecoder` class + helpers (Fourier features, FiLM block). |
| `ss3dm_prior/training/spcarnet_autodecoder.py` | Standalone trainer module. Owns the per-object `Z` parameter, the optimisers, the loss assembly, and the eval entrypoint. |
| `configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml` | Decoder + loss config. |
| `configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml` | Optimiser, schedule, eval cadence. |
| `scripts/car_model/train_spcarnet_shape_field_autodecoder.sh` | Launcher (CUDA + WANDB online). |
| `scripts/car_model/eval_spcarnet_shape_field_autodecoder.py` | Eval entrypoint (Marching-Cubes mesh extraction + metrics). |
| `scripts/car_model/smoke_test_spcarnet_stage2.py` | 2-object × 2-iter smoke test. |
| `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md` | Closing report (filled after smoke). |

No existing file is modified. CarNet v0.x configs, the patch-centric trainer, the v11 OccupancyHead, and Stage 1 modules remain untouched.

---

## 10. Smoke test contract

`scripts/car_model/smoke_test_spcarnet_stage2.py` must:

1. Build the Stage 1 object index against the full cache (cached on disk; smoke reuses if present).
2. Construct `SPCarObjectDataset(splits=("train",), …)` and select the first 2 objects.
3. Build a tiny `SPCarShapeFieldDecoder` (width 64, depth 3, latent_dim 32 — overridden for smoke).
4. Run **2 training iterations** (full forward + backward + Adam step) on these 2 objects.
5. Verify: total loss is finite (not NaN/Inf) on both iterations; loss decreases or stays flat (no explosion).
6. Run Marching-Cubes extraction at `resolution=16` on object 0; verify the call does not raise. If skimage is missing the `mesh=None` fallback is acceptable (logged as a warning, not a smoke failure).
7. If a mesh was produced, sample 256 points from it and compute a one-shot bidirectional chamfer L1 to `clean_points`. Print the value (any finite number passes; quality is not a smoke gate).

Expected output line: `[stage2-smoke] PASS …`

---

## 11. Out of scope (deferred)

- **Hyperparameter sweep** (latent_dim ∈ {128, 256, 512}, decoder depth, FiLM vs hyper-network) — Stage 2 ablation table after the headline run lands.
- **Validation-time MAP fitting of `z`** — required for clean-val mesh-IoU on objects whose latent was not in `Z` (val/test): implemented in the eval script as a 100-step Adam minimisation on `z` only with the trained decoder frozen. Listed under Stage 2 but logged as a separate ablation row in the report.
- **Symmetry / retrieval / corruption integration** — Stage 3+.
- **Stage-2 transfer to Semantic-KITTI** — Stage 7.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Per-object latent `Z` table grows linearly with dataset (1854 × 256 × 4 B ≈ 1.9 MB) — fine on a single GPU but introduces a sharded-state hazard if we later go multi-GPU. | Keep `Z` on rank-0 only; multi-GPU plan deferred to Stage 3. |
| Smoke at width=64 depth=3 may fit too quickly to detect bugs in the FiLM gradient path. | Smoke also asserts that `decoder.parameters()` and `Z` both receive non-zero gradients on at least one of the two iterations. |
| MC at resolution 32 may produce empty meshes for under-trained latents. | The MC call's `mesh=None` return is treated as a successful smoke (we are testing the call, not the geometry). The real geometry quality is a Stage-2 eval concern, not a smoke concern. |
| Latent L2 too strong → all latents collapse to 0 → decoder learns the *mean shape*. | `λ_z = 1e-4` is the DeepSDF default and well-studied; if it bites, ablate `λ_z ∈ {1e-5, 1e-3}` in the headline run. |
| Auto-decoder training is per-object and patch-centric trainer assumes per-batch loss aggregation; reusing the existing trainer would force pseudo-classes through the codepath. | **Do not reuse `engine/trainer.py`.** Stage 2 ships its own minimal training loop in `ss3dm_prior/training/spcarnet_autodecoder.py`. |

_End of design._
