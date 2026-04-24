# CarNet_v0 Update Log

Chronological record of changes made under the CarNet_v0 research line.
Companion to the master plan at `docs/car_model/CarNet_v0.md`. Entries are
most-recent-first within each date.

---

## 2026-04-17

### Session summary

One session, mostly landing the three architectural contributions
(A1 flow matching / A2 learned symmetry / A3 retrieval) plus two supporting
subsystems (LiDAR-realistic corruption, Marching-Cubes mesh extraction) and
two AMP bug fixes surfaced by the first multi-GPU launch.

### Commits / changes

1. **CarNet_v0 plan doc** — new `docs/car_model/CarNet_v0.md` establishing
   lineage (v8_car_novis → v3_5 trio → v4 trio → CarNet_v0), paper story
   (SR-AFM: Symmetric Retrieval-Augmented Flow Matching), 5-phase work
   plan, and five locked design decisions D1–D5.

2. **Phase 1 — flow matching baseline.**
   - Configs: `configs/ss3dm_prior/carnet_v0/model_carnet_v0_flow.yaml` +
     `train_carnet_v0_flow.yaml` (60 epochs, EMA+AMP, flow loss weight 0.1,
     stochastic K=[1,4,8], 16 ODE steps).
   - Trainer: added `flow_matching_start_epoch` curriculum gate (default
     active from `main_start_epoch`).
   - Launcher: `scripts/car_model/train_carnet_v0_flow.sh`.
   - Param count: 64.6 M; smoke-tested forward produces
     `latent_flow_matching_loss` scalar on batch of 2.

3. **Phase 2 — learned symmetry.**
   - New `ss3dm_prior/data/symmetry_targets.py` with
     `estimate_symmetry_plane()` — closed-form PCA+Chamfer fit + soft
     confidence σ. Sanity: σ=1.00 on mirror-augmented data, σ=0.02 on
     random clouds, σ=0.0003 on single-side partial patches.
   - New `ss3dm_prior/models/symmetry_head.py` — MLP head producing
     `(n, d, σ)` from fused pre-VQ latent, plus a `reflect_points()`
     utility. ~0.3 M params.
   - `ss3dm_prior/losses.py::symmetry_consistency_loss` — three-component
     loss (self-symmetry Chamfer weighted by σ predicted, plane regression
     weighted by σ target, confidence BCE).
   - `TeacherPatchSample` + `PatchIndexRecord` extended with four
     symmetry fields. **Cache format bumped 2 → 3.**
   - Cache builder (`build_car_mesh_patch_cache.py`) now calls the
     estimator on every sample and persists targets into NPZ.
   - `train_dataset.py` emits the fields in each batch dict.
   - `trainer.py::_collate_samples` tensor_keys set extended.
   - `trainer.py` curriculum adds `symmetry_start_epoch` (default 4).
   - Configs: `model_carnet_v0_flow_sym.yaml` +
     `train_carnet_v0_flow_sym.yaml` enable `use_symmetry_head: true`.
   - Smoke: gradients reach the symmetry head; all three loss components
     non-zero on random batches.

4. **Phase 3 — LiDAR-realistic corruption.**
   - `ss3dm_prior/data/corruptions.py` gains four physically-motivated
     corruption modes:
     * `beam_occlusion` — drop points whose surface normal points away
       from the simulated scanner (self-occlusion).
     * `incidence_angle_dropout` — retain with probability
       `cos(θ_incidence)^k`, simulating grazing-angle return weakness.
     * `range_dependent_noise` — per-point σ scales with distance² to
       the scanner (inverse-square intensity).
     * `azimuthal_ring_sparsity` — drop points in narrow azimuth bands
       to simulate column-wise beam spacing.
   - Scanner pose sampling: hemispherical prior with configurable radius
     and elevation range; fixed-position override.
   - Smoke on a unit-sphere mesh with scanner at `[3, 0, 0.6]`: 100 % of
     remaining points lie on the +x (near-facing) side, mean range to
     scanner ≈ 2.28.

5. **Phase 3b — Marching-Cubes mesh extractor (D4).**
   - New `ss3dm_prior/mesh/marching_cubes.py` exposes
     `extract_patch_mesh()`, `save_patch_mesh()`, `mesh_iou_at_iso()`,
     `surface_normal_consistency()`.
   - Queries an occupancy closure on a sparse 3-D grid and delegates to
     `trimesh.voxel.ops.matrix_to_marching_cubes`.
   - Graceful fallback when `scikit-image` is not installed: logs a single
     warning, returns `mesh=None`, IoU + volume metrics remain available.
   - Eval-only; no changes to training.

6. **Phase 4 — retrieval-augmented decoder (A3).**
   - New `ss3dm_prior/retrieval/anchor_bank.py`:
     `AnchorBank(embedding_dim, latent_dim)` with FAISS `IndexFlatIP`
     when available, NumPy-cosine fallback otherwise. Supports save/load
     to NPZ. Source-agnostic API (MeshFleet-only bank now; ShapeNetCore
     mix deferred to CarNet_v0.1 per D3).
   - v11 model extended with `use_retrieval_anchors`,
     `num_retrieval_anchors`, an anchor-token projection MLP, and a
     learnable anchor type embedding. Anchor tokens enter the
     cross-attention context alongside corrupted/observed/visible/hidden/
     query tokens.
   - Training-time bootstrap: in-batch shortcut using cyclic-shifted
     `latent_seed` tensors, so the RAG pathway trains from epoch 0
     without a pre-built bank.
   - Persistent end-of-epoch bank rebuild is intentionally deferred —
     the in-batch shortcut is cleaner for the first submission cut.

7. **Six-variant ablation launcher.**
   - `scripts/car_model/train_carnet_v0_ablation.sh` launches the matrix
     from §6.1 of the plan on six GPUs (0/1/2/3/5/6 default; GPU 4
     skipped because of existing usage). wandb online, project
     `carnet_v0`.
   - Configs dropped under `configs/ss3dm_prior/carnet_v0/`:
     `det`, `flow`, `flow_sym`, `flow_rag`, `full`, `full_lidar`. Param
     counts 64.6–65.3 M across variants.

### Bugs found & fixes

- **B1 — `torch.linalg.eigh` has no fp16 CUDA kernel**
  (`RuntimeError: "linalg_eigh_cuda" not implemented for 'Half'`).
  All six variants crashed at the end of epoch 0's first step because
  the v10 backbone's `_compute_local_frame` ran under the trainer's
  AMP autocast.
  **Fix:** `CrossAttentionHybridPatchPriorV10._compute_local_frame` now
  wraps the PCA block in `torch.autocast(enabled=False)` and explicitly
  casts the covariance input to float32, then casts eigenvectors back
  to the original dtype. Casting alone wasn't enough because the outer
  autocast demotes matmul inside the same context.

- **B2 — `torch.cdist` has no fp16 CUDA kernel**
  (`RuntimeError: "cdist_cuda" not implemented for 'Half'`).
  Surfaced in `_sample_visibility_metrics` where metrics are computed
  on model fp16 outputs outside the autocast scope. The loss-side
  Chamfer happens to avoid the crash because autocast's promotion rule
  for `torch.cdist` covers the in-autocast case, but eval / metric
  paths don't.
  **Fix:** introduced `_cdist_fp32_safe(x, y, p)` in both
  `metrics.py` and `losses.py`. Wraps `torch.cdist` in
  `autocast(enabled=False)` and upcasts inputs to float32. Both
  production copies of the function are behaviour-identical, so a
  later refactor can consolidate.

- **Side effect:** the AMP fixes don't touch CPU / non-autocast code
  paths. Unit-level smoke tests (CPU-only) continued to pass without
  modification.

### Wandb cleanup (previous session, landed before Phase 1 configs)

- Dropped literal-duplicate `epoch/{split}_loss_raw/*` and
  `train_step/loss_raw/*` series in favour of a single canonical
  `epoch/{split}/X_loss` plus the weighted-contribution view.
- Silenced `score_spearman` constant-input warnings and matplotlib
  tight-layout / legend spam.
- Added `epoch_visualization_interval_epochs` knob (default 5) so
  per-epoch panel uploads are throttled; end-of-training and first-epoch
  panels still render.
- Whole-car runs skip `retrieval_top5_*` and
  `retrieval_top1_self_aligned` (saturate to 1.0 by design when each
  sample is its own sequence).

### Outstanding follow-ups

- Wire `extract_patch_mesh` into `ss3dm_prior/eval.py` so mesh IoU and
  normal-consistency numbers land in the formal eval report without
  manual scripting.
- End-of-epoch persistent anchor-bank rebuild (Phase 4 follow-up). The
  in-batch shortcut is functional for training; the bank is still
  required for transfer-time (zero-shot) inference.
- Semantic-KITTI car-instance extraction pipeline (Phase 5 gate).
- Lightweight `_cdist_fp32_safe` consolidation: same helper is
  duplicated in `metrics.py` and `losses.py`; move to
  `ss3dm_prior/utils/amp_safe.py` and re-import.
