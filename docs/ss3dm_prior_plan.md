# SS3DM Prior Plan

## Goal

Build a standalone local mesh distribution learning system that uses `/data2/peilincai/SS3DM_raw` directly as the raw source of truth, while remaining decoupled from the main mesh-splatting training pipeline.

## Why V1 Uses `SS3DM_raw` Directly

- `SS3DM_raw` is the canonical local dataset root that already contains the multimodal sequence tree and the town-level mesh assets.
- Keeping V1 directly anchored to `SS3DM_raw` avoids accidental mixing with unrelated local datasets such as `/data2/peilincai/Mesh_Dataset`.
- A direct raw-data discovery layer makes later cache generation reproducible, auditable, and independent from mesh-splatting scene loaders.

## Why The Training Unit Is Not Full OBJ

- Town meshes are extremely large ASCII OBJ files, so they are unsuitable for repeated online parsing during training.
- The intended training unit is `sequence -> offline local patch cache`, not `full town OBJ -> online parse`.
- Step 1 therefore only establishes sequence-level discovery and indexing; later steps will build cacheable local patch units from that index.

## Why Split Must Be Town Holdout

- The objective is to learn a transferable local prior rather than memorize town-specific geometry.
- Random patch split would leak local geometry across train/val/test.
- Random frame split would also leak scene identity because frames from the same town and sequence share the same underlying mesh and layout.
- V1 therefore requires town-level holdout as the default split boundary.

## Phased Plan

1. Step 1: discovery / manifest / split skeleton for `SS3DM_raw`. Status: completed.
2. Step 2: raw sequence metadata parsing and LiDAR-based observed cache builder. Status: completed.
3. Step 3: town OBJ to binary mesh cache conversion for later local patch extraction. Status: completed.
4. Step 4: local clean teacher patch cache construction from observed cache and town mesh cache. Status: completed.
5. Step 5: online corruption, training dataset, local patch model, losses, and metrics. Status: completed.
6. Step 6: training script, checkpointing, wandb logging, and qualitative visualization. Status: completed.
7. Step 7: standalone test evaluation script. Status: completed.

## Step 1 Scope Boundary

- Included:
  - town and sequence discovery
  - sequence-level manifest building
  - town mesh path registration
  - sensor directory and frame count indexing
  - town-holdout split configs
  - manifest validation CLI
- Excluded:
  - offline patch cache generation
  - model training
  - wandb logging
  - test-set evaluation
  - any mesh-splatting integration

## Step 2 Scope Boundary

- Included:
  - `scenario.pt` parsing with a `torch.load` first path and a non-torch fallback
  - unified camera and LiDAR metadata access
  - sequence-level LiDAR fusion into observed point clouds
  - occupancy-aware tile center generation
  - offline observed cache writing and cache validation CLI
- Excluded:
  - town OBJ processing
  - clean teacher patch extraction
  - training code
  - wandb logging
  - any mesh-splatting integration

## Why V1 Builds Observed Occupancy From LiDAR

- LiDAR `.npz` already exposes `rays_o`, `rays_d`, and `ranges`, which makes geometry extraction direct and low-ambiguity.
- This avoids early entanglement with depth-scale handling and camera backprojection details.
- The resulting observed occupancy map is enough to define sequence-level support regions and candidate local tile centers for the next stage.

## Why Scenario Is Parsed Before RGB/Depth Training Use

- Camera metadata still matters even when RGB and depth are not used in Step 2, because later steps will need intrinsics, poses, and frame-aligned sensor metadata.
- Parsing `scenario.pt` now provides a stable metadata contract without forcing premature camera/depth training logic into V1.
- This keeps Step 2 focused on observed geometry while preserving forward compatibility.

## Step 3 Scope Boundary

- Included:
  - offline conversion of `Town*_obj.obj` into compact binary cache files
  - memmap-friendly mesh array loading
  - precomputed face centroids, normals, areas, and bbox metadata
  - simple centroid-radius face queries for future local patch extraction
  - town mesh cache validation CLI
- Excluded:
  - local patch extraction
  - clean/observed patch pairing
  - model training
  - wandb logging
  - any mesh-splatting integration

## Why Binary Mesh Cache Is Required

- The source town meshes are huge ASCII OBJ files, so repeated parsing would dominate both patch-cache generation and any later training-time teacher lookup.
- A binary cache amortizes OBJ parsing into a one-time offline step and makes downstream reads memmap-friendly.
- Precomputing face geometry lets later patch extractors skip repeated centroid, normal, area, and bbox recomputation.

## Why Training-Adjacent Stages Must Not Read OBJ Directly

- ASCII OBJ parsing is slow, text-heavy, and memory-inefficient for repeated access.
- Patch extraction needs only a spatial subset of faces, so loading dense precomputed arrays is a better fit than reparsing text geometry.
- This separation keeps future local patch logic deterministic, faster, and easier to validate.

## Step 4 Scope Boundary

- Included:
  - local clean teacher patch extraction from town mesh cache
  - local observed patch extraction from sequence observed cache
  - patch-local coordinate normalization
  - patch cache serialization and global patch index generation
  - static PNG inspection for sampled patches
- Excluded:
  - synthetic corruption
  - dataset-time augmentation
  - model code
  - training
  - wandb logging

## Why Clean Teacher Comes From Town Mesh Cache

- The teacher geometry should come from the clean GT town mesh, not from sparse observed LiDAR.
- Using the binary town mesh cache avoids repeated ASCII OBJ parsing and makes repeated local queries practical.
- Sampling from the cached local mesh gives dense clean supervision while still keeping the extraction stage offline and reproducible.

## Why Observed Patch Is Only A Local Anchor

- The observed patch is intentionally the sparse local evidence around a tile center, not the full target geometry.
- Its role is to anchor local occupancy and later provide noisy partial input for denoising or critic-style training.
- Synthetic corruption is deferred to dataset/training time so the offline patch cache stays canonical and reusable.

## Step 5 Scope Boundary

- Included:
  - online synthetic corruption on top of cached clean patches
  - training dataset for clean/observed/corrupted patch tuples
  - lightweight PointNet-style denoiser/distribution learner
  - reconstruction, defect, score, and latent-alignment losses
  - reconstruction, quality, and retrieval metrics
- Excluded:
  - training loop
  - optimizer/scheduler wiring
  - wandb logging
  - evaluation scripts

## What This Model Learns

- This version learns a `clean local geometry distribution`, not a whole-scene generator.
- The target is a canonical clean local surface patch around a tile anchor.
- The corrupted patch is a noisy or incomplete local observation sampled from that same clean distribution.

## Why The Task Is `Corrupted Patch -> Clean Patch`

- The denoising task directly matches the intended prior: recover plausible clean local geometry from partial or damaged local input.
- It naturally supports per-point defect prediction and patch-level quality scoring as auxiliary heads.
- The clean teacher patch is dense and local, so the supervision is far more stable than trying to predict an entire town or sequence geometry at once.

## Why This Is Better Than Whole-Scene Generation For V1

- Whole-scene generation would entangle global layout, large memory cost, and much harder supervision.
- Local patches are cheaper to cache, easier to corrupt online, and easier to batch consistently.
- A local clean-geometry prior is also more reusable later as a denoiser, critic, retrieval model, or teacher module inside larger systems.

## Step 6 Scope Boundary

- Included:
  - formal training CLI
  - trainer and checkpoint management
  - offline/disabled-safe wandb integration
  - static qualitative visualization panels and sequence maps
  - local debug training scripts
- Excluded:
  - standalone test evaluation script
  - held-out test reporting

## What The Training Loop Now Optimizes

- The trainer now optimizes the local denoising objective end to end on cached clean teacher patches and online-corrupted inputs.
- Validation tracks both direct reconstruction quality and actual denoise gain relative to the corrupted input.
- Best checkpoints are selected separately for `best_recon` and `best_gain`, because reconstruction fidelity and practical denoising improvement are both important.

## What To Look At In wandb

- Scalar curves:
  - total loss and each sub-loss
  - validation `recon_chamfer_l1`
  - validation `denoise_gain_chamfer`
  - validation score and retrieval metrics
- Qualitative panels:
  - `patch_denoise_panel`
  - `sequence_improvement_map`
  - `retrieval_gallery`

These three image families are the clearest way to verify that the model is learning a clean local geometry prior rather than just memorizing inputs.

## Step 7 Scope Boundary

- Included:
  - standalone held-out test evaluation CLI
  - JSON/CSV metric export
  - Markdown report generation
  - patch-panel, sequence-map, and retrieval-gallery export for test results
  - optional wandb eval logging
- Excluded:
  - any mesh-splatting interface
  - any change to the main mesh-splatting training flow

## Why Step 7 Emits Structured Reports

- Test evaluation needs to be reproducible and comparable across checkpoints, not just visible in transient console output.
- JSON and CSV outputs make it easy to aggregate results later or compare multiple eval runs programmatically.
- A generated Markdown report keeps the most important metrics, qualitative figures, and provenance in one shareable place beside the output images.

## What Step 7 Standardizes

- Global metrics summarize overall reconstruction, denoise gain, score prediction, defect regression, and retrieval quality on the test split.
- Per-town and per-sequence tables make it easier to spot generalization gaps that would be hidden by a single global average.
- Curated qualitative exports highlight best gain, worst gain, largest score error, and retrieval behavior so model failure modes remain inspectable after training.
