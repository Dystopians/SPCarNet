# SS3DM Prior Data Schema

## Raw Root Contract

The Step 1 discovery layer assumes the raw dataset root is:

`/data2/peilincai/SS3DM_raw`

Expected subtrees:

- `DATA/TownXX/<N>_streetsurf`
- `meshes/mesh/TownXX_obj.obj`

This schema explicitly excludes unrelated local datasets, especially `/data2/peilincai/Mesh_Dataset`.

## Manifest Unit

- Primary unit: `sequence`
- Not allowed as the primary unit in Step 1:
  - frame
  - patch
  - full-town mesh

The later training system will consume offline patch caches derived from sequence-level indexing, because training must not parse full ASCII OBJ files online.

## Manifest Root Fields

- `schema_version`: manifest schema identifier
- `dataset_name`: dataset label
- `dataset_root`: absolute path to `SS3DM_raw`
- `generated_at_utc`: build timestamp
- `unit_of_index`: always `sequence` for this version
- `split_policy`: always `town_holdout_only` for this version
- `num_sequences`: number of sequence entries
- `town_ids`: discovered town ids
- `entries`: list of sequence entries

## Sequence Entry Fields

Each manifest entry contains at least:

- `town_id`
- `sequence_id`
- `sequence_root`
- `town_mesh_obj_path`
- `scenario_pt_path`
- `scenario_txt_path`
- `num_frames_from_name`
- `camera_names`
- `lidar_names`
- `image_dir_map`
- `mask_dir_map`
- `depth_dir_map`
- `lidar_dir_map`
- `frame_count_summary`
- `all_required_paths_exist`
- `notes`

Recommended `sequence_id` format:

- `Town01__550_streetsurf`

## `frame_count_summary` Schema

`frame_count_summary` is a dict with four required sub-dicts:

- `images_jpg`
- `depth_png`
- `masks_npz`
- `lidars_npz`

Example shape:

```json
{
  "images_jpg": {
    "camera_FRONT": 550
  },
  "depth_png": {
    "camera_FRONT": 550
  },
  "masks_npz": {
    "camera_FRONT": 550
  },
  "lidars_npz": {
    "lidar_TOP": 550
  }
}
```

## Split Policy

Step 1 provides town-level split configs only.

Required rule:

- First version must split by whole towns.

Explicitly forbidden:

- random patch split
- random frame split used as a substitute for town holdout

Default split:

- train: `Town01` to `Town06`
- val: `Town07`
- test: `Town10`

Debug split:

- train: `Town01`
- val: `Town02`
- test: `Town03`

## Why This Schema Exists

- Directly using `SS3DM_raw` keeps the source of truth simple and reproducible.
- Sequence-level discovery is the right boundary for future `sequence -> patch cache` preprocessing.
- Town-level split is required to prevent geometry leakage across train, validation, and test.

## Step 2 Scenario Metadata Schema

Step 2 adds a parsed scenario view for each raw sequence.

Required top-level summary fields:

- `scene_id`
- `num_frames`
- `source_format`
- `camera_names`
- `lidar_names`
- `warnings`

Per-camera metadata interface:

- `intr`
- `c2w`
- `hw`

Per-lidar metadata interface when available:

- `l2v`
- `sensor_v2w`
- `timestamp`

Loader behavior:

- First try `torch.load(..., map_location="cpu", weights_only=False)`.
- If that is unavailable or fails, emit a clear warning and fall back to a non-torch loader path.
- If full parsing still fails, return a degraded `scenario.txt` summary rather than silently failing.

## Why RGB/Depth Are Not In Step 2 Training Inputs

- Step 2 is only about building observed geometry support from LiDAR.
- RGB and depth are intentionally deferred so the first cache format stays lightweight and geometry-focused.
- Camera/depth integration remains possible later because the scenario loader already exposes the relevant metadata contract.

## Step 2 Observed Cache Schema

Each selected sequence writes:

- `<out_dir>/<town_id>/<sequence_id>/observed_cache.npz`
- `<out_dir>/<town_id>/<sequence_id>/sequence_stats.json`

`observed_cache.npz` contains:

- `observed_points`
- `tile_centers`
- `camera_centers`
- `sequence_stats_json`

## Observed Point Construction

V1 uses the LiDAR assumption:

`points = rays_o + rays_d * ranges[:, None]`

With config control:

- `lidar_rays_world_frame`
- `min_range`
- `max_range`
- `max_points_per_frame`
- `seed`

Invalid samples are removed when:

- any component is `NaN`
- any component is `inf`
- `range <= 0`
- `range` falls outside configured limits

## Tile Center Definition

Tile centers are occupancy-aware support points derived only from observed LiDAR geometry.

Construction:

1. Quantize `observed_points` into a 3D grid with `tile_stride_m`.
2. Count points per grid cell.
3. Keep only cells with at least `tile_min_points`.
4. Define each tile center as either:
   - the mean of points in the occupied cell, or
   - the geometric center of the grid cell

Meaning:

- They approximate locations with repeated observed support.
- They provide stable local anchors for future patch extraction.
- They deliberately avoid unobserved empty space and are therefore better suited than naive uniform tiling.

## Step 3 Binary Town Mesh Cache Schema

Step 3 adds a town-level binary cache under:

- `<cache_root>/<town_id>/`

Required files:

- `vertices.npy`
- `faces.npy`
- `face_centroids.npy`
- `face_normals.npy`
- `face_areas.npy`
- `bbox.json`
- `mesh_meta.json`

Recommended dtypes:

- `vertices.npy`: `float32`, shape `[V, 3]`
- `faces.npy`: `int32` or `int64`, shape `[F, 3]`
- `face_centroids.npy`: `float32`, shape `[F, 3]`
- `face_normals.npy`: `float32`, shape `[F, 3]`
- `face_areas.npy`: `float32`, shape `[F]`

`mesh_meta.json` records at least:

- `source_obj_path`
- `num_vertices`
- `num_faces`
- `vertex_dtype`
- `face_dtype`
- `conversion_command`
- `conversion_time_sec`
- `converted_at_utc`
- `town_id`

## Why Binary Cache Must Exist Before Patch Work

- Patch extraction will need repeated spatial queries against the same town mesh.
- Re-reading huge ASCII OBJ files would waste both CPU and I/O.
- The binary cache moves all expensive text parsing and per-face geometry precomputation into a one-time offline stage.

## Step 3 Read API Contract

Step 3 introduces a lightweight mesh cache API for future local patch extraction:

- `load_town_mesh_cache(cache_dir, mmap=True)`
- `query_faces_in_radius(center, radius, margin=...)`
- `build_local_mesh_from_face_mask(...)`

Current query behavior:

- The first query path is based on `face_centroids`.
- Radius filtering is a simple linear scan over centroids.
- The structure is intentionally simple now, but leaves room for a future KD-tree or similar spatial index.

## Binary Mesh Cache Field Purpose

- `vertices.npy`: canonical vertex positions for rebuilding local submeshes
- `faces.npy`: triangle connectivity
- `face_centroids.npy`: cheap first-stage spatial query key
- `face_normals.npy`: later geometric supervision or filtering
- `face_areas.npy`: area-aware weighting and patch statistics
- `bbox.json`: coarse spatial bounds and validation
- `mesh_meta.json`: reproducibility and conversion provenance

## Step 4 Teacher Patch Cache Schema

Each generated patch is anchored by one `tile_center` from the sequence observed cache.

Patch root layout:

- `<patch_cache_root>/<town_id>/<sequence_id>/<patch_id>.npz`
- `<patch_cache_root>/patch_index.jsonl`

Recommended config:

- `town_mesh_unit_scale: 0.01`
- `patch_radius_m: 3.0`
- `observed_min_points: 64`
- `clean_min_faces: 20`
- `clean_sample_count: 2048`
- `observed_sample_count: 512`
- `face_query_margin_m: 1.0`

`town_mesh_unit_scale` exists because the town OBJ geometry is stored in a larger raw unit scale than the observed cache world coordinates, so local teacher extraction must explicitly rescale cached mesh vertices and centroids before querying.

## Patch File Fields

Each patch `.npz` stores at least:

- `clean_points`
- `clean_normals`
- `observed_points`
- `patch_center_world`
- `patch_radius_m`
- `town_id`
- `sequence_id`
- `tile_id`
- `patch_id`
- `num_local_faces`
- `num_observed_points_raw`
- `teacher_area_local`
- `source_town_mesh_cache_dir`
- `source_sequence_observed_cache`

This implementation also stores:

- `patch_metadata_json`

## Local Frame Definition

All patch point coordinates are normalized into a local frame:

1. subtract `patch_center_world`
2. divide by `patch_radius_m`

Consequences:

- local coordinates are roughly radius-normalized
- `clean_points` and `observed_points` live in the same canonical patch frame
- normals are kept directional and renormalized, but not translated or scaled

## Clean Teacher Patch Construction

For each tile center:

1. query town-mesh faces by centroid distance within `patch_radius_m + face_query_margin_m`
2. rebuild a local submesh from the selected faces
3. reject the tile if the local face count is below `clean_min_faces`
4. sample `clean_sample_count` points on the local mesh surface
5. attach sampled face normals as `clean_normals`

Stored clean-patch statistics include:

- `num_local_faces`
- `teacher_area_local`
- local bbox
- mean normal
- planarity hint

## Observed Patch Construction

For each tile center:

1. crop sequence observed points inside `patch_radius_m`
2. reject the tile if fewer than `observed_min_points` raw points remain
3. resample to exactly `observed_sample_count`
4. convert to local frame

The observed patch remains a sparse local anchor rather than a clean target.

## Patch Index Schema

`patch_index.jsonl` contains one JSON record per written patch with at least:

- `patch_id`
- `town_id`
- `sequence_id`
- `tile_id`
- `patch_file`
- `num_local_faces`
- `num_observed_points_raw`
- `num_clean_points`
- `num_observed_points`
- `teacher_area_local`
- `planarity_hint`

This file is the dataset-wide lookup table for later training and evaluation stages.

## Step 5 Online Training Sample Schema

Step 5 does not pre-store corrupted patches. Instead, the dataset reads clean teacher patches and generates corrupted inputs online.

Per-sample fields returned by the dataset:

- `clean_points`
- `clean_normals`
- `observed_points`
- `corrupted_points`
- `corrupted_normals`
- `point_defect_target`
- `corruption_score_target`
- `town_id`
- `sequence_id`
- `patch_id`
- `patch_metadata`
- `corruption_metadata`

## Step 5 Corruption Schema

Online corruptions include:

- `point_dropout`
- `gaussian_jitter`
- `normal_noise`
- `local_hole_mask`
- `outlier_cluster`
- `density_imbalance`

Generated targets:

- `corruption_score_target`
- `point_defect_target`

`point_defect_target` is built from corrupted-point nearest-neighbor distance to the clean patch plus corruption-specific defect flags.

`corruption_score_target` is a continuous patch-level severity target derived from per-point defects and corruption severity terms.

## Step 5 Model Output Schema

The local patch model returns a dict with:

- `recon_points`
- `recon_normals`
- `point_defect_pred`
- `patch_score_pred`
- `corrupted_latent`
- `clean_latent`
- `retrieval_embedding`
- `observed_latent`

Interpretation:

- `recon_points` / `recon_normals`: predicted clean local geometry
- `point_defect_pred`: per-corrupted-point defect regression
- `patch_score_pred`: patch-level corruption or quality estimate
- `corrupted_latent` / `clean_latent`: latent features for alignment
- `retrieval_embedding`: normalized embedding for patch retrieval-style evaluation

## Step 5 Loss Schema

Implemented losses:

- `recon_chamfer_loss`
- `recon_normal_loss`
- `point_defect_loss`
- `patch_score_loss`
- `latent_align_loss`
- `total_loss`

Training rationale:

- `recon_chamfer_loss` teaches shape recovery
- `recon_normal_loss` stabilizes local surface orientation
- `point_defect_loss` teaches local defect awareness
- `patch_score_loss` teaches patch-level severity estimation
- `latent_align_loss` encourages corrupted and clean representations to share the same clean local geometry manifold

## Step 5 Metric Schema

Implemented metrics:

- `recon_chamfer_l1`
- `recon_normal_cosine`
- `denoise_gain_chamfer`
- `score_mae`
- `score_spearman`
- `point_defect_mae`
- `retrieval_top1`
- `retrieval_top5`

This metric set is deliberately local-patch-centric and aligned with the clean-local-geometry objective rather than whole-scene generation quality.
