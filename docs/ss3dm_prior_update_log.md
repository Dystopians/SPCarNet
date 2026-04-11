# SS3DM Prior Update Log

## Step 1 - Discovery / Manifest / Split Skeleton

### Modified Files

- `ss3dm_prior/__init__.py`
- `ss3dm_prior/data/__init__.py`
- `ss3dm_prior/data/discovery.py`
- `ss3dm_prior/data/manifest.py`
- `ss3dm_prior/tools/__init__.py`
- `ss3dm_prior/tools/build_manifest.py`
- `ss3dm_prior/tools/check_manifest.py`
- `ss3dm_prior/utils/__init__.py`
- `ss3dm_prior/utils/io.py`
- `configs/ss3dm_prior/data_default.yaml`
- `configs/ss3dm_prior/splits/default_town_split.yaml`
- `configs/ss3dm_prior/splits/debug_town_split.yaml`
- `tests/ss3dm_prior/conftest.py`
- `tests/ss3dm_prior/test_manifest.py`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_update_log.md`
- `docs/ss3dm_prior_data_schema.md`

### Design Rationale

- Built a standalone `ss3dm_prior` package so Step 1 stays fully isolated from the mesh-splatting training codepath.
- Indexed `SS3DM_raw` at the sequence level because the manifest unit must be sequence, not frame or patch.
- Recorded town mesh paths without ever opening the large OBJ files, which keeps discovery cheap and consistent with the future offline-cache design.
- Added town-level split configs because V1 must forbid random patch split and random frame split.
- Added a validation CLI so later steps can treat the manifest as a contract and quickly catch missing paths or inconsistent sensor counts.

### Actual Commands

- `python -m ss3dm_prior.tools.build_manifest --help`
- `python -m ss3dm_prior.tools.check_manifest --help`
- `python -m ss3dm_prior.tools.build_manifest --root /data2/peilincai/SS3DM_raw --out /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json`
- `python -m ss3dm_prior.tools.check_manifest --manifest /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json`
- `pytest tests/ss3dm_prior/test_manifest.py -q`
- `python -m py_compile ss3dm_prior/__init__.py ss3dm_prior/data/__init__.py ss3dm_prior/data/discovery.py ss3dm_prior/data/manifest.py ss3dm_prior/tools/__init__.py ss3dm_prior/tools/build_manifest.py ss3dm_prior/tools/check_manifest.py ss3dm_prior/utils/__init__.py ss3dm_prior/utils/io.py tests/ss3dm_prior/conftest.py tests/ss3dm_prior/test_manifest.py`

### Result

- Discovery successfully indexes the SS3DM raw root by town and sequence.
- Manifest generation writes a JSON file with the required sequence-level fields and frame count summaries.
- Manifest validation on the real dataset reports 28 sequences across 8 towns, complete camera/lidar sets, and no warnings or errors.
- Default and debug split configs are now available and explicitly enforce town holdout.
- Tests pass after adding a local `conftest.py` that places the repo root on `sys.path` for this non-packaged codebase.

### Remaining Risks / TODO

- Step 1 intentionally does not parse `scenario.pt` or `scenario.txt`; future cache/training steps will need a robust metadata reader.
- Discovery currently trusts file naming conventions and directory layout; later steps may need stronger schema validation for scenario metadata contents.
- Manifest validation checks path existence and count consistency, but it does not inspect file payload integrity.

## Step 2 - Scenario Parsing And Observed Cache Skeleton

### Modified Files

- `ss3dm_prior/data/__init__.py`
- `ss3dm_prior/data/scenario_loader.py`
- `ss3dm_prior/data/raw_sequence.py`
- `ss3dm_prior/data/lidar_io.py`
- `ss3dm_prior/data/observed_fusion.py`
- `ss3dm_prior/tools/inspect_raw_sequence.py`
- `ss3dm_prior/tools/build_observed_cache.py`
- `ss3dm_prior/tools/check_observed_cache.py`
- `configs/ss3dm_prior/observed_cache_default.yaml`
- `tests/ss3dm_prior/test_scenario_loader.py`
- `tests/ss3dm_prior/test_observed_fusion.py`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_update_log.md`
- `docs/ss3dm_prior_data_schema.md`

### Design Rationale

- Used LiDAR as the V1 observed geometry source because `rays_o`, `rays_d`, and `ranges` define points more directly than early RGB/depth backprojection.
- Parsed `scenario.pt` now so camera and LiDAR metadata become available through a stable interface even before RGB/depth enter training.
- Kept the cache unit at `sequence` level so later patch extraction can operate from compact observed geometry support rather than re-reading raw LiDAR every epoch.
- Defined occupancy-aware tile centers by thresholding populated 3D grid cells, which gives future local patch samplers meaningful anchors in observed space.
- Kept all logic inside `ss3dm_prior` and did not touch town OBJ handling or any training entrypoint.

### Actual Commands

- `python -m ss3dm_prior.tools.inspect_raw_sequence --help`
- `python -m ss3dm_prior.tools.build_observed_cache --help`
- `python -m ss3dm_prior.tools.check_observed_cache --help`
- `python -m ss3dm_prior.tools.inspect_raw_sequence --sequence_root /data2/peilincai/SS3DM_raw/DATA/Town01/150_streetsurf --frame_stride 50 --max_points_per_frame 5000`
- `python -m ss3dm_prior.tools.build_observed_cache --manifest /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/debug_town_split.yaml --config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/observed_cache_default.yaml --out_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache_debug`
- `python -m ss3dm_prior.tools.check_observed_cache --cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache_debug`
- `pytest tests/ss3dm_prior/test_scenario_loader.py -q`
- `pytest tests/ss3dm_prior/test_observed_fusion.py -q`
- `python -m py_compile ss3dm_prior/__init__.py ss3dm_prior/data/__init__.py ss3dm_prior/data/discovery.py ss3dm_prior/data/manifest.py ss3dm_prior/data/scenario_loader.py ss3dm_prior/data/raw_sequence.py ss3dm_prior/data/lidar_io.py ss3dm_prior/data/observed_fusion.py ss3dm_prior/tools/__init__.py ss3dm_prior/tools/build_manifest.py ss3dm_prior/tools/check_manifest.py ss3dm_prior/tools/inspect_raw_sequence.py ss3dm_prior/tools/build_observed_cache.py ss3dm_prior/tools/check_observed_cache.py ss3dm_prior/utils/__init__.py ss3dm_prior/utils/io.py tests/ss3dm_prior/conftest.py tests/ss3dm_prior/test_manifest.py tests/ss3dm_prior/test_scenario_loader.py tests/ss3dm_prior/test_observed_fusion.py`

### Result

- Real raw sequence inspection succeeded and reported plausible bbox, range, and frame-to-frame drift statistics under the world-frame LiDAR assumption.
- `scenario.pt` parsing works in the current environment via a pickle fallback after an explicit `torch.load` warning path.
- Debug split observed cache generation succeeded for 12 sequences across `Town01` to `Town03`.
- Built cache summary:
  - `processed_sequences: 12`
  - `total_observed_points: 5495033`
  - `total_tile_centers: 44517`
- Cache validation succeeded with no warnings.
- Both new unit-test files passed.

### Remaining Risks / TODO

- The current environment lacks `torch`, so Step 2 uses a warning-backed fallback path instead of the preferred `torch.load` execution path.
- `lidar_rays_world_frame=true` looks plausible from sanity stats, but this remains an assumption that should be rechecked before patch supervision depends on it.
- Camera centers are stored for future use, but RGB/depth observations are still not fused into the cache.
- Sequence-level observed caches do not yet include clean teacher geometry or patch extraction outputs.

## Step 3 - Town OBJ Binary Cache Conversion

### Modified Files

- `ss3dm_prior/data/__init__.py`
- `ss3dm_prior/data/obj_converter.py`
- `ss3dm_prior/data/town_mesh_cache.py`
- `ss3dm_prior/tools/convert_town_obj.py`
- `ss3dm_prior/tools/check_town_mesh_cache.py`
- `configs/ss3dm_prior/town_mesh_cache_default.yaml`
- `tests/ss3dm_prior/test_obj_converter.py`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_update_log.md`
- `docs/ss3dm_prior_data_schema.md`

### Design Rationale

- Added a one-time OBJ conversion stage because downstream local patch work should not keep reparsing huge ASCII mesh files.
- Used `trimesh` with `process=False` to preserve a lightweight, non-repairing conversion path.
- Saved mesh arrays in `.npy` format so later stages can use `mmap_mode="r"` for low-overhead reads.
- Precomputed face centroids, normals, and areas because these are core geometry features for later local patch extraction and query.
- Implemented the first query API on top of `face_centroids` so the later patch step has a simple, working spatial entrypoint without forcing a complex accelerator now.

### Actual Commands

- `python -m ss3dm_prior.tools.convert_town_obj --help`
- `python -m ss3dm_prior.tools.check_town_mesh_cache --help`
- `pytest tests/ss3dm_prior/test_obj_converter.py -q`
- `python -m py_compile ss3dm_prior/__init__.py ss3dm_prior/data/__init__.py ss3dm_prior/data/discovery.py ss3dm_prior/data/manifest.py ss3dm_prior/data/scenario_loader.py ss3dm_prior/data/raw_sequence.py ss3dm_prior/data/lidar_io.py ss3dm_prior/data/observed_fusion.py ss3dm_prior/data/obj_converter.py ss3dm_prior/data/town_mesh_cache.py ss3dm_prior/tools/__init__.py ss3dm_prior/tools/build_manifest.py ss3dm_prior/tools/check_manifest.py ss3dm_prior/tools/inspect_raw_sequence.py ss3dm_prior/tools/build_observed_cache.py ss3dm_prior/tools/check_observed_cache.py ss3dm_prior/tools/convert_town_obj.py ss3dm_prior/tools/check_town_mesh_cache.py ss3dm_prior/utils/__init__.py ss3dm_prior/utils/io.py tests/ss3dm_prior/conftest.py tests/ss3dm_prior/test_manifest.py tests/ss3dm_prior/test_scenario_loader.py tests/ss3dm_prior/test_observed_fusion.py tests/ss3dm_prior/test_obj_converter.py`
- `python -m ss3dm_prior.tools.convert_town_obj --town_id Town02 --obj_path /data2/peilincai/SS3DM_raw/meshes/mesh/Town02_obj.obj --out_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache_smoke --config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/town_mesh_cache_default.yaml`
- `python -m ss3dm_prior.tools.check_town_mesh_cache --cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache_smoke/Town02`

### Result

- Town mesh binary cache conversion works on a toy OBJ and on a real smoke-test town.
- Real smoke-test conversion on `Town02` succeeded with:
  - `num_vertices: 9588162`
  - `num_faces: 3274484`
  - output cache at `outputs/ss3dm_prior/town_mesh_cache_smoke/Town02`
- Cache validation succeeded with no warnings.
- The new cache API supports memmap loading, centroid-radius face queries, and local submesh rebuilding from a face mask.
- The new Step 3 unit test passed.

### Remaining Risks / TODO

- The current radius query is a centroid linear scan, so it is simple but not yet optimized for repeated large-scale query workloads.
- Conversion is still a full-mesh offline operation, so larger towns than `Town02` may require noticeably more time and disk.
- Step 3 intentionally stops at town-level binary cache creation; clean patch extraction and patch dataset construction remain for a later step.

## Step 4 - Teacher Patch Cache Construction

### Modified Files

- `ss3dm_prior/data/__init__.py`
- `ss3dm_prior/data/patch_types.py`
- `ss3dm_prior/data/teacher_patch_builder.py`
- `ss3dm_prior/data/patch_index.py`
- `ss3dm_prior/data/town_mesh_cache.py`
- `ss3dm_prior/tools/build_teacher_patch_cache.py`
- `ss3dm_prior/tools/check_teacher_patch_cache.py`
- `configs/ss3dm_prior/teacher_patch_default.yaml`
- `tests/ss3dm_prior/test_teacher_patch_builder.py`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_update_log.md`
- `docs/ss3dm_prior_data_schema.md`

### Design Rationale

- Used Step 2 tile centers as patch anchors so every patch is tied to an actually observed local support region.
- Built clean teacher patches from the town mesh cache rather than from raw OBJ so local extraction is offline, repeatable, and fast enough for later dataset iteration.
- Kept observed patches as sparse local anchors only; synthetic corruption is intentionally deferred to dataset/training time.
- Normalized both clean and observed points into the same local frame by subtracting the world patch center and dividing by the patch radius.
- Added an explicit `town_mesh_unit_scale` because the cached town mesh geometry is stored in a different raw scale than the observed cache world coordinates, and real-data patch extraction required this alignment.

### Actual Commands

- `python -m ss3dm_prior.tools.build_teacher_patch_cache --help`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache --help`
- `pytest tests/ss3dm_prior/test_teacher_patch_builder.py -q`
- `python -m py_compile ss3dm_prior/__init__.py ss3dm_prior/data/__init__.py ss3dm_prior/data/discovery.py ss3dm_prior/data/manifest.py ss3dm_prior/data/scenario_loader.py ss3dm_prior/data/raw_sequence.py ss3dm_prior/data/lidar_io.py ss3dm_prior/data/observed_fusion.py ss3dm_prior/data/obj_converter.py ss3dm_prior/data/town_mesh_cache.py ss3dm_prior/data/patch_types.py ss3dm_prior/data/patch_index.py ss3dm_prior/data/teacher_patch_builder.py ss3dm_prior/tools/__init__.py ss3dm_prior/tools/build_manifest.py ss3dm_prior/tools/check_manifest.py ss3dm_prior/tools/inspect_raw_sequence.py ss3dm_prior/tools/build_observed_cache.py ss3dm_prior/tools/check_observed_cache.py ss3dm_prior/tools/convert_town_obj.py ss3dm_prior/tools/check_town_mesh_cache.py ss3dm_prior/tools/build_teacher_patch_cache.py ss3dm_prior/tools/check_teacher_patch_cache.py ss3dm_prior/utils/__init__.py ss3dm_prior/utils/io.py tests/ss3dm_prior/conftest.py tests/ss3dm_prior/test_manifest.py tests/ss3dm_prior/test_scenario_loader.py tests/ss3dm_prior/test_observed_fusion.py tests/ss3dm_prior/test_obj_converter.py tests/ss3dm_prior/test_teacher_patch_builder.py`
- `python -m ss3dm_prior.tools.build_teacher_patch_cache --manifest /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/debug_town_split.yaml --config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/teacher_patch_default.yaml --observed_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache_debug --town_mesh_cache_root /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache_smoke --out_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_debug_val --subsets val --debug_max_sequences 1 --debug_max_tiles_per_sequence 8 --seed 0`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache --patch_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_debug_val --num_visualizations 1`

### Result

- Real debug patch cache generation succeeded on the `debug_town_split` val subset with one `Town02` sequence.
- Build summary:
  - `processed_sequences: 1`
  - `written_patches: 8`
  - patch index at `outputs/ss3dm_prior/teacher_patch_cache_debug_val/patch_index.jsonl`
- Patch validation summary:
  - `patch_count_total: 8`
  - `clean_point_count_stats: min=2048 max=2048 mean=2048.00`
  - `observed_point_count_stats: min=512 max=512 mean=512.00`
  - `observed_raw_point_count_stats: min=96 max=170 mean=117.38`
- At least one static visualization PNG was generated:
  - `outputs/ss3dm_prior/teacher_patch_cache_debug_val/visualizations/Town02__150_streetsurf__tile_000000.png`
- The Step 4 unit test passed.

### Remaining Risks / TODO

- The current clean-face query still relies on centroid radius filtering, so it is correct enough for this stage but not yet an optimized spatial index.
- The mesh-to-observed alignment currently depends on `town_mesh_unit_scale=0.01`; this works for the real smoke test but should be revalidated across all towns before large-scale cache generation.
- Patch cache construction currently stores canonical clean/observed pairs only; synthetic corruption and training-time augmentation remain deferred to later steps.

## Step 5 - Online Corruption, Dataset, Model, And Losses

### Modified Files

- `ss3dm_prior/data/__init__.py`
- `ss3dm_prior/data/corruptions.py`
- `ss3dm_prior/data/train_dataset.py`
- `ss3dm_prior/models/__init__.py`
- `ss3dm_prior/models/pointnet.py`
- `ss3dm_prior/models/patch_denoiser.py`
- `ss3dm_prior/losses.py`
- `ss3dm_prior/metrics.py`
- `configs/ss3dm_prior/model_default.yaml`
- `tests/ss3dm_prior/test_corruptions.py`
- `tests/ss3dm_prior/test_model_forward.py`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_data_schema.md`
- `docs/ss3dm_prior_update_log.md`

### Design Rationale

- Kept corruption online so the cached patch corpus remains canonical and reusable while training can sample infinitely many defect variants.
- Defined the learning target as `corrupted patch -> clean patch` because the first-version prior should model clean local geometry distribution, not full-scene generation.
- Used a lightweight PointNet-style architecture for stability, simplicity, and easy conditioning on both corrupted and observed local geometry.
- Added both per-point and patch-level defect heads so the model can act as a denoiser and a local quality assessor.
- Included latent alignment and retrieval metrics so the representation can later support retrieval, critic-style scoring, or auxiliary supervision beyond pure reconstruction.

### Actual Commands

- `python -m pip install torch`
- `python - <<'PY' ... import torch; print(torch.__version__) ... PY`
- `pytest tests/ss3dm_prior/test_corruptions.py -q`
- `pytest tests/ss3dm_prior/test_model_forward.py -q`
- `python - <<'PY' ... TeacherPatchTrainDataset -> DataLoader -> LocalPatchDenoiser -> compute_patch_losses ... PY`
- `python -m py_compile ss3dm_prior/__init__.py ss3dm_prior/data/__init__.py ss3dm_prior/data/discovery.py ss3dm_prior/data/manifest.py ss3dm_prior/data/scenario_loader.py ss3dm_prior/data/raw_sequence.py ss3dm_prior/data/lidar_io.py ss3dm_prior/data/observed_fusion.py ss3dm_prior/data/obj_converter.py ss3dm_prior/data/town_mesh_cache.py ss3dm_prior/data/patch_types.py ss3dm_prior/data/patch_index.py ss3dm_prior/data/teacher_patch_builder.py ss3dm_prior/data/corruptions.py ss3dm_prior/data/train_dataset.py ss3dm_prior/models/__init__.py ss3dm_prior/models/pointnet.py ss3dm_prior/models/patch_denoiser.py ss3dm_prior/losses.py ss3dm_prior/metrics.py ss3dm_prior/tools/__init__.py ss3dm_prior/tools/build_manifest.py ss3dm_prior/tools/check_manifest.py ss3dm_prior/tools/inspect_raw_sequence.py ss3dm_prior/tools/build_observed_cache.py ss3dm_prior/tools/check_observed_cache.py ss3dm_prior/tools/convert_town_obj.py ss3dm_prior/tools/check_town_mesh_cache.py ss3dm_prior/tools/build_teacher_patch_cache.py ss3dm_prior/tools/check_teacher_patch_cache.py ss3dm_prior/utils/__init__.py ss3dm_prior/utils/io.py tests/ss3dm_prior/conftest.py tests/ss3dm_prior/test_manifest.py tests/ss3dm_prior/test_scenario_loader.py tests/ss3dm_prior/test_observed_fusion.py tests/ss3dm_prior/test_obj_converter.py tests/ss3dm_prior/test_teacher_patch_builder.py tests/ss3dm_prior/test_corruptions.py tests/ss3dm_prior/test_model_forward.py`

### Result

- Online corruption now produces corrupted local patches, per-point defect targets, and continuous corruption severity scores.
- The dataset reads teacher patch cache records, filters by split, and yields clean/observed/corrupted sample tuples ready for batching.
- The local patch model returns reconstruction outputs, per-point defect predictions, patch-level scores, latents, and retrieval embeddings in a dict-based API.
- Losses and metrics now cover reconstruction, normal alignment, defect regression, patch scoring, latent alignment, denoise gain, and retrieval quality.
- Real smoke test on the debug patch cache succeeded:
  - `dataset_len: 8`
  - `batch_clean: (2, 2048, 3)`
  - `batch_corrupted: (2, 2048, 3)`
  - `recon_points: (2, 2048, 3)`
  - `point_defect_pred: (2, 2048)`
  - `patch_score_pred: (2,)`
  - `total_loss: 1.7796505689620972`
- Both new Step 5 unit tests passed.

### Remaining Risks / TODO

- The corruption targets are practical and stable for V1, but the exact severity calibration may need tuning once real training curves are available.
- Chamfer-based reconstruction and nearest-neighbor-derived defect targets are correct but may become expensive at larger batch sizes.
- Step 5 intentionally stops before any training script, optimizer setup, wandb integration, or evaluation loop.

## Step 6 - Trainer, Checkpoints, wandb, And Qualitative Visualization

### Modified Files

- `ss3dm_prior/engine/__init__.py`
- `ss3dm_prior/engine/trainer.py`
- `ss3dm_prior/engine/checkpoint.py`
- `ss3dm_prior/viz/__init__.py`
- `ss3dm_prior/viz/render_patch_panels.py`
- `ss3dm_prior/viz/render_sequence_maps.py`
- `ss3dm_prior/train.py`
- `scripts/ss3dm_prior/train_debug.sh`
- `scripts/ss3dm_prior/train_default.sh`
- `configs/ss3dm_prior/train_default.yaml`
- `tests/ss3dm_prior/test_train_smoke.py`
- `docs/ss3dm_prior_experiments.md`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_update_log.md`

### Design Rationale

- Added a standalone trainer so Step 6 can execute the local clean-geometry learning objective without touching the mesh-splatting main training codepath.
- Kept checkpoint selection split into `best_recon` and `best_gain` because reconstruction fidelity and denoising usefulness are related but not identical.
- Implemented static patch panels, sequence maps, and retrieval galleries because these are the most direct visual evidence that the model is learning a clean local geometry prior.
- Made wandb robust to offline or disabled modes so the pipeline remains usable even without login or network sync.
- Added a debug fallback that can split a small patch cache into train/val automatically, which makes smoke testing possible before a full train patch cache is built.

### Actual Commands

- `python -m ss3dm_prior.train --help`
- `python -m pip install wandb`
- `pytest tests/ss3dm_prior/test_train_smoke.py -q`
- `python -m py_compile ss3dm_prior/__init__.py ss3dm_prior/data/__init__.py ss3dm_prior/data/discovery.py ss3dm_prior/data/manifest.py ss3dm_prior/data/scenario_loader.py ss3dm_prior/data/raw_sequence.py ss3dm_prior/data/lidar_io.py ss3dm_prior/data/observed_fusion.py ss3dm_prior/data/obj_converter.py ss3dm_prior/data/town_mesh_cache.py ss3dm_prior/data/patch_types.py ss3dm_prior/data/patch_index.py ss3dm_prior/data/teacher_patch_builder.py ss3dm_prior/data/corruptions.py ss3dm_prior/data/train_dataset.py ss3dm_prior/models/__init__.py ss3dm_prior/models/pointnet.py ss3dm_prior/models/patch_denoiser.py ss3dm_prior/engine/__init__.py ss3dm_prior/engine/checkpoint.py ss3dm_prior/engine/trainer.py ss3dm_prior/viz/__init__.py ss3dm_prior/viz/render_patch_panels.py ss3dm_prior/viz/render_sequence_maps.py ss3dm_prior/losses.py ss3dm_prior/metrics.py ss3dm_prior/train.py ss3dm_prior/tools/__init__.py ss3dm_prior/tools/build_manifest.py ss3dm_prior/tools/check_manifest.py ss3dm_prior/tools/inspect_raw_sequence.py ss3dm_prior/tools/build_observed_cache.py ss3dm_prior/tools/check_observed_cache.py ss3dm_prior/tools/convert_town_obj.py ss3dm_prior/tools/check_town_mesh_cache.py ss3dm_prior/tools/build_teacher_patch_cache.py ss3dm_prior/tools/check_teacher_patch_cache.py ss3dm_prior/utils/__init__.py ss3dm_prior/utils/io.py tests/ss3dm_prior/conftest.py tests/ss3dm_prior/test_manifest.py tests/ss3dm_prior/test_scenario_loader.py tests/ss3dm_prior/test_observed_fusion.py tests/ss3dm_prior/test_obj_converter.py tests/ss3dm_prior/test_teacher_patch_builder.py tests/ss3dm_prior/test_corruptions.py tests/ss3dm_prior/test_model_forward.py tests/ss3dm_prior/test_train_smoke.py`
- `python -m ss3dm_prior.train --data_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/data_default.yaml --model_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/model_default.yaml --train_config /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_debug_real/debug_train_config.yaml --manifest_path /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json --observed_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache_debug --town_mesh_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache_smoke --patch_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_debug_val --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/debug_town_split.yaml --run_name ss3dm_prior_step6_debug --output_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_debug_real --wandb_project ss3dm_prior --wandb_mode offline`

### Result

- Formal training CLI now works and produces:
  - `history.json`
  - `checkpoints/last.pt`
  - `checkpoints/best_recon.pt`
  - `checkpoints/best_gain.pt`
  - local visualization PNGs
  - offline wandb run files
- Real one-epoch debug training succeeded on the debug patch cache.
- Real output artifacts include:
  - `outputs/ss3dm_prior/train_debug_real/checkpoints/last.pt`
  - `outputs/ss3dm_prior/train_debug_real/checkpoints/best_recon.pt`
  - `outputs/ss3dm_prior/train_debug_real/checkpoints/best_gain.pt`
  - `outputs/ss3dm_prior/train_debug_real/visualizations/epoch_000/Town02__150_streetsurf__tile_000002_panel.png`
  - `outputs/ss3dm_prior/train_debug_real/visualizations/epoch_000/Town02__150_streetsurf_sequence_map.png`
  - `outputs/ss3dm_prior/train_debug_real/visualizations/epoch_000/Town02__150_streetsurf__tile_000002_retrieval.png`
- Debug run summary:
  - `best_recon: 1.0407075881958008`
  - `best_gain: -0.9008909314870834`
- The new training smoke test passed.

### Remaining Risks / TODO

- The current debug run uses a very small patch cache, so the best-checkpoint numbers only validate pipeline functionality, not model quality.
- Validation retrieval and rank-based metrics can still become noisy or undefined on tiny validation sets; the code now degrades gracefully with warnings.
- Step 6 intentionally stops before any standalone test evaluation script or held-out test report generation.

## Step 7 - Standalone Test Eval, Export, And Reporting

### Modified Files

- `ss3dm_prior/eval.py`
- `ss3dm_prior/reporting.py`
- `scripts/ss3dm_prior/eval_default.sh`
- `tests/ss3dm_prior/test_eval_smoke.py`
- `docs/ss3dm_prior_experiments.md`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_update_log.md`

### Design Rationale

- Added a standalone eval entrypoint so held-out test reporting stays decoupled from the main training codepath and from any mesh-splatting interfaces.
- Reused the existing checkpoint format and model config stored inside the checkpoint so eval can run without extra architecture arguments.
- Exported JSON, CSV, and Markdown together because test evaluation needs both machine-readable summaries and a human-readable report with linked figures.
- Curated patch panels for `best_gain`, `worst_gain`, and `largest_score_error` so the output highlights both successes and failure modes rather than only averages.
- Generated at least one sequence map per test town and one retrieval gallery so the learned prior can be inspected spatially and in embedding space.

### Actual Commands

- `python -m ss3dm_prior.eval --help`
- `pytest tests/ss3dm_prior/test_eval_smoke.py -q`
- `python - <<'PY' ... dump_yaml(outputs/ss3dm_prior/train_debug_real/eval_debug_town02.yaml, {'test_towns': ['Town02']}) ... PY`
- `python -m ss3dm_prior.eval --checkpoint /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_debug_real/checkpoints/best_recon.pt --manifest_path /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json --patch_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_debug_val --split_config /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_debug_real/eval_debug_town02.yaml --output_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior_eval --eval_name smoke_debug_eval --wandb_project ss3dm_prior_eval --wandb_mode disabled`
- `python -m py_compile ss3dm_prior/eval.py ss3dm_prior/reporting.py tests/ss3dm_prior/test_eval_smoke.py`

### Result

- The new eval CLI runs on a checkpoint and writes the requested standardized output directory structure.
- Smoke eval artifacts were produced under `outputs/ss3dm_prior_eval/smoke_debug_eval`:
  - `metrics_summary.json`
  - `metrics_per_town.csv`
  - `metrics_per_sequence.csv`
  - `patch_predictions.csv`
  - `report.md`
  - `patch_panels/best_gain__Town02__150_streetsurf__tile_000001.png`
  - `patch_panels/worst_gain__Town02__150_streetsurf__tile_000005.png`
  - `patch_panels/largest_score_error__Town02__150_streetsurf__tile_000005.png`
  - `sequence_maps/Town02__150_streetsurf.png`
  - `retrieval_gallery/Town02__150_streetsurf__tile_000004_retrieval.png`
- Smoke eval global summary on the debug checkpoint:
  - `recon_chamfer_l1: 1.0020790174603462`
  - `recon_normal_cosine: 0.2569358628243208`
  - `denoise_gain_chamfer: -0.8718227623030543`
  - `score_mae: 0.15290501154959202`
  - `score_spearman: 0.07142857142857144`
  - `point_defect_mae: 0.09508668165653944`
  - `retrieval_top1: 0.125`
  - `retrieval_top5: 0.625`
- The new Step 7 smoke test passed.

### Remaining Risks / TODO

- Eval currently mirrors the training-time online corruption recipe from the checkpoint config, so exact test numbers still depend on that corruption policy and seed.
- Per-town and per-sequence retrieval metrics are not exported yet; only global retrieval is summarized in this first Step 7 version.
- The current smoke eval uses the small debug patch cache, so it validates the reporting pipeline but not full held-out generalization quality.
