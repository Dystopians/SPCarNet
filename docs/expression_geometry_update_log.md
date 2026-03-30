# Expression Geometry Optimization - Update Log

## Date
2026-03-11

## Background
Initial run with strict planar constraints produced:

- `merged_regions = 0`
- no reduction in vertices/triangles between iterations `30000` and `30001`

This indicates the pipeline worked, but candidate/region gating was too strict for the parking-lot scene.

---

## Code Changes in This Update

### 1) `scene/triangle_model.py`

Enhanced `optimize_ground_planar_patches(...)` with parking-oriented controls:

- `up_axis="auto"` support (area-weighted axis selection from current mesh normals)
- near-field filtering support (`near_center`, `near_radius`)
- adjacency gating now includes:
  - normal consistency (`max_neighbor_normal_deg`)
  - local height continuity (`max_neighbor_height_delta`)
- robust residual gate:
  - configurable `residual_quantile` (default `0.95`)
- optional cross-region consolidation:
  - `enable_global_snap`
  - `global_height_bin`
- optional conservative boundary participation:
  - `allow_boundary_snap`
  - `boundary_snap_max_shift`
- richer diagnostics returned in `stats`:
  - `candidate_triangles`, `regions_total`
  - rejection counters:
    - `rejected_small_region`
    - `rejected_small_area`
    - `rejected_plane_tilt`
    - `rejected_plane_residual`

### 2) `optimize_expression_geometry.py`

- Added `--preset {default, parking_lot}` (default `parking_lot`)
- Added/updated args:
  - `--up_axis auto|x|y|z`
  - `--max_neighbor_height_delta`
  - `--residual_quantile`
  - `--near_field_radius`
  - `--enable_global_snap`
  - `--global_height_bin`
  - `--allow_boundary_snap`
  - `--boundary_snap_max_shift`
- Added parking-lot preset tuning behavior
- Automatically estimates `near_center` from training camera centers when `near_field_radius > 0`

### 3) Docs

- Updated `docs/expression_geometry_optimization.md` with:
  - new controls
  - revised examples
  - tuning guidance for near-field and robust residual gating

---

## Expected Impact

Compared to the previous version, this update should:

1. avoid wrong up-axis assumptions (`auto`)
2. avoid over-expansion across curbs/walls (height continuity gate)
3. focus optimization on relevant parking-space neighborhood (near-field filter)
4. provide actionable diagnostics when no merge occurs

---

## Next Recommended Validation

1. Run optimizer and inspect the returned rejection counters.
2. If still `merged_regions=0`, loosen parameters in this order:
   1) increase `max_ground_tilt_deg`
   2) increase `max_neighbor_normal_deg`
   3) increase `max_plane_residual`
   4) increase `snap_cell_size`
3. Compare before/after:
   - triangle count reduction
   - rendering fidelity near boundaries
   - mesh export quality in near field

---

## Run Record: Parking-Lot Aggressive Pass

### Command profile

- preset: `parking_lot`
- up axis: `auto` (resolved to `z`)
- near field: disabled (`near_field_radius=-1`, full-scene)
- global snap: enabled
- boundary snap: enabled (max shift `0.015`)

### Output checkpoint

- Saved iteration: `point_cloud/iteration_30030/point_cloud_state_dict.pt`

### Quantitative result

- Triangles: `5,322,338 -> 5,144,189` (reduced `178,149`, **3.347%**)
- Vertices: `2,992,267 -> 2,894,286` (reduced `97,981`, **3.275%**)
- Merged regions: `7,263`
- Candidate triangles: `1,400,000`

### Visualization artifacts (CPU-generated)

All files are under:

`models/parking_phone_tiny/comparisons/expression_30030`

- `summary_30030.png` (numeric summary card)
- `density_before_30000.png` (triangle centroid density before)
- `density_after_30030.png` (triangle centroid density after)
- `density_delta_30030_minus_30000.png` (delta heatmap)

### Note

Render-based comparison (`render.py` for iteration 30030) was blocked during this run due external GPU memory pressure (OOM from concurrent process). The expression-level optimization itself completed successfully on CPU path.

---

## Run Record: Render + Mesh Export (iteration 30030)

### 1) Test rendering completed

Command completed successfully with `CUDA_VISIBLE_DEVICES=4`:

- `render.py --iteration 30030 --eval --skip_train`

Outputs generated under:

- `models/parking_phone_tiny/test/ours_30030/renders`
- `models/parking_phone_tiny/test/ours_30030/gt`

### 2) Side-by-side qualitative comparison generated

Top-10 triplets (`iter30000 render | iter30030 render | gt`) generated under:

- `models/parking_phone_tiny/comparisons/render_30000_vs_30030_top10`
  - `00000.png` ... `00009.png`
  - `sheet_top10.png`

### 3) Mesh export for iteration 30030

Command completed successfully with `CUDA_VISIBLE_DEVICES=4`:

- `mesh.py --iteration 30030 --skip_train --skip_test --depth_trunc 0.4 --voxel_size 0.004 --sdf_trunc 0.02 --num_cluster 10`

Outputs:

- `models/parking_phone_tiny/train/ours_30030/fuse.ply`
- `models/parking_phone_tiny/train/ours_30030/fuse_post.ply`

Observed behavior:

- Mesh topology remained small in this bounded TSDF setting (raw vertices around 2.7k after extraction).
- This is consistent with the previously observed bounded meshing truncation behavior and should be interpreted separately from expression-level triangle-count reduction metrics.

---

## Run Record: Quality-First Edge-Collapse Pass (iteration 30040)

### Objective

Prioritize visual quality and topology safety, with conservative geometry reduction.

### Expression optimization result

- Checkpoint: `point_cloud/iteration_30040/point_cloud_state_dict.pt`
- Triangles: `5,322,338 -> 5,321,953` (reduced `385`, `0.007%`)
- Vertices: `2,992,267 -> 2,992,107` (reduced `160`, `0.005%`)
- Key mode: `merge_mode=edge_collapse`, `project_to_plane=True`, boundary snap disabled

### Render outputs

- `test/ours_30040/renders`
- `test/ours_30040/gt`

### Visual comparison sheet

- `models/parking_phone_tiny/comparisons/render_30000_vs_30040_top10/00000.png` ... `00009.png`
- `models/parking_phone_tiny/comparisons/render_30000_vs_30040_top10/sheet_top10.png`

### Quantitative quality check (test set, 54 views)

- Mean PSNR (iter30000): `17.65708`
- Mean PSNR (iter30040): `17.65701`
- Delta: `-0.00008` (negligible)

Interpretation: this pass preserves quality almost exactly, but reduction is very small.

### Mesh export (iteration 30040)

- `train/ours_30040/fuse.ply`
- `train/ours_30040/fuse_post.ply`

As before, bounded TSDF export remains a small mesh artifact in this pipeline configuration and should not be used as the sole quality proxy for expression-layer optimization.

---

## Run Record: Ground-Aware Pipeline Dry-Run + Render Comparison (2026-03-11)

### Objective

- Provide a practical command set for next training stage with ground-aware regularization.
- Run a reliability pre-check for ground mask + plane fit + mesh assignment.
- Generate a full visual comparison package against the previous render result.

### Environment note

- GPU contention is high on shared cards. In this run, `render.py` on `iteration_30040` failed with CUDA OOM because the selected device became occupied by another process.
- Existing rendered outputs for `iteration_30030` and `iteration_30040` were already present and used for comparison generation.

### Reliability pre-check command

```bash
CUDA_VISIBLE_DEVICES=4 micromamba run -n mesh_splatting python inspect_ground_reliability.py \
  -s /data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix \
  -m /data2/peilincai/mesh-splatting/models/parking_phone_tiny \
  --iteration 30040 \
  --eval \
  --enable_ground_masks \
  --ground_mask_dir ../SegmentationClass \
  --ground_mask_label_rgb 197,248,171 \
  --enable_ground_plane_fit \
  --max_views 12
```

### Reliability check output (key values)

- Plane fit: `ok=True`, `enabled_for_loss=True`
- Plane normal: `(-0.0329, 0.9762, 0.2145)`, `d=-0.3412`
- Plane inlier ratio: `0.6904`
- Ground assignment coverage:
  - reliable triangles: `155,160`
  - ground-classified triangles: `395`
  - boundary-uncertain triangles: `44`
  - total triangles: `5,321,953`
- Script verdict: likely reliable for enabling ground-aware regularization.

### Full visual comparison package (generated)

Output directory:

- `models/parking_phone_tiny/comparisons/render_30030_vs_30040_groundaware_review`

Generated artifacts:

- `triplets/*.png` (54 per-view panels: `old render | new render | gt | abs(new-old) heatmap`)
- `sheet_top20_delta.png` (top-20 changed views)
- `sheet_all_compact.png` (all 54 views compact sheet)
- `metrics_summary.txt` (per-view and mean MAE summary)

### Quantitative summary from `metrics_summary.txt`

- Mean MAE (iter30030): `0.087727`
- Mean MAE (iter30040): `0.085917`
- Mean improvement (`old-new`): `+0.001811`
- Mean render delta (`|new-old|`): `0.004408`

Interpretation:

- Current `30040` render set is overall better than `30030` under MAE-to-GT.
- Most views show small positive improvements; large regressions were not observed in this comparison.

---

## Run Record: Dual Ground-Mask Full Try (no mesh export, 2026-03-11)

### Objective

- Verify two independent ground-mask sets are both trainable end-to-end.
- Run one complete training+render attempt per mask set (same config for fair comparison).
- Produce side-by-side visualization and quantitative comparison.

### Mask sets checked

- ADE mask set:
  - dir: `parking_phone_tiny_anonymized/colmap_undistorted_fix/ground_masks`
  - report model: `nvidia/segformer-b5-finetuned-ade-640-640`
  - report non-empty coverage: `408 / 425` (`96.0%`)
- Cityscapes mask set:
  - dir: `parking_phone_tiny_anonymized/colmap_undistorted_fix/ground_masks__nvidia__segformer_b5_finetuned_cityscapes_1024_1024/masks`
  - report model: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`
  - report non-empty coverage: `418 / 425` (`98.35%`)

### Attempt A (ADE masks): command and result

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True micromamba run -n mesh_splatting python train.py \
  -s /data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix \
  -m /data2/peilincai/mesh-splatting/models/parking_phone_tiny_ground_ade_try \
  -i images --eval --iterations 300 --save_iterations 300 --resolution 4 \
  --enable_ground_masks --ground_mask_dir ground_masks --ground_mask_matching exact --ground_mask_suffix .png \
  --ground_mask_missing_strategy empty \
  --enable_ground_plane_fit --enable_ground_mesh_assignment --enable_ground_regularization \
  --enable_ground_plane_loss --enable_ground_normal_loss --enable_ground_smoothness_loss \
  --lambda_ground_plane 0.01 --lambda_ground_normal 0.005 --lambda_ground_smoothness 0.005 \
  --ground_reg_start_iter 60 --ground_reg_warmup_iters 100 \
  --ground_assoc_cache_every 100 --debug_save_ground_visualizations --ground_debug_vis_every 150
```

Then:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True micromamba run -n mesh_splatting python render.py \
  -s /data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix \
  -m /data2/peilincai/mesh-splatting/models/parking_phone_tiny_ground_ade_try \
  --iteration 300 --eval --skip_train
```

Key run outputs:

- train+render: success (`exit_code=0`)
- ground-plane fit: `ok=True`, `inlier_ratio=0.8351`
- plane normal: `(-0.00550, 0.98175, 0.19010)`
- rendered test views: `54`

### Attempt B (Cityscapes masks): command and result

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True micromamba run -n mesh_splatting python train.py \
  -s /data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix \
  -m /data2/peilincai/mesh-splatting/models/parking_phone_tiny_ground_city_try \
  -i images --eval --iterations 300 --save_iterations 300 --resolution 4 \
  --enable_ground_masks \
  --ground_mask_dir ground_masks__nvidia__segformer_b5_finetuned_cityscapes_1024_1024/masks \
  --ground_mask_matching exact --ground_mask_suffix .png --ground_mask_missing_strategy empty \
  --enable_ground_plane_fit --enable_ground_mesh_assignment --enable_ground_regularization \
  --enable_ground_plane_loss --enable_ground_normal_loss --enable_ground_smoothness_loss \
  --lambda_ground_plane 0.01 --lambda_ground_normal 0.005 --lambda_ground_smoothness 0.005 \
  --ground_reg_start_iter 60 --ground_reg_warmup_iters 100 \
  --ground_assoc_cache_every 100 --debug_save_ground_visualizations --ground_debug_vis_every 150
```

Then:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True micromamba run -n mesh_splatting python render.py \
  -s /data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix \
  -m /data2/peilincai/mesh-splatting/models/parking_phone_tiny_ground_city_try \
  --iteration 300 --eval --skip_train
```

Key run outputs:

- train+render: success (`exit_code=0`)
- ground-plane fit: `ok=True`, `inlier_ratio=0.7451`
- plane normal: `(-0.00658, 0.98282, 0.18443)`
- rendered test views: `54`

### Visualization + quantitative comparison generated

Output dir:

- `models/parking_phone_tiny/comparisons/groundmask_dual_try_300`

Artifacts:

- `comparison_summary.json` (all 54 views)
- `triplets/*.png` (6 representative panels, each contains:
  `GT | ADE render | Cityscapes render | ADE abs error heatmap | City abs error heatmap`)

Mean metrics (54 views, from `comparison_summary.json`):

- mean PSNR (ADE): `6.4321`
- mean PSNR (Cityscapes): `6.4995`
- mean PSNR delta (`city - ade`): `+0.0674`
- mean MAE (ADE): `0.41209`
- mean MAE (Cityscapes): `0.40831`
- mean MAE delta (`city - ade`): `-0.00378` (lower is better)

Quick interpretation:

- In this matched 300-iter trial, Cityscapes masks are slightly better on average than ADE masks.
- Both mask sets are stable for full train+render loop with ground-aware options enabled.

---

## Run Record: Baseline-Resume Ground Try (new model dirs, 2026-03-11/12)

### Why this rerun

- Previous quick runs started from scratch (`iter=0`) and produced non-comparable broken visuals.
- This rerun starts from baseline checkpoint (`iteration_30040`) and writes to new model directories.

### New model directories

- ADE resume dir: `models/parking_phone_tiny_ground_resume30040_ade`
- Cityscapes resume dir: `models/parking_phone_tiny_ground_resume30040_city`

### Resume protocol

- Copied baseline checkpoint to both new dirs:
  - `point_cloud/iteration_30040/point_cloud_state_dict.pt`
- Trained with:
  - `--load_iteration 30040`
  - short continuation to `--iterations 30100`
  - saved at `iteration_30100`
- Then rendered:
  - `render.py --iteration 30100 --eval --skip_train`

### Key sanity checks from logs

- Both runs loaded baseline topology correctly before optimization:
  - `triangles: torch.Size([5321953, 3])`
  - `vertices: torch.Size([2992107, 3])`
- Both runs finished successfully with `exit_code=0` (train + render).

### Comparison against baseline (`ours_30040`, test set 54 views)

Source file:

- `models/parking_phone_tiny/comparisons/ground_resume30040_short30100/comparison_summary.json`

Mean metrics:

- Baseline PSNR: `17.65701`
- ADE-resume PSNR: `17.62789` (delta `-0.02912`)
- City-resume PSNR: `17.62834` (delta `-0.02866`)
- Baseline MAE: `0.0859167`
- ADE-resume MAE: `0.0861244` (delta `+0.0002077`)
- City-resume MAE: `0.0861214` (delta `+0.0002047`)

Interpretation:

- Under this short resume setting, both ground-regularized runs are very close to baseline but slightly worse on average.
- Crucially, they are now visually valid and comparable (no isolated-triangle collapse from cold-start training).

### Visualization package

- `models/parking_phone_tiny/comparisons/ground_resume30040_short30100/triplets`
  - 6 representative panels
  - each panel: `GT | Base30040 | ADE30100 | City30100`

---

## Run Record: Evaluation Protocol Upgrade + Original Baseline Audit (2026-03-27)

### Why this update

Recent comparisons exposed that "baseline" naming had drifted across experiments. We therefore:

1. explicitly restored and rendered the **original pure baseline**;
2. added a switchable **out-of-train split** pipeline;
3. added **COLMAP-based geometry evaluation** (depth + sparse normal consistency).

### 1) Original baseline identification

Confirmed original pure baseline model:

- `models/parking_phone_tiny_baseline_fastcmp_30000`
- `cfg_args`: `enable_ground_masks=False` (no ground-aware training switches)

Rendered checkpoint used for comparisons:

- `iteration_15000`
- output: `models/parking_phone_tiny_baseline_fastcmp_30000/test/ours_15000`

### 2) Original baseline vs current method checkpoints

Compared against:

- `models/parking_ground_resume_16500` @ `iteration_16000`
- `models/parking_ground_resume_16500` @ `iteration_16500`

Generated artifacts:

- `models/parking_ground_resume_16500/visual_comparisons_original_baseline/`
  - `comparison_overview_8frames.png`
  - `per_frame/compare_*.png` (54 views)
  - `error_report.csv`

Observed (resized-to-baseline-resolution visual protocol):

- mean MAE:
  - original baseline: `0.08282`
  - method_16000: `0.09260`
  - method_16500: `0.09556`
- per-view MAE wins (baseline better):
  - vs method_16000: `50 / 54`
  - vs method_16500: `50 / 54`

Performance-only side:

- render wall-time (54 views):
  - original baseline: `141.91s` (~`0.38 FPS`)
  - method_16000: `57.74s` (~`0.94 FPS`)
  - method_16500: `41.38s` (~`1.31 FPS`)
- topology:
  - baseline: triangles `721,930`, vertices `234,401`
  - method_16000: triangles `626,830`, vertices `204,175`
  - method_16500: triangles `181,288`, vertices `149,033`

Interpretation:

- Current method branch is clearly better on speed/compactness.
- In this audit, original baseline is better on visual fidelity.

### 3) Out-of-train split support (switchable)

Code changes:

- `arguments/__init__.py`: add
  - `--split_strategy` (`llff` | `file`, default `llff`)
  - `--split_file`
- `scene/__init__.py`: forward split args to COLMAP loader
- `scene/dataset_readers.py`: support explicit split file parsing (`train/test/dropped`)

New split generator:

- `create_colmap_outoftrain_split.py`

Generated split:

- `.../sparse/0/split_outoftrain_v1.json`
- counts: total `425`, train `361`, test `51`, dropped `13`
- separation:
  - out-of-train split: min `0.6728`, median `1.4921`
  - old llff-hold8: min `0.0247`, median `0.1530`

Interpretation:

- old alternating split is near-neighbor validation, not strict extrapolation.
- new split is materially farther from train distribution.

### 4) COLMAP geometry realism evaluation added

New script:

- `evaluate_geometry_colmap.py`

What it evaluates:

1. **Depth realism** using COLMAP sparse correspondences (projected sparse points):
   - MAE, RMSE, AbsRel, delta-thresholds.
2. **Normal realism proxy**:
   - local PCA normals on sparse COLMAP points vs rendered normals.
   - reports mean angular error and threshold percentages.

Saved reports:

- baseline:
  - `models/parking_phone_tiny_baseline_fastcmp_30000/geometry_eval_colmap/iter_15000_llff.json`
- method:
  - `models/parking_ground_resume_16500/geometry_eval_colmap/iter_16000_llff.json`
  - `models/parking_ground_resume_16500/geometry_eval_colmap/iter_16500_llff.json`

Key results (LLFF split):

- Depth AbsRel:
  - baseline: `0.05666`
  - method_16000: `0.05833`
  - method_16500: `0.05945`
- Depth delta<1.25:
  - baseline: `0.9439`
  - method_16000: `0.9395`
  - method_16500: `0.9385`
- Normal mean angle (deg, lower is better):
  - baseline: `39.13`
  - method_16000: `41.14`
  - method_16500: `40.68`

Interpretation:

- Under this geometry protocol, baseline currently remains stronger.
- Method does not yet convert its speed gains into geometry-quality gains.
