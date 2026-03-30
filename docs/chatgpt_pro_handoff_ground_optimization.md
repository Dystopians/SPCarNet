# Ground-Aware Optimization Handoff (for GPT-Pro, 2026-03-27 Updated)

## 1) Core Requirement (Updated)

I need a method that **wins both geometry and FPS** over baseline on this parking-lot scene.

Minimum acceptance criteria (hard constraints):

- Geometry must beat baseline (not tie):
  - depth quality: lower `AbsRel` and/or higher `delta<1.25`
  - normal quality: lower mean angular error (or higher cosine alignment)
- FPS must beat baseline clearly.
- Visual quality should not collapse:
  - no obvious degradation in per-view comparison
  - global image metrics should not regress significantly.

I am frustrated because the current branch mostly wins only in speed/compactness.

---

## 2) Scene / Environment

- Scene root: `/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix`
- Images: `images`
- COLMAP sparse available:
  - `sparse/0/images.bin`
  - `sparse/0/points3D.bin`
- Total cameras: `425`
- Eval test views in LLFF protocol: `54`
- Python 3.11 + CUDA setup (`mesh_splatting` env)

---

## 3) What Has Been Implemented in Code

### 3.1 Ground-aware training pipeline

Relevant files:

- `train.py`
- `arguments/__init__.py`
- `utils/ground_plane_utils.py`
- `utils/ground_association_utils.py`
- `utils/ground_regularization_utils.py`
- `utils/ground_mask_utils.py`
- `inspect_ground_reliability.py`

Main mechanisms:

- ground-mask loading and matching
- ground-plane estimation/fitting
- mesh-space ground association
- ground losses (plane/normal/smoothness)
- warmup and diagnostics logs

### 3.2 Robustness fixes already applied

- normal-loss height gate bug fixed:
  - `ground_normal_max_abs_height` now actually gates normal supervision.
- tracker reset after topology mutation:
  - association state reset after prune/split/delaunay topology changes.
- adaptive scaling was tested and found unstable in this scene.

### 3.3 New split and geometry-eval utilities added (this update)

- `--split_strategy` + `--split_file` (switchable split behavior)
  - default `llff`
  - optional `file`
- `create_colmap_outoftrain_split.py`
  - creates pose-separated out-of-train split JSON
- `evaluate_geometry_colmap.py`
  - evaluates geometry against COLMAP sparse correspondences
  - depth metrics + sparse normal consistency proxy

---

## 4) Train/Test Split Status (Important)

### Old default split (LLFF holdout)

- train/test are interleaved (`idx % 8`)
- measured separation:
  - min test-to-train pose distance: `0.0247`
  - median: `0.1530`
- this is near-neighbor validation, not strict extrapolation.

### New out-of-train split (generated)

- file: `.../sparse/0/split_outoftrain_v1.json`
- counts:
  - total `425`
  - train `361`
  - test `51`
  - dropped buffer `13`
- separation:
  - min `0.6728`
  - median `1.4921`

Interpretation:

- new split is much more out-of-distribution than LLFF holdout.

---

## 5) Full Experimental Status (Visual / Geometry / FPS)

## 5.1 Historical 16000 training runs (visual metrics focus, from previous logs)

Best test PSNR among major runs:

- baseline best: `18.8198` (`parking_phone_tiny_baseline_16000_pure`, iter 15000)
- ground v1 best: `18.8294` (+0.0096)
- ground v2 conservative best: `18.8471` (+0.0273)  <- best observed gain
- ground v3 best: `18.7744` (worse than baseline)
- recover best: `18.8199` (~tie)
- fix-normalgate best: `18.7729` (worse)
- toporeset best: `18.7700` (worse)
- adaptive best: `18.2398` (collapsed)

Conclusion from this phase:

- only v2 had a tiny gain; most variants did not robustly beat baseline.

## 5.2 Recent fast comparison protocol (same-resolution family)

Compared:

- `parking_base_16000` @ 15000
- `parking_ground_resume_16500` @ 16000
- `parking_ground_resume_16500` @ 16500

Image metrics (54 views):

- baseline_15000:
  - PSNR `17.8351`, SSIM `0.5589`, LPIPS `0.4402`
- method_16000:
  - PSNR `18.0843`, SSIM `0.5673`, LPIPS `0.4353` (better than this baseline variant)
- method_16500:
  - PSNR `17.9103`, SSIM `0.5575`, LPIPS `0.4416`

Performance:

- baseline_15000: `64.73s / 54` (~`0.83 FPS`)
- method_16000: `57.74s / 54` (~`0.94 FPS`)
- method_16500: `41.38s / 54` (~`1.31 FPS`)

Topology:

- baseline_15000: triangles `626,830`, vertices `204,175`
- method_16000: triangles `626,830`, vertices `204,175`
- method_16500: triangles `181,288`, vertices `149,033`

## 5.3 Original baseline audit (most important correction)

True original baseline used for audit:

- `parking_phone_tiny_baseline_fastcmp_30000` @ `iteration_15000`
- this run is pure baseline (`enable_ground_masks=False`)

Against methods (`method_16000` / `method_16500`), resized visual protocol:

- mean MAE:
  - original baseline: `0.08282`
  - method_16000: `0.09260`
  - method_16500: `0.09556`
- per-view wins (54 views, lower MAE better):
  - baseline better vs method_16000: `50/54`
  - baseline better vs method_16500: `50/54`

Approx resized-image quality summary:

- original baseline: PSNR `17.764`, MAE `0.08282`
- method_16000: PSNR `17.309`, MAE `0.09260`
- method_16500: PSNR `17.147`, MAE `0.09556`

Performance for original baseline:

- `141.91s / 54` (~`0.38 FPS`) -> much slower than methods.

## 5.4 Geometry realism (new COLMAP-based evaluation)

Script:

- `evaluate_geometry_colmap.py`

Protocol:

- depth GT proxy: COLMAP sparse point reprojection depths
- normal GT proxy: local PCA normals on sparse COLMAP points

Results (LLFF split):

- original baseline (`parking_phone_tiny_baseline_fastcmp_30000@15000`)
  - Depth:
    - AbsRel `0.05666`
    - MAE `1.2496`
    - delta<1.25 `0.9439`
  - Normal:
    - mean angle `39.13 deg`
    - mean abs-cos `0.7106`

- method_16000 (`parking_ground_resume_16500@16000`)
  - Depth:
    - AbsRel `0.05833`
    - MAE `1.2595`
    - delta<1.25 `0.9395`
  - Normal:
    - mean angle `41.14 deg`
    - mean abs-cos `0.6896`

- method_16500 (`parking_ground_resume_16500@16500`)
  - Depth:
    - AbsRel `0.05945`
    - MAE `1.2666`
    - delta<1.25 `0.9385`
  - Normal:
    - mean angle `40.68 deg`
    - mean abs-cos `0.6939`

Current geometry conclusion:

- baseline is still better on these geometry proxies.
- methods are faster but currently weaker geometrically.

---

## 6) The Actual Conflict

Current branch gives:

- clear speed/compactness gains
- but geometry and (against original baseline) visual fidelity regressions

Need:

- keep speed gains
- recover and surpass baseline geometry
- avoid visual regression

This is now a constrained multi-objective optimization problem, not a single-metric tuning issue.

---

## 7) What I Need GPT-Pro To Produce

Please provide a concrete plan to reach:

1. **FPS > baseline** and
2. **Geometry > baseline** (depth + normal proxy) and
3. **No meaningful visual collapse**.

I need:

- a precise training recipe (stage-wise schedule, loss weights, warmup design)
- target parameter ranges with rationale
- early-stop and rollback criteria tied to geometry metrics
- minimal ablation matrix (few runs, high information gain)
- which operations should be frozen/disabled in stage B (e.g., topology mutations)

---

## 8) Reorganized Questions for GPT-Pro (copy-ready)

1. Given the results above, design a **two-stage or multi-stage recipe** that can beat baseline on geometry while preserving the method's FPS advantage.
2. How should ground regularization be scheduled to avoid the warmup shock observed in adaptive runs?
3. Should geometry supervision use soft confidence weighting from association stats instead of hard thresholds, and how exactly?
4. How to align optimization target with near-field parking-ground quality (ROI weighting) without sacrificing global quality?
5. Which metrics should gate checkpoint acceptance during training (depth AbsRel, delta<1.25, normal mean angle, PSNR, LPIPS, FPS), and what thresholds are realistic?
6. How to run a fair evaluation when resolutions differ across legacy runs, and what exact re-render protocol should be enforced?
7. On top of LLFF split, how should out-of-train split be integrated into training/eval loops for robust claims?

---

## 9) Key Paths

Code:

- `train.py`
- `arguments/__init__.py`
- `scene/dataset_readers.py`
- `utils/ground_regularization_utils.py`
- `utils/ground_association_utils.py`
- `create_colmap_outoftrain_split.py`
- `evaluate_geometry_colmap.py`

Important reports/artifacts:

- Original baseline visual audit:
  - `models/parking_ground_resume_16500/visual_comparisons_original_baseline/`
- Geometry eval JSONs:
  - `models/parking_phone_tiny_baseline_fastcmp_30000/geometry_eval_colmap/iter_15000_llff.json`
  - `models/parking_ground_resume_16500/geometry_eval_colmap/iter_16000_llff.json`
  - `models/parking_ground_resume_16500/geometry_eval_colmap/iter_16500_llff.json`
- Out-of-train split:
  - `.../sparse/0/split_outoftrain_v1.json`

---

## 10) Key Code Snippets (Integrated)

### 10.1 Split switch (argument layer)

From `arguments/__init__.py`:

```python
# Camera split configuration for COLMAP scenes.
# split_strategy:
# - "llff": default every-N holdout
# - "file": load explicit train/test split from split_file
self.split_strategy = "llff"
self.split_file = ""
```

### 10.2 Split forwarding into scene loader

From `scene/__init__.py`:

```python
scene_info = sceneLoadTypeCallbacks["Colmap"](
    args.source_path,
    args.images,
    args.eval,
    split_strategy=getattr(args, "split_strategy", "llff"),
    split_file=getattr(args, "split_file", ""),
)
```

### 10.3 File-based split logic (COLMAP loader)

From `scene/dataset_readers.py`:

```python
if eval:
    if split_strategy == "file":
        train_keys, test_keys, dropped_keys = _load_colmap_split_file(split_file)
        ...
        for c in cam_infos:
            key = _normalize_image_key(c.image_name)
            if key in dropped_keys:
                continue
            if key in test_keys:
                test_cam_infos.append(c)
            elif key in train_keys:
                train_cam_infos.append(c)
            else:
                train_cam_infos.append(c)  # unspecified -> train
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
```

### 10.4 Out-of-train split generation (pose-space)

From `create_colmap_outoftrain_split.py`:

```python
d = np.linalg.norm(centers - centers[anchor], axis=1)
order = np.argsort(d)
test_idx = order[:test_count]
dropped_idx = order[test_count : test_count + gap_count]
train_idx = order[test_count + gap_count :]
```

and selection objective:

```python
cross = np.linalg.norm(
    centers[test_idx][:, None, :] - centers[train_idx][None, :, :], axis=2
)
min_cross = float(cross.min())
med_cross = float(np.median(cross.min(axis=1)))
score = (min_cross, med_cross)  # maximize separation
```

### 10.5 COLMAP geometry evaluation core

From `evaluate_geometry_colmap.py`:

```python
render_pkg = render(view, triangles, pipe, background)
pred_depth = render_pkg["surf_depth"][0].detach().cpu().numpy()
pred_normal = render_pkg["rend_normal"].detach().cpu().numpy().transpose(1, 2, 0)
```

sparse depth GT projection:

```python
xyz_cam = xyz @ np.array(cam_info.R, dtype=np.float64) + np.array(cam_info.T, dtype=np.float64)[None, :]
gt_depth = xyz_cam[:, 2]
pd = pred_depth[py, px]
```

depth metrics:

```python
abs_rel = np.mean(np.abs(pred - gt) / np.clip(gt, 1e-8, None))
ratio = np.maximum(pred / np.clip(gt, 1e-8, None), gt / np.clip(pred, 1e-8, None))
d1 = np.mean(ratio < 1.25)
```

normal proxy:

```python
# estimate sparse GT normal via local PCA on COLMAP points
w, v = np.linalg.eigh(cov)
gn = v[:, np.argmin(w)]
cos_abs = abs(np.dot(pred_normal_at_pixel, gn))
ang_deg = np.degrees(np.arccos(np.clip(cos_abs, 0.0, 1.0)))
```

### 10.6 Ground normal height-gating fix context

From `utils/ground_regularization_utils.py` (logic now active):

```python
if float(cfg.normal_max_abs_height) > 0:
    signed_h = point_to_plane_signed_distance(centroids, plane_normal, plane_offset)
    keep_h = signed_h.abs() <= float(cfg.normal_max_abs_height)
    valid = valid & keep_h
```

### 10.7 Topology-change safety reset (context)

Training loop now resets ground supervision state after topology-changing events
(prune / split / delaunay) to avoid stale triangle-index statistics.


## 11) 2026-03-27 Incremental Update Log (PRISM + Training Ops)

### 11.1 PRISM validation / rollback (global gate)

Implemented global dev-validation + stage-best rollback integration:

- new module: `utils/prism_validation.py`
- train integration: `train.py`
- new args in `arguments/__init__.py`:
  - `prism_validation_interval`
  - `prism_validation_max_views`
  - `prism_rollback_absrel_rel_thresh`
  - `prism_rollback_mean_angle_thresh`
  - `prism_rollback_psnr_drop_thresh`
  - `prism_rollback_mae_increase_thresh`

Behavior:

- if `split_strategy=file` and split has `dropped`, use dropped buffer as PRISM dev-validation set
- periodic validation summary dump to:
  - `<model_path>/prism_validation/validation_iter_*.json`
  - `<model_path>/prism_validation/validation_iter_*.md`
- at prune-recovery round end, compare current validation vs stage-best and rollback round snapshot when gate fails
- optional ROI-only metrics are exported as analysis-only breakdown (global gate remains global)

### 11.2 PRISM experiment automation + benchmark outputs

Added parking-ground experiment scripts:

- `scripts/parking_ground/run_case.sh`
- `scripts/parking_ground/run_geom_first_no_prism.sh`
- `scripts/parking_ground/run_geom_first_dead_only.sh`
- `scripts/parking_ground/run_geom_first_full_prism.sh`
- `scripts/parking_ground/run_geom_first_full_prism_ground_protect.sh`
- `scripts/parking_ground/run_geom_first_grounding.sh`
- `scripts/parking_ground/run_full_practice_suite.sh`
- `scripts/parking_ground/benchmark_prism_runs.py`
- `scripts/parking_ground/make_qualitative_panels.py`

Benchmark outputs:

- machine-readable json: `benchmarks/prism_parking_ground/<timestamp>/benchmark_results.json`
- markdown summary: `benchmarks/prism_parking_ground/<timestamp>/benchmark_summary.md`
- qualitative panels + index:
  - `benchmarks/prism_parking_ground/<timestamp>/qualitative_panels/*.png`
  - `qualitative_summary.md`
  - `qualitative_summary.json`

### 11.3 Auto-GPU parallel launcher

Added:

- `scripts/parking_ground/run_full_practice_suite_auto_gpu.sh`

Behavior:

- auto-select 4 least-busy GPUs from `nvidia-smi` (utilization + memory)
- run 4 training jobs in parallel (baseline / grounding / full PRISM / PRISM+ground protect)
- wait for completion, then run benchmark + qualitative export

Bugfix applied:

- fixed PID capture issue (`wait: not a pid`) by redirecting launch-info to stderr so stdout returns pure PID.

### 11.4 Training speed/stability fixes and optional knobs

1) baseline PRISM-overhead regression fix:

- baseline (PRISM disabled) no longer triggers PRISM validation evaluation path.

2) WandB overhead control:

- scalar logging dedup/filter in `train.py` (`_wandb_log_filtered`)
- `--wandb_scalar_log_interval` added
- suite default raised `WANDB_IMAGE_LOG_INTERVAL` from 1000 to 5000
- `WANDB_ENABLE` gate added to full-suite launcher

3) ground smoothness configurability (default behavior unchanged):

- added `ground_smooth_tri_adj_max_triangles` (default `4096`)
- replaces hardcoded tri-adjacency threshold in `utils/ground_regularization_utils.py`

### 11.5 WandB test-metrics regression fix

Issue introduced during logging refactor:

- disabling fixed-view image logging also unintentionally disabled `test/*` and `train/*` evaluation metrics.

Fix:

- decoupled fixed-view image upload from numerical validation reporting in `training_report`:
  - fixed images controlled by `wandb_disable_fixed_views`
  - test/train scalar evaluation controlled by `testing_iterations` (independent path)

### 11.6 Ground-mask alignment utility

Added utility:

- `scripts/parking_ground/align_ground_masks.py`

Purpose:

- align sparse segmentation masks to image stems expected by training
- e.g. source `00001.png` -> aligned `images_00001.png`
- supports `symlink|hardlink|copy` output modes
- writes alignment coverage reports:
  - `alignment_summary.json`
  - `alignment_summary.md`

Current parking dataset alignment result:

- source masks: `173`
- train/eval images: `425`
- matched after alignment: `173/425` (coverage `0.4071`)

### 11.7 Dense mask preparation for grounding reruns

Added:

- `scripts/parking_ground/prepare_ground_masks_dense.py`

Purpose:

- produce one mask per training image stem (`images_XXXXX.png`)
- fill missing masks via nearest numeric-id source mask
- save as binary single-channel PNG for faster loading

Current generated dense set:

- output: `/data2/peilincai/parking_phone_tiny_anonymized/SegmentationClass_dense_for_training`
- matched exact: `173`
- matched nearest-fill: `252`
- missing: `0`
- final coverage: `425/425` (`1.0000`)

### 11.8 PRISM 12k-stage slowdown root-cause fix

Observed issue:

- PRISM runs appeared to stall around ~12000 steps (stats-collection phase entry).
- process alive, but step advancement became extremely slow.

Root cause:

- heavy PRISM feature recomputation (structure metrics + sparse support) could still run too frequently
  for large meshes in post-12k phases.

Fix:

- added `prism_score_recompute_interval` (default `500`) in `arguments/__init__.py`
- in `train.py`, heavy PRISM features are now cached and recomputed only when:
  - forced (before prune attempt)
  - topology count changes
  - recompute interval elapsed
- prune attempts still force a fresh recompute to preserve decision quality.

Additional state-machine correction:

- in `utils/prism_pipeline.py`, prune attempt scheduling is now one-shot per round
  (instead of repeatedly attempting every iteration while in prune phase).
- this avoids pathological repeated heavy prune checks after stats-collection boundary.
