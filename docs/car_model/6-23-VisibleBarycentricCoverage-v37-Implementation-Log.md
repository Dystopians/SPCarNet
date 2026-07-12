# v37 Visible Barycentric Coverage Fix and Atlas Target Gate

日期：2026-06-23  
状态：implementation + smoke + Bonsai full-res target/train/eval verified  
结论：v37 修复了 v35/v36 暴露出的 evidence coverage 机制短板，并完成 Bonsai full-res target/train atlas replay；但最终 held-out 指标退化，因此不是新的 promoted method endpoint。

## 1. Motivation

v35/v36 的 residual atlas 分支出现了一个明确模式：

- train-only policy-val gate 很强；
- residual atlas technically accepted；
- held-out target render 几乎没有被改动；
- full-res metrics 接近 compact parent，远低于 Phase-J。

v36 的 target apply 只修改了 `205 / 59,932,637` pixels，changed fraction 为 `0.00000342`。这说明瓶颈不是 atlas 容量本身，而是 held-out target surface support / barycentric coverage 太稀疏。

## 2. Code Changes

### 2.1 Visible-Scope Barycentric Evidence

Modified:

```text
scripts/car_model/ecsr_build_surface_evidence_cache.py
```

New behavior:

- `--save_barycentric` now supports `--barycentric_scope visible`;
- visible mode writes `barycentric` and `barycentric_valid` into each view NPZ immediately during per-view save;
- this avoids the old late-write failure mode where barycentric was only backfilled after global top-support aggregation;
- visible mode computes barycentric coordinates for all valid visible pixels instead of only global top residual supports;
- `barycentric_valid` is now strict: finite coordinates and within `[-0.05, 1.05]`;
- summary/report now include `barycentric_scope` and `barycentric_valid_pixel_fraction`.

The legacy compact mode is preserved:

```text
--barycentric_scope top_residual_supports
```

New representation-level mode:

```text
--barycentric_scope visible
```

### 2.2 Target Coverage Audit

Added:

```text
scripts/car_model/ecsr_audit_surface_residual_atlas_coverage.py
```

Purpose:

- before fitting/applying a residual atlas, quantify whether target evidence actually contains usable support for candidate atlas faces;
- report total candidate face pixels, barycentric-valid pixels, candidate barycentric-valid pixels, and actionable fraction;
- make near-noop target coverage visible before spending time on full metrics.

### 2.3 Atlas Target Changed-Fraction Gate

Modified:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

New optional gate:

```text
--min_target_changed_fraction
```

If policy-val accepts but target changed fraction is below this threshold, the audit now records:

```text
accepted_before_target_coverage: true
accepted: false
reject_reason: target_changed_fraction ... < min_target_changed_fraction ...
```

This prevents a candidate like v36 from being misread as accepted when it barely changes held-out renders.

## 3. Verification

### 3.1 Static Checks

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  scripts/car_model/ecsr_audit_surface_residual_atlas_coverage.py \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

Result: passed.

```bash
git diff --check -- \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  scripts/car_model/ecsr_audit_surface_residual_atlas_coverage.py
```

Result: passed.

### 3.2 Synthetic Barycentric Unit Smoke

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import numpy as np
p = Path('scripts/car_model/ecsr_build_surface_evidence_cache.py')
spec = importlib.util.spec_from_file_location('cache', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
face_ids = np.array([[0,0,-1],[0,0,0]], dtype=np.int32)
verts = np.array([[0,0],[2,0],[0,2]], dtype=np.float32)
faces = np.array([[0,1,2]], dtype=np.int64)
bary, valid, used = mod._compute_visible_face_barycentric(face_ids, verts, faces, selected_faces=None, chunk_pixels=2)
print('used', used)
print('valid_sum', int(valid.sum()))
print('sum_minmax', float(bary[:,valid].sum(axis=0).min()), float(bary[:,valid].sum(axis=0).max()))
print('center_like', bary[:,1,1].round(4).tolist())
PY
```

Result:

```text
used 1
valid_sum 5
sum_minmax 1.0 1.0
center_like [0.0, 0.5, 0.5]
```

### 3.3 Real Renderer Smoke: Bonsai `images_4`, One Test View

Command:

```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --iteration 26000 --split test --scene_name bonsai \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke \
  --max_views 1 --view_indices 0 \
  --save_view_npz --save_residual_rgb --save_rgb --save_barycentric \
  --barycentric_scope visible \
  --barycentric_chunk_pixels 200000 \
  --images images_4 --quiet
```

Output:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke/bonsai
```

Key checks:

```text
summary_scope: visible
summary_written: 1
mean_valid_face_id_fraction: 0.9986834319526627
barycentric_valid_pixel_fraction: 0.8102021696252465
fields include barycentric: true
barycentric shape: (3, 520, 780)
finite_all: true
barycentric sum min/max/mean: 0.999496 / 1.000493 / 1.000000
barycentric min/max on valid pixels: -0.049988 / 1.049805
covered faces: 112882
```

This is a large improvement over the old target evidence, where candidate actionable coverage was around `1e-5`.

### 3.4 Full-Resolution `images_2` Smoke Attempt

Command:

```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --iteration 26000 --split test --scene_name bonsai \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke_images2 \
  --max_views 1 --view_indices 0 \
  --save_view_npz --save_residual_rgb --save_rgb --save_barycentric \
  --barycentric_scope visible \
  --barycentric_chunk_pixels 400000 \
  --images images_2 --quiet
```

Result: blocked by GPU memory on GPU2, not by code logic.

Error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 694.00 MiB.
GPU 0 ... 650.50 MiB is free.
```

Log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke_images2/logs/build_bonsai_test_view0_images2_visible_bary_gpu2.log
```

## 4. Existing v36 Target Coverage Audit

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_audit_surface_residual_atlas_coverage.py \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/bonsai_target_surface_evidence_images2/bonsai \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_render_visible_region_carriers_images2_train46_s2_alpha1.json \
  --out_json outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke/v36_target_existing_coverage_audit.json \
  --out_md outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke/v36_target_existing_coverage_audit.md \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_faces 4096 \
  --barycentric_tolerance 0.05 \
  --min_alpha 0.03
```

Result:

```text
views: 37
views with barycentric: 22
views missing barycentric: 15
candidate faces: 2247
valid face fraction: 0.99968198
candidate face fraction: 0.00986114
barycentric valid fraction: 0.00017635
candidate barycentric valid fraction: 0.00001126
actionable fraction: 0.00001126
actionable / candidate fraction: 0.00114212
actionable pixels: 675
```

Interpretation:

> Existing v36 target evidence does contain candidate faces in about `0.986%` of pixels, but only `0.001126%` of all pixels have both candidate face support and valid barycentric coordinates. This directly explains why v36 changed only 205 pixels.

## 5. Coverage-Gate Replay

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_surface_evidence_images2_train46_s2_alpha1 \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/bonsai_target_surface_evidence_images2/bonsai \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_render_visible_region_carriers_images2_train46_s2_alpha1.json \
  --output_model outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke/bonsai_teacher_region_texture_adapter_v37_coveragegate \
  --target_split test \
  --method_name ours_26000_teacher_region_texture_adapter_v37_coveragegate \
  --texture_size 16 \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_faces 4096 \
  --policy_val_stride 4 \
  --alpha_grid 0,0.125,0.25,0.5,0.75,1.0 \
  --min_l1 0.001 \
  --min_alpha 0.03 \
  --max_samples_per_view 240000 \
  --max_abs_delta_rgb 0.12 \
  --min_policy_val_samples 1024 \
  --min_policy_val_relative_gain 0.001 \
  --min_target_changed_fraction 0.0001 \
  --force
```

Expected / observed behavior:

```text
accepted_before_target_coverage: true
accepted: false
selected_alpha: 1.0
policy_val_relative_gain: 0.7518149841
target_changed_fraction: 0.0000034205
reject_reason: target_changed_fraction 0.00000342 < min_target_changed_fraction 0.00010000
```

Audit:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke/bonsai_teacher_region_texture_adapter_v37_coveragegate/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke/bonsai_teacher_region_texture_adapter_v37_coveragegate/surface_residual_region_texture_adapter_audit.md
```

## 6. What This Fixes

Before v37:

- barycentric maps were late-written after global top-support aggregation;
- interrupted evidence builds could leave many target NPZs without barycentric;
- even successful builds only stored barycentric for top residual supports;
- atlas selection could pass train policy-val while changing almost no held-out pixels.

After v37:

- visible-scope barycentric is written immediately into each view NPZ;
- output summary exposes true barycentric valid pixel fraction;
- coverage audit quantifies target support before full metrics;
- adapter can reject target no-op candidates with `--min_target_changed_fraction`.

## 7. Full-Resolution Bonsai Rerun

The full-resolution rerun has now been completed. The old v36 failure mode was a target-coverage/no-op failure; v37 fixes that part but exposes a deeper residual-transfer generalization problem.

### 7.1 Target Evidence

Command:

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --iteration 26000 --split test --scene_name bonsai \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2 \
  --max_views 37 --view_stride 1 --view_offset 0 \
  --save_view_npz --save_residual_rgb --save_rgb --save_barycentric \
  --barycentric_scope visible \
  --barycentric_chunk_pixels 400000 \
  --images images_2 --quiet
```

Output:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/bonsai
```

Summary:

```text
num_views: 37
barycentric_scope: visible
barycentric_written_views: 37
barycentric_valid_pixel_fraction: 0.9303566268909542
mean_valid_face_id_fraction: 0.9996711474584372
```

Coverage audit:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_audit_surface_residual_atlas_coverage.py \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/bonsai \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_render_visible_region_carriers_images2_train46_s2_alpha1.json \
  --out_json outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/v37_target_coverage_audit.json \
  --out_md outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/v37_target_coverage_audit.md \
  --max_carriers 64 --max_faces_per_carrier 128 --max_faces 4096 \
  --barycentric_tolerance 0.05 --min_alpha 0.03
```

Result:

```text
views: 37
views with barycentric: 37
views missing barycentric: 0
candidate faces: 2247
valid face fraction: 0.99967115
candidate face fraction: 0.00986114
barycentric valid fraction: 0.93035663
candidate/actionable fraction: 0.00968427
actionable / candidate fraction: 0.98206442
actionable pixels: 580404
```

Relative to the old target evidence:

```text
actionable pixels: 675 -> 580404
candidate/actionable fraction: 0.00001126 -> 0.00968427
```

This confirms that v37 fixes the target-coverage bottleneck.

### 7.2 Train Evidence and Teacher Residual Cache

Train evidence command:

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --iteration 26000 --split train --scene_name bonsai \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2 \
  --max_views 46 --view_stride 2 --view_offset 0 \
  --save_view_npz --save_residual_rgb --save_rgb --save_barycentric \
  --barycentric_scope visible \
  --barycentric_chunk_pixels 400000 \
  --images images_2 --quiet
```

Summary:

```text
num_views: 46
barycentric_scope: visible
barycentric_written_views: 46
barycentric_valid_pixel_fraction: 0.9268109772904739
mean_valid_face_id_fraction: 0.9999197700694475
```

Teacher evidence summary:

```text
processed_views: 46
mean_active_fraction: 0.1937276084611897
mean_target_l1: 0.005636017870805834
mean_raw_parent_delta_l1: 0.00902401079909633
mean_positive_teacher_gain_l1: 0.005435154808725676
```

Region carriers:

```text
carriers: 64
raw regions: 552
evidence faces: 2048
```

### 7.3 Visible-Train / Visible-Target Atlas

Adapter audit:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_region_texture_adapter_v37_visible_train_target/surface_residual_region_texture_adapter_audit.json
```

Key values:

```text
accepted: true
selected_alpha: 0.75
candidate_faces: 2247
atlas_faces: 2208
fit_samples: 205165
policy_val_relative_gain: 0.3374342158
target changed pixels: 578910
target changed fraction: 0.0096593447
target coverage gate: passed
```

This is a large materialization improvement over v36:

```text
v36 target changed pixels: 205
v37 target changed pixels: 578910
```

### 7.4 Full-Resolution Held-Out Metrics

Metric command:

```bash
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py \
  -m outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_region_texture_adapter_v37_visible_train_target \
  > outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/logs/metrics_bonsai_teacher_region_texture_adapter_v37_visible_train_target_gpu3.log 2>&1
```

Result:

```text
PSNR: 28.801197052001953
SSIM: 0.8915395736694336
LPIPS: 0.264999657869339
```

Comparison:

| method on Bonsai | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| selected clean `ours_26000` | 28.8952 | 0.8964 | 0.2595 |
| compact parent | 28.8643 | 0.8960 | 0.2593 |
| v36 matched-res atlas | 28.8648 | 0.8960 | 0.2593 |
| v37 old-train visible-target atlas | 28.8628 | 0.8959 | 0.2594 |
| v37 visible-train visible-target atlas | 28.8012 | 0.8915 | 0.2650 |
| Phase-J render-time ELA | 31.8620 | 0.9303 | 0.1726 |

## 8. Updated Diagnosis

v37 should not be promoted as a method endpoint. It is still important because it cleanly separates two failure modes:

1. **Target coverage failure is fixed.** Visible barycentric support now reaches the held-out target pixels, and the atlas changes almost `0.966%` of test pixels instead of `0.000342%`.
2. **Residual-transfer generalization is still unsolved.** The larger target edit worsens full-res PSNR/SSIM/LPIPS, so the residual texture is not yet safe across view, depth, occlusion, and material changes.

Next method direction:

- split carrier fitting and carrier holdout rather than only view holdout;
- use view-stratified policy-val bins for normal/view-angle/depth discontinuity;
- downweight high-variance support residuals;
- add residual smoothness or low-rank constraints to avoid high-frequency texture drift;
- add target-risk proxy from support count, barycentric stability, normal angle, and edge proximity.
