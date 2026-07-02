# GEMS — ASSET_MAP.md

Status: **DRAFT (GOAL #001, 2026-07-02)** — compiled from code reading by three parallel surveys.
Items tagged `[VERIFY-M0]` must be confirmed by execution before AT0 is accepted.

---

## 1. Core entry points

### Training — `train.py`
- `python train.py -s <source> -m <model_dir> --images images_4 -r -1 --eval --iterations 30000 [--indoor]`
- Config system: argparse groups in `arguments/__init__.py` (`ModelParams:56`, `OptimizationParams:101`, `PipelineParams:93`); render/eval re-read `<model>/cfg_args` via `get_combined_args:554`.
- Core loss: `(1-λ_dssim)·L1 + λ_dssim·(1-SSIM)` (`train.py:3640`, λ_dssim=0.2). Optional: LPIPS loss, weight/opacity reg, vertex-depth reg, mono-depth L1, sparse-COLMAP depth loss (`_compute_sparse_colmap_depth_loss:1221`), 2DGS-style normal loss (`train.py:3900`), teacher render loss (`_compute_teacher_render_loss:906`), parent-render rollback, checkpoint geometry/render anchors.
- Resume: `--start_checkpoint <chkpntN.pth>` (full optimizer state) or `--load_iteration N` (loads `point_cloud_state_dict.pt`, fresh optimizer).
- Topology control: legacy densify/prune loop `train.py:4701-4815` (every 500 it until `densify_until_iter`; restricted-Delaunay retopo at `densify_until_iter+1000`); freeze via `--freeze_topology_updates --skip_restricted_delaunay`; PRISM subsystem off unless `enable_prism_pruning`.

### Checkpoint format
- Model save: `<model>/point_cloud/iteration_N/point_cloud_state_dict.pt` (`scene/triangle_model.py::save_parameters:188`): `triangles_points [V,3]`, `_triangle_indices [T,3] int32` (T implicit in shape), `vertex_weight [V]` (sigmoid-activated opacity, per-VERTEX), `sigma` (global scalar softness), `features_dc [V,1,3]` + `features_rest` (per-vertex SH), `active_sh_degree`, `importance_score`, `image_size`, `pixel_count`.
- Resume ckpt: `<model>/chkpntN.pth` = `(triangles.capture(), iter)`.
- Model dir also holds `cfg_args`, `cameras.json`, `input.ply` (init cloud), `results.json`, `test|train/ours_N/{renders,gt}` after render.

### Rendering — `render.py`
- `python render.py -m <model_dir> --iteration N --skip_train [--quiet]` → `<model>/test/ours_N/{renders,gt}/*.png`. Supersampling `pc.scaling=4`.
- Renderer `triangle_renderer/__init__.py::render:103` returns dict incl. `render`, `expected_depth` (mean), `surf_depth` (median), `rend_alpha` (accumulated α; 1−α = transmittance), `rend_normal`, `surf_normal` (from depth), `rend_ids` (per-pixel triangle id), per-triangle `radii/scaling/max_blending/triangle_was_rendered`. Grads flow to vertices, vertex_weights, SH; NOT to sigma or indices.

### Evaluation (legacy mouth — replaced by run_eval.py in M1)
- `python metrics.py -m <model_dir>` → walks `test/<method>/{gt,renders}`, computes PSNR (`utils/image_utils.py:30`), SSIM (`utils/loss_utils.py:110`), LPIPS-vgg (`lpipsPyTorch`) → `results.json`, `per_view.json`.
- `full_eval.py` = orchestrator; `eval.py` = DTU-style Chamfer mesh eval; `evaluate_geometry_colmap.py` = COLMAP-points geometry eval (used by recovery contract).

### Legacy compaction (E1 baseline reference)
- **Selection**: `ss3dm_prior/meshsplatopt/compact_selector.py` — CSEF score (`build_score_table:147`): `1.25·topology_cost + 1.0·redundancy + 0.75·negative_free_space + 0.35·uncertainty − 1.25·positive_evidence − 1.10·explanation_debt − 0.65·boundary_risk`; modes incl. `area_smallest`, `csef_low_evidence[_boundary_protected]`, `csef_adaptive_policy`, `pareto_area_csef`, **`random_same_count`**. On clean checkpoints only `render_contribution = importance_score` is populated (`meshsplatopt_select_compaction_candidates.py:33`); sparse/normal/free-space/debt terms dormant.
- **Prune execution**: `ss3dm_prior/meshsplatopt/checkpoint_compaction.py::apply_compaction:137` — keep-mask tensor surgery + vertex GC + index remap. **NO optimization.**
- **Post-prune fine-tune (exists!)**: `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py` — continues train.py from compacted ckpt, topology frozen (`--densify_until_iter=load_iter --skip_restricted_delaunay --freeze_topology_updates`); presets `compact_render_only` / `compact_sparse_low_lambda` / `compact_sparse_decay`; optional teacher-render/LPIPS/anchor losses; post-hoc `topology_audit`. → E1's "legacy = prune WITHOUT fine-tune" row means the pruner alone; this script is the existing FT recipe.

### ELA / Phase-J teacher (M4)
- ELA core: `utils/evidence_lumigraph_adapter.py::adapt_frame:2037` — target-GT-free: warps k nearest train-view residuals (GT−render) into target via rendered-depth reprojection with depth-consistency confidence; `adapted = clamp(base + α·signal)`.
- Needs per support frame: `renders/ gt/ depths/(.npy) camera_index.json` under `<model>/<split>/<method>/`. Target needs only base render + depth → **arbitrary novel poses OK** (pattern: `IntegratedFrameLoader.set_target` in `scripts/car_model/benchmark_phasej_integrated_runtime.py:61,285`).
- Teacher bake + distill orchestration: `scripts/car_model/ecsr_run_phaseg_teacher_bake_recovery.py` (`_ela_train_command:200` bakes teacher renders via `meshsplatopt_apply_evidence_lumigraph_adapter.py`; `_train_command:304` runs train.py `--enable_teacher_render_loss --teacher_render_dir --lambda_teacher_render`).

### Evidence statistics (E1 importance inputs)
- Sparse geometric support: `utils/triangle_sparse_support.py::TriangleSparseSupportEstimator.compute:170` → per-triangle `support_count`, `plane_residual_mean/median`, `normal_angle_residual_deg`, `geometry_support_score_base`.
- Render residual evidence: `scripts/car_model/ecsr_build_surface_evidence_cache.py::build_cache:346` → per-face `residual_pixel_sum/view_mean/view_norm`, barycentric footprints (train views only).
- Sparse-depth sentinels: `utils/sparse_depth_sentinel_cache.py::build_sparse_depth_sentinel_cache:53` (rejects test split).
- Render importance: `importance_score`/`pixel_count`/`image_size` stored in ckpt.

### Geometry losses (M3 status)
- Existing in mesh optimizer: sparse-COLMAP depth loss, checkpoint depth/normal render anchors, vertex anchors, sparse-depth parent rollback (`utils/sparse_depth_parent_rollback.py:144`).
- **No mesh-space free-space (ray-carving) loss exists.** `ss3dm_prior/losses.py::free_space_violation_loss:277` is occupancy-space (SPCarNet autodecoder only). Renderer exposes depth/alpha/ids → both sanctioned E2 routes implementable.

## 2. Data

| Dataset | Path | Notes |
|---|---|---|
| Mip-NeRF360 full9 | `/data/peilincai/mesh_datasets/mipnerf360/{bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai}` | images + images_{2,4,8} + COLMAP sparse. Outdoor=first5 (train at images_4), indoor=last4 (images_2), `-r -1` |
| ETH3D | `/data/peilincai/mesh_datasets/eth3d[_colmap]/courtyard` | `[VERIFY-M0]` GT laser scan present on disk? (dev_drive_A candidate) |
| Tanks&Temples | `/data/peilincai/mesh_datasets/tanks_and_temples[_colmap]/{barn,truck}` | `[VERIFY-M0]` GT availability |
| Parking phone | `/data/peilincai/parking_phone_tiny_anonymized/` | domain-perfect, no GT mesh (dev_drive_A candidate) |
| SS3DM | **NOT ON DISK** (`/data2/peilincai/SS3DM_raw` absent) | schema doc: `docs/ss3dm_prior_data_schema.md` |
| MeshFleet car prior | `outputs/ss3dm_prior_car/meshfleet_car_cache_ext_v02/` | SPCarNet prior data, not scene data |
| toy_parking | TO BUILD (M1b) | Blender/procedural sanctioned |

## 3. Trained checkpoints (clean baseline, DEC-005)

`outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/<scene>/` — 9/9 scenes, iterations 26000 + 30000, `.pt` ≈ 0.7–0.95 GB each, dir total 23G. Reproduction matches paper (`repro_metrics_vs_paper_iter30000.csv`): garden 24.71/0.762/0.216, bonsai 28.38/0.879/0.290, kitchen 27.30/0.858/0.226.
- `[VERIFY-M0]` per-scene triangle counts (load `.pt`, read `_triangle_indices.shape[0]`); mid-training wandb reference ~625k tris (garden-family, iter 7k).
- `[VERIFY-M0]` end-to-end render+metrics runs green today on garden.

## 4. Costs

- Training: 30k iters/scene; wall-clock NOT in retained logs (0-byte logs, overwritten wandb summaries). `[VERIFY-M0]` estimate via timed short run.
- `[VERIFY-M0]` fine-tune cost/1k iters, render FPS at dev res, ELA teacher cost/view (`benchmark_phasej_integrated_runtime.py` reports render_ms/adapter_ms).

## 5. Compute & storage

- GPUs: 8× RTX 6000 Ada 48GB (driver 565.57.01, CUDA 12.7). Free-ish: **3**, 1, 2, 5. Busy: 0, 6, 7 (other users).
- Env: micromamba `mesh_splatting` (py3.10).
- Storage: `/data` 28T, **42G free (100%) — BELOW D6 FLOOR**; `/` 14T, 5.9T free ✓; `/dev/shm` FULL (banned anyway). GEMS output root → `/home/peilincai/gems_stage1/` (DEC-001). Repo `outputs/` holds 2.2T historical artifacts (cleanup = human decision).
