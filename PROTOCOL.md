# GEMS STAGE ONE — PROTOCOL.md

**Version 1.1.0** (semantic versioning; metric-definition changes bump MAJOR-or-MINOR and force re-run of all affected rows) · 2026-07-02

**Changelog 1.1.0** (same-day amendment after adversarial review of the harness, BEFORE any affected rows existed — zero re-runs required; garden g1/g3/render rows from 1.0.0 are definitionally unchanged):
- §4.3 "opaque surface" for g4/d1/d2 redefined = **all checkpoint triangles**. Reason: `TriangleModel` pins render-time opacity to ≥ 0.999 for every triangle (opacity floor, `triangle_model.py:347`), so the renderer draws all triangles near-opaque and a sigmoid(logit)≥0.5 mask (which selects ~0.26% on garden) does not describe the rendered surface.
- §4.1 documents the 8-bit quantization convention (parity with legacy `metrics.py`).
- §4.3 documents g1's depth-map convention and silhouette-edge caveat; g4 skip-rule when a scan-GT ROI is unfrozen; g4 pairing rule (only gt→recon is paired).
- §5 documents memory-bounded (chunked) bootstrap implementation; statistics unchanged.

**Changelog 1.1.1** (2026-07-02, GT-asset correction): ETH3D courtyard laser scans must be loaded through the `scan_alignment.mlp` per-scan rigid transforms (frozen in `scenes.py` as `gt.scan_transforms`); raw vertices are ~1.2 m misaligned from the camera frame (verified via COLMAP sparse points: median sparse→scan 0.036 m transformed vs 1.19 m raw). The courtyard ROI is the transformed-scan AABB + 0.3 m. The single g4 row computed with raw scans (`courtyard_clean30k_v2`) is VOID and superseded; no other rows touched scan GT.
Constitution for all reported numbers. The single evaluation mouth is `run_eval.py` (D5). Numbers not produced by it do not exist (exception: LEGACY-labeled numbers from `metrics.py` produced before this protocol, used only for cross-checks, never in result tables).

## 0. Frozen claim under test

As in `docs/GEMS_Stage1_Prompt.md` §0, unmodified. Demotions (if any) will be recorded here with links to the killing evidence. Current demotions: **none**.

## 1. Datasets (dev only; holdout untouched)

| Scene | Source | Ingestion config (FROZEN) | Split | Test views | GT assets for metrics |
|---|---|---|---|---|---|
| `toy_parking` | Procedural, built in M1b (spec §1.1) | COLMAP-format dir, `--images images -r -1`, ~1000×750 px | file split: every 5th view → test | ~12–24 | GT mesh (.obj), per-view GT depth (.npy), exact poses |
| `dev_real_A` = garden | `/data/peilincai/mesh_datasets/mipnerf360/garden` | `--images images_4 -r -1 --eval` (= clean baseline config) | llff every-8 | 24 | COLMAP sparse points (metric-only) |
| `dev_drive_A` = courtyard | `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard` | `--images images -r 8 --eval` (= stageR precedent) | llff every-8 | 5 | ETH3D laser scans `scan{1,2}.ply` (metric-only), COLMAP sparse |

- Courtyard has only 5 test views: rendering CIs there are reported but **underpowered by design**; render-quality pass/fail criteria (E1/E3) bind on garden + toy_parking. Courtyard's role is geometry (g1–g4) and downstream (d1–d2), where samples number in the thousands.
- **toy_parking spec (§1.1):** meters as units; ground plane ≥ 30×30 m; ≥2 parked vehicle meshes; ≥1 thin structure (pole/fence, ≤10 cm diameter members); ≥1 textureless wall; 60–120 posed views on ring/arc trajectories at 1.4–2.0 m height; renders at ~1000×750; trains < 30 min at this resolution; GT mesh + GT depth per view exported at build time. Procedural/Blender generation sanctioned here ONLY.

## 2. Budgets

B ∈ {50%, 25%} of the scene's clean triangle count (clean counts: see ASSET_MAP census; toy/courtyard counts fixed when their clean baselines exist). Optional stretch after E1: B=100% reallocation-only. A model "at budget B" must satisfy `N_triangles ≤ B · N_clean` (hard).

## 3. Training / model-selection rules (D4, D7)

- One frozen config per experiment row across ALL dev scenes (D7). Per-scene numbers are analysis only.
- Seeds: `--seed 0` everywhere. Fine-tunes: fixed iteration count; **the final iterate is THE model** — no checkpoint selection, no early stopping, no test-view influence anywhere (D4).
- Every run records: git commit, full command line, config hash = sha256 of the exact train command string (reported by the pipeline runner), durable output path, and a `tools/storage_preflight.py` result.

## 4. The single mouth: `run_eval.py`

`python run_eval.py --checkpoint <point_cloud_state_dict.pt> --scene <toy_parking|garden|courtyard> --out <dir>` → writes `<out>/metrics.json` + panels. Scene registry (paths, split, GT assets, ROI, units) lives in `tools/gems/scenes.py` and is part of this protocol.

Renders test views in memory via `triangle_renderer.render` with the training-time settings (supersampling ×4, background per config). Consumes ONLY: checkpoint, camera poses, eval images, and the declared metric-only GT assets (§1). Importing any ELA/teacher/selector module is a protocol violation — enforced by `tools/audit_test_path.py`.

### 4.1 Rendering metrics (per test view, then mean)
Renders and GT are quantized to 8-bit exactly as `torchvision.utils.save_image` would (mul 255, add 0.5, clamp, uint8, /255) before metric computation — bit-for-bit parity with the legacy render.py→PNG→metrics.py path. Applied symmetrically to every model; cannot leak GT.
- PSNR: `utils/image_utils.psnr` (20·log10(1/√MSE), RGB in [0,1]).
- SSIM: `utils/loss_utils.ssim` (Gaussian 11×11, σ=1.5).
- LPIPS: `lpipsPyTorch`, net `vgg`.
Per-view values are stored in `metrics.json` (they feed the paired bootstrap).

### 4.2 Cost metrics
- `n_triangles` = `_triangle_indices.shape[0]`; `n_vertices`; `disk_mb` = checkpoint file size.
- `peak_vram_mb` = `torch.cuda.max_memory_allocated()` over the render pass (reset before).
- `render_fps` = pure forward renders of all test views, no image I/O; 3 warmup renders excluded; median over 3 repeats of the full pass.
- `finetune_wallclock_min`: reported by the pipeline runner from its own logs (not measured by run_eval).

### 4.3 Geometry metrics
Reconstructed surface for g4/d1/d2 = **all triangles of the checkpoint whose vertices are finite** (the renderer draws every triangle at opacity ≥ 0.999 due to the model's opacity floor; there is no meaningful translucency in this representation — see changelog 1.1.0). Faces touching non-finite vertices — a rare training pathology the rasterizer silently culls (observed: 13/6.59M on toy_parking clean) — are excluded from g3/g4/d1/d2 surfaces and reported as `n_nonfinite_faces_excluded`.
Depth convention: "rendered median depth" = the renderer's `surf_depth` product (4× supersampled median depth, area-downsampled to camera resolution with background subpixels contributing 0). At silhouette edges this dilutes depth toward 0 and can inflate g1 slightly; the bias is deterministic and identical across models, so paired comparisons are valid. The alpha ≥ 0.5 gate is retained as written (near-vacuous under the opacity floor).
- **g1 free-space violation rate**: sample pairs (camera c, 3D point p with depth d_p in c): toy → all test-view GT-depth pixels subsampled to ≤20k/view; garden/courtyard → COLMAP sparse points visible in each train camera (all, capped 100k total, seed 0). Violation iff rendered **median depth** at p's pixel < 0.95·d_p AND rendered alpha ≥ 0.5. g1 = violations / samples. Lower is better.
- **g2 held-out depth L1** (toy only): mean |median-rendered depth − GT depth| over pixels with valid GT and rendered alpha ≥ 0.5, averaged over test views (meters).
- **g3 floater score**: connected components of the triangle adjacency graph (triangles sharing a vertex). Per-triangle train-support = number of TRAIN views in which the triangle contributes ≥1 pixel (from a train-view render pass reading `rend_ids`; capped at 60 evenly-spaced train views for cost). A component is a **floater** iff every member triangle has support ≤ 1 AND component size < 10,000 triangles. Report: floater component count and floater triangle fraction.
- **g4 Chamfer-L1 / F-score@τ** (toy: vs GT mesh; courtyard: vs laser-scan point cloud): sample 1M points area-weighted on the opaque reconstructed triangles; toy GT: 1M points on GT mesh; courtyard GT: scan points (subsampled 2M, seed 0). Chamfer-L1 = mean bidirectional nearest-neighbor distance; F@τ with τ = 0.05 m. Scene ROI (frozen axis-aligned box in `scenes.py`) crops both clouds; for courtyard, the recon→scan direction only counts recon points inside the scan cloud's AABB expanded by 0.3 m (avoids punishing sky/unscanned areas) — the exact ROI is frozen in `scenes.py` at first courtyard eval and never edited (edits = MAJOR bump). **When the GT is a scan and the scene ROI is not yet frozen, g4 is reported as `skipped` — never as a protocol number.** Pairing rule for CIs: only the gt→recon per-sample distances are a pairable bootstrap unit across models (GT sample points are model-independent); recon→gt is reported as an unpaired summary.

### 4.4 Downstream proxy v0 (toy_parking + courtyard)
- **d1 occupancy agreement**: voxelize opaque recon triangles and GT surface at **0.10 m** into the scene ROI (ground-relevant z-band frozen in `scenes.py`). GT-occupied voxel = contains GT surface; recon likewise. Report `false_free_rate` = P(recon free | GT occupied) (safety-critical) and `false_occupied_rate` = P(recon occupied | GT free) separately.
- **d2 collision-verdict agreement**: 200 trajectories (seed 0): straight lines and constant-curvature arcs at ground level (ground = `z_band[0]` of the scene ROI; vehicle band = [z_band[0]+0.1, z_band[0]+1.5] m), vehicle footprint 4.5×1.8 m, sampled within the ROI. Voxelization sampling density must satisfy spacing ≤ voxel/2 for every triangle (implementation must guard, not silently cap).
- d1/d2 require a gravity-aligned `z_band`. Courtyard's scan frame is not verified axis-aligned: its ROI is frozen for g4 (scan AABB + 0.3 m, see `scenes.py`), but `z_band=None` gates d1/d2 until a documented up-axis derivation freezes it (M5). Toy_parking is built z-up; its `z_band` is frozen. Verdict = swept footprint intersects occupied voxels (from d1 grids). Report overall agreement rate and `unsafe_disagreement_rate` = P(recon says free | GT says collision).

### 4.5 Panels
Per eval: for 6 evenly-spaced test views (or all if fewer): RGB render | GT | ×5 error heatmap | median-depth map; plus one floater overlay (triangles of floater components in red over a representative view). PNG, written under `<out>/panels/`.

## 5. Statistics

`tools/gems/paired_bootstrap.py` — the ONLY CI implementation. Paired bootstrap on per-unit differences (units: test views for rendering; sampled segments/points/voxels/trajectories for g/d metrics), **10,000 resamples**, numpy seed 0, percentile 95% CI of the mean difference. The implementation is memory-bounded (chunked resampling with a fixed deterministic stream per seed) so protocol-scale unit counts (10^6–10^7) run within RAM; chunking is an implementation detail, not a statistic change. Self-test: synthetic paired data with known effect must recover nominal coverage ±2% (runs in CI of the tool itself).
Reporting language: "improves/reduces" ONLY when the 95% CI excludes 0 AND the effect clears the D3 floor. Otherwise: "DIAGNOSTIC". Oracle analyses are labeled "ORACLE — NOT A RESULT".

## 6. Rules restated (binding)

- **Effect-size floors (D3)**: rendering ≥ +0.10 dB PSNR or ≥ 0.005 LPIPS; compaction ≥ 20% triangle reduction at iso-quality (ΔPSNR ≥ −0.10 dB AND ΔLPIPS ≤ +0.005 vs clean); geometry ≥ 20% relative on the pre-registered metric. Below floor = DIAGNOSTIC; at most ONE follow-up per mechanism.
- **Sunset rule**: 3 consecutive below-floor results on one mechanism → tombstone in LEDGER, thread closed.
- **Iteration budget**: ≤ 2 tuning-flavored /goals per mechanism (schedules are mechanisms; thresholds are not experiments at all, D1).
- **Storage (D6 + LEDGER DEC-007)**: `tools/storage_preflight.py` before every run; 50 GB floor on target volumes; GB-scale artifacts blocked until the DEC-007 human decision; retention = latest + milestone checkpoints only; no >1h jobs or checkpoints in `/dev/shm`; long runs resumable.
- **Purity (D4)**: `tools/audit_test_path.py` must be green in every /goal that reports numbers.
- **One mouth (D5)**: this file + `run_eval.py`. A second metric path is a violation.
