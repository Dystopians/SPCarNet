# GEMS STAGE ONE — LEDGER

Persistent memory for the GEMS Stage One loop (`docs/GEMS_Stage1_Prompt.md`, v1.0, 2026-07-02).
At every /goal: read this file top-to-bottom, then PROTOCOL.md (once it exists), then act.
Never trust chat memory over this file.

---

## FROZEN DECISIONS (change only with explicit justification logged here)

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-001 | GEMS durable output root = `/data/peilincai/gems_stage1/` **only if** /data freed above 50 GB; otherwise `/home/peilincai/gems_stage1/` (root volume, 5.9T free). Currently: **`/home/peilincai/gems_stage1/`** | `/data` is 100% full (42 GB free) — below the D6 50 GB floor. `/dev/shm` also full (never use per D6). | 2026-07-02 |
| DEC-002 | `dev_real_A` = Mip-NeRF360 **garden**, dev resolution = `images_4`, `--resolution -1` (matches existing clean checkpoint config) | Clean 30k checkpoint + reproduction metrics already exist at this config (PSNR 24.71 / SSIM 0.762 / LPIPS 0.216, iter 30000). Avoids retraining; fine-tunes are affordable. | 2026-07-02 |
| DEC-003 | SS3DM town data is **NOT on disk** (`/data2/peilincai/SS3DM_raw` gone). `dev_drive_A` must use a fallback; candidates: ETH3D courtyard (GT laser scan? verify), T&T barn/truck (COLMAP present; GT?), `parking_phone_tiny_anonymized` (domain-perfect, no GT mesh). Final pick in M0. | Per prompt §2 "facts M0 must verify" — verified absent. | 2026-07-02 |
| DEC-004 | Python env = micromamba `mesh_splatting` (`/home/peilincai/micromamba/envs/mesh_splatting`, py3.10). GPUs: prefer 3, then 1/2/5 (0/6/7 busy with other users' jobs). | Environment survey. | 2026-07-02 |
| DEC-005 | Clean baseline = existing `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/<scene>/` checkpoints (iteration_30000; 26000 also retained). Do NOT retrain full9. | 9/9 scenes trained, metrics recorded, match paper. Retraining wastes days and /data has no space. | 2026-07-02 |

## STANDING CONSTRAINTS (from Prime Directives — restated for quick check)

- D1: independent variable ∈ {triangle set/topology, triangle params, losses, supervision data, budget schedule}. NO selector/gate/threshold experiments (1 protocol-regression gate per module allowed).
- D3 floors: rendering +0.10 dB PSNR or 0.005 LPIPS; compaction ≥20% tri reduction at iso-quality (ΔPSNR ≥ −0.10, ΔLPIPS ≤ +0.005); geometry ≥20% relative. Below floor = DIAGNOSTIC.
- D4: run `tools/audit_test_path.py` in every /goal that reports numbers (exists after M1).
- D5: all numbers from `run_eval.py` (exists after M1). Until then, only legacy `metrics.py` numbers, labeled LEGACY.
- D6: preflight `df -h` before every run; abort if target volume < 50 GB. No >1h jobs in `/dev/shm`. Resumable long runs. Retention: latest + milestone checkpoints only.
- D8: Stage One scope = {toy_parking, dev_real_A, dev_drive_A} × B ∈ {50%, 25%}. No full9 sweeps, no holdout.
- Sunset rule: 3 consecutive below-floor results on one mechanism → tombstone here, close thread.

---

## MILESTONE STATUS BOARD

| Milestone | Status | % | Blockers / notes |
|-----------|--------|---|------------------|
| M0 Reproduce & Audit (AT0) | IN PROGRESS | 60 | Asset map drafted from code reading; still need: executed end-to-end eval on garden, triangle counts from .pt, storage preflight tool, legacy-compaction FT recipe documented (done in ASSET_MAP), wall-clock estimate |
| M1 Protocol & Harness (AT1) | NOT STARTED | 0 | PROTOCOL.md, run_eval.py, bootstrap tool, audit_test_path.py, toy_parking (M1b) |
| M2 Budget Engine (E1) | NOT STARTED | 0 | depends M1 |
| M3 Geometry Objectives (E2) | NOT STARTED | 0 | depends M2; renderer already exposes expected/median depth, alpha, normals, tri-ids |
| M4 Teacher Distillation (E3) | NOT STARTED | 0 | depends M2; teacher-render-loss hook already exists in train.py |
| M5 Downstream Proxy & Efficiency (AT5) | NOT STARTED | 0 | depends M2 |
| M6 Integration & Report (AT6) | NOT STARTED | 0 | depends all |

## ITERATION BUDGETS & SUNSET WATCH

| Mechanism | Tuning-flavored goals used (max 2) | Consecutive below-floor (sunset at 3) | Status |
|-----------|-----------------------------------|----------------------------------------|--------|
| (none yet) | — | — | — |

---

## KEY VERIFIED FACTS (asset survey, 2026-07-02 — details in ASSET_MAP.md)

1. **Post-prune fine-tuning already exists**: `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py` = topology-frozen continued training (`--load_iteration`, `--freeze_topology_updates --skip_restricted_delaunay`, `--densify_until_iter = load_iteration`). The legacy *pruner* itself (`ss3dm_prior/meshsplatopt/checkpoint_compaction.py`) does pure tensor surgery, NO optimization. So E1's "prune WITHOUT fine-tune ≈ legacy behavior" row = pruner alone; the recovery script is the existing FT recipe.
2. **Importance machinery exists**: CSEF selector `ss3dm_prior/meshsplatopt/compact_selector.py` (modes incl. `random_same_count` — usable for E1's random-prune row). On clean checkpoints only `render_contribution`=importance is populated; sparse/normal/free-space/debt terms are dormant → E1 importance definition should populate them from `utils/triangle_sparse_support.py` + `ecsr_build_surface_evidence_cache.py`.
3. **Teacher distillation hook exists**: `train.py --enable_teacher_render_loss --teacher_render_dir`; baking of ELA/Phase-J teacher renders via `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`; orchestration example `ecsr_run_phaseg_teacher_bake_recovery.py`. ELA (`utils/evidence_lumigraph_adapter.py::adapt_frame`) can render arbitrary novel poses from train-only evidence (needs base render+depth at pose + train residual/depth caches) — pseudo-view factory is plumbing, not research.
4. **No mesh-space free-space loss exists** (only occupancy-space `free_space_violation_loss` in ss3dm_prior, unused by mesh optimizer). Renderer returns `expected_depth`, `surf_depth` (median), `rend_alpha`, `rend_normal`, `rend_ids` per pixel → both sanctioned M3 routes are implementable.
5. **Checkpoint format**: `<model>/point_cloud/iteration_N/point_cloud_state_dict.pt` with `triangles_points`, `_triangle_indices` (count implicit in shape), `vertex_weight`, `sigma`, per-vertex SH (`features_dc/rest`), `importance_score`. Eval mouth today: render.py → `test/ours_N/{renders,gt}` → metrics.py (PSNR/SSIM/LPIPS-vgg) → results.json.
6. **Data**: all 9 mipnerf360 scenes at `/data/peilincai/mesh_datasets/mipnerf360/`; clean full9 checkpoints at `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/` (23G). SS3DM ABSENT. ETH3D courtyard + T&T barn/truck + parking_phone_tiny available as dev_drive_A fallbacks.
7. **Settled post-mortems** (do not re-litigate): selector/arbitration dead (oracle headroom +0.0086 dB); static baking dead (4.76% gap capture, cos 0.21); Phase-J teacher = +1.33 dB upper bound; honest gates converge to no-op; micro-delta trap → D3.

## OPEN RISKS

- R1: `/data` at 100% (42 GB free) — every run writing under the repo's `outputs/` risks death-by-quota (v168 pattern). Mitigation: DEC-001 output root on `/` volume; preflight tool in M0. 2.2T of `outputs/` is historical — cleanup needs human approval, flag in report.
- R2: SS3DM absent → `dev_drive_A` degraded; g4 (Chamfer vs GT mesh) may only be possible on toy_parking (+ ETH3D courtyard if its laser GT is on disk). PROTOCOL must scope g4 honestly.
- R3: per-scene training wall-clock unknown (logs empty) — estimate in M0 from a short timed run before committing to fine-tune budgets.
- R4: toy_parking must be ingested via Blender/transforms_train.json path or synthetic COLMAP; Blender reader forces white bg / random init cloud if no points3d.ply — supply a GT-derived init cloud to avoid junk init.

---

## GOAL LOG

### GOAL #001 — Bootstrap [M0] — 2026-07-02 — DONE
- Infra goal (no hypothesis). Created LEDGER.md, ASSET_MAP.md (pre-verified draft), internalized §2 context via 3 parallel code surveys (train/render/eval path; compaction+ELA+evidence; data/checkpoints/storage).
- Storage preflight (manual): /data 42G free ⚠ BELOW 50G FLOOR; / 5.9T free ✓; /dev/shm full ✗. → DEC-001.
- SS3DM verified ABSENT → DEC-003.
- Next: GOAL #002 — complete M0/AT0: triangle counts from .pt, executed end-to-end render+metrics on garden clean checkpoint, `tools/storage_preflight.py`, wall-clock estimate, dev_drive_A candidate verification (ETH3D GT availability), finalize ASSET_MAP.
