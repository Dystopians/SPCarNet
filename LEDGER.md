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
| DEC-006 | `dev_drive_A` = **ETH3D courtyard** (`/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`, 38 imgs, COLMAP + points3D.ply; GT laser scans at `mesh_datasets/eth3d/courtyard/courtyard/{scan_clean,dslr_scan_eval}/scan{1,2}.ply`). T&T barn/truck have NO GT on disk; parking_phone_tiny has no GT mesh (keep as optional qualitative extra). Caveat: llff every-8 split → ~5 test views; PROTOCOL must set split honestly (candidate: every-4). | Only available scene with GT geometry → enables g1/g4. SS3DM absent (DEC-003). | 2026-07-02 |
| DEC-008 | **Storage RESOLVED by human decision (2026-07-02)**: user approved deleting `outputs/carnet/meshsplatopt/ecsr_phase_{s,f,r}` (~1.1 TB, decommissioned v1xx–v3xx axis). Deletion manifest: `/home/peilincai/gems_stage1/deletion_manifest_20260702.txt`. After deletion, GEMS durable output root moves to **`/data/peilincai/gems_stage1/`** (updates DEC-001); preflight still mandatory before every run. `official_clean30k` untouched. | User selected "Delete ecsr_phase_s/f/r" via AskUserQuestion. | 2026-07-02 |
| DEC-007 | **STORAGE CRISIS (superseded by DEC-008).** Verified: `/` volume has a hard 100 GiB **user quota**, currently ~1 GiB headroom (torch.save died mid-write; dd probe stopped at 941 MiB). `/data` = 41 GB free, no user quota, below the 50 GB D6 floor. No other writable volume. Interim interpretation of D6 (documented, not a waiver): KB–MB artifacts (code, docs, metrics.json, panels) may be written; **GB-scale artifacts (checkpoints, render sets, teacher caches, toy dataset training) are BLOCKED** until human chooses: (a) approve retention-policy cleanup of decommissioned v1xx–v3xx dirs in `outputs/` (2.2 TB, reclaim ≥300 GB), (b) raise home quota, or (c) explicitly waive the 50 GB floor for /data with a tight per-run budget. | D6 + honesty; deleting user research artifacts autonomously is forbidden. | 2026-07-02 |

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
| M0 Reproduce & Audit (AT0) | **DONE** | 100 | Eval reproduced exactly (garden 24.7120/0.7618/0.2163); tri counts censused; preflight tool demonstrated (caught 2 violations); FT cost measured 19.3 it/s; storage blocker escalated (DEC-007) |
| M1 Protocol & Harness (AT1) | IN PROGRESS | 90 | PROTOCOL v1.1.0 + run_eval.py + bootstrap tool + audit all GREEN on garden; toy_parking built, clean30k training running; REMAINS: toy eval proving g1/g2/g4/d1/d2 on clean baseline (M1b acceptance) |
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

### GOAL #003 — M1: PROTOCOL v1.1.0 + eval harness, validated + adversarially reviewed [M1] — 2026-07-02 — DONE
- Infra goal. Built via 8-agent workflow (5 implementers, integrator, 2 adversarial reviewers) + 1 fixer agent.
- Artifacts: `PROTOCOL.md` (v1.0.0→v1.1.0 same-day pre-first-row amendments, changelog inside), `run_eval.py` (single mouth), `tools/gems/{scenes,eval_context,panels,geometry_metrics,downstream_metrics,paired_bootstrap,test_paired_bootstrap}.py`, `tools/audit_test_path.py`.
- Validation (durable): `/data/peilincai/gems_stage1/eval/garden_clean30k_v2/metrics.json` — PSNR 24.7120037 (bit-identical to legacy mouth), SSIM 0.76179, LPIPS 0.21626; g1=0.0045972 (99,626 SfM samples, 458 violations); g3=1941 floater comps / 0.13754% tris (60 support views); cost: 11.57M tris, 942MB, 6.9GB VRAM, 32.2 FPS; full eval 81 s. Bootstrap self-test: coverage 0.948, n=2M×10k resamples in 214 s / 1.59GB RAM, chunked==full-matrix bit-identical. Audit: GREEN (3443 modules, 4166 traced reads); hardened after review caught a real evasion hole (slash-form/bare-name car_model imports).
- Reviews: protocol-compliance + numerical-correctness. All blocking findings fixed & revalidated (chunked bootstrap; g4 ROI gate; surface=all-triangles per opacity floor; audit blocklist; voxel sampling guard). Deferred (documented, benign): per-view LPIPS construction (slow, bit-exact); g1 silhouette-edge bias (deterministic, model-invariant, in PROTOCOL).
- VERDICT: PASS.

### GOAL #004 — M1b: toy_parking dataset + clean baseline [M1] — 2026-07-02 — IN PROGRESS
- Infra goal. `tools/gems/build_toy_parking.py`: procedural GT scene (meters): 34×34m mottled ground + lot markings, 2 cars, 10cm pole, 8-post fence + rails, textureless wall, curb; GT mesh 54,513 V / 105,982 F; 90 views @1000×750 (72/18 file split), exact GT depth per view; COLMAP-text export verified against reader; element coverage ≥8 views PASS (min: pole 34).
- Renderer facts discovered (recorded in dataset_manifest.json): tile compositor sorts by triangle-center depth, NO backface culling → builder bakes per-view backface masks; screen-space culls (>1600px, <1px, inradius<1px) → tessellation ≤0.5m. sigma=1e-4 (log-stored) for hard edges; vertex_weight stored pre-sigmoid; render opacity = 0.999+0.001·sigmoid (floor pinned by load_parameters).
- Ingestion smoke 300 iters PASS (~93 it/s). Clean30k training detached on GPU 1 (PID 4050215, wandb run gems_toy_parking_clean30k, online). Toy ROI frozen in scenes.py; z_band[0] corrected 0.1→0.0 (true ground) pre-first-row.
- REMAINS: training completes → `run_eval.py` on toy clean ckpt proving g1(GT-depth)/g2/g4/d1/d2 (= M1b acceptance).

### GOAL #002 — M0 completion: verified reproduction, census, costs, storage [M0] — 2026-07-02 — DONE
- Infra goal. Evidence (durable): `/home/peilincai/gems_stage1/{m0_triangle_census.txt, logs_m0_garden_e2e.log, m0_repro/garden/results.json}`.
- **Eval path GREEN & exact**: render.py + metrics.py on copied garden ckpt (GPU 3) → PSNR 24.7120 / SSIM 0.76179 / LPIPS 0.21626 = recorded baseline (24.71/0.762/0.216). Render: 24 test views in 34 s incl. PNG I/O (×4 supersampling).
- **Trainer GREEN**: 150-iter topology-frozen fine-tune from garden@30000: **19.3 it/s** (11.57M tris, images_4). → 10k-iter fine-tune ≈ 9 min/scene. Full-train 30k ESTIMATE ≈ 40–80 min/scene (rate varies over training; not directly measured — logs empty; toy scene will be timed in M1b).
- **Triangle census (30000 = 26000, topology frozen)**: bicycle 9.42M, flowers 9.65M, garden 11.57M, stump 9.28M, treehill 9.53M, room 11.17M, counter 9.85M, kitchen 9.72M, bonsai 10.83M; verts 2.45–3.61M; .pt 743–988 MB; SH deg 3. → B=50% ≈ 4.6–5.8M tris, B=25% ≈ 2.3–2.9M.
- `tools/storage_preflight.py` written + demonstrated: PASS on quota-free target, FAIL on `outputs/` (41 GB < 50). Then the FT-probe's final save **died on the home quota** (torch.save mid-write) — which is how DEC-007's quota was discovered. Crash root cause: 100 GiB uid quota on `/`; fix: deleted my ckpt copies (~2 GB), rerouted all GB-scale work to pending DEC-007.
- Env correction: mesh_splatting env is Python **3.11** (not 3.10).
- VERDICT: PASS (AT0 satisfied; training demonstrated via resume-probe per DEC-005 justification).

### GOAL #001 — Bootstrap [M0] — 2026-07-02 — DONE
- Infra goal (no hypothesis). Created LEDGER.md, ASSET_MAP.md (pre-verified draft), internalized §2 context via 3 parallel code surveys (train/render/eval path; compaction+ELA+evidence; data/checkpoints/storage).
- Storage preflight (manual): /data 42G free ⚠ BELOW 50G FLOOR; / 5.9T free ✓; /dev/shm full ✗. → DEC-001.
- SS3DM verified ABSENT → DEC-003.
- Next: GOAL #002 — complete M0/AT0: triangle counts from .pt, executed end-to-end render+metrics on garden clean checkpoint, `tools/storage_preflight.py`, wall-clock estimate, dev_drive_A candidate verification (ETH3D GT availability), finalize ASSET_MAP.
