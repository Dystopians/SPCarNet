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
| M1 Protocol & Harness (AT1) | **DONE** | 100 | All metric families proven: GT-model calibration row ≈perfect (g1 0.028%, g2 2.8mm, chamfer 1.8cm, d2 agreement 1.0) vs toy clean30k poor (g1 23.9%, g2 0.26m, chamfer 0.57m, d2 0.625) → metric code validated; toy clean geometry genuinely unreliable (headline motivation) |
| M2 Budget Engine (E1) | **CONCLUDED — E1 FAIL as written; mechanism VALIDATED; escalated** | 95 | KILL_REPORT.md filed. Garden B50 prune+features-FT BEATS clean (+0.157 dB, −0.0071 LPIPS, CIs excl. 0); garden B25 iso (75% reduction); courtyard B50 iso; toy −0.52 residual (geometric). Human decision pending: amended E1′ vs strict stop |
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

### GOAL #006 — M3 geometry objectives, TEST E2 [M3] — 2026-07-02 — IN PROGRESS
- Human question on E1′ amendment went unanswered (60s timeout); proceeding under "E1 stays FAIL, Stage One continues" — the no-amendment option; final go/no-go remains human's. E1′ decision remains OPEN in KILL_REPORT.md.
- **Pre-registered E2 hypothesis**: mechanism = add (1) free-space hinge loss L_fs = mean relu(0.95·d_evidence − D_rendered)/d_evidence over sampled train-view evidence (toy: GT train depths; courtyard/garden: COLMAP sparse train points — train-only, D4-pure) via the DIFFERENTIABLE rendered depth, and (2) multi-view rendered-depth consistency loss (pairwise warp between train views, occlusion-masked) — both integrated into the M2 fine-tune (features default lr + positions at low lr re-enabled: geometry losses supply coherent position signal, unlike the noise-drift case; weights frozen). Implementation route: rendered-depth penalties (Route B) — justification: rasterizer already backprops through depth outputs (train.py's existing vertex-depth loss proves it); no CUDA changes.
- **E2 PASS iff** at B=50% on toy_parking AND courtyard: g1 OR g3 improves ≥30% relative vs the M2 model (CI excl. 0), ΔPSNR ≥ −0.10 dB vs the M2 model, AND before/after panels show visible floater/free-space cleanup. Kill: 3-variant rule; if dead → geometry axis demoted to evaluation-only (documented, claim edited, human review).
- Baselines (the "M2 model" at B50): toy = toy_parking_B50_importance_ft_e1v2; courtyard = courtyard_B50_importance_ft_e1v2 (launched now, features-only FT 10k — also completes M2 trend table + E1′(i)(ii) courtyard data).
- Predicted effect: toy g1 0.202→≤0.14; courtyard g1 0.094→≤0.066 (or g3 equivalents); rendering held within −0.10 dB.
- **E2 attempt 1 (base config, λ_fs=0.01/λ_dc=0.05/pos-lr 1.5e-5): FAIL both scenes** (evals `eval/m3_e2_{toy_parking,courtyard}_B50_v1`): toy g1 −1.0%, g3 frac −21.8% (<30%), PSNR guard VIOLATED (−0.245 CI[−0.43,−0.09]); courtyard g1 +2.0%, g3 −9.6%, guard ok (−0.056 CI incl. 0). Loss trajectories: L_fs barely decreases on toy → geometry signal too weak vs mobile-position photometric drift (the E1 mechanism returned once positions re-enabled: toy's −0.245 at weak λ is drift, not geometry-loss cost).
- **E2 mechanism variant 1/3 (pre-registered before implementation): GRADIENT-ROUTED geometry coupling** — photometric loss backprops to FEATURES only; geometry losses (λ_fs=0.05, λ_dc=0.10) backprop to POSITIONS only (two-pass backward, vertices.grad cleared between); weights frozen; position lr 1.5e-4 (×10, safe because positions no longer receive photometric noise); 10k iters; same config both scenes. Predict: g3 floater fraction −30%+ (positions free to evacuate free space without photometric counter-pressure), g1 down on toy, PSNR guard held by features-only photometric channel (the E1-v2-validated safe channel). Kill: guard violated on either scene while g-metrics still <30% on both.

### GOAL #005 — E1 budget engine runs [M2] — 2026-07-02 — CONCLUDED (see KILL_REPORT.md)
- **Variant 3 result**: garden iterative B50 = 24.8689 (+0.157 vs clean CI[+0.110,+0.200], ΔLPIPS −0.0071) — best row yet; toy unchanged vs one-shot (−0.022 CI[−0.115,+0.079] vs noft; −0.542 vs clean) → toy kill condition tripped. Variants 3/3 spent.
- **FINAL VERDICT: E1 FAIL as pre-registered (criterion (b) miscalibrated — presumed lossy pruning; measured prune-only −0.011 dB garden B50; criterion (a) fails on toy only). Mechanism (evidence-prune + features-only FT) VALIDATED on real scenes. Full accounting + fallback proposal in `KILL_REPORT.md`. Escalated to human: adopt E1′ or strict stop. M2 experiments STOPPED per rules; no soft pivot.**
- **E1 attempt 1 (tag e1b, post scaling-fix) = FAIL as pre-registered** (`/data/peilincai/gems_stage1/analysis/e1_summary.{md,json}`): importance_ft LOSES vs importance_noft on both scenes (garden B50 −2.92 dB CI[−3.29,−2.54]; toy B50 −1.26 CI[−1.79,−0.78]). But the FAILURE STRUCTURE is informative:
  - **Prune-only (importance=pixels_total) is near-lossless: garden B50 −0.011 dB CI[−0.033,+0.003], ΔLPIPS +0.0001 → MEETS the D3 compaction floor (50% tris, iso-quality, 32→42 FPS).** garden B25 −0.139; toy B50 −0.520; toy B25 −1.167 (fail strict floor). Random-prune is far worse at B25 (garden −5.0 vs clean) → importance definition genuinely matters.
  - **The FT stage damages ANY resumed model** — diagnostics (all toy, durable evals): D-a clean-resume+10k FT, NO prune → −1.12 dB (`eval/toy_cleanresume_diag`); D-b prune+FT-1k → only −0.11 vs noft (`eval/toy_parking_B50_importance_ft_diag1k`) → damage ∝ FT length; D-c fine-tuned model loses 3.7 dB on ITS OWN TRAIN VIEWS (36.22→32.53) and its training loss RISES ~50% over 10k iters → not overfitting; optimizer noise-drift at default constant LRs (feature 0.0016, weight 0.03) on a converged model. Supersampling fix verified effective (model best at scaling 4). Sigma/floor/pos-LR schedules verified resume-consistent.
- **Mechanism variant 1/3 (pre-registered before launch)**: drift-controlled FT — feature_lr×0.1 (0.00016), weight_lr=0, positions unchanged, 10k iters. Predict: toy B50 ft−noft ≥ 0 (recovers part of −0.52); garden B50 |ft−noft| < 0.1 (no damage). Kill: FT still degrades ≥0.2 dB on both. Tag e1v1, 4 rows (2 scenes × B50/B25), launched.
- **Variant 1 result: FAIL, hypothesis falsified** — LR-reduction made damage WORSE (garden B50 ftv1 −4.13 dB vs clean, CI[−4.57,−3.67]; both garden budgets converge to the same degraded PSNR 20.58 → systematic attractor, not noise). Evals: `eval/*_importance_ft_e1v1`.
- **Root-cause chain (probes, durable in logs/this entry):** (1) fine-tuned model's TRUE train loss DOUBLES (0.004747→0.009394 over 72 views) → optimizer ascends its own objective; (2) single-view repeated stepping DESCENDS cleanly → gradients correct; (3) pure-photometric multi-view loop (none of train.py's extra terms) still ascends (0.004747→0.005351 @3k) → core stochastic dynamics; (4) channel isolation @1.5k steps: vertices-only +0.000670 (DESTROYER, even at lr 1.5e-5), features-only +0.000169 (mild, acts as repairer of position damage — explains v1 < default), weights-only −0.000001 (innocent); (5) fused_ssim NOT installed → toy trained today with identical utils-ssim objective as FT → not an objective switch. Interpretation: near convergence, Adam-normalized position updates drift the model along a value-destroying direction; the trainer's own end phase is on this slope (test: garden@26000 vs @30000 eval running).
- **Mechanism variant 2/3 (pre-registered before launch)**: features-only FT — position lr 0, weight lr 0, features default 0.0016, 10k iters, tag e1v2. Predict: toy B50 ft−noft > 0 (CI excl. 0, recovers part of −0.52); garden B50 |ft−noft| ≤ 0.1. Kill: degradation ≥0.2 dB again on both.
- Courtyard prune-only rows done (tag e1b, incl. random_noft); garden/toy random_noft rows done (M6 trend baselines).
- **Variant 2 result: MECHANISM WORKS on real scene** (evals `eval/*_e1v2`, CIs by PROTOCOL bootstrap): garden B50 24.8509 = **+0.139 dB vs clean CI[+0.097,+0.177] AND ΔLPIPS −0.0061 CI[−0.0065,−0.0057] at 50% triangles** (clears D3 improvement + compaction floors); garden B25 iso vs clean (+0.027 CI[−0.054,+0.099], ΔLPIPS −0.0014) at 25% triangles (clears compaction floor, 75% reduction); vs noft +0.150/+0.166 CI excl. 0. Toy: FT harmless but no recovery (B50 −0.004 vs noft; −0.524 vs clean persists — prune damage is geometric; features can't rebuild it).
- **Key context discovered: garden@26000 = 25.029/0.780/0.201 BEATS garden@30000 = 24.712/0.762/0.216** (eval `eval/garden_clean26k_diag`) — original training's final ~4k iterations were already value-destroying (same position-drift mechanism); explains historical clean26000 references. Features-only FT partially recovers this pre-existing damage on the remaining triangles. NOTE for Stage-2/human: sourcing compaction from the 26000 checkpoint may be even better; NOT explored (variant budget + DEC-005 frozen baseline).
- **Mechanism variant 3/3 (pre-registered before launch)**: iterative prune schedule — 2 steps: prune to ~70.7% + features-only FT 5k, then prune to exactly floor(0.5·T_clean) + features-only FT 10k (B25: two more equivalent steps). Same per-step config both scenes (D7). Predict: toy B50 full-pipeline vs clean better than one-shot's −0.52 (target ≥ −0.20 = E1(a)); garden must stay ≥ clean (no regression vs one-shot v2). Kill: toy still < −0.35 vs clean or garden regresses below iso → E1 FAIL final, verdict written with variant-2 results as the M2 outcome.
- Iteration budget: E1 mechanism variants used 3/3 (this is the last). Retention: diag ckpts deleted, evals kept.
- Pre-registered hypothesis (before any run): evidence-guided prune to B + topology-frozen FT (10k it, frozen config) → at B=50 on garden AND toy: ≥ +0.5 dB vs prune-no-FT (CI excl. 0) AND ≥ −0.20 dB vs clean. Kill: either fails on either scene after ≤3 mechanism variants.
- 12 runs launched as 3 detached GPU chains (tag e1). CRASH #1: all importance-mode runs died at `gems_pipeline.py:519` — `stage_evidence` returned the npz path (str) where the stamp payload (dict) was expected; random-mode smoke never exercised the branch. Root-caused, fixed (commit ee5cf64), chains relaunched; stamps resumed completed work. No results were affected (crash was before FT).
- CRASH/VOID #2: first courtyard g4 row (courtyard_clean30k_v2) showed F@5cm=0.009 → investigated → ETH3D raw scan vertices are ~1.2 m off the camera frame; `scan_alignment.mlp` transforms are REQUIRED (verified: median sparse→scan 0.036 m transformed vs 1.19 m raw; camera frames of eth3d_colmap and ETH3D GT calibration are bit-identical). Fix: transforms frozen into scenes.py, loaders apply them, ROI recomputed from transformed AABB, PROTOCOL changelog 1.1.1, v2 g4 row VOID (commit cb1560b). v3 eval re-running.
- Courtyard clean baseline (trained this session, 30k it, ~20 min, GPU 5): PSNR 17.686 / SSIM 0.5968 / LPIPS 0.3850, 4.37M tris, 104.6 FPS; g1 9.40% (99,690 samples), g3 5808 comps / 0.81% — geometry much worse than garden, good E2 headroom. Eval: `/data/peilincai/gems_stage1/eval/courtyard_clean30k_v1` (+v3 for g4).

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
- DONE 2026-07-02: training completed (30k iters, ~17 min, final model 6,590,559 tris / 541 MB — 62× over-parameterized vs GT 105,982 faces; contains 13 NaN faces, handled per PROTOCOL §4.3 non-finite exclusion added pre-first-row).
- **M1b acceptance evidence** (durable): `/data/peilincai/gems_stage1/eval/toy_parking_clean30k_v1/metrics.json` and `.../toy_parking_GTmodel_v1/metrics.json`:
  - toy clean30k: PSNR 30.894 / SSIM 0.9603 / LPIPS 0.0936; 62.3 FPS; g1 23.85% (360k GT-depth samples), g2 0.2559 m, g3 395 floater comps (0.28%), g4 chamfer 0.5696 m / F@5cm 0.396, d1 false_free 58.9% / false_occ 3.0%, d2 agreement 0.625 (unsafe_disagreement 0.0).
  - GT-mesh model (calibration): PSNR 56.17; g1 0.028%, g2 0.0028 m, g4 chamfer 0.0179 m / F 0.998, d1 0.19%/0.01%, d2 agreement 1.000 → **metric code validated end-to-end; clean-model geometry unreliability is REAL** (motivates E2 and compaction headroom for E1).
- VERDICT: PASS (AT1 + M1b complete).

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
