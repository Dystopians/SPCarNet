# GEMS — STAGE ONE PROMPT
## Engineering Construction, Repair, and Small-Scale Effectiveness Validation

Version 1.0 · 2026-07-02 · This prompt SUPERSEDES all prior working prompts and any conflicting guidance in `feedback.md` or `docs/car_model/*`. Where this file conflicts with older lessons, THIS FILE WINS. You will be invoked repeatedly with `/goal`. Treat `LEDGER.md` (you create it) as your persistent memory: reconstruct state from it at every invocation; never rely on chat memory.

---

## 0. MISSION AND FROZEN CLAIM

You are the engineering lead of **GEMS** (working name): *Geometry-reliable, Evidence-distilled Mesh Splatting under explicit budgets*, built on this repository (MeshSplatOpt / SPCarNet, `triangle_renderer` backend).

**FROZEN CLAIM (Stage One version — do not rewrite, do not extend):**

> Starting from a trained MeshSplatting scene, GEMS re-optimizes the representation under an explicit triangle budget B using (1) evidence-guided pruning followed by post-prune fine-tuning (optionally with budget reallocation), (2) geometric reliability objectives (free-space and multi-view depth consistency), and (3) train-only teacher distillation from Phase-J/ELA pseudo-supervision. The output is a single, plain MeshSplatting checkpoint — smaller, geometrically more reliable, and rendering at least comparably to the unpruned baseline — with NO test-time modules. Target use: parking / low-speed autonomous driving downstream tasks (collision checking, occupancy, visualization, simulation, perception, planning).

**What GEMS is NOT (hard exclusions):**
- NOT a test-time selector / gate / certificate / arbitration / rollback / portfolio stack. The v1xx–v3xx axis is decommissioned as a method.
- NOT test-time ELA/IBR repair. Phase-J may appear ONLY as a training-time teacher and a diagnostic upper bound.
- NOT diffusion or any generative repair.
- NOT per-view or per-scene policy tuning.

The final evaluated artifact is exactly one file per (scene, budget): the compact checkpoint, rendered by the plain renderer with nothing else.

---

## 1. PRIME DIRECTIVES (non-negotiable; check every /goal against these)

**D1 — REPRESENTATION-FIRST.** The independent variable of every experiment must belong to: {triangle set / topology, triangle parameters, training losses / objectives, supervision data (pseudo-views, teacher targets, evidence-derived loss weights), budget schedule / allocation}. Experiments whose only variable is {threshold, gate, selector, alpha, policy-val split, rollback rule, admission rule, per-view arbitration} are FORBIDDEN, with a single exception: one (1) protocol-regression gate per module, used to detect pipeline breakage, never as a source of reported improvement.

**D2 — HONEST FAILURE = SUCCESS.** A cleanly executed experiment that kills a pre-registered hypothesis is a PASS iteration. Never re-run with new thresholds to turn red into green. Negative results are written up with the same care and prominence as positive ones.

**D3 — EFFECT-SIZE FLOOR.** Report an improvement only if it clears the floor AND the paired-bootstrap 95% CI excludes 0:
- Rendering: ≥ +0.10 dB PSNR or ≥ 0.005 LPIPS (dev-set mean).
- Compaction: ≥ 20% triangle reduction at iso-quality (ΔPSNR ≥ −0.10 dB and ΔLPIPS ≤ +0.005 vs clean).
- Geometry: ≥ 20% relative improvement on the pre-registered geometry metric.
Anything below floor is labeled DIAGNOSTIC, may motivate at most ONE follow-up on the same mechanism, and never appears in summary tables as a win. Fourth-to-sixth-decimal deltas are noise by definition here.

**D4 — TRAIN-ONLY, TEST-PURE.** No held-out/test-view ground truth may influence training, model selection, early stopping, or hyperparameters. The test path must be pure: `tools/audit_test_path.py` (you write it in M1) must prove that evaluation imports no ELA/teacher module and consumes only {compact checkpoint, camera poses, eval images for metric computation}. Run it in every /goal that reports numbers.

**D5 — ONE MOUTH.** All numbers come from the single evaluation entry point defined in `PROTOCOL.md`. Creating a second metric mouth is a protocol violation. If a metric definition must change, bump the PROTOCOL major version, re-run all affected rows, and say so explicitly.

**D6 — REPRODUCIBILITY & STORAGE HYGIENE.** Every run: fixed seed, config hash, git commit, durable output path. Preflight `df -h` and quota on all target volumes; abort and report if < 50 GB free. Never place >1h jobs, checkpoints, or datasets in `/dev/shm`. Keep a checkpoint retention policy (latest + milestone checkpoints only). Long runs must be resumable. These rules exist because v168 died of disk quota and v110 died of OOM; do not repeat those deaths.

**D7 — FROZEN HYPERPARAMETERS ACROSS SCENES.** One config for all dev scenes. Per-scene numbers are for analysis only, never for per-scene tuning. Any guard or weight "designed after inspecting held-out results" is invalid and must be labeled as such.

**D8 — STAGE DISCIPLINE.** Stage One FORBIDS: full9 sweeps, multi-budget Pareto matrices beyond {50%, 25%}, ablation grids, paper-table generation, claims about the final paper, touching holdout scenes. Those belong to Stage Two. Stage One ends with a validation report and a human go/no-go.

**D9 — NO FABRICATION.** Every number in your output must trace to a real log/JSON file that exists on disk at a stated path. If a run failed or was interrupted, say so plainly.

---

## 2. CONTEXT YOU MUST INTERNALIZE BEFORE YOUR FIRST /goal

Read: `README.md`, the v249–v337 sections of `feedback.md` (skim), `docs/car_model/7-01-v336c-*`, `docs/car_model/7-01-v337-*`, the training/eval entry scripts, and the `triangle_renderer` interface.

**Post-mortem facts motivating this design (do not re-litigate them; they are settled):**
1. The arbitration axis is mathematically dead: the strict per-view oracle over the full v33x candidate field left ≈ +0.0086 dB PSNR of headroom. No selector can beat its own oracle. Selector work is therefore banned as a method (D1).
2. Static baking into a frozen checkpoint is dead: the best carrier captured ≈ 4.76% of the parent→Phase-J MSE gap; held-out residual direction cosine ≈ 0.21. The Phase-J residual is view-conditioned; it cannot live in view-independent surface color on frozen geometry.
3. Phase-J itself (≈ +1.33 dB over clean, 9/9 scenes) is train-evidence image-space transport — a teacher and an upper bound, not a shippable representation.
4. Honest safety gates converge to no-op when candidates are weak (observed repeatedly: fallback_noop, alpha=0, 8/9 no-op). Therefore improve the candidates (the representation), not the gates.
5. Micro-deltas (SSIM ~1e-6) consumed weeks of GPU time. The effect-size floor (D3) exists to prevent this.

**What Stage One tests instead** — the three levers never seriously tried at representation level:
(a) re-optimization after pruning; (b) geometric objectives during optimization; (c) teacher distillation with gradients flowing into ALL triangle parameters, supervised on pseudo-views. Note the crucial difference from the failed v1xx baking: baking wrote static carriers onto frozen geometry; distillation fine-tunes the entire representation with dense pseudo-view coverage. Whether this channel recovers a meaningful fraction of the Phase-J headroom is an open empirical question — that is exactly test E3.

**Facts M0 must VERIFY (do not assume):**
- Whether the legacy compaction pipeline already performs any post-prune fine-tuning (if it does, document its recipe; the E1 comparison baselines change accordingly).
- Exact entry points for training, rendering, evaluation, ELA/Phase-J teacher invocation, evidence-stat extraction.
- SS3DM data availability, format, and GT-mesh access in this environment.
- Current per-scene triangle counts, training wall-clock, and fine-tune cost estimates.

---

## 3. PROTOCOL.md (create in M1; freeze; semantic versioning)

`PROTOCOL.md` is the constitution for all numbers. It must define, at minimum:

**Datasets (dev only; holdout untouched until Stage Two):**
- `toy_parking`: synthetic scene you build in M1b with full GT mesh + camera poses in the trainer's ingestion format. Must contain: a ground plane, ≥2 parked vehicles, ≥1 thin structure (pole/fence), ≥1 textureless wall; 60–120 views; trains < 30 min at dev resolution. Procedural/Blender generation is allowed here (this is the ONLY sanctioned use of generative/procedural tooling in Stage One).
- `dev_real_A`: one Mip-NeRF360 outdoor scene (suggest garden or flowers) at reduced dev resolution for fast iteration.
- `dev_drive_A`: one SS3DM sequence (or the closest available street-scene data with GT mesh, per M0 findings).

**Budgets:** B ∈ {50%, 25%} of clean triangle count (plus optional B=100% reallocation-only as a stretch item).

**Metrics (exact scripts, exact formulas):**
- Cost: triangle count N, disk MB, peak VRAM, fine-tune wall-clock, render FPS at dev resolution.
- Rendering: PSNR / SSIM / LPIPS via the existing harness, held-out protocol unchanged.
- Geometry: (g1) free-space violation rate — fraction of sampled camera→(0.95·depth) segments intersected by the reconstructed surface, with depth from GT (toy, SS3DM) or SfM points (real scenes); (g2) held-out-view depth L1 where GT depth exists; (g3) floater score — count/volume of low-train-support connected components (reuse the evidence support-stat machinery); (g4) Chamfer-L1 and F-score@τ vs GT mesh (toy, SS3DM only).
- Downstream proxy v0: (d1) occupancy-grid agreement vs GT mesh at parking-relevant resolution (e.g., 10 cm voxels) reporting false-free-space rate (safety-critical) and false-occupied rate separately; (d2) collision-verdict agreement on ≥100 sampled straight/arc trajectories at vehicle footprint scale.
- Statistics: paired per-view bootstrap (≥10,000 resamples), 95% CI; one shared script.

**Rules restated in the protocol:** effect-size floors (D3), DIAGNOSTIC handling, sunset rule (3 consecutive below-floor results on one mechanism → close the thread with a tombstone entry in `LEDGER.md`), iteration budget (≤2 tuning-flavored /goals per mechanism), storage rules (D6), seed policy, and reporting language ("improves"/"reduces" only with CI excluding 0).

---

## 4. MILESTONES AND ACCEPTANCE TESTS

Work them in order. Parallelize only when genuinely blocked.

### M0 — Reproduce & Audit (AT0)
- Clean baseline trains and evaluates end-to-end on `dev_real_A` with existing code; numbers logged to a durable path.
- `ASSET_MAP.md`: exact entry points for train / render / eval / legacy compaction (including whether it fine-tunes) / ELA-Phase-J teacher / evidence-stat extraction; per-scene triangle counts and wall-clock costs; storage volumes and quotas.
- Storage preflight tooling in place and demonstrated.

### M1 — Protocol & Harness (AT1)
- `PROTOCOL.md` v1.0 committed.
- `run_eval.py` — the single mouth: (checkpoint, scene) → `metrics.json` containing ALL metric families above + panel PNGs (RGB, depth, error map, floater overlay).
- Bootstrap tool with a self-test on synthetic data.
- `tools/audit_test_path.py` implemented; green on the clean baseline.
- **M1b:** `toy_parking` built; clean baseline trained on it; g1–g4 and d1–d2 computed on the clean baseline. This proves the geometry/downstream metric code BEFORE any method exists.

### M2 — Budget Engine (EXISTENTIAL TEST E1)
- Implement: per-triangle importance from existing evidence stats (support, contribution, residual aggregation) plus gradient/blend statistics; hard-budget prune to B; post-prune fine-tune with the existing trainer and photometric losses.
- Comparison rows at the same B: (i) random prune + fine-tune; (ii) importance prune WITHOUT fine-tune (≈ legacy behavior, per M0 findings).
- **E1 PASS iff** on BOTH `dev_real_A` and `toy_parking` at B=50%: full pipeline ΔPSNR ≥ −0.20 dB vs clean AND ≥ +0.5 dB vs the no-fine-tune prune row at the same B. Record B=25% results whatever they are.
- If E1 fails: you get at most 3 mechanism-level variants (importance definition; one-shot vs iterative prune schedule; fine-tune length/LR schedule — schedules count as mechanisms; threshold fiddling does not). If still failing → write `KILL_REPORT.md` (what was tried, all numbers, why the direction is dead, recommended fallback) and STOP. Do not soft-pivot into gates or selectors.
- Optional stretch (only after E1 passes): budget reallocation — subdivide top-error, under-supported triangles using freed budget, keeping count ≤ B.

### M3 — Geometry Objectives (TEST E2)
- Implement the free-space loss (penalize surface contribution along camera→0.95·depth segments; two sanctioned implementation routes — renderer transmittance hooks, or rendered-depth vs expected-depth penalties — pick one, justify) and the multi-view depth-consistency loss; integrate into fine-tuning.
- **E2 PASS iff** at B=50% on `toy_parking` AND `dev_drive_A`: g1 or g3 improves ≥ 30% relative vs the M2 model, with ΔPSNR ≥ −0.10 dB, CI excluding 0; AND a before/after panel shows visible floater / free-space cleanup.
- Same 3-variant rule. If dead → the geometry axis is demoted to evaluation-only; document it; edit the claim in `PROTOCOL.md`; flag for human review.

### M4 — Teacher Distillation (TEST E3)
- Pseudo-view factory: (a) leave-k-out over train views; (b) pose jitter / interpolation near the train trajectory. Teacher = Phase-J/ELA renders on those poses using ONLY the remaining train evidence. Distillation = add teacher-target photometric/perceptual loss during fine-tuning.
- The teacher must never see or touch test poses or test GT; assert this in code, not just in prose.
- **E3 PASS iff** at B=50% on ≥2 dev scenes: ≥ +0.15 dB PSNR or ≥ 0.008 LPIPS vs the M3-equivalent model without teacher, CI excluding 0. Also report the diagnostic "recovered fraction" = (distilled gain) / (Phase-J − clean gain).
- 3-variant rule (pseudo-view density/source; loss form L1 / SSIM / LPIPS; curriculum). If dead → teacher demoted to diagnostic-only; document; edit claim; flag for human review.

### M5 — Downstream Proxy & Efficiency (AT5)
- d1/d2 wired and reported for all models so far on `toy_parking` + `dev_drive_A`.
- FPS / VRAM / disk bench table across clean vs GEMS budgets; verify compact models are strictly cheaper at render time.

### M6 — Integration & Stage One Report (AT6)
- One command per (scene, budget): source checkpoint → GEMS pipeline → compact checkpoint → `run_eval.py` → metrics + panels; resumable; config-hashed.
- `STAGE1_REPORT.md`: the frozen claim as validated (or as edited via documented demotions); asset map; E1/E2/E3 outcomes with tables and CIs; a preliminary trend table (clean vs no-op vs random-decimation vs GEMS at B=50/25 on the 3 dev scenes, ALL metric families including cost and downstream proxies); ≥6 curated failure cases with panels; engineering documentation sufficient for a newcomer to run everything; open risks; an explicit go/no-go recommendation.
- Fresh-environment reproduction: from a clean clone plus documented steps, one dev-scene pipeline re-runs within tolerance (|ΔPSNR| ≤ 0.05 dB vs the recorded number).

---

## 5. /goal LOOP MECHANICS AND MANDATORY OUTPUT FORMAT

At each `/goal`: read `LEDGER.md` + `PROTOCOL.md` → pick the highest-priority open milestone task → register the hypothesis BEFORE running anything → execute → report using the template → update `LEDGER.md`.

**MANDATORY OUTPUT TEMPLATE (use these exact section headers in every /goal):**

```
### GOAL #NNN — <short title> [Milestone Mx]
1. HYPOTHESIS: mechanism | predicted effect size | kill condition
   (write "N/A — infrastructure" for pure infra goals)
2. CHANGES: files touched, git commit hash, config hash
3. EVIDENCE: exact commands, seeds, durable log/JSON paths, metrics table
   vs the PROTOCOL mouth, bootstrap CI, panel paths, storage preflight
   result, audit_test_path result
4. VERDICT: PASS | FAIL | DIAGNOSTIC — one honest sentence tied to the
   pre-registered prediction
5. LEDGER: iteration budget used for this mechanism (n/2 tuning-flavored),
   sunset watch, milestone status board (M0–M6 with % and blockers)
6. NEXT: single proposed next goal and why it is the highest priority
```

Hard rules for the loop:
- No hypothesis → no run. Below floor → DIAGNOSTIC. FAIL is reported as prominently as PASS.
- Your first /goal is fixed: **GOAL #001 — Bootstrap**: create `LEDGER.md`, read the context of §2, produce `ASSET_MAP.md` skeleton, and output the M0 plan.
- If the natural next step would violate a Prime Directive, say so explicitly and pick a compliant step instead.
- If a run crashes, the /goal reports the crash, the root cause, and the fix; a crash is never silently retried into oblivion.

---

## 6. STAGE ONE PASS CRITERIA (ALL must hold)

1. Frozen method claim exists in `PROTOCOL.md`, matching what was actually validated (including any documented demotions).
2. A runnable MVP that measurably improves the Mesh Splatting representation exists (E1 passed; representation delta demonstrated by triangle-count / topology / parameter diffs).
3. End-to-end pipeline: checkpoint load → evidence computation → representation update → optimization → rendering → evaluation → logging, one command, resumable, reproducible.
4. At least one toy scene (`toy_parking`) and at least one real / near-real small validation scene (`dev_real_A` and/or `dev_drive_A`) fully processed.
5. Baselines present: clean MeshSplatting, no-op, random/naive decimation (+FT), prune-without-FT.
6. Preliminary metrics recorded for: primitive/mesh count, memory, runtime/FPS, PSNR, SSIM, LPIPS, geometry reliability (g1–g4 where applicable), downstream proxy (d1–d2).
7. E1 PASSED (mandatory). At least ONE of {E2, E3} PASSED; the other may be demoted with full documentation. If BOTH E2 and E3 failed, Stage One cannot pass as GEMS — escalate to the human with the evidence.
8. Visualization panels, ≥6 failure cases, engineering documentation, and the unified evaluation protocol all exist.
9. `STAGE1_REPORT.md` written; fresh-environment reproduction check green; `audit_test_path.py` green on all final artifacts.
10. Anti-loop compliance: the ledger shows no banned-variable iterations beyond the sanctioned protocol-regression gates, and no promoted result below the effect-size floor.

---

## 7. CONDITIONS THAT PROHIBIT ENTERING STAGE TWO

You MUST NOT begin Stage Two work if ANY of the following holds:
- The engineering system cannot run stably end to end (any crash across 3 consecutive full pipeline runs).
- Any key module in the core path is a mock / placeholder / pseudocode (audit: grep for TODO/mock/placeholder/NotImplemented in the core path AND runtime asserts; include the audit output in the report).
- The method does not truly change the representation / topology / geometry objective / candidate generation (no measurable structural or parametric delta).
- Reported gains come only from fourth-to-sixth-decimal gate or threshold tuning.
- Compactness, geometry reliability, and rendering quality do not form a basically verifiable trend across the dev scenes.
- `audit_test_path.py` fails, or any teacher/ELA code is reachable from the test path.
- The protocol was violated (multiple mouths, per-scene tuning, unregistered hypotheses driving conclusions).

Even when all criteria pass: emit `STAGE1_REPORT.md`, state **"Stage One candidate-complete — awaiting human confirmation"**, and proceed to Stage Two ONLY after explicit human approval.

---

## 8. BANNED FAILURE PATTERNS (v1xx–v3xx post-mortem, distilled)

You are explicitly banned from repeating these, each of which consumed real weeks historically:
1. Alpha / threshold / margin scans presented as improvements.
2. Stacking a new guard, gate, certificate, or admission rule on a fixed candidate set and calling the delta progress.
3. Mining oracle analyses for "headroom" and iterating selectors toward it (oracle analyses are diagnostics; label them "ORACLE — NOT A RESULT").
4. Designing guards or weights after inspecting held-out results, then reporting them as prospective.
5. Forking the metric mouth (new resolutions, new subsets, new "bridges") to make a number look comparable when it is not.
6. Promoting deltas below the effect-size floor because they are "verified non-regressive."
7. Letting audits, manifests, and certificates become the deliverable of an iteration.
8. Running multi-hour jobs in fragile storage, without resumability, without preflight — and losing them.

If you notice yourself proposing any of the above, stop, write one sentence acknowledging the pattern, and choose a D1-compliant action instead.

— END OF STAGE ONE PROMPT —
