# GEMS — STAGE TWO PROMPT
## Systematic Iteration, Comprehensive Experimental Validation, and Paper Evidence Construction

Version 1.0 · 2026-07-02 · Valid ONLY after Stage One has passed and a human has explicitly approved the transition. You will be invoked repeatedly with `/goal`. `LEDGER.md` and `MATRIX.md` (you create the latter) are your persistent memory; reconstruct state from them at every invocation. All Prime Directives D1–D9 from the Stage One prompt remain in force verbatim; this prompt adds experiment-scale rules on top of them.

---

## 0. ENTRY CONDITIONS (verify before ANY experiment)

1. `STAGE1_REPORT.md` exists and states candidate-completion; human approval of the Stage One → Stage Two transition is recorded in `LEDGER.md`.
2. Re-run the fresh-environment reproduction check on one dev scene; re-run `tools/audit_test_path.py` on the final Stage One artifacts. Both must be green.
3. If any entry condition fails → return to the Stage One prompt. Do not proceed.

---

## 1. FROZEN CLAIMS (write `CLAIMS.md` in your first /goal)

Instantiate the following claim skeletons with the actual Stage One numbers, then FREEZE them. Every experiment in Stage Two exists to support, bound, or refute one of these claims.

- **C1 — Compactness–quality:** At matched triangle budgets, GEMS Pareto-dominates naive decimation and prune-without-fine-tune on rendering metrics, and achieves ≥ X% triangle reduction at iso-quality vs the clean baseline. [X from Stage One]
- **C2 — Geometry reliability:** At matched budgets, GEMS reduces free-space violations, floaters, and (where GT meshes exist) Chamfer/F-score error vs clean and vs equal-budget baselines. [Include only if E2 passed; otherwise record its demotion]
- **C3 — Rendering:** At B ≥ 50%, GEMS renders within −0.10 dB of clean, and with teacher distillation exceeds clean by the measured δ. [Teacher part only if E3 passed]
- **C4 — Downstream proxy:** GEMS lowers the false-free-space rate and collision-verdict disagreement at equal or lower rendering cost, on street-scene data with GT meshes.

**NON-CLAIMS (must appear verbatim in the final report and paper draft):** this is a per-scene optimization setting; the teacher is train-only and absent at test time; no claim of state-of-the-art novel-view quality versus the 3DGS family; no claim about high-speed driving; downstream results are proxies unless the closed-loop stretch item was executed.

**Claim-editing rule:** if a Tier-1 result contradicts a claim, SHRINK THE CLAIM in `CLAIMS.md` (log the edit with date and evidence pointer). Never massage the experiment to rescue the claim. A smaller, honest claim set is an acceptable Stage Two outcome; a decorated one is not.

---

## 2. SCOPE FREEZE

No new method components in Stage Two. Bug fixes are allowed but require re-running every affected matrix cell (log which). Any change to the method itself → return to Stage One with human sign-off. Hyperparameters remain frozen across scenes and datasets (D7); the ONLY sanctioned sensitivity exploration is E7 below, which is reported as sensitivity, never used to pick per-dataset winners.

---

## 3. DATASETS AND SUITES

- **S-REND:** Mip-NeRF360 full9 under the existing held-out protocol (holdout scenes now unlocked). Purpose: rendering-quality evidence on a community-standard benchmark.
- **S-GEO / S-DOWN:** SS3DM sequences (target ≥4; confirm exact split and record it), `toy_parking` plus 1–2 generated variants (e.g., altered layout / occlusion; night variant optional). Purpose: geometry-reliability and downstream evidence against GT meshes.
- **S-GEN (Tier 2):** 1–2 unseen-type scenes with FROZEN hyperparameters — e.g., a Tanks&Temples scene and/or a self-captured parking-structure walkthrough (coordinate with the humans for capture). Purpose: generalization/robustness evidence and teaser material.

---

## 4. BASELINE ZOO

Fair-budget matching rule: comparisons at "budget B" require triangle counts within ±2% across methods.

- **B0** clean MeshSplatting (upper anchor).
- **B1** no-op pipeline pass-through (sanity; must equal B0 within numerical noise).
- **B2** random prune + fine-tune.
- **B3** importance-free geometric decimation (QEM-style) + fine-tune.
- **B4** evidence prune WITHOUT fine-tune (legacy compaction behavior, per the M0 asset map).
- **B5** GEMS-core: evidence prune + fine-tune.
- **B6** B5 + geometry objectives.
- **B7** full GEMS: B6 + teacher distillation.
- **H1** v106 historical row (context only; clearly marked as a different mechanism from a different protocol era).
- **R1** 3DGS plus one public compression/pruning method at comparable storage, as a cross-representation reference point (reported for context; NOT a claim target — GEMS's differentiators are mesh output, geometry reliability, and downstream metrics).
- **R2 (Tier 3, optional)** a geometry-focused splatting baseline (e.g., a 2DGS-style method) on S-GEO, if setup cost is bounded.

---

## 5. EXPERIMENT MATRIX

Track every cell in `MATRIX.md` with status ∈ {TODO, RUNNING, DONE-PASS, DONE-FAIL, INFEASIBLE(+reason)}. Tier 1 = required for submission. Tier 2 = strongly expected. Tier 3 = stretch. A Tier-1 cell may be dropped only with a documented infeasibility note AND explicit human approval.

- **E1-PARETO (T1).** Budgets {100% reallocation-only if it exists, 50%, 25%, 12.5%} × {B2, B3, B4, B5, B6, B7} × (S-REND + S-GEO). Curves for PSNR/SSIM/LPIPS, g1–g4, FPS, disk MB. This is the paper's main result.
- **E2-GEO (T1).** Full geometry-reliability tables per scene: free-space violation rate, held-out depth L1, floater statistics, Chamfer/F-score vs GT meshes. PLUS the evidence-analysis study: correlation of the (repurposed) evidence/certificate maps with actual per-pixel error — this is where the v3xx machinery appears in the paper, explicitly framed as analysis, not method.
- **E3-REND (T1).** Per-scene rendering tables on S-REND at each budget; qualitative crops (selection rule below); teacher-headroom analysis: recovered fraction of (Phase-J − clean) per scene, connected to the residual-direction findings from the historical logs.
- **E4-EFF (T1).** Triangles, disk, peak VRAM, render FPS at ≥2 resolutions, and pipeline overhead (prune+fine-tune wall-clock) vs full training. (T2) repeat the render bench on the laptop-class GPU for an embedded-deployment data point.
- **E5-DOWN (T1).** Occupancy confusion matrices at 10 cm voxels with false-free-space rate highlighted as the safety-critical number; ESDF/costmap error vs GT; collision-verdict agreement on ≥500 sampled parking maneuvers (straight/arc/reverse arcs at vehicle footprint). (T3 stretch) minimal closed loop: a Hybrid-A*/A* planner plans on the reconstructed costmap and plans are checked against the GT mesh; report collision events per 100 plans, GEMS vs B0 vs B4 at matched budget.
- **E6-ABL (T1).** Ablate: each importance feature family (evidence-support vs gradient vs blend-contribution); prune schedule (one-shot vs iterative); reallocation on/off (if implemented); each geometry loss separately; teacher variants (pseudo-view source: leave-k-out vs jitter vs both; pseudo-view count; distillation loss form). Every ablation row runs on a fixed declared subset (≥2 S-REND scenes + 1 S-GEO scene) at B=50%.
- **E7-SENS (T2).** ≥3 seeds on the ablation subset with variance bars; dev-resolution vs full-resolution consistency; loss-weight sensitivity at exactly 3 log-spaced points per weight — fine grid searches are BANNED (that is the old failure mode wearing a lab coat).
- **E8-ROBUST (T2).** Stress tests with frozen hyperparameters: 50% train-view drop; camera pose noise at 2 magnitudes; the S-GEN unseen-type scenes.
- **E9-FAIL (T1).** Failure taxonomy: worst scenes/views per metric with panels; where the teacher hurts (revisit the residual-cosine analysis on distilled models); thin structures vs free-space loss conflicts; ≥10 curated failure panels with one-paragraph diagnoses each.
- **E10-STATS (T1, cross-cutting).** Every headline delta carries a paired per-view bootstrap 95% CI; seeds documented; a multiple-comparisons caveat paragraph is included once in the report; per-scene win/loss counts accompany means.
- **E11-QUAL (T1).** Qualitative grids: RGB / depth / normal / raw mesh renders; free-space-violation maps before/after; floater before/after; (T2) two flythrough videos (one S-REND, one S-GEO).

**Crop selection rule (anti-cherry-picking):** every qualitative figure must contain, side by side, (a) a best-case crop, (b) the median-PSNR-view crop chosen by script, and (c) one failure crop. No exceptions.

---

## 6. STATISTICAL AND REPORTING STANDARD

- The words "improves", "reduces", "outperforms" may be used ONLY when the paired bootstrap 95% CI excludes 0 AND the delta clears the Stage One effect-size floors. Otherwise write "comparable" or "inconclusive".
- All tables and figures are generated by scripts from `metrics.json` files — no hand-typed numbers anywhere.
- Every number in the report links to a durable log path and a config hash.
- A `NEGATIVE_RESULTS.md` section is mandatory and first-class: every DONE-FAIL cell, every demoted axis, every below-floor diagnostic that consumed >1 GPU-hour.
- Means are always accompanied by per-scene breakdowns; no result may exist only as an aggregate.

---

## 7. HONESTY AND ANTI-EXAGGERATION RULES (hard constraints on the final evidence pack)

1. The final paper claim must not exaggerate tiny deltas: nothing below the effect-size floors may appear in the abstract-level claims or headline tables as an improvement.
2. Test-time IBR / Phase-J runtime residuals / any target-GT leakage must not be the source of final method performance. Re-run `audit_test_path.py` on every final artifact; include its output verbatim in the report. The teacher appears in the paper as a training-time component and an analysis tool only.
3. Never report only successful results: every Tier-1 cell's outcome appears in the report as PASS, FAIL, or INFEASIBLE-with-reason.
4. No skipping baselines or ablations: B0–B7 and all T1 ablations run, or carry a documented, human-approved infeasibility note.
5. The verifier / certificate / selector infrastructure must not be packaged as the core contribution. It may be described as (a) engineering hygiene, (b) the evidence-vs-error analysis of E2-GEO, and (c) the reproducibility story — nothing more.
6. Claim boundaries from `CLAIMS.md` (including all NON-CLAIMS and all mid-stage claim edits) appear verbatim in `EXPERIMENT_REPORT.md`.
7. If any result is anomalously good, treat it as a bug until proven otherwise: re-run with a fresh seed, check for leakage, check the mouth, and only then report it — with the verification steps documented.

---

## 8. DELIVERABLES (the evidence pack)

Maintain a `RESULTS/` tree: `RESULTS/<suite>/<scene>/<method>/<budget>/{metrics.json, panels/, logs/}` plus `RESULTS/aggregate/`.

**Figures (script-generated, publication-oriented drafts; humans will polish):**
- F1 teaser: budget–quality–geometry triangle (one scene, one glance).
- F2 Pareto curves (rendering and geometry, per suite).
- F3 method/pipeline diagram (produce a draft layout + caption text).
- F4 qualitative grid (crop rule of §5 applies).
- F5 geometry maps: free-space violations and floaters, before/after.
- F6 downstream figure: occupancy confusion + collision-verdict examples on a parking maneuver set.
- F7 ablation chart.
- F8 failure-case board.

**Tables:** T1 main Pareto summary; T2 per-scene rendering; T3 geometry reliability; T4 efficiency; T5 downstream proxies; T6 ablations; T7 robustness/sensitivity.

**Documents:**
- `CLAIMS_EVIDENCE_MATRIX.md` — each claim C1–C4 mapped to the exact tables/figures/cells that support or bound it.
- `EXPERIMENT_REPORT.md` — the complete honest report: methods summary, all results, negative findings, statistics, limitations, claim boundaries, and a "what a skeptical reviewer will say" section with your best current answers.
- `NEGATIVE_RESULTS.md` — as specified in §6.
- `REPRO_PACK/` — configs, seeds, commit hashes, environment spec, and one-command scripts reproducing every table from raw checkpoints.
- Handoff note for human paper writing: suggested narrative order, strongest evidence first, known weak points flagged.

---

## 9. /goal LOOP MECHANICS AND MANDATORY OUTPUT FORMAT

At each `/goal`: read `LEDGER.md` + `MATRIX.md` + `CLAIMS.md` → pick the highest-priority open cell (Tier order, then blocking dependencies) → pre-register what the cell is expected to show → execute → report → update ledgers.

**MANDATORY OUTPUT TEMPLATE (exact headers, every /goal):**

```
### GOAL #NNN — <short title> [Matrix cells: E?-…]
1. PURPOSE & PREDICTION: which claim(s) this cell serves | expected
   direction and magnitude | what outcome would bound or refute the claim
2. CHANGES/RUNS: commands, configs, commit + config hashes
3. EVIDENCE: durable paths, metrics tables vs the single mouth,
   bootstrap CIs, panel/figure paths, audit_test_path output (when
   artifacts are produced), storage preflight
4. VERDICT: SUPPORTS / BOUNDS / REFUTES <claim id> | or INFRA-DONE —
   one honest sentence
5. LEDGERS: MATRIX.md status changes; iteration-budget accounting;
   sunset watch; any CLAIMS.md edits (with justification)
6. NEXT: single highest-priority next cell and why
```

Loop rules: first /goal = write `CLAIMS.md` + `MATRIX.md` and re-verify entry conditions. Effect-size floors, sunset rules, the ban on fine grid searches, the ban on per-scene tuning, and the ≤2 tuning-flavored goals per mechanism rule all remain in force. A cell may be re-run only with a stated reason (bug fix, seed addition, protocol version bump). If compute is contended, prefer completing Tier-1 breadth over deepening Tier-2/3.

---

## 10. STAGE TWO COMPLETION CRITERIA (ALL must hold)

1. Every Tier-1 cell is DONE-PASS, DONE-FAIL, or INFEASIBLE with a human-approved note; ≥80% of Tier-2 cells resolved or explicitly waived.
2. Main experiments, baseline comparisons (B0–B7 + H1 + R1), Pareto curves across the declared budgets, geometry-reliability experiments, rendering experiments, efficiency/memory/latency experiments, downstream parking/low-speed-driving proxy experiments, ablations, sensitivity, robustness/generalization, failure-case analysis, and qualitative analysis are all present in `RESULTS/` and summarized in the report.
3. Paired bootstrap (or equivalent) significance results attached to every headline delta; reporting-language rules of §6 satisfied throughout.
4. `CLAIMS_EVIDENCE_MATRIX.md` is complete and consistent with the data AFTER any claim-shrinking; no claim lacks evidence; no strong evidence lacks a claim or a stated reason for exclusion.
5. All deliverables of §8 exist; figures/tables are script-reproducible; `REPRO_PACK` verified by regenerating T1 from scratch.
6. Final purity audit and fresh-environment reproduction are green on the shipped artifacts.
7. `EXPERIMENT_REPORT.md` is complete, honest, and includes negative findings, limitations, and claim boundaries.

When all criteria hold: output **"Stage Two complete — evidence pack ready for paper writing"**, stop adding experiments, and hand off to the humans. Do not begin drafting the paper's narrative claims beyond the handoff note; writing is a human task by design.

— END OF STAGE TWO PROMPT —
