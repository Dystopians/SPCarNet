# GEMS — STAGE 4 PROMPT: EVIDENCE-CACHED RENDERING (ECR)
## Exceed Phase-J · Positive Deliverable by Construction · Honest Hybrid Positioning

Version 1.0 · 2026-07-09 · **This prompt OPENS a new track on top of the closed GEMS program.** The Stage-2/3 evidence pack (v4, declared) is FROZEN and untouched; Stage-4 work lives in new rows, new tools, and a new claims file (`CLAIMS_ECR.md`). All Prime Directives, the single-mouth principle, pre-registration, CI discipline, sunset rules, iteration budgets, storage hygiene, and the /goal template remain in force. Log goals as `GOAL #E-xx`.

Before your first E-goal, re-read: `STATUS_AUDIT_20260709.md`, `CLAIMS.md` v1.3, `EXPERIMENT_REPORT.md` §4/§8, LEDGER #007 (E3), the v1xx/Phase-J sections of `feedback.md` and `docs/Latest.md`, `analysis/e2geo_evidence_vs_error/`, and the ELA implementation (`utils/evidence_lumigraph_adapter.py`, `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`, `tools/gems/teacher_factory.py`).

---

## §0. HUMAN SIGN-OFF, MISSION, AND THE DELIVERY BAR

**Human sign-off (this paragraph is the authorization):** the prior prohibition "the final method must not be test-time evidence transport" is hereby LIFTED for the Stage-4 track. The program's own measurements justify it: every pure-representation channel for view-conditioned information is quantified dead (baking captures 4.76% of the Phase-J gap; full-parameter distillation 5–16%; residual direction cosine ≈ 0.21), while the render-time evidence channel is a measured +1.33 dB existence proof. The Stage-4 shipped artifact per scene is therefore the triple **{mesh-splat checkpoint (full-budget and/or B5@B50) + evidence cache (train-view images/renders/depths/residuals + cameras) + transport renderer}**, positioned honestly as **per-scene evidence-cached hybrid rendering** — the same representation class as surface light fields / unstructured-lumigraph / deep-blending systems, and it will be compared as such (§4).

**Mission:** deliver a system that **strictly exceeds Phase-J** under the modern protocol, with every exceedance CI-backed, plus the paper-grade evidence around it.

**DELIVERY BAR (the definition of a shippable Stage-4 result):**
- Floor row **PJ-2026** (§2) reproduced under the single mouth — the guaranteed-by-construction positive deliverable.
- Final ECR stack > PJ-2026 on full9 mean **PSNR AND LPIPS**, paired CIs excl. 0; target ≥ **+0.15 dB**; stretch ≥ +0.30.
- All quality claims reported against THREE references: legacy anchor (clean@30k), PRIMARY anchor (clean-fixed@30k), and PJ-2026.
- If, after the full ladder (§3), no rung exceeds the floor: **do NOT ship an exceedance claim** — escalate to the human with the evidence. (This outcome is considered unlikely — five independent, evidence-backed levers — but honesty outranks the bar.)

**What "negative results" means in Stage 4:** the headline deliverable is positive by construction (the floor exists). Failed rungs are recorded first-class in the ledger and ablations as always — they are never deleted, hidden, or excluded from the paper's analysis sections. The Stage-1/2/3 falsification corpus is REUSED as the paper's motivation: it is the quantitative proof of *why* evidence must be cached rather than baked.

---

## §1. WHAT CHANGES AND WHAT DOES NOT

**D4 is REDEFINED, not weakened:**
- STILL ABSOLUTE: no test-view ground truth touches anything — not training, not model selection, not thresholds, not stopping, not per-test-view hyperparameters. No per-TEST-view tuning of any kind.
- NOW LEGAL: train-view images, train renders/depths/residuals, and train cameras are render-time inputs, because they are part of the shipped artifact.
- `tools/audit_test_path.py` gets a Stage-4 mode (`--ecr`): whitelist = {checkpoint, evidence cache manifest, camera poses, eval images for metric computation}; it must PROVE no test-GT dependency and no per-test-view parameter injection. Run it on every reported row. The Stage-2 mode remains untouched for legacy rows.

**The single mouth is EXTENDED, not forked:** `run_eval.py --renderer ecr` renders test views through the full stack (base render → transport → fuse) and reports the same metrics PLUS new cost columns: `cache_mb_raw`, `cache_mb_compressed`, `transport_ms_per_frame`, `end_to_end_fps`. One mouth, two renderer modes, identical metric code. PROTOCOL gets a MINOR bump documenting this (v1.2.0); no legacy rows are re-run or re-interpreted.

**Effect floors (raised — this is a headline hunt):** a ladder rung is PROMOTED into the incumbent stack only if full9 mean ΔPSNR ≥ +0.10 dB OR ΔLPIPS ≥ 0.004 vs the current incumbent, CI excl. 0. Below floor = DIAGNOSTIC, one follow-up max, sunset at 3. Per-scene tuning banned as always: one frozen config across scenes.

**Learned components vs the banned species — the boundary, verbatim:** a per-pixel fusion network trained ONCE per scene on train views only, then FROZEN and applied identically to every test view, is a MODEL (legal; it is the Deep-Blending move). The banned species remains: any per-test-view candidate arbitration, any decision informed by test metrics, any gate/threshold iterated against evaluation numbers. If a proposal selects among outputs USING test-side information, it is dead on arrival.

---

## §2. M-E0 — THE FLOOR: PJ-2026 (do this first; ~days)

Re-run Phase-J/ELA exactly as archived, but through the modern mouth:
- Bases: (a) PRIMARY anchor `clean-fixed@30k` (primary), (b) `B5@B50` compact (secondary row).
- Suites: full9 S-REND; 4 SS3DM towns; toy_parking (expect weakness — ring coverage; report honestly).
- Report: quality vs all three references; per-scene tables; AND the honest cost row — cache size raw/compressed, transport ms/frame, FPS with transport.
- **Acceptance AT-E0:** PJ-2026 ≥ +1.0 dB over the PRIMARY anchor on full9 mean, CI excl. 0 (expected ≈ +1.2–1.33 given the archived +1.33 vs the weaker legacy clean). If Phase-J fails to reproduce at ≥ +1.0 under the modern mouth → STOP the ladder, root-cause, escalate. That would be a program-level surprise and changes everything downstream.
- Freeze `PJ-2026` as the incumbent stack v0 and the guaranteed deliverable floor.

---

## §3. THE EXCEEDANCE LADDER (pre-register each rung; ≤2 mechanisms per rung; gates per §1)

Ordered by banked-evidence expected value. Each rung is evaluated against the CURRENT incumbent stack; promotion updates the incumbent.

- **L1 — Base composition.** (a) already in E0: PJ on the primary anchor. (b) PJ on a distilled base: run ONE full-budget E3-style distillation (teacher pseudo-views, features+SH channel — machinery exists) to prepare a better base, then transport on top. Rationale: the +0.04..+0.13 distillation channel and the +1.33 transport channel have never been composed; residuals shrink on a better base. Predict composite +0.05..+0.15 over E0.
- **L2 — K-source occlusion-aware aggregation (classical, high reliability).** Replace single/naive source use with nearest-K train views (pre-register K ∈ {2,4,8}, choose ONCE on train-side leave-k-out, freeze), mesh-depth soft z-tests for occlusion, distance/angle-weighted contributions, Laplacian multi-band blending for seams. Predict +0.15..+0.40 over the incumbent; this is standard ULR-class engineering with decades of precedent.
- **L3 — LEARNED per-pixel fusion (the method core; the Deep-Blending move).** A small U-Net takes {base render, K warped sources or residuals, per-pixel confidence features} → per-pixel blend weights (and optionally a bounded correction). Confidence inputs come from the EXISTING evidence machinery: support counts, residual statistics, warp agreement, TNC features — the ρ≈0.7 evidence-vs-error result is the banked justification that these features carry signal. Training: per scene, leave-k-out over TRAIN views only (`teacher_factory` / source-heldout protocol exists); frozen at test. Predict +0.20..+0.50 over L2. Two mechanisms max (architecture/inputs count as one axis; loss form as the other).
- **L4 — Residual-vs-RGB routing.** Let L3's head also choose, per pixel, between transporting full source RGB (where confidence high and occlusion clean) vs residual correction (elsewhere). Folded into L3's second mechanism if not free.
- **L5 — Cache Pareto (the honest storage axis).** Compress the cache (JPEG quality sweep at 3 pre-registered points, resolution halving, K-subset selection by train-side coverage) → quality-vs-TOTAL-artifact-MB curve. Then RERUN the R1 comparison honestly: 3DGS at matched TOTAL storage (checkpoint + cache) on the same 3 scenes. Report whatever it says.
- **L6 (T2) — Compact variant.** Full incumbent stack on the B5@B50 base → the "half the triangles AND above Phase-J" row, tying Stage-4 back to the compaction result.

Ladder bookkeeping: `LEDGER` gets a STACK BOARD — incumbent composition, per-rung deltas with CIs, cumulative delta vs PJ-2026 and vs both anchors. The final shipped stack is the incumbent at ladder close.

---

## §4. BASELINES, COMPARISONS, AND THE NOVELTY DEFENSE (mandatory; this is where Stage 4 lives or dies at review)

- **Internal:** PJ-2026 (floor); L2-heuristic stack (the "no learning" ablation-baseline); per-rung ablations (K, confidence inputs on/off, residual-vs-RGB, cache size, learned-vs-heuristic weights).
- **External (bounded cells):** (a) ONE per-scene generative enhancer comparison point — a Difix3D+-style single-step fixer, IF the existing `Difix` env supports inference within a 2-day cell; else file INFEASIBLE with the attempt log. (b) 3DGS at matched TOTAL storage (§3 L5). (c) OPTIONAL T3: a classical ULR/Deep-Blending reimplementation point if L2 doesn't already serve as it (it usually does — justify either way).
- **RED-TEAM additions (write into `REBUTTAL_BANK.md` §Stage-4):**
  - **A10 "this is ULR / Deep Blending 2.0":** answer = mesh-splat base with a retained mesh artifact; evidence-CERTIFIED confidence inputs with a measured error-prediction law (ρ≈0.7) and a measured failure boundary (coverage gaps, 5–11× tails); strict train-only guarantees enforced by an audit tool; honest TOTAL-storage accounting; and a falsification corpus proving the baked alternative is information-theoretically closed (cos 0.21, 4.76%, 5–16%). The negatives ARE the motivation section.
  - **A11 "why not 3DGS + an enhancer":** answered by measured rows (§4 external cells), plus the mesh-artifact and preservation-exactness story.
  - **A12 "per-scene learned fusion doesn't generalize":** correct — the claim is per-scene by design (NON-CLAIM); cross-scene training is future work (T3 note only).

---

## §5. EXPERIMENT SET (Stage-4 matrix; append to MATRIX.md under a new `ECR` section)

Tier-1: E0 floor rows (all suites, both bases) · ladder rungs with gates · final-stack full9 + S-GEO tables vs three references · per-scene win/loss + CIs · efficiency table (FPS w/ transport, cache MB, transport ms) · cache Pareto + matched-TOTAL-storage 3DGS rerun · ablation table · qualitative grids under the existing crop rule (best/median/failure) incl. transport-confidence visualizations · E9-style failure cases for the transport (occlusion errors, seam cases, coverage-gap views) · statistics discipline throughout.
Tier-2: L6 compact variant · flythrough video with the final stack · Difix-style comparison cell.
Downstream note (write once, don't re-run): transport does not alter geometry — g/d metrics and the R3/impossibility results are unchanged by construction; the ECR paper cites them as scope, not new work.

## §6. OPTIONAL, HUMAN-DELETABLE: DS-1 dense-carve retry (T2; does NOT gate the mainline)

Sign-off: R3-FINAL's permanent closure is narrowly reopened for ONE mechanism, justified by the audit's diagnosis (P1 was MET — FREE-set false-free 9.75%; the failure was UNKNOWN starvation from stride-16 sampling plus UNKNOWN-as-obstacle semantics, not map unsafety). Mechanism: ray stride 16→2 (or full), FREE-region dilation by r_inf, planner semantics UNKNOWN = traversable at high cost (never free). PASS = the original frozen fix-target (courtyard ≥30/100 at ≤3.0 coll/100). ONE variant, hard kill, verdict either way updates `CONSUMPTION_IMPOSSIBILITY.md` (strengthened or amended). If the human deletes this section, nothing else changes.

## §7. CLAIMS (create `CLAIMS_ECR.md`; instantiate from evidence only)

- **CR1 — Quality:** the ECR stack renders full9 at [measured] dB above the PRIMARY anchor and [measured] above PJ-2026 (CIs), with per-scene wins [n]/9.
- **CR2 — Honest cost:** at matched TOTAL artifact storage, ECR vs 3DGS = [measured]; the cache-quality Pareto is [curve].
- **CR3 — Mesh retained:** the compact variant delivers [measured] above Phase-J at 50% triangles; geometry/downstream metrics preserved per the frozen Stage-2/3 results.
- **CR4 — Certified transport:** confidence inputs predict transport benefit (train-side analysis), with train-only guarantees audited per row.
- **NON-CLAIMS:** per-scene method; no cross-scene generalization claim; no test-GT anywhere; no claim that ECR fixes geometry or planning consumability (cite the impossibility result as scope); rendering comparisons name their references explicitly.

## §8. TIMELINE AND VENUE GATES

- Week 1–2: E0 floor + audits + mouth extension. Week 2–3: L1+L2. **3DV go/no-go at end of week 3:** if incumbent ≥ PJ-2026 + 0.15 (CI) and pack machinery is folding cleanly → sprint for the 3DV 2027 deadline (≈ Aug 28; human verifies the exact date); else default to the CVPR 2027 schedule (≈ mid-Nov): weeks 3–6 L3/L4, weeks 6–8 L5/L6 + external cells, weeks 8–10 pack + handoff refresh.
- Either way, refresh `SUBMISSION_HANDOFF/` (venue memo rerun with the positive-result branch; rebuttal bank A10–A12; abstract skeleton re-slotted with ECR numbers).

## §9. BOUNDARIES (the dead list persists, with one lift)

LIFTED: the test-time-evidence prohibition (per §0 sign-off), for the ECR track only.
STILL DEAD, verbatim: test-GT influence of any kind; per-TEST-view arbitration/selection/tuning; metric-mouth forks; gate/threshold tuning presented as progress; static baking and full-parameter fine-tuning as method claims (citable falsifications); E2 loss routes; R3.a/R3.b (citable only); reopening anchors; per-scene hyperparameters. New non-ECR ideas → one line in `FUTURE_WORK.md`, never an experiment.

## §10. /goal MECHANICS AND THE DELIVERY CHECKLIST

Same mandatory template, plus the STACK BOARD in section 5 of every /goal. First goal is fixed: **GOAL #E-00** — mouth extension (`--renderer ecr` + `--ecr` audit mode + PROTOCOL v1.2.0 changelog) with a self-test on one scene, then E0 pre-registration.

**DELIVERY CHECKLIST (all boxes, then stop):**
- [ ] AT-E0 passed; PJ-2026 floor rows banked on all suites with cost columns.
- [ ] Ladder closed; final incumbent > PJ-2026 on full9 PSNR AND LPIPS, CIs excl. 0 (target ≥ +0.15) — or the honest escalation was made instead of a decorated claim.
- [ ] Tier-1 experiment set complete; ablations, failure cases, qualitative, efficiency, cache Pareto, matched-TOTAL-storage comparison all banked.
- [ ] `CLAIMS_ECR.md` instantiated from evidence; RED-TEAM A10–A12 written; audits green on every reported row; pack folded and byte-verified; handoff refreshed.
- [ ] Final line, verbatim: **"Stage 4 closed: ECR final stack = PJ-2026 + [Δ dB / ΔLPIPS] (CIs); deliverable exceeds Phase-J = <YES by margin / ESCALATED>; evidence and handoff ready for writing."**

— END OF STAGE 4 PROMPT —
