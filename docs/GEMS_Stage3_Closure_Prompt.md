# GEMS — STAGE 3 CLOSURE PROMPT
## Downstream Consumption Resolution · Waiver Execution · Declaration · Submission Handoff

Version 1.0 · 2026-07-03 · **This prompt EXTENDS `docs/GEMS_Stage2_Prompt.md` and the Stage-1R Insert Directive; it does not replace them.** All Prime Directives D1–D9, the single mouth, effect floors, pre-registration, CI discipline, sunset rules, and the /goal template remain in force verbatim. Log goals as `GOAL #C-xx` (Closure) in `LEDGER.md`. This document carries the HUMAN RULINGS required by EXPERIMENT_REPORT §7 (W1–W6) — execute them as written in §1 unless the human edits them before launch.

Before your first C-goal, re-read: `EXPERIMENT_REPORT.md` §4 (RED-TEAM) + §7 (waivers, §10 table) + §8, `CLAIMS.md` v1.2 (esp. the C4′ trilogy and the frozen fix-target), `MATRIX.md` R3 rows, `DELTA_MEMO_002.md`, `RESULTS/HANDOFF.md`, `LEDGER.md` #R-02/#R-03/#R-06/#015.

---

## §0. MISSION AND DEFINITION OF DONE

Two gaps stand between the current state and program closure:

- **G-A (the downstream application verdict).** The frozen fix-target — **courtyard ≥30/100 problems found feasible AND ≤3.0 collisions/100 simultaneously** — is unmet, and exactly ONE principled consumer class was never tested (§2). The user's final objective explicitly includes a working downstream deliverable; this is its last untested path.
- **G-B (the declaration).** Stage-2 §10 is PASS everywhere except the W1-residue and W2 human rulings (§1 resolves them).

**DONE means ALL of:**
1. R3-FINAL (§2) has a definitive verdict: **PASS** → C4″ instantiated as an application claim; or **FAIL after ≤3 pre-registered mechanisms** → `RESULTS/CONSUMPTION_IMPOSSIBILITY.md` written, strengthening the negative and closing the axis permanently.
2. §1 rulings executed and folded.
3. Evidence pack v4 regenerated, `verify_t1.sh` byte-diff PASS, §10 table shows zero PARTIAL rows.
4. The declaration line issued verbatim: **"Stage Two complete — evidence pack ready for paper writing."**
5. `SUBMISSION_HANDOFF/` delivered (§4).

Both R3-FINAL outcomes are acceptable closures. An unfinished, hedged, or decorated one is not. Honest negatives remain first-class.

---

## §1. HUMAN RULINGS ON WAIVERS (execute as written; the human may edit before launch)

- **W1 (E7-SENS): RUN THE CHEAP SUBSTITUTE, THEN WAIVE THE RESIDUE.** One full re-train pair on garden at seed 1 (clean 30k + B5@B50; ~1 GPU-h per the waive draft). Pre-register: |clean(seed1) − clean(seed0)| and |B5Δ(seed1) − B5Δ(seed0)| both ≤ 0.15 dB → seed-sensitivity bounded; record either way, then the full 3-seed×subset grid is WAIVED with this run + the 1.6e-5 dB repeat floor + the measured seed-pair on record. T7 updated from placeholder to substitute-evidence table.
- **W2 (E8-ROBUST + S-GEN): RUN ONE ARM, THEN WAIVE THE REST.** Garden 50% train-view drop: retrain clean-half + run B5@B50-half (frozen hyperparameters). Pre-register the direction from the selection-effect family: B50 residual vs clean-half worsens relative to full-view garden; report whatever happens. Pose-noise and S-GEN (T&T B0s) are then WAIVED with the §8 limitation note strengthened by this datum.
- **W4 / W4a / W5: GRANTED as drafted** (scope-freeze on E1 breadth; B6.25 and S-GEO B2 already run in #019 — fold, no further breadth).
- **W3: already withdrawn** (videos delivered) — no action.
- **W6 (SS3DM planner cells): SUPERSEDED by §2.4.** They are run ONLY with the R3-FINAL consumer and ONLY if it passes. The three falsified routes are never re-run anywhere.

---

## §2. R3-FINAL — THREE-STATE EVIDENCE-CARVED OCCUPANCY (the last untested consumer class)

### 2.0 Sanction and framing guards
This is a ONE-TIME, GLOBAL, TRAIN-EVIDENCE-ONLY map-building step (D4-pure): inputs are the checkpoint, train-view cameras, and the model's own rendered median depth at train views. It is NOT a selector: no per-view or per-query logic exists at consumption time; the artifact is a static three-state voxel map per (scene, model). All thresholds are calibrated ONCE on `toy_parking` against its GT, FROZEN, and applied unchanged to courtyard and SS3DM (calibrate-once/test-elsewhere; log the frozen values). Nothing is ever deleted from any checkpoint — states live in the MAP, not the model.

### 2.1 Why this class is distinct from the three falsified routes (internalize before pre-registering)
Each falsified mechanism maps 1:1 onto a design element here: route-i declared every non-surface voxel FREE → coverage gaps became false-free and junk clutter blocked planning (88–100% spurious infeasibility); here, unobserved space is **UNKNOWN**, and junk is **outvoted** by the many free-carving rays that pass through it. Route-ii averaged signed distances → grazing-ray bias; here there is **no averaging**, only occupancy votes. R3.b deleted model content → shed load-bearing walls; here **nothing is deleted**. The ESDF clearance under-estimate (1.3–2.9 m) becomes a **measured inflation calibration input** instead of a silent hazard.

### 2.2 V1 — log-odds visibility carving (pre-register FULLY before code)
- Map: voxel grid at the d1 resolution (plus a planner-resolution copy if they differ), three states {FREE, OCCUPIED, UNKNOWN} from accumulated log-odds.
- Votes, per train view (pixel rays on a declared subsample grid): FREE votes along [camera, 0.95·d_pix] where d_pix = the model's rendered median `surf_depth`; one OCCUPIED vote in the hit cell at d_pix; declared per-vote weights; cells traversed by no ray remain UNKNOWN.
- Thresholds θ_occ / θ_free / v_min (minimum evidence count) calibrated once on toy vs GT occupancy, frozen.
- Planner semantics: traversable = FREE only; OCCUPIED ∪ UNKNOWN = obstacle; inflation radius r_inf calibrated on toy from the measured surface misplacement (g2 ≈ 0.26 m) and the ESDF under-estimate evidence, frozen. Same Hybrid-A*-lite harness, same 100 seed-0 problems (replay-exact), plus the N=500 set where it exists.
- **Pre-registered predictions (write them down, with the banked evidence pointer, BEFORE running):**
  - P1: toy FREE-set false-free ≤ 10% (vs 59% under route-i) — coverage gaps become UNKNOWN.
  - P2: courtyard found ≥ 30/100 at ≤ 3.0 coll/100 — 33 cameras heavily observe the drivable corridor.
  - P3: spurious infeasibility collapses vs route-i (junk outvoted by free rays).
- **PASS bar (the frozen fix-target + transfer):** courtyard ≥30/100 found AND ≤3.0 coll/100 SIMULTANEOUSLY; toy found ≥ 0.5× the GTREF found-rate at ≤3.0 coll/100; median path-length inflation ≤ 1.5× GTREF. Report per cell: found/100, coll/100, inflation, map state fractions (FREE/OCC/UNKNOWN), and a d1-style confusion of the FREE set vs GT.

### 2.3 V2 / V3 (only on near-miss; ≤3 mechanisms TOTAL for R3-FINAL; each pre-registered with its own kill condition)
- V2: support-weighted OCCUPIED votes — down-weight hits whose source triangles have train-support ≤ k (via `rend_ids`; junk suppression at the vote level), and/or hit-cell dilation by the measured g2.
- V3: conservative hybrid — OCCUPIED = route-i surface voxels confirmed by ≥ m carving hits; FREE = carved-free; everything else UNKNOWN.
- **Kill:** if after 3 mechanisms the fix-target is unmet → write `RESULTS/CONSUMPTION_IMPOSSIBILITY.md`: *no one-time train-evidence consumer among {surface voxelization, TSDF fusion, certified sub-mesh, visibility-carved three-state (×3 variants)} meets parking-grade bars on these checkpoints; the quantified blocker is baseline checkpoint geometry plus train-coverage limits (invariant to compaction).* Update T5b, F6, RED-TEAM A4, NON-CLAIMS. The axis is then closed PERMANENTLY. This outcome is a valid, citable closure — do not soften it and do not continue past it.

### 2.4 If PASS (execute all, in order)
1. Run the map+planner cells on ≥2 feasible SS3DM towns (Town06 stays INFEASIBLE per #R-08) and the toy variants; report the same per-cell metrics.
2. Verify compaction invariance under the new consumer: clean ↔ B5@B50 maps and planner outcomes (predict EXACT, given bit-identical rendered depth; report CIs).
3. Instantiate **C4″** via the claim-edit rule, dated, evidence-pointed: *a visibility-aware three-state consumer meets the frozen parking-grade bar on X/Y scenes; compaction remains outcome-invariant under it.* NON-CLAIMS updated (the "no tested route" sentence is replaced by the bounded positive with its scene scope).
4. `parking_phone_tiny_anonymized` qualitative demo: build the map, plan 3 maneuvers, render one figure + one short clip (no GT mesh → qualitative only; say so in the caption).
5. Regenerate T5/T5b/F6; final-answer RED-TEAM A4 with the positive.

### 2.5 Scope and cost guards
No retraining anywhere in §2 (W1/W2 runs are §1's, not §2's). Map building is render+CPU work, hours per scene; storage preflight per run; all maps and planner logs durable under the DEC-007 root; every number through declared analysis scripts (the planner harness is the existing single-harness — extend, don't fork).

---

## §3. FINALIZATION AND DECLARATION

1. Fold §1 and §2 outputs into the corpus; regenerate T1–T7, F1–F8, `CLAIMS_EVIDENCE_MATRIX.md`, `NEGATIVE_RESULTS.md` (add any new negatives, including a failed R3-FINAL), `EXPERIMENT_REPORT.md` §7 table.
2. Pack v4: `verify_t1.sh` from-scratch byte-diff PASS; purity audit on every new artifact (the carving tool must be audited like any consumer: train-only inputs, no test poses).
3. §10 re-verification: every row PASS, zero PARTIAL. Then — and only then — output the declaration line verbatim: **"Stage Two complete — evidence pack ready for paper writing."**
4. Tag the release (`gems-evidence-v1.0`), write `RESULTS/ARCHIVE.md` (release manifest, how to freeze a citable archive, exact commit + environment pins), and record the declaration in `LEDGER.md`.

---

## §4. SUBMISSION HANDOFF (evidence-locked support; narrative prose remains human work by design)

Deliver `SUBMISSION_HANDOFF/` containing:

- **`VENUE_MEMO.md`** — honest fit analysis keyed to the R3-FINAL outcome. Include the deadline table with [HUMAN-VERIFY] tags: 3DV 2027 ≈ Aug 25–28, 2026; ICRA 2027 ≈ Sep 15, 2026; WACV 2027 R2 ≈ Aug–Sep 2026; CVPR 2027 ≈ mid-Nov 2026; NeurIPS 2027 D&B ≈ May 2027. Recommendation logic: R3-FINAL PASS → ICRA (application + analysis) and CVPR become primary; FAIL → 3DV / ICRA-analysis / WACV primary, CVPR high-variance, NeurIPS-D&B fits the measurement-suite framing. For the top-2 choices, produce a claim→section page-budget map.
- **`REBUTTAL_BANK.md`** — expand RED-TEAM A1–A7 into rebuttal-ready paragraphs with exact table/figure/cell pointers; ADD **A8** ("why not 3DGS + meshing, e.g., SuGaR/2DGS-style?" — answer honestly: not compared; R1 covers rendering context only; listed as limitation/future work) and **A9** (seed/robustness — answered by the §1 runs, whatever they showed).
- **`FIGURE_NOTES.md`** — per-figure art direction for the humans; include the Delta-Memo-#002 recommendation to promote the evidence-vs-error curve to main text, and (if §2 passed) a three-state map figure spec.
- **`ABSTRACT_SKELETON.md`** — numbered, claim-safe sentence slots with numbers drawn only from tables; CI-backed language rules apply; no adjectives the evidence does not own. Skeleton, not prose.

---

## §5. BOUNDARIES (extended dead list — verbatim)

Everything in the Stage-1R boundary section, PLUS: R3.a and R3.b are citable, never re-runnable; E2R/B6R is cited as the courtyard-bounded positive, no v2, no new geometry work; teacher stays diagnostic; no selectors / per-view arbitration / gates; no new representation work of any kind; no anchor reopening; no new metric mouths; no new datasets beyond those named here. Any new idea that arises during closure becomes a one-line entry in `FUTURE_WORK.md` — never an experiment. If a proposal touches this list, name this section and pick a compliant action.

---

## §6. /goal MECHANICS AND COMPLETION CHECKLIST

Same mandatory template as Stage 2, with one addition to section 5 (LEDGERS): a **closure board** — the explicit list of remaining blockers to DONE (§0), which must shrink monotonically; if a /goal does not shrink it, justify why it was still the highest priority.

**COMPLETION CHECKLIST (all boxes, then stop):**
- [ ] W1 substitute run + residue waived; W2 arm run + rest waived; W4/W4a/W5 recorded as granted; T7 updated.
- [ ] R3-FINAL verdict: PASS (with §2.4 items 1–5 done) OR IMPOSSIBILITY addendum written; ≤3 mechanisms used; all pre-registrations honored.
- [ ] Pack v4 regenerated; byte-diff PASS; purity audits green incl. the carving tool; §10 all-PASS.
- [ ] Declaration line issued and logged.
- [ ] Release tagged + ARCHIVE.md; SUBMISSION_HANDOFF/ complete (4 documents).
- [ ] Final /goal outputs, verbatim: **"GEMS program closed: declaration issued; downstream verdict = <PASS at bar / IMPOSSIBILITY×4 routes>; submission handoff delivered."**

After that line: stop. No further experiments, no further optimization, no epilogue. The paper is written by the humans from what survives — and everything that survives is honest.

— END OF STAGE 3 CLOSURE PROMPT —
