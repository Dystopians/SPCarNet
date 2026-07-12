# Stage-4 Paper Plan — "Caching Evidence Beats Baking It"

Written 2026-07-11, immediately after Stage-4 closure. Companion to `docs/stage4_sum.md`.
Rule inherited from the program: **every numeric clause in the paper points to a banked artifact**
(paths given per table/figure below); prose is written around numbers, never the reverse.

---

## 0. The one-sentence thesis

> For mesh-first pipelines, **caching** training evidence and transporting it at render time — through a
> certified, train-only, per-pixel-gated transport — beats **baking** it into the representation: the same
> compact mesh whose baked variants plateaued at Phase-J gains +0.36 dB / −0.017 LPIPS over that
> plateau (CIs excl. 0), and at HALF the triangle budget still exceeds the full-budget baked anchor by
> +1.49 dB — at an honestly reported storage and latency price.

Working titles (pick at draft 2, not before):
1. *Evidence-Cached Rendering: Certified Train-View Transport for Compact Mesh-Splat Scenes*
2. *Cache, Don't Bake: Evidence Transport Beats Evidence Distillation for Mesh Rendering*
3. *ECR: A Certified Per-Scene Evidence Cache that Outperforms Its Baked Baseline at Half the Mesh Budget*

## 1. Contribution list (each maps to a claim + banked evidence)

- **C1 (= CR1).** A per-scene evidence-cached rendering system exceeding a strong frozen transport
  baseline on PSNR AND LPIPS with paired CIs (full9 +0.361/−0.0169 vs PJ-2026; +1.67 vs primary anchor;
  extends to 3/5 held-out suites). Evidence: `RESULTS/STAGE4_ECR/tables/final_stack_tables.md`.
- **C2 (= CR3).** The compact result: half the triangles AND above the full-budget anchor
  (+1.488/−0.0748, 9/9); the transport's margin is base-independent (+0.336 vs Phase-J on B50).
  Evidence: L6 tie-back, same table file.
- **C3 (= CR4 shrunk).** Certified purity + structural safety: a render-time transport whose no-test-GT
  property is proven per row (96/96 audits: read confinement, split disjointness, frozen kwargs), and
  whose worst case is bounded by a STRUCTURAL gate (β·valid), not by learned confidence — demonstrated
  by the conf-input ablation (quality unchanged) and the failure-case census (1 negative view in 139,
  −0.06 dB). Evidence: audit reports, `ecr_failure_cases.md`, E-08 rows.
- **C4 (= CR2 + negatives as method guidance).** The honest trade map: cache-quality Pareto (jpeg95
  ~free; halfres −0.9 dB at ~50% TOTAL), matched-TOTAL-storage 3DGS (still ahead +0.32..+1.53 dB at
  40–70× fps — stated, not hidden), a generative enhancer measured and dominated (Difix3D+ PSNR-negative
  on all 3 scenes), and the distillation/baking negative corpus that MOTIVATES caching
  (L1 subsumption; Stage-1/2/3: baking 4.76%, distillation 5–16%, residual cosine ≈ 0.21).
  Evidence: `l5_pareto.md`, `e07_matched_total_3dgs.md`, `difix_table.md`, `l1_gate.json`.

Framing note: C3 is what elevates this above "yet another ULR/Deep-Blending variant" — the certified-purity
audit machinery and the structural-gate result are the parts the lineage papers do not have. See §7 (A10).

## 2. Venue decision + timeline

Primary target: **3DV 2027** (deadline ~Aug 25–28 2026 — HUMAN-VERIFY before planning backwards).
Backup: **WACV 2027 R2** (Aug–Sep). CVPR 2027 (mid-Nov) is the stretch venue if the mentor wants it;
it buys ~10 extra weeks but demands the sharpest novelty framing (A10 paragraph + F2/F3 visuals carry it).

Backward plan for 3DV (≈6 weeks from now):
- **W1:** skeleton + all tables/figures generated (everything mechanical, §5–6) + Experiments section drafted.
- **W2:** Method section + Preliminaries; F1 pipeline art v1.
- **W3:** Intro + Related Work + Limitations/Scope; full draft v1 to mentor.
- **W4:** revision pass; supplementary assembled (per-scene tables, protocol/audit appendix, extra quals).
- **W5:** polish, internal red-team vs REBUTTAL_BANK; freeze numbers (re-run generators once, diff-check).
- **W6:** buffer + submission mechanics.

## 3. Paper skeleton (8 pages + supp, 3DV/WACV format)

1. **Introduction (0.9 pg).** Hook = the bake-vs-cache dichotomy: three stages of falsified attempts to
   push train evidence INTO the mesh (numbers from the corpus), then the reframe — keep the mesh compact,
   ship the evidence beside it, transport at render time under a purity contract. End with C1–C4 and the
   headline numbers. Write LAST.
2. **Related Work (0.9 pg).** Three threads: (a) image-based rendering lineage — surface light fields,
   ULR, Deep Blending, recent per-scene hybrid systems (position ECR as "ULR 2.0 with certified purity and
   a learned, structurally-gated router"); (b) 3DGS-family + compression (we do NOT compete on
   quality/speed — the R1/E-07 numbers stated up front, disarming the obvious review); (c) enhancer/
   diffusion fixers (Difix3D+ — we measure it). One paragraph on evaluation discipline (pre-registration,
   paired CIs, single mouth) as a differentiator.
3. **Preliminaries (0.5 pg).** The GEMS artifact (mesh-splat checkpoint, B50 compaction), the frozen
   protocol (PROTOCOL v1.2.0: one mouth, 8-bit metric convention, paired bootstrap), and the D4 rights
   statement: train-view evidence is a DECLARED render-time input of the shipped artifact; test GT is
   unreachable by construction.
4. **Method: Evidence-Cached Rendering (2.0 pg).**
   4.1 Evidence cache (what ships: train renders + train GT + median depths + cameras + frozen config;
       sizes; build-time train-LOO calibration of (K, α)).
   4.2 Transport: K-nearest selection (distance+direction score), depth-reprojection warp with soft
       z-test, confidence accumulation; multiband (Laplacian) confidence-weighted fusion.
   4.3 Learned fusion + routing: FusionNet (844k), 12-ch inputs, α (residual gain) + β (direct-RGB route),
       compose = clamp((1−β·valid)(base+α·signal) + β·valid·color); trained per scene on train-LOO,
       3000 steps, last iterate — no selection.
   4.4 Certification: the audit contract (pose-primitive boundary, confined reads vs manifest, frozen
       per-view kwargs hash, checkpoint fingerprint, split disjointness) — 1 figure inset (audit wall).
   Method text discipline: present the LADDER as the design methodology (each mechanism admitted only on
   a CI gate; failed mechanisms reported), which is itself a contribution of practice.
5. **Experiments (2.3 pg).** Tables T1–T4 + figures F2, F4, F5 (see §5–6). Order: main quality (T1) →
   ladder/ablations (T2) → compact L6 (T3 headline row) → cost/storage/external (T4 + F5).
6. **Analysis (0.8 pg).** β/confidence visualizations (F3), failure-case census (F6): 1 negative view in
   139; coverage-gap anatomy; the E-08 structural-gate finding; town06/toy honest boundary cases and what
   they say about coverage as the binding resource.
7. **Limitations & Scope (0.5 pg).** Per-scene (no generalization claim); storage/latency honestly framed
   (the artifact is 0.8–3.5 GB and ~1–9 fps e2e); 3DGS trade stated once more in one sentence; the
   consumption impossibility (now DS-1-strengthened) quoted as the boundary: the same evidence that fails
   as a world model succeeds as photometric evidence — representation-vs-consumption is the paper's
   closing thought.
8. **Conclusion (0.1 pg).** The thesis sentence, past tense.

Supplementary: full per-scene tables (all 14 scenes × 3 references), suite results, protocol details +
PROTOCOL §4E text, audit mechanics + an example audit report, cache format spec, L5 full Pareto table,
Difix cell provenance (incl. the dead-code note — a service to reproducers), additional quals, DS-1 detail.

## 4. Paper tables (generator → paper mapping; NO retyping — extract via script)

| Paper table | Content | Banked source |
|---|---|---|
| T1 main quality | full9 per-scene + stratified means vs {legacy, primary, PJ-2026}; suite block below | `STAGE4_ECR/tables/final_stack_tables.md` (T-ECR-1) |
| T2 ladder + ablations | L1(neg)/L2/L3/L4 per-rung CIs; E-08 conf-off rows; final-vs-floor | T-ECR-3 + `gates/*.json` + `eval/abl_*_confoff_v1` |
| T3 compact (L6) | final-on-B50 vs full-budget anchor AND vs PJ-2026-on-B50 | L6 tie-back section + LEDGER #E-05/L6 numbers |
| T4 cost + external | per-scene TOTAL MB / transport ms / e2e fps; matched-3DGS; Difix rows | T-ECR-2, `e07_matched_total_3dgs.md`, `difix/difix_table.md` |
| S-tables | E0 floor (28 rows), full L5 Pareto, per-scene suite detail | `e0_{primary,b50}_table.md`, `l5_pareto.md` |

Action item (mechanical, delegable): `tools/analysis/paper_tables.py` — emits LaTeX bodies from the
banked json/md so draft revisions never retype a number.

## 5. Paper figures (existing assets vs to-build)

| Fig | Content | Source / to-build |
|---|---|---|
| F1 | ECR pipeline + audit wall (cache → warp → fuse → route; purity boundary as a hard wall; Stage-3 branches as a small "scope" box) | NEW ART (human or vector tool); layout spec in `SUBMISSION_HANDOFF/FIGURE_NOTES.md` Stage-4 section |
| F2 | Qual grids, 2–3 scenes in main paper (bicycle = coverage-gap story, bonsai = routing recovery, garden = flagship), rest in supp | EXISTS: `RESULTS/figures/ecr_qual/*_ecr_qual_grid.png` (+ manifest); regenerate with prettier row labels ("β routing map", "transport confidence") |
| F3 | β/confidence close-ups (poster child: garden DSC07956 β map) | EXISTS: `analysis/quals/<scene>_final/<view>/{beta,conf}.png` |
| F4 | Ladder bar chart with CI whiskers (incl. L1 negative below zero; promotion floor line) | TO BUILD from `gates/*.json` (small matplotlib script) |
| F5 | Storage-quality Pareto scatter: L5 points + uncompressed + L6 + matched-3DGS points + Difix point, per scene or normalized | TO BUILD from `l5_pareto.json` + `e07_*.json` + `difix_*.json` |
| F6 | Failure-case panel: the 1 negative view (occlusion seam) + the coverage-gap archetype, each with base/final/err/conf strips | EXISTS (compose from `analysis/quals/.../` planes per `ecr_failure_cases.md` pointers) |

## 6. Writing order (do experiments first, intro last)

1. Generate F4/F5 + `paper_tables.py` (mechanical; delegate to Codex with exact specs; verify outputs
   against source jsons before use — same division-of-labor rule as the program).
2. Draft **Experiments** while the tables are fresh — this locks which numbers appear and kills scope creep.
3. Draft **Method** against the actual frozen configs (lift from LEDGER pre-registrations — they are
   already precise method prose).
4. Draft **Analysis**, **Limitations** (the honest paragraphs are pre-written in CLAIMS/NON-CLAIMS and the
   DS-1 addendum — adapt, don't invent).
5. Draft **Related Work** with the A10–A12 rebuttals open in a side window: every positioning sentence
   should pre-answer one of them.
6. Draft **Intro** last, from the finished experiments; abstract from `ABSTRACT_SKELETON.md` Stage-4
   re-slot (8 slots already filled with numbers).
7. Red-team pass: read the draft against `REBUTTAL_BANK.md` A1–A12 and the NON-CLAIMS; strike any
   sentence that overclaims (language rules: "preserves/bounds/exposes/falsifies"; never
   "solves/safe/guarantees/SOTA/robust").
8. Number-freeze pass: re-run every generator once; `diff` against the copies quoted in the draft.

## 7. Pre-empted reviews (write these INTO the paper, not just the rebuttal)

- **A10 "this is just ULR/Deep Blending":** yes — deliberately in that lineage; the deltas are
  (i) certified train-only purity per row (no prior system proves it), (ii) the structural β·valid gate
  with a measured 1-in-139 worst case, (iii) the CI-gated ladder methodology, (iv) the mesh artifact +
  compaction result it composes with (L6). L2 IS the classical point — and the ladder shows +0.26 dB of
  headroom above it (v3 − v1 ≈ +0.264/-0.0122).
- **A11 "why not 3DGS + enhancer":** measured, twice — matched-TOTAL 3DGS reported plainly (we lose
  quality/speed, win mesh-artifact axes); Difix3D+ on our base is PSNR-negative and dominated by ECR.
- **A12 "per-scene method":** by design; declared in NON-CLAIMS; per-scene is the deployment model of the
  artifact class (a scene ships with its cache).
- **"Storage is huge":** the Pareto is the answer (jpeg95 ~free, halfres −0.9 dB at ~50%), plus
  lossless-compressed sizes in every row; never hide TOTAL.
- **"Test-time use of training images is cheating":** the D4 rights paragraph + audit wall figure +
  the reference-class argument (ULR/DB/light fields all do this — we are the ones who PROVE no test leak).

## 8. Guardrails (carry-over program law)

- All claims trace to `CLAIMS_ECR.md` v0.5 — if a sentence isn't covered by CR1–CR4 or NON-CLAIMS, cut it
  or file a claim edit with evidence.
- Three references named in every quality statement (legacy / primary / PJ-2026); no unlabeled deltas.
- Negatives are content, not confessions: L1, E-08, town06/toy, DS-1 each get affirmative sentences.
- Any new number needed during writing → generate through the existing mouth/tools, bank it, then quote.

## 9. Division of labor

- **Human (mentor/authors):** venue confirmation + deadline verify; F1 art direction; final prose voice;
  camera-ready decisions.
- **Claude (next sessions):** section drafts in the order of §6; `paper_tables.py` spec + verification;
  F4/F5 plot specs + verification; red-team + number-freeze passes.
- **Codex (delegated, tightly specced):** LaTeX table generator implementation, F4/F5 matplotlib scripts,
  qual-grid label polish, supplementary assembly mechanics — all verified against banked sources before use.

---

## Addendum (2026-07-12): the editing section (Route A, GOAL #E-15)

Insert as §6.5 "Editing the scene without losing the evidence" (~0.75 pg): the two-failure-modes
evidence matrix (rebuild inversion + depth-consistency asymmetry), the one-mechanism fix (masks at the
single warp site), preservation + cost numbers, and F-E7. This section IS the "why retain the mesh"
answer (face identity → exact invalidation) — position it as completing the thesis, claims bounded per
CR5 (deletion + recolor only). Verdict provenance: docs/EDIT_AWARE_ECR_VALUE_REPORT.md.
