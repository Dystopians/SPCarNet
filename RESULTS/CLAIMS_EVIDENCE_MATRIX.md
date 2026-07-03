# GEMS — CLAIMS_EVIDENCE_MATRIX.md

Generated 2026-07-03 (Stage-2 §8 evidence-pack assembly). Maps every frozen
claim in `CLAIMS.md` (v1.2, incl. edits of 2026-07-03) to the exact tables /
figures / LEDGER goals / durable artifacts that support or bound it, and flags
(per §10.4) every claim component lacking evidence and every strong result
lacking a claim. No numbers are asserted here beyond quoting claim text; all
evidence numbers live in the script-generated tables under
`RESULTS/aggregate/` (source: the metrics.json corpus at
`/data/peilincai/gems_stage1/eval/`).

Anchor vocabulary: **B0** = clean@30k (legacy/deployed default); **B0′** =
clean-fixed@30k (PRIMARY anchor, LEDGER GOAL#R-01); B0-26k = 26k context row.

---

## C1′ — Compactness–quality (anchored re-instantiation, CLAIMS v1.1)

| Claim component | Evidence | Status |
|---|---|---|
| B50 iso-or-better vs B0 (deployed default) on 8/9 S-REND scenes; LPIPS better 9/9 at half triangles | `T1_main_pareto.md` (S-REND/B50/B5 row: iso-floor + w/i/l columns), `T2_rendering.md`, `F2_pareto_srend_{psnr,lpips}`; LEDGER GOAL#009 | **SUPPORTED** |
| B50 vs B0′ (primary anchor): iso-or-better 4/9, garden strictly above | `T1_main_pareto.md` ("iso-floor pass (vs B0′)" column), `T1_per_scene_detail.csv` (per-scene CIs); LEDGER GOAL#R-01 | **SUPPORTED (bounded, anchor named)** |
| ≥50% triangle reduction at iso-quality on real scenes | same T1/T2 rows + `T4_efficiency.md` (tri/disk/VRAM/FPS halved) | **SUPPORTED vs B0; bounded vs B0′** |
| Dominates **random pruning** | `T6_ablations.md` (importance-family block), `F7_ablations` (B12.5: 9/9 CIs excl. 0), T1 B2 rows at B50/B25 (garden/toy/courtyard); LEDGER GOAL#011; NON-CLAIM covers courtyard-B50-with-FT (CI incl. 0) | **SUPPORTED in the aggressive-budget regime; NON-CLAIM at moderate budget + safe FT** |
| Dominates **QEM-style decimation (B3)** | — none — B3 was never run (MATRIX E1: "remaining: B2/B3 columns") | **⚠ EVIDENCE GAP: claim component currently UNSUPPORTED. Action: run B3 or shrink C1′ wording to "random pruning and prune-without-FT".** |
| Budget-regime framing (B25 floor 3/9; B12.5 0/9 with graceful degradation; FT value grows as budget shrinks, B5>B4 9/9) | T1 B25/B12.5 rows; LEDGER GOAL#R-00 (B25 verdict), #010, #011 | **SUPPORTED** |
| toy_parking B50 residual (known bound, selection effect) | T1/T2 S-DEV rows; LEDGER GOAL#005/#008; E9 taxonomy case (family A) | **BOUND documented** |

Also missing (context rows promised by MATRIX, not claim-bearing): **B1 no-op**
(sanity), **H1** v106 historical, **R1** 3DGS+compression reference — all
still TODO in `MATRIX.md`; flagged here so the gap is explicit.

## C2 — Geometry reliability (DEMOTED to measurement claim, CLAIMS v1.0)

| Claim component | Evidence | Status |
|---|---|---|
| Metric suite is valid (GT-calibration row ≈ perfect) | `T3_geometry.md` GT-CAL row; LEDGER GOAL#004 | **SUPPORTED** |
| Photometric quality masks geometric unreliability (toy/courtyard/SS3DM absolutes) | `T3_geometry.md` B0 rows, `T5_downstream.md` d1 false-free rows; LEDGER GOAL#004 (toy), #005 (courtyard), #R-08 (SS3DM measurements) | **SUPPORTED** |
| Compaction preserves g/d metrics at half budget | LEDGER GOAL#R-08 (pre-registered, 20/20 CI arms preserved) + durable `analysis/r08_sgeo/{r08_table.md,r08_summary.json}`; T3/T5 paired rows | **SUPPORTED** |
| NON-claim: GEMS does NOT improve geometry vs clean | honored everywhere; see also F1 caption caveat (garden g3 fragmentation RISES — E9 family E) and NEGATIVE_RESULTS.md §E2 | **RESPECTED** |

## C3′ — Rendering (anchor-named, CLAIMS v1.1; teacher part DEMOTED)

| Claim component | Evidence | Status |
|---|---|---|
| B≥50: within −0.10 dB of clean@30k on 8/9 S-REND | `T1_main_pareto.md` iso-floor column, `T2_rendering.md` per-scene CIs | **SUPPORTED (vs B0; flowers is the 9th, in NEGATIVE_RESULTS + E9)** |
| Same claim vs B0′ holds 4/9 — claim text names its anchor | T1 "vs B0′" columns; LEDGER GOAL#R-01 | **SUPPORTED as bounded/re-worded** |
| Teacher distillation = diagnostic channel only (real, sub-floor, view-conditioning-capped) | `T6_ablations.md` teacher block (6 distill−control CIs), `F7_ablations`; LEDGER GOAL#007 (E3 sunset); eval rows `e3*` | **SUPPORTED as demoted diagnostic; no headline claim** |
| End-phase decline: features-only end-phase improves baseline 9/9, yet 26k > fixed-30k 9/9 (trainer property, NON-CLAIM) | T2 B0-26k and B0′ rows; LEDGER GOAL#R-01 | **SUPPORTED, kept as NON-CLAIM** |

## C4′ — Downstream (three components, CLAIMS v1.2)

| Claim component | Evidence | Status |
|---|---|---|
| (1) Preservation-exactness (bit-identical rendered depth; identical certification sets; planner outcomes CI [0,0]; SS3DM 20/20 arms) | `T5_downstream.md` paired rows, `T5b_r3_trilogy.md`; LEDGER GOAL#R-02 (side finding), #R-03 (P1), #R-06 (positive side finding), #R-08; durable `analysis/r3{a,c,b}_*/summary.json`, `analysis/r08_sgeo/` | **SUPPORTED** |
| (2) Consumption-route falsifications with mechanisms (TSDF grazing-ray bias; raw-grid spurious infeasibility; certification sheds load-bearing surface; frozen fix-target on courtyard) | `T5b_r3_trilogy.md` (verdicts quoted verbatim from summary.json); NEGATIVE_RESULTS.md §§8–10; LEDGER GOAL#R-02/#R-03/#R-06 | **SUPPORTED (citable negatives)** |
| (3) Bounded geometry positive B6R (courtyard opacity-release; single real scene; joint bar FAIL stands) | T1 S-DEV B6R row, `T3_geometry.md`/`T5_downstream.md` B6R rows, `T6_ablations.md` geometry-mechanism block; LEDGER GOAL#R-04; eval rows `e2r_*` | **SUPPORTED as bounded (scene-scoped, not bar-clearing)** |
| NON-CLAIM: no tested one-time consumption route supports parking-grade closed-loop planning; blocker is baseline checkpoint geometry | `T5b_r3_trilogy.md` + LEDGER GOAL#R-06 consumability conclusion | **RESPECTED** |

## NON-CLAIMS (must appear verbatim in report/paper — from CLAIMS.md)

> This is a per-scene optimization setting; the teacher is train-only and
> absent at test time; no claim of state-of-the-art novel-view quality versus
> the 3DGS family; no claim about high-speed driving; downstream results are
> proxies unless the closed-loop stretch item was executed. Additionally
> (Stage-One additions): no claim that GEMS improves geometry or downstream
> metrics vs clean; no claim that evidence-guided importance dominates random
> pruning under safe fine-tuning at moderate budgets on small scenes; the
> end-phase-decline finding (clean26k > clean30k) is reported as a property of
> the baseline trainer, not a GEMS contribution. (v1.2 addition:) no tested
> one-time occupancy-consumption route currently supports parking-grade
> closed-loop planning from these checkpoints.

Where honored in the pack: T1/T2 dual anchors + explicit anchor naming; T6
teacher block labeled DEMOTED; T5/T5b preservation-only language; F1 caption
carries the geometry caveat; T7 placeholder admits robustness cells are
pending.

---

## §10.4 audit

**Claims lacking evidence (must be fixed by experiment or claim-shrink):**
1. C1′ "Pareto-dominates QEM-style decimation" — **B3 never run** (Tier-1
   MATRIX cell open). Until B3 lands, the wording must be read as bounded to
   the baselines actually run (B2, B4).
2. C1′/C3′ at B≥50 implicitly cover S-GEO via GOAL#R-05/#R-08 (B50 3/4 towns);
   town06 is the documented failure — bounded, not silent.
3. E1-PARETO S-GEO B2 column and toy variants (D-2) not run — the S-GEO Pareto
   rests on B4/B5 only.

**Strong evidence lacking a claim (stated home or reason for exclusion):**
1. B12.5 importance-vs-random dominance (9/9 CIs, large margins — T6/F7,
   GOAL#011): lives inside C1′'s budget-regime framing.
2. Bit-identical-depth / certification invariance (GOAL#R-02/#R-06): folded
   into C4′(1) preservation-exactness.
3. Features-only end-phase repair beats clean@30k on 9/9 (GOAL#R-01): kept as
   a baseline-trainer recommendation NON-CLAIM by pre-registered policy.
4. Efficiency gains (FPS/disk/VRAM roughly track budget, T4 incl. the second-
   resolution bench): cost side of C1′; no standalone efficiency claim made.
5. E9 failure taxonomy (13 cases, 5 families; GOAL#R-09): analysis product
   feeding limitations; not a claim.
