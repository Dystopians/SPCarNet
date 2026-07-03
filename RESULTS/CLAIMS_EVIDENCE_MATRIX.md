# GEMS — CLAIMS_EVIDENCE_MATRIX.md

Regenerated 2026-07-03 (Stage-2 §8 final assembly, GOAL#018; supersedes the
pack-v1 matrix). Maps every frozen claim in `CLAIMS.md` (v1.2, incl. edits of
2026-07-03) to the exact tables / figures / LEDGER goals / durable artifacts
that support or bound it, and flags (per §10.4) every claim component lacking
evidence and every strong result lacking a claim. No numbers are asserted
here beyond quoting claim text / LEDGER verdicts; all evidence numbers live
in the script-generated tables under `RESULTS/aggregate/` (source: the
metrics.json corpus at `/data/peilincai/gems_stage1/eval/`, 217 rows).

Anchor vocabulary: **B0** = clean@30k (legacy/deployed default); **B0′** =
clean-fixed@30k (PRIMARY anchor, LEDGER GOAL#R-01); B0-26k = 26k context row.

---

## C1′ — Compactness–quality (anchored re-instantiation, CLAIMS v1.1)

| Claim component | Evidence | Status |
|---|---|---|
| B50 iso-or-better vs B0 (deployed default) on 8/9 S-REND scenes; LPIPS better 9/9 at half triangles | `T1_main_pareto.md` (S-REND/B50/B5 row: iso-floor + w/i/l columns), `T2_rendering.md`, `F2_pareto_srend_{psnr,lpips}`; LEDGER GOAL#009 | **SUPPORTED** |
| B50 vs B0′ (primary anchor): iso-or-better 4/9, garden strictly above | `T1_main_pareto.md` ("iso-floor pass (vs B0′)" column), `T1_per_scene_detail.csv` (per-scene CIs); LEDGER GOAL#R-01 | **SUPPORTED (bounded, anchor named)** |
| ≥50% triangle reduction at iso-quality on real scenes | same T1/T2 rows + `T4_efficiency.md` (tri/disk/VRAM/FPS halved) | **SUPPORTED vs B0; bounded vs B0′** |
| Dominates **random pruning** | `T6_ablations.md` (importance-family block), `F7_ablations` (B12.5: +3.6..+8.2 dB, 9/9 CIs excl. 0), T1 B2 rows; LEDGER GOAL#011; NON-CLAIM covers courtyard-B50-with-FT (CI incl. 0) | **SUPPORTED in the aggressive-budget regime; NON-CLAIM at moderate budget + safe FT** |
| Dominates **QEM-style decimation (B3)** | T1/T2 B3 rows @B50, `F2` B3 series, `F7` B3-vs-B5 block; LEDGER GOAL#013 **DONE-PASS**: B3 < B5 in mean 3/3 scenes (garden −3.394 CI[−3.62,−3.15], toy −2.723 CI[−3.52,−1.94]; ≥0.10 dB margin CI-certified 2/3 — courtyard's 5-view PSNR CI is wide but its LPIPS margin +0.0615 IS CI-certified); mechanism: QEM moves appearance carriers, features-FT cannot repaint; `analysis/b3_qem/{b3_table.md,b3_summary.json}`; audit GREEN `eval/b3_audit_garden` | **SUPPORTED (the last naked C1′ arm is dressed; scope: B50 × 3 dev scenes)** |
| Importance-DEFINITION robustness (which evidence column) | `T6_ablations.md` importance-definition block, `F7`; LEDGER GOAL#012: axis flat (all pairwise \|dPSNR\| ≤ 0.052 dB, 40× below the revision trigger); pixels_total stands; measured pipeline noise floor 1.6e-5 dB; `analysis/e6_abl/e6_table.md` | **SUPPORTED as robustness evidence: *having* render evidence is load-bearing, the column choice is not** |
| Budget-regime framing (B25 floor 3/9; B12.5 0/9 with graceful degradation; FT value grows as budget shrinks, B5>B4 9/9) | T1 B25/B12.5 rows; LEDGER GOAL#R-00 (B25 verdict), #010, #011 | **SUPPORTED** |
| toy_parking B50 residual (known bound, selection effect) — now a FAMILY property | T1/T2 S-DEV rows incl. toy_parking_v2/occl; LEDGER GOAL#005/#008/#016 (D-2: P1 in-range on v2, occl misses by 0.053 dB via one −6.21 dB tail view — the iso bound formally excludes occlusion-heavy toy scenes); E9 family A | **BOUND documented + replicated** |

Context rows (not claim-bearing): **H1** v106 historical and **R1**
3DGS+compression — both RECORDED as clearly-marked context appendices in
T1/T4 (LEDGER GOAL#013/#017; `analysis/h1_v106_context/`,
`analysis/r1_3dgs_reference/`). **B1 no-op was never run** (sanity cell;
honest open gap, flagged in T1's header — the near-lossless B4-vs-B0
prune-only deltas at B50 bound the pipeline's pass-through fidelity
indirectly, but that is not a substitute row).

## C2 — Geometry reliability (DEMOTED to measurement claim, CLAIMS v1.0)

| Claim component | Evidence | Status |
|---|---|---|
| Metric suite is valid (GT-calibration row ≈ perfect) | `T3_geometry.md` GT-CAL row; LEDGER GOAL#004 | **SUPPORTED** |
| Photometric quality masks geometric unreliability — now a FAMILY property, three suites | `T3_geometry.md` B0 rows, `T5_downstream.md` d1 false-free rows, `F6` panel (a); LEDGER GOAL#004 (toy), #005 (courtyard), #R-08 (SS3DM), #016 (D-2: clean d1 false-free 58.5/59.3% on both variants — replication 2/2, P2 met) | **SUPPORTED + replicated** |
| Compaction preserves g/d metrics at half budget | LEDGER GOAL#R-08 (pre-registered, 20/20 CI arms preserved) + `analysis/r08_sgeo/{r08_table.md,r08_summary.json}`; T3/T5 paired rows; D-2 variants preserve the profile (GOAL#016 table) | **SUPPORTED** |
| Evidence-vs-error analysis (the v3xx machinery's honest paper role, §7.5 ANALYSIS framing) | LEDGER GOAL#015-A: train residual_view_mean predicts per-triangle test error, Spearman ρ = +0.686/+0.729/+0.736 (garden/kitchen/town01, P-A1 3/3); selection-effect tail quantified (test/train error ratio 1.42→1.03 with coverage; never-train-visible triangles have the HIGHEST test error 3/3); coverage-column global-ρ arm REFUTED (recorded); `analysis/e2geo_evidence_vs_error/{summary.json,plots/}` | **SUPPORTED as analysis (explicitly not method); the honest bound on any evidence-as-certificate reading** |
| NON-claim: GEMS does NOT improve geometry vs clean | honored everywhere; F1 caption caveat (garden g3 fragmentation RISES — E9 family E); g3 fragmentation 1.9–3.1× on all 4 towns (GOAL#R-08 surprise 1); NEGATIVE_RESULTS.md §§3–6 | **RESPECTED** |

## C3′ — Rendering (anchor-named, CLAIMS v1.1; teacher part DEMOTED)

| Claim component | Evidence | Status |
|---|---|---|
| B≥50: within −0.10 dB of clean@30k on 8/9 S-REND | `T1_main_pareto.md` iso-floor column, `T2_rendering.md` per-scene CIs | **SUPPORTED (vs B0; flowers is the 9th, in NEGATIVE_RESULTS + E9 case 1)** |
| Same claim vs B0′ holds 4/9 — claim text names its anchor | T1 "vs B0′" columns; LEDGER GOAL#R-01 | **SUPPORTED as bounded/re-worded** |
| Teacher distillation = diagnostic channel only (real, sub-floor, view-conditioning-capped ~5–16%) | `T6_ablations.md` teacher block (6 distill−control CIs), `F7_ablations`; LEDGER GOAL#007 (E3 sunset); eval rows `e3*` | **SUPPORTED as demoted diagnostic; no headline claim** |
| End-phase decline: features-only end-phase improves baseline 9/9, yet 26k > fixed-30k 9/9 (trainer property, NON-CLAIM) | T2 B0-26k and B0′ rows; LEDGER GOAL#R-01; NEGATIVE_RESULTS §12 | **SUPPORTED, kept as NON-CLAIM** |

## C4′ — Downstream (three components, CLAIMS v1.2)

| Claim component | Evidence | Status |
|---|---|---|
| (1) Preservation-exactness (bit-identical rendered depth; identical certification sets; planner outcomes CI [0,0]; SS3DM 20/20 arms) — now at N=500 | `T5_downstream.md` paired rows, `T5b_r3_trilogy.md`, `F6`; LEDGER GOAL#R-02/#R-03/#R-06/#R-08 + **GOAL#015-B: N=500 outcome-level EXACT (identical found/collision sets, CIs [0,0]; sole deviation = 2/500 label-only reason flips on courtyard, no outcome differs)**; `analysis/e5_down_ext/summary.json` (replay-verified vs R3.c first-100) | **SUPPORTED (strongest form measured)** |
| (2) Consumption-route falsifications with mechanisms (TSDF grazing-ray bias; raw-grid spurious infeasibility; certification sheds load-bearing surface; frozen fix-target on courtyard) | `T5b_r3_trilogy.md`, `F6` panels (b)–(d); NEGATIVE_RESULTS §§9–11; LEDGER GOAL#R-02/#R-03/#R-06; **N=500 refinement (GOAL#015-B): route-i spurious infeasibility 88.4%/99.6% (P-B2's frozen ≥90%/=100% letter FALSIFIED at better resolution — substance survives); NEW damning datum: courtyard route-i's only 2/500 plans BOTH hit GT; route-ii 5.93 coll/100 = 3.66× GTREF floor (P-B3's 5× letter falsified, direction holds); ESDF clearance under-estimated 1.0–2.9 m mean everywhere (P-B4 confirmed)** | **SUPPORTED (citable negatives; falsified numeric letters reported as falsified)** |
| (3) Bounded geometry positive B6R (courtyard opacity-release; single real scene; joint bar FAIL stands) | T1 S-DEV/S-GEO B6R rows, T3/T5 B6R rows, T6 geometry-mechanism block incl. **B6R-on-SS3DM (GOAL#014 DONE-FAIL as pre-registered: guard held 3/3, LPIPS/g1/g3-components better CI-excl-0 3/3, but g3-FRACTION arm 0/3 and d1-ff worse CI-excl-0 3/3 — NOT claim-grade, stays courtyard-scoped)**; LEDGER GOAL#R-04/#014; `analysis/b6r_ss3dm/` | **SUPPORTED as bounded (scene-scoped); the generalization test was run and failed honestly** |
| NON-CLAIM: no tested one-time consumption route supports parking-grade closed-loop planning; blocker is baseline checkpoint geometry | `T5b_r3_trilogy.md` + LEDGER GOAL#R-06 consumability conclusion + #015-B N=500 sharpening | **RESPECTED** |

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

Where honored in the pack: T1/T2 dual anchors + explicit anchor naming; the
R1 context appendix measures the 3DGS trade plainly (2.1–3.4 dB, 3.1–4.4×
FPS at matched storage) instead of hiding it; T6 teacher block labeled
DEMOTED; T5/T5b/F6 preservation-only language; F1 caption carries the
geometry caveat; T7 admits robustness cells were not run.

---

## §10.4 audit (regenerated at final assembly)

**Claims lacking evidence:**
1. — none at the claim-text level. Every sentence in CLAIMS.md v1.2 now has
   at least one supporting-or-bounding table/figure/goal above, including the
   formerly naked "QEM-style decimation" arm (B3, GOAL#013).
2. Residual SCOPE caveats (documented, not silent): the B3 column exists
   only @B50 on 3 scenes (margin CI-certified on 2/3; courtyard 5-view CI
   wide); B2 at B50/B25 exists only on garden/toy/courtyard and S-GEO B2 was
   never run — C1′'s "dominates random pruning" therefore rests on the
   B12.5 9/9 result plus the moderate-budget NON-CLAIM, which is exactly how
   CLAIMS.md words it; B1 no-op sanity row never run. These are flagged in
   T1's header and EXPERIMENT_REPORT §7 as human-decision items (complete or
   waive), not hidden.

**Strong evidence lacking a claim (stated home or reason for exclusion):**
1. B12.5 importance-vs-random dominance (+5.23 dB mean, 9/9 CIs — T6/F7,
   GOAL#011): lives inside C1′'s budget-regime framing.
2. Bit-identical-depth / certification invariance / N=500 outcome-exactness
   (GOAL#R-02/#R-06/#015-B): folded into C4′(1) preservation-exactness.
3. Features-only end-phase repair beats clean@30k 9/9 (GOAL#R-01): kept as a
   baseline-trainer recommendation NON-CLAIM by pre-registered policy.
4. Evidence-vs-error correlation ρ≈0.7 + coverage-gap tail (GOAL#015-A): the
   E2-GEO analysis arm inside C2; deliberately NOT a method/certificate claim
   (§7.5 framing guard — the same analysis quantifies why it cannot be one).
5. Efficiency gains (FPS/disk/VRAM track budget, T4 + half-res bench): cost
   side of C1′; no standalone efficiency claim made.
6. E9 failure taxonomy (13 cases, 5 families; GOAL#R-09; F8): analysis
   product feeding limitations; not a claim.
7. B6R-on-SS3DM direction transfer (LPIPS/g1/g3-components better 3/3 CI,
   GOAL#014): excluded from claims BY PRE-REGISTRATION — the magnitude arm
   failed 0/3, so it stays a bounded courtyard-scoped observation in C4′(3).
8. R1 measured trade (GOAL#017): context only; NON-CLAIMS disclaim the axis.
9. D-2 family replication (GOAL#016): folded into C2 (measurement claim) and
   the C1′ toy bound; no new claim minted.

**Verdict on §10.4:** no claim lacks evidence; every strong result has a
named claim-home or a stated exclusion reason. The matrix is consistent with
the post-shrink CLAIMS.md v1.2.
