# Stage 5-A — Metric Movement After Route A + Current Situation Report

Written 2026-07-12, immediately after Route-A (edit-aware ECR) closure. Companion to
`docs/stage4_sum.md` (Stage-4 closure), `docs/stage5_summary.md` (TOPCONF hardening), and the four
`docs/EDIT_AWARE_ECR_*.md` governance docs. Every number is read from banked artifacts
(`analysis/edit_aware/*/edit_eval.{md,json}`, `analysis/edit_aware/garden_temporal/temporal.json`);
statistics are paired per-view bootstrap CIs (10k, seed 0).

---

## 1. The direct answer: what rose, what declined, what stayed exactly the same

### 1.1 The banked Stage-4/5 system metrics: **UNCHANGED — bit-identical, by construction and by test**

Route A added exactly one mechanism to the audited render path: an optional per-source-pixel
evidence-validity mask, sampled at the transport's single warp site and multiplied into confidence.
It is **default-off** (`FrameRecord.mask_path = None` for every pre-edit cache), and before any
experiment ran, the banked garden view DSC07956 was re-rendered through the modified code and
reproduced its banked PSNR **to the last float (22.319719314575195 = bit-stable)**. Therefore:

- full9 final stack vs PJ-2026 (+0.3607 / −0.01686), vs primary anchor (+1.666), L6 compact
  (+1.4877 / −0.07478), suite and T&T/DB rows, all 96/96 audits — **none of these moved**. Route A
  neither improved nor degraded the core system; it added a capability alongside it.

### 1.2 On EDITED scenes — the metrics that ROSE (good) with our method (C5, local invalidation)

| Metric (edited scenes) | Value | Reading |
|---|---|---|
| ECR gain over the edited base, unaffected region (true GT), garden | 26.034 vs 24.389 = **+1.65 dB retained** | the transport's photometric advantage SURVIVES editing |
| Same, toy_parking | 32.238 vs 31.296 = **+0.94 dB retained** | idem |
| Same, garden recolor | 26.046 vs 24.365 = **+1.68 dB retained** | idem |
| Edit-region cleanliness vs the stale cache (garden deletion) | leak 0.0065 vs 0.0088 = **−26% deviation** | principled masking beats the accidental z-test protection |
| Edit-region cleanliness vs stale cache (recolor) | leak 0.0034 vs 0.0346 = **10× cleaner** | the decisive case: depth-consistent staleness has NO accidental protection |
| Edit-region cleanliness vs full rebuild (deletion) | 0.0065 vs 0.0840 (garden), 0.0032 vs 0.0221 (toy) = **13× / 7× cleaner** | naive rebuild is the WORST strategy |
| Update speed vs full rebuild | 108 s vs ~40 min (garden), 34 s vs ~15–20 min (toy) = **~20–30× faster** | no α recalibration, no net retrain, GT untouched |

### 1.3 On EDITED scenes — the metrics that DECLINED (all small, all honestly quantified)

| Metric | Movement | Severity |
|---|---|---|
| Unaffected-region true-GT PSNR, garden deletion (C5 vs original ECR) | 26.054 → 26.034 = **−0.020 dB [−0.034, −0.008]** | statistically detectable, practically negligible; caused by masked-out evidence near the region boundary slightly thinning support |
| Same, toy deletion | 32.240 → 32.238 = −0.002 [−0.004, +0.000] | no change within CI |
| Same, garden recolor | 26.031 → 26.046 = **+0.015 [+0.005, +0.023]** | a small RISE (recolored evidence masked, but regenerated renders slightly cleaner) |
| Temporal roughness ratio, edited garden path | 0.988 (pre-edit) → **0.982** (post-edit) | no decline — masking introduces no flicker; both PASS the ≤1.5 bar |
| Edit-region ghost above the edited-base floor, C5 | +0.059 [+0.038,+0.081] (garden del) / +0.058 [+0.048,+0.069] (recolor) | a small residual similarity-to-original offset, mostly the 8-px context ring + the ground-contact seam; an order of magnitude below the failing controls (+3.09 rebuild, +1.96 stale-recolor) |

### 1.4 The CONTROL metrics that exploded (the point of the experiment)

- **C4 full rebuild, deletion:** ghost **+3.088 dB [+2.568, +3.616]** above the edited base; in-region
  leak 0.0840; a visible translucent table apparition (panels + F-E7 grids). The "obvious fix" fails
  because refreshed depths make the stale PHOTOGRAPHS depth-consistent.
- **C2 stale cache, recolor:** ghost **+1.964 [+1.869, +2.061]**; leak 0.0346 — the old color repaints.
- Toy metric caveat banked: `ghost_psnr_R` is non-monotone when the ghost is blurry (C4 toy reads
  −1.70 wide-CI while its leak is 7× ours) — `leak_R` is the robust cross-scene metric and is reported
  as primary.

## 2. Summary table — the full evidence matrix

| Edit | Method | ghost vs floor (CI) | leak_R | U-PSNR (true GT) | vs ORIG-ECR U |
|---|---|---|---|---|---|
| garden deletion (2,037,550 faces) | C1 edited base | floor | 0 | 24.389 | −1.665 |
| | C2 stale cache | +0.085 [+0.066,+0.105] | 0.0088 | 26.046 | −0.008 |
| | C4 full rebuild | **+3.088 [+2.568,+3.616]** | 0.0840 | 25.803 | −0.251 |
| | **C5 ours** | +0.059 [+0.038,+0.081] | **0.0065** | 26.034 | −0.020 [−0.034,−0.008] |
| toy car_0 deletion (711,609 faces) | C2 stale | (metric non-monotone, see §1.4) | 0.0033 | 32.239 | −0.001 |
| | C4 rebuild | " | 0.0221 | 31.330 | −0.910 |
| | **C5 ours** | " | **0.0032** | 32.238 | −0.002 [−0.004,+0.000] |
| garden recolor | C2 stale | **+1.964 [+1.869,+2.061]** | 0.0346 | 26.057 | +0.026 |
| | **C5 ours** | +0.058 [+0.048,+0.069] | **0.0034** | 26.046 | +0.015 [+0.005,+0.023] |

Update cost (C5): garden deletion 108 s / 1053 MB rewritten; toy 34 s / 231 MB; recolor 117 s /
1036 MB — vs ~15–40 min full rebuild per scene. Honest locality caveat: both objects are centrally
visible, so ALL train views were affected (161/161, 72/72); savings come from bytes-per-view and
skipped recalibration/retraining, not from view sparsity.

## 3. Current situation — the whole project at a glance

1. **Stage 4 (ECR) — CLOSED.** Final stack exceeds PJ-2026 at the stretch bar (+0.361/−0.0169, CIs
   excl. 0); L6 compact +1.488 over the full-budget anchor; delivery checklist all boxes
   (`docs/stage4_sum.md`).
2. **Stage 5 (TOPCONF hardening) — CLOSED at 87/100, TOP-CONFERENCE READY.** Community suite (T&T+DB)
   transfer with CIs; IBRNet head-to-head (ECR +2.6..+5.9 above it); temporal PASS 3/3; scene-cluster
   bootstrap PASS 8/8; threat model + "audited" terminology; plots/tables/README
   (`docs/stage5_summary.md`, `docs/TOPCONF_READINESS_REPORT.md`).
3. **Route A (edit-aware ECR) — CLOSED, CONDITIONAL-GO executed.** Deletion + recolor validated on a
   frozen protocol with two non-obvious findings (rebuild inversion; depth-consistency asymmetry);
   translation deferred with a written path; deformation declared a boundary. Integrated as the ECR
   paper's editing section: CR5 added tightly bounded (CLAIMS v0.9), NON-CLAIMS extended
   (no-arbitrary-editing), LEDGER #E-15 complete, 9 figure grids + abstract slot + paper-plan §6.5,
   35 artifacts byte-verified in `RESULTS/STAGE4_ECR` (`docs/EDIT_AWARE_ECR_VALUE_REPORT.md`).
4. **Claims state:** CLAIMS_ECR v0.9 — CR1 (quality, 3 references + 2 suites + community set),
   CR2 (honest cost + Pareto + matched 3DGS), CR3 (compact), CR4 (audited transport, structural gate),
   CR5 (edit-consistent evidence, deletion+recolor only). All instantiated, none open.
5. **What remains: paper writing only** — per `docs/stage4_paper_plan.md` (+§6.5 addendum): experiments
   section first from the banked tables (`RESULTS/tables_tex/`), method from the LEDGER
   pre-registrations, F1 pipeline art (human), venue deadline verification (3DV primary / WACV backup /
   CVPR stretch), red-team + number-freeze passes.

## 4. One-line answer to the question

**Nothing in the banked system regressed — the pre-edit metrics are bit-identical.** On edited scenes,
the metrics that matter ROSE with our method (the ECR gain survives editing; the edit region is 7–13×
cleaner than any naive strategy at 20–30× lower update cost), and the only measurable DECLINE is a
−0.02 dB true-GT dip outside the garden edit region (toy: none; recolor: a +0.015 rise) — while both
naive alternatives fail catastrophically on at least one edit class.

---

**PRE-SUBMISSION ADDENDUM (2026-07-12):** the in-region leak/ghost numbers in this report are
SECONDARY metrics per the Route-A red-team (leak_R ρ=0.502 vs true edited GT; it penalizes legitimate
improvements). Canonical in-region evidence = the oracle-scored master table
(`RESULTS/STAGE4_ECR/edit_aware/routeA_master_table.md`). Unaffected-region (true-GT) and cost numbers
in this report are unchanged and remain primary.
