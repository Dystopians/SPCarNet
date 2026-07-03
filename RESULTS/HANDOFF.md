# GEMS — HANDOFF note for human paper writing

2026-07-03 · Stage-2 §8 final deliverable (GOAL#018). Suggested narrative
order, strongest evidence first, known weak points flagged. Writing the
paper's narrative claims is a human task by design (Stage-2 §10); this note
is the map, not the draft. Companion: `EXPERIMENT_REPORT.md` (complete
honest report + §7 completion table), `CLAIMS_EVIDENCE_MATRIX.md`,
`NEGATIVE_RESULTS.md`, `REPRO_PACK/`.

## The paper's spine (per Delta Memo #002 steering)

Three-regime compaction story + preservation-exactness + honest
consumption/geometry negatives WITH mechanisms + the measurement suite.
Lead with T1 + F2. All three tombstoned axes (geometry repair, teacher,
consumption certification) share ONE mechanism family — train-coverage
selection effects — write them as ONE section, not three apologies.

## Suggested narrative order (strongest first)

1. **The three-regime compaction result (C1′)** — T1, F2, F1.
   - B50: iso-or-better vs the deployed-default anchor on 8/9 M360 scenes
     (LPIPS better 9/9) at HALF the triangles/disk, ~1.3× FPS; garden
     strictly above BOTH anchors. Transfers to driving (3/4 SS3DM towns).
   - B25/B12.5: graceful degradation; fine-tune value GROWS as budget
     shrinks (B5>B4 9/9 at both).
   - B12.5: evidence-vs-random **+5.23 dB mean, 9/9 CIs** — the definitive
     "importance matters" datum. B3 QEM: **−2.7..−3.4 dB below B5** at
     matched budget — geometric parsimony does not rescue appearance;
     evidence + carrier preservation does. Importance-DEFINITION axis is
     flat (±0.05 dB) — an honest simplification: *having* render evidence is
     what matters.
   - Anchor honesty is a FEATURE: lead with the dual-anchor table (A1
     pre-empted; the clean-fixed@30k re-anchor is itself a publishable
     one-line trainer recommendation, kept as NON-CLAIM).
2. **Preservation-exactness (C4′(1))** — T5, F6, T5b.
   - The strongest single sentence in the pack: at B50, rendered median
     depth is BIT-IDENTICAL, train-evidence certification sets are
     FLOAT-EXACT identical, and all 500 seed-0 planner outcomes are
     outcome-level EXACT clean↔B50 (CIs [0,0]). Compaction is
     downstream-invariant — deploy the half-size artifact and nothing a
     planner sees changes.
3. **The measurement suite + "photometric quality masks geometric
   unreliability" (C2)** — T3, F6(a), GT-CAL row.
   - Metric validity via the GT-model calibration row; then 30.9 dB toy with
     59% false-free space, 65% courtyard, SS3DM far-field collapse — now a
     FAMILY property (D-2 ×3 instances, 4 towns). This motivates everything.
4. **The evidence-vs-error bridge (E2-GEO analysis)** — promote
   `analysis/e2geo_evidence_vs_error/plots/` to a main-text figure if space
   allows (Delta Memo #002 recommends): train residual evidence predicts
   test error (ρ≈0.7, 3/3) BUT is silent exactly in the coverage gaps where
   error is largest (5–11×). This is simultaneously "why evidence-guided
   pruning works" and "why it cannot be a certificate" — and it is the ONE
   mechanism behind every negative below. Frame as analysis, never method
   (§7.5 guard).
5. **The honest negatives with mechanisms (one section)** — F8, F6(b–d),
   NEGATIVE_RESULTS.md, T5b.
   - Consumption trilogy: TSDF votes near-surface voxels free (grazing-ray);
     raw grids blanket free space (88–100% spurious infeasibility, and the
     rare courtyard plans collide); certification sheds real walls with junk.
     Frozen fix-target: courtyard ≥30/100 found at ≤3.0 coll/100.
   - Geometry-repair and teacher axes: tombstoned with measured mechanisms
     (position drift; view-conditioning cap 5–16%). B6R = the one bounded
     positive (courtyard), generalization tested and failed honestly.
6. **Cross-representation context (R1) + efficiency (T4)** — place late,
   plainly: 3DGS is 2.1–3.4 dB better at matched storage and 3–4× faster;
   the contribution is the mesh artifact + reliability/consumability
   measurements, not rendering parity. Never let this table look hidden.
7. **Failure taxonomy (F8/E9)** as the limitations backbone — 5 families,
   each with a design implication; reviewers reward this.

## Known weak points (flag, don't bury)

- **W-1 Anchor duality:** vs the PRIMARY (drift-repaired) anchor, B50 iso is
  4/9, not 8/9. The claim text names its anchor; a reviewer may still call
  the headline soft. Mitigation: lead with the dual-row table itself.
- **W-2 Single seed:** no E7; base-training seed sensitivity
  uncharacterized. Measured 1.6e-5 dB pipeline repeat floor + 15-scene
  replication is the current answer (EXPERIMENT_REPORT §4-A5, waive W1).
- **W-3 No robustness/generalization column (E8/S-GEN):** three dataset
  families only; T&T data on disk but never run (waive W2).
- **W-4 Baseline breadth:** B1 never run; B3 only @B50×3 scenes (courtyard
  PSNR CI wide — its LPIPS margin is the certified arm); B2 moderate-budget
  columns only on 3 scenes; S-GEO B2 absent (waives W4/W5).
- **W-5 Courtyard rendering CIs are 5-view** (by design; geometry/downstream
  carry that scene) — say it before the reviewer does.
- **W-6 toy-family LPIPS caveat:** features-FT trades LPIPS for PSNR on toy
  scenes only (B5 LPIPS > B4 there; real scenes go the other way) — one
  honest sentence in the ablation section.
- **W-7 SS3DM absolutes** (g1 far-field, g4 sampling-density) are not
  cross-dataset comparable; only paired deltas are load-bearing (T3 header).
- **W-8 H1/R1 are context, not baselines-in-the-CI-sense** — keep their
  appendix framing verbatim to avoid an "unfair comparison" review thread.

## Mechanical notes for writing

- Regenerate everything: `bash RESULTS/REPRO_PACK/regenerate_all.sh`
  (CPU, minutes). T1 from-scratch byte-diff: `verify_t1.sh` (PASS on file).
- NON-CLAIMS block must appear verbatim (it already does in
  EXPERIMENT_REPORT §6 — copy from there, not from memory).
- Numbers in prose: quote from T1–T7/analysis JSONs only; the §6 no
  hand-typed-numbers rule applies to the paper draft too.
- Figure inventory + captions: F1/F3/F6 have caption .txt files as draft
  caption text; F4 grids embed the §5 crop rule (best/median/failure —
  manifest.json proves script selection, cite it in the reproducibility
  statement).
- Open human decisions before "Stage Two complete" can be declared:
  approve/reject waive drafts W1–W6 in EXPERIMENT_REPORT §7.
