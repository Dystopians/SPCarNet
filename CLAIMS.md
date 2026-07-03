# GEMS — CLAIMS.md (Stage Two, FROZEN)

v1.0 · 2026-07-02 · Instantiated from Stage One evidence (`STAGE1_REPORT.md`, `LEDGER.md`); every Stage Two experiment supports, bounds, or refutes one of these. Edits only via the claim-editing rule (shrink with dated evidence pointer; never massage experiments).

## Claims

- **C1 — Compactness–quality.** At matched triangle budgets, GEMS-core (evidence prune + drift-safe features-only fine-tune) Pareto-dominates random pruning and QEM-style decimation on rendering metrics, and achieves **≥50% triangle reduction at iso-quality** vs clean on real scenes (Stage-One instantiation: garden B50 +0.157 dB / −0.0071 LPIPS BETTER than clean, CIs excl. 0; garden B25 iso; courtyard B50 iso). Known bound: on a heavily over-parameterized synthetic scene (toy_parking) B50 costs −0.52 dB (train-coverage selection effect, measured).
- **C2 — Geometry reliability: DEMOTED to measurement claim** (E2 failed 3/3 variants incl. sanctioned topology mechanism; PROTOCOL §0). The claim becomes: *GEMS compaction preserves geometry-reliability metrics (g1–g4) at half budget, and the GEMS metric suite exposes that photometric quality masks geometric unreliability in splatting models* (toy clean: 30.9 dB PSNR yet 23.9% free-space violations, 59% d1 false-free; courtyard 65%; GT-calibration row proves the metrics). NON-claim: GEMS does not improve geometry vs clean.
- **C3 — Rendering.** At B ≥ 50%, GEMS renders within −0.10 dB of clean on real dev scenes (measured: garden ABOVE clean; courtyard −0.04 prune-only). Teacher part DEMOTED (E3 sunset): distillation is reported as a diagnostic channel (+0.04..+0.13 dB, CIs excl. 0, below floor; recovered fraction ~5–16% — the view-conditioning bound, connected to the v1xx residual-cosine analysis).
- **C4 — Downstream proxy: BOUNDED to preservation.** At ≤50% budget and lower rendering cost, GEMS *preserves* downstream proxies exactly (courtyard d2 collision agreement 0.895, unsafe-disagreement 0.000, identical clean/B50/B25; toy d2 0.625→0.625). NON-claim: GEMS does not *lower* false-free-space rates vs clean (Stage-One evidence shows preservation, not improvement). The safety-relevant contribution is the measurement suite + the false-free-space exposure.

## NON-CLAIMS (verbatim in final report and paper draft)

This is a per-scene optimization setting; the teacher is train-only and absent at test time; no claim of state-of-the-art novel-view quality versus the 3DGS family; no claim about high-speed driving; downstream results are proxies unless the closed-loop stretch item was executed. Additionally (Stage-One additions): no claim that GEMS improves geometry or downstream metrics vs clean; no claim that evidence-guided importance dominates random pruning under safe fine-tuning at moderate budgets on small scenes (courtyard B50: +0.13 CI incl. 0); the end-phase-decline finding (clean26k > clean30k) is reported as a property of the baseline trainer, not a GEMS contribution.

## Claim-edit log

- 2026-07-02 v1.0: instantiated with C2 demoted, C3 teacher-part demoted, C4 bounded — all from Stage One pre-registered outcomes (not post-hoc edits).
