# GEMS — CLAIMS_ECR.md (Stage 4: Evidence-Cached Rendering)

v0.1 · 2026-07-09 · Opened per `docs/GEMS_Stage4_ECR_Prompt.md` §7. The Stage-2/3
`CLAIMS.md` (v1.3) is FROZEN and untouched; this file holds only Stage-4 ECR
claims. Claims are instantiated FROM EVIDENCE ONLY — the bracketed [measured]
slots stay empty until the corresponding CI-backed rows exist. Edits follow the
same claim-editing rule (shrink with dated evidence pointer; never massage
experiments).

## Positioning (verbatim, from the §0 human sign-off)

The Stage-4 shipped artifact per scene is the triple **{mesh-splat checkpoint
(full-budget and/or B5@B50) + evidence cache (train-view images/renders/depths/
residuals + cameras) + transport renderer}**, positioned honestly as
**per-scene evidence-cached hybrid rendering** — the same representation class
as surface light fields / unstructured-lumigraph / deep-blending systems, and
compared as such.

## Claims (instantiated 2026-07-10 from banked rows; stratified
## mean-of-scene-means paired bootstrap, 10k, seed 0, throughout)

- **CR1 — Quality (INSTANTIATED).** The ECR final stack (v3: multiband
  K-source transport + learned per-pixel fusion + residual-vs-RGB routing)
  renders full9 at **+1.666 dB [+1.567, +1.766]** above the PRIMARY anchor
  (clean-fixed@30k; 9/9 per-scene PSNR CI-wins), **+1.809 dB [+1.706,
  +1.911]** above the legacy anchor (clean@30k; 9/9), and **+0.361 dB
  [+0.316, +0.407]** above PJ-2026 with **ΔLPIPS −0.01686 [−0.01880,
  −0.01495]** (8/9 per-scene PSNR CI-wins) — the §0 delivery bar (exceed
  PJ-2026 on BOTH metrics, CIs excl. 0, target ≥ +0.15) is met at the
  stretch level (CI-low +0.316 ≥ +0.30). Evidence:
  `analysis/final_stack/final_stack_tables.md` (T-ECR-1), gates
  `analysis/e0_pj2026/l4_{gate,vs_floor}.json`. Suite rows (banked
  2026-07-10): vs PJ-2026 town01 +0.412 / town02 +0.298 / town03 +0.519
  (CIs excl. 0), town06 +0.095 (CI excl. 0, LPIPS worse — honest boundary
  case), toy_parking +0.029 (CI incl. 0 — the pre-registered ring-coverage
  saturation).
- **CR2 — Honest cost (matched-storage cell INSTANTIATED; Pareto pending
  E-06).** At matched TOTAL artifact storage (checkpoint + raw cache = 2.1–
  3.3 GB/scene), stock 3DGS-30k uses only 11–55% of the budget and stays
  ahead by +0.32 (kitchen) to +1.53 dB (bicycle) PSNR at ~40–70× the
  end-to-end FPS — context-only cross-representation reference, no CIs,
  exactly the frozen NON-CLAIMS trade; the Stage-2 R1 gap (2.1–3.4 dB) is
  nonetheless mostly closed (`analysis/final_stack/e07_matched_total_3dgs.md`,
  GOAL #E-07). The cache-quality Pareto (GOAL #E-06, 15 points, 15/15
  audits): **jpeg95 re-encoding is nearly free** (garden −0.10 / kitchen
  −0.08 dB, bicycle +0.32 dB) at ~22% TOTAL savings; halfres buys 44–58%
  savings at −0.85..−0.95 dB; K-subset-50 is the dominated axis
  (`analysis/final_stack/l5_pareto.md`). Every ECR row carries cache_mb_raw
  / cache_mb_compressed / transport_ms_per_frame / end_to_end_fps /
  total_artifact_mb.
- **CR3 — Mesh retained (INSTANTIATED).** At 50% triangles (B5@B50 base),
  the full ECR stack delivers **+1.488 dB [+1.379, +1.593] AND ΔLPIPS
  −0.0748 [−0.0775, −0.0720]** over the FULL-BUDGET primary anchor (full9
  stratified, 9/9), and **+0.336 dB [+0.294, +0.380] / −0.0176** over
  PJ-2026 on the same B50 base — the ladder's margin over Phase-J is
  base-independent (full-budget margin +0.361/−0.0169). Geometry/downstream
  metrics preserved per the frozen Stage-2/3 results (cited, not re-run;
  the transport leaves rendered depth untouched by construction). Evidence:
  `final_<scene>_B50_v1` rows + L6 tie-back in
  `analysis/final_stack/final_stack_tables.md` (GOAL L6, 9/9 audits GREEN).
- **CR4 — Audited transport (INSTANTIATED; SHRUNK v0.3; terminology v0.6).**
  "Audited" is the defined term of PROTOCOL §4E.1 (threat model: what is
  PROVEN per row / ASSUMED / NOT CLAIMED — no formal-verification connotation).
  Train-only guarantees are audited per reported row: **96/96 `--ecr` audits
  GREEN**
  (all transport reads ⊆ cache manifest; manifest train views disjoint from
  the independently recomputed test split; checkpoint fingerprint match;
  frozen per-view kwargs hash — no per-test-view parameters anywhere).
  Confidence predicts transport benefit (banked precedent: evidence-vs-error
  Spearman ρ ≈ 0.69–0.74 on 3 scenes), and the audited gating is
  STRUCTURAL: evidence RGB can only be routed where the transport physically
  has support (compose = β·valid with valid = warp-confidence mask), which
  is why β vanishes at occlusion boundaries and coverage gaps degrade to
  the base render instead of hallucinating (failure-case set: 1
  transport-negative view in 139, −0.06 dB). SHRUNK per GOAL #E-08
  (2026-07-10): the net's EXPLICIT confidence input planes are redundant —
  zeroing them at train+test leaves quality statistically unchanged on 2
  scenes (CIs incl. 0) — so the audit claim attaches to the audit checks
  + the structural compose gate, NOT to the net's use of confidence input
  channels.

## NON-CLAIMS (verbatim in any Stage-4 report)

Per-scene method — no cross-scene generalization claim; no test-GT anywhere
(train-view evidence is a declared render-time input of the shipped artifact);
no claim that ECR fixes geometry or planning consumability (the Stage-3
IMPOSSIBILITY×4 result is cited as scope, not re-litigated); rendering
comparisons name their references explicitly; the Stage-1/2/3 falsification
corpus (baking 4.76%, distillation 5–16%, residual cosine ≈ 0.21) is the
MOTIVATION for caching evidence rather than baking it, reported first-class.

## Claim-edit log

- 2026-07-09 v0.1: file opened at GOAL #E-00 (mouth extension); claim slots
  registered, none instantiated. Delivery bar per prompt §0: floor row PJ-2026
  reproduced under the single mouth; final stack > PJ-2026 on full9 mean PSNR
  AND LPIPS, CIs excl. 0, target ≥ +0.15 dB — or honest escalation.
- 2026-07-10 v0.2: CR1 instantiated (ladder closed at v3, GOAL #E-05 —
  delivery bar met at stretch); CR2 matched-storage cell instantiated (GOAL
  #E-07; Pareto slot open pending #E-06); CR3 partial (E-01 secondary base
  banked; L6 slot open); CR4 instantiated (65/65 ecr audits GREEN + routing
  qualitative evidence). No claim shrunk; two slots remain open on in-flight
  rows only.
- 2026-07-10 v0.3: **CR4 SHRUNK** per GOAL #E-08 (confidence-inputs-off
  ablation: quality unchanged, CIs incl. 0 on 2 scenes) — certification now
  attaches to the audits + the structural compose valid-mask, not to the
  net's confidence input channels; audit tally 69/69. External context
  banked alongside (GOAL #E-09): Difix3D+ single-step on the same base
  renders is PSNR-negative on all 3 scenes and dominated by the ECR final
  stack on both metrics — recorded as A11 rebuttal evidence, not a claim.
- 2026-07-10 v0.4: CR1 extended with the banked suite rows (3/5 CI-positive
  over PJ-2026; town06/toy honest boundary cases); **CR3 fully
  instantiated** from the L6 rows (compact stack +1.488/−0.0748 over the
  full-budget anchor; +0.336/−0.0176 over PJ-2026 on the same base, 9/9;
  14/14 chain audits GREEN). Open slot remaining: CR2's Pareto curve
  (E-06 chains running).
- 2026-07-10 v0.5: **CR2 complete** — L5 Pareto banked (15 points, 15/15
  audits GREEN; jpeg95 ~free, halfres −0.9 dB at ~50% TOTAL savings). All
  four claims now fully instantiated; total ecr-audit trail 96/96 GREEN.
  No open slots.
- 2026-07-11 v0.6 (TOPCONF hardening): terminology — "certified" retired
  from claim-bearing text, replaced by the DEFINED term "audited"
  (PROTOCOL §4E.1 threat model added: PROVEN / ASSUMED / NOT CLAIMED);
  stale CR4 figures refreshed (96/96 audits; 1-in-139 failure census).
  **Robustness addendum:** every headline CI ALSO excludes 0 under a
  two-stage scene-cluster bootstrap (scenes resampled with replacement,
  then views; 10k, seed 0) — e.g. final vs PJ-2026 dPSNR hierarchical CI
  [+0.218, +0.527], dLPIPS [−0.0258, −0.0098]
  (`analysis/final_stack/hierarchical_cis.md`, EXP-HBOOT). The stratified
  interval remains the pre-registered primary form.
