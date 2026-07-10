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
  `analysis/e0_pj2026/l4_{gate,vs_floor}.json`. Suite rows (4 SS3DM towns +
  toy_parking) extend the same stack and are logged as they bank.
- **CR2 — Honest cost (matched-storage cell INSTANTIATED; Pareto pending
  E-06).** At matched TOTAL artifact storage (checkpoint + raw cache = 2.1–
  3.3 GB/scene), stock 3DGS-30k uses only 11–55% of the budget and stays
  ahead by +0.32 (kitchen) to +1.53 dB (bicycle) PSNR at ~40–70× the
  end-to-end FPS — context-only cross-representation reference, no CIs,
  exactly the frozen NON-CLAIMS trade; the Stage-2 R1 gap (2.1–3.4 dB) is
  nonetheless mostly closed (`analysis/final_stack/e07_matched_total_3dgs.md`,
  GOAL #E-07). The cache-quality Pareto is [curve — GOAL #E-06 chains in
  flight]. Every ECR row carries cache_mb_raw / cache_mb_compressed /
  transport_ms_per_frame / end_to_end_fps / total_artifact_mb.
- **CR3 — Mesh retained (partial; L6 rows in flight).** At 50% triangles
  (B5@B50 base), the frozen Phase-J transport alone already delivers
  **+1.151 dB [+1.062, +1.238]** over the FULL-BUDGET primary anchor (GOAL
  #E-01 secondary base, 9/9 CI-positive); the final-stack-on-B50 tie-back
  (GOAL #E-05/L6) is [measured — `final_<scene>_B50_v1` rows in flight].
  Geometry/downstream metrics preserved per the frozen Stage-2/3 results
  (cited, not re-run; transport leaves rendered depth untouched by
  construction).
- **CR4 — Certified transport (INSTANTIATED).** Train-only guarantees are
  audited per reported row: **65/65 `--ecr` audits GREEN** (all transport
  reads ⊆ cache manifest; manifest train views disjoint from the
  independently recomputed test split; checkpoint fingerprint match; frozen
  per-view kwargs hash — no per-test-view parameters anywhere). Confidence
  inputs predict transport benefit (banked precedent: evidence-vs-error
  Spearman ρ ≈ 0.69–0.74 on 3 scenes), and the L4 routing head consumes
  exactly those confidence features to gate direct-evidence RGB (β maps
  concentrate on high-frequency, well-supported regions and vanish at
  occlusion boundaries — `analysis/quals/<scene>_final/`).

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
