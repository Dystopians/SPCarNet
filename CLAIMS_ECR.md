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

## Claims (templates — instantiate from evidence only)

- **CR1 — Quality.** The ECR stack renders full9 at [measured] dB above the
  PRIMARY anchor (clean-fixed@30k) and [measured] above PJ-2026 (paired 95%
  CIs), with per-scene wins [n]/9. All quality rows are reported against THREE
  references: legacy anchor (clean@30k), PRIMARY anchor (clean-fixed@30k), and
  PJ-2026.
- **CR2 — Honest cost.** At matched TOTAL artifact storage (checkpoint +
  cache), ECR vs 3DGS = [measured]; the cache-quality Pareto is [curve].
  Every ECR row carries cache_mb_raw / cache_mb_compressed /
  transport_ms_per_frame / end_to_end_fps.
- **CR3 — Mesh retained.** The compact variant (B5@B50 base) delivers
  [measured] above Phase-J at 50% triangles; geometry/downstream metrics
  preserved per the frozen Stage-2/3 results (cited, not re-run).
- **CR4 — Certified transport.** Confidence inputs predict transport benefit
  (train-side analysis; banked precedent: evidence-vs-error Spearman ρ ≈
  0.69–0.74 on 3 scenes), with train-only guarantees audited per row
  (`tools/audit_test_path.py --ecr`).

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
