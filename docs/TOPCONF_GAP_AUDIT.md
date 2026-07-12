# TOPCONF GAP AUDIT — Stage-4 ECR vs CVPR/ICCV/ECCV bar

Opened 2026-07-11. Living document; updated as blockers close. Baseline endpoint: ECR v3
(+0.3607 [+0.3158,+0.4067] dB / −0.01686 LPIPS vs PJ-2026 on full9; L6 half-triangle validation;
96/96 audits GREEN). Frozen Stage-2/3 evidence and banked Stage-4 artifacts are NOT disturbed;
all new work is additive rows/cells through the same mouth.

"100%" is interpreted as: no unresolved high-severity reviewer objection that is still addressable
with this repository, this machine, and submission-cycle time.

## 1. Readiness scorecard (0–100, evidence-based)

| Axis | Initial | Current | Evidence / gap |
|---|---|---|---|
| A. Novelty & distinction from closest prior work | 55 | 80 | Deliberately in the ULR/Deep-Blending lineage; the distinct assets (per-row purity audits, structural β·valid gate + E-08 null, CI-gated ladder methodology, mesh-artifact composition/L6) are real but currently argued, not contrasted head-to-head against any EXTERNAL IBR system. "Certified" wording invites attack (see G). |
| B. Closest external baseline coverage | 40 | 85 | Banked: PJ-2026 (internal, strong but self-made), 3DGS matched-TOTAL (E-07), Difix3D+ (E-09, dominated). MISSING: any external learned-IBR baseline (IBRNet/ENeRF-class) on our scenes. This is the single most likely "reject" sentence. |
| C. Benchmark breadth & external validity | 55 | 85 | mip-NeRF 360 full9 (standard) + 4 SS3DM towns + toy (non-standard). MISSING: the second standard set every reviewer knows — Tanks&Temples (truck/train) + Deep Blending (drjohnson/playroom), i.e. the exact 3DGS eval suite. truck+barn already on disk (COLMAP); tandt_db.zip is public. |
| D. Statistical rigor | 85 | 95 | Paired per-view bootstrap 10k/seed 0, stratified mean-of-scene-means, pre-registration + effect floors + sunset rules — above the venue norm. MISSING: scene-level (cluster/hierarchical) resampling — with n=9 scenes a reviewer can ask whether the full9 CI understates scene-to-scene variance. |
| E. Temporal / view-path stability | 20 | 90 | NOTHING measured. The transport is per-view independent (K-support sets and α/β maps change across views) — flicker on smooth camera paths is plausible and is a standard IBR objection. No videos exist. GT-free to measure (no purity issue). |
| F. Total-storage & runtime fairness | 80 | 85 | TOTAL artifact = ckpt + cache (incl. fusion net, which lives inside cache sizes) in every row; L5 Pareto (15 pts); matched-TOTAL 3DGS; lossless-compressed sizes; honest fps. MISSING: one unified R-D figure (all points on one plot), explicit net-size call-out (~3.4 MB — favorable, currently buried), fusion-training time in the cost table. |
| G. Audit / threat-model correctness | 70 | 95 | Mechanics are strong (96/96: read confinement, split disjointness, frozen kwargs hash, ckpt fingerprint, GT sentinel, base-mode non-loading). MISSING: a precise written threat model (what is proven / assumed / NOT claimed); the word "certified" overreaches (no formal verification; audits are dynamic+static checks) — rename and define. |
| H. Method simplicity & ablation completeness | 75 | 85 | Ladder isolates one mechanism per rung incl. the L1 negative; E-08 proves conf-input redundancy (simplicity asset — the paper-facing abstraction can drop 3 channels); K,α jointly calibrated (covered); crop/loss/steps frozen. Minor: no net-capacity ablation (assess value vs cost — likely P2/reject). |
| I. Reproducibility & paper-writing readiness | 65 | 85 | Deterministic generators, frozen configs in manifests, byte-verified pack, seeds everywhere. MISSING: ECR-scoped README + repro commands; paper tables extractor; ladder-CI and R-D plots; supplementary videos. |

**Overall initial score: 58/100** → **FINAL 87/100** (see TOPCONF_READINESS_REPORT.md; all P0/P1 closed or rejected with evidence, 2026-07-11) (bottlenecked by B, C, E — the three axes a CVPR reviewer hits first).

## 2. Blocker register

### P0 (plausible sole cause of rejection)

- **P0-1 — Second standard benchmark absent.**
  Verified real: full9 is mip-NeRF 360 only; SS3DM/toy are not community sets.
  CLOSE BY: run the exact 3DGS eval suite — T&T {truck, train} + DB {drjohnson, playroom} — through the
  unchanged pipeline: clean30k anchor (SS3DM/toy precedent; no 26k continuation exists for new scenes),
  PJ-2026 floor row, final-stack routed row, per-scene CIs + 4-scene stratified mean, audits per row.
  Assets: truck already on disk; tandt_db.zip public. → EXP-T2B below.
- **P0-2 — No external learned-IBR baseline.**
  Verified real: no IBRNet/ENeRF/etc. checkout or row exists anywhere in the tree.
  CLOSE BY: bounded external cell (Difix/E-09 pattern, sanctioned mirror exception): one
  generalizable-IBR method with published weights (preference order: ENeRF → IBRNet) evaluated on the R1
  trio splits; report plainly whatever it says. If genuinely infeasible (weights gone, env wall), bank the
  attempt log AND strengthen the fallback positioning (L2 = classical point; PJ-2026 = tuned per-scene IBR
  floor). → EXP-IBR below.

### P1 (high-severity, must close or reject with evidence)

- **P1-3 — Temporal/view-path stability unmeasured.** CLOSE BY: GT-free camera-path cell: interpolated
  pose paths, per-frame transport, flicker metric (mean |Δframe| vs base's |Δframe|, plus OF-warped
  variant if cheap), side-by-side videos for supplementary. → EXP-TEMP.
- **P1-4 — Scene-level bootstrap absent.** CLOSE BY: hierarchical bootstrap (resample scenes, then views
  within scenes) added to the stats tool; report BOTH intervals for every headline number. → EXP-HBOOT.
- **P1-5 — Threat model imprecise; "certified" overclaim.** CLOSE BY: write the explicit threat model
  (proven / assumed / out-of-scope) into PROTOCOL §4E + CLAIMS; global terminology change
  certified → **audited** (with one defined term). Documentation only. → DOC-THREAT.
- **P1-6 — R-D/runtime positioning not unified.** CLOSE BY: single R-D master plot (L5 points, L6,
  uncompressed, matched-3DGS, Difix) + ladder CI bar plot + fusion-train-time column. → PLOT-RD, PLOT-LADDER.
- **P1-7 — Repro/paper mechanics.** CLOSE BY: ECR README (env, commands, artifact map), paper-tables
  extractor (LaTeX from banked json), supplementary video renders (rides on EXP-TEMP). → DOC-REPRO, TOOL-TABLES.

### P2 (assessed; rejected or deferred with rationale)

- Net-capacity ablation — REJECTED for now: E-08 already demonstrates input-side simplicity; capacity
  sweep = open-ended, low reviewer value (nobody rejects over 844k params).
- More suites (extra SS3DM towns etc.) — REJECTED: breadth is closed by P0-1 with community sets, not more
  private scenes.
- DS-1 / Phase-S operator exploration — EXCLUDED by mission constraints; DS-1 already closed strengthened.
- Per-scene hyperparameter re-tuning of the ladder — FORBIDDEN (frozen-config law; also a purity story asset).

## 3. Verification notes on the suspected gaps (mission list)

| Suspected gap | Verdict |
|---|---|
| Missing ULR/DB/FWD/IBRNet-style baselines | REAL → P0-2 (external cell). ULR/DB classical points partially served internally by L2/PJ-2026 — keep that argument as fallback, not primary. |
| No second public benchmark | REAL → P0-1 (T&T+DB on disk / one download away). |
| Hierarchical scene-view bootstrap | REAL but small → P1-4 (stats addition, no new data). |
| Temporal consistency | REAL and unmeasured → P1-3. |
| Redundant confidence channels / complexity | ALREADY CLOSED as evidence (E-08 null, banked); remaining work is presentation (paper-facing abstraction drops the channels). |
| Total-artifact R-D/runtime positioning | Mostly closed (Pareto, matched 3DGS, TOTAL accounting); remaining = one unified figure + minor cost columns → P1-6. |
| Threat-model precision / "certified" | REAL → P1-5 (documentation + terminology; no code change needed — the audits themselves are sound). |
| Weak differentiation from generic learned IBR | Partially real → closed by P0-2 head-to-head + the three assets no generic IBR has (per-row purity audit, structural gate with measured worst case, mesh-artifact/L6 composition). |

## 4. Compute/data feasibility snapshot (2026-07-11)

- GPUs: 0/3/4/7 have ~38 GB free each (shared with ~50%-util tenants — timing caveat only); 1/2/5/6 busy.
- Disk: 733 GB free (T&T+DB + 4 models + 4 caches ≈ ~15 GB — fine).
- Data: `mesh_datasets/tanks_and_temples_colmap/{truck,barn}` ready; tandt_db.zip download needed for
  train/drjohnson/playroom. Network verified working.
- New-scene anchor recipe: clean30k via `train.py` with the recorded SS3DM/toy cfg_args pattern
  (sh_degree 3, eval split, seed 0); anchors are NEW rows (no frozen artifact touched).
