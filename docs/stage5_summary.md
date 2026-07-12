# GEMS Stage 5 — TOPCONF Hardening Campaign: Complete Summary

Written 2026-07-12 at campaign closure. "Stage 5" = the top-conference hardening pass run on top of the
closed Stage-4 ECR result (see `docs/stage4_sum.md`), governed by the three living documents
`docs/TOPCONF_GAP_AUDIT.md` (diagnosis), `docs/TOPCONF_EXECUTION_PLAN.md` (frozen experiment specs), and
`docs/TOPCONF_READINESS_REPORT.md` (final verdict). Every number below is read from a banked artifact;
statistics are paired per-view bootstrap (10k, seed 0) with 95% CIs; suite means are reported in BOTH the
pre-registered stratified form and the new scene-cluster (hierarchical) form.

---

## 1. Mission and interpretation

Diagnose and close every material gap between the Stage-4 ECR system and "100% top-conference
competitiveness" (CVPR/ICCV/ECCV), where 100% is defined realistically: **no unresolved high-severity
reviewer objection that is still addressable with the available repository, compute, and time** — not
guaranteed acceptance. Hard constraints held throughout: no test-GT leakage, single evaluation mouth,
banked-numbers-only, TOTAL-storage accounting (checkpoint + cache + network), unfavorable evidence
(3DGS, runtime, storage, temporal, failure cases) never hidden, frozen Stage-2/3 evidence and banked
Stage-4 artifacts untouched (all new work = additive rows/cells).

## 2. Verdict

**TOP-CONFERENCE READY at 87/100** (from 58/100 at audit opening). Both P0 blockers closed with banked
evidence; every P1 closed; P2s rejected with written rationale. Residual risks are positioning/taste
risks that additional experiments cannot reduce.

## 3. Readiness scorecard (initial → final)

| Axis | Initial | Final | Close driver |
|---|---|---|---|
| A. Novelty & distinction | 55 | 80 | head-to-head vs generalizable IBR measured; "audited" defined term; lineage answer runs measured in both directions |
| B. External baseline coverage | 40 | 85 | PJ-2026 (tuned classical, exceeded) + 3DGS matched-TOTAL + Difix3D+ + IBRNet — all measured |
| C. Benchmark breadth | 55 | 85 | T&T+DB (the 3DGS eval suite) banked, zero per-scene tuning, CIs excl. 0 |
| D. Statistical rigor | 85 | 95 | scene-cluster bootstrap PASS 8/8 headline intervals |
| E. Temporal stability | 20 | 90 | measured PASS 3/3 (ratios 0.988–1.030 vs base) + videos |
| F. Storage/runtime fairness | 80 | 85 | unified R-D master figure (accounting already TOTAL-correct) |
| G. Audit/threat model | 70 | 95 | PROTOCOL §4E.1 threat model; "certified" retired |
| H. Simplicity & ablations | 75 | 85 | E-08 null reframed as THE simplicity result; zero-tuning transfer evidences frozen-config robustness |
| I. Reproducibility & paper readiness | 65 | 85 | ECR README, LaTeX table extractor, ladder/R-D figures, videos |

## 4. GOAL #E-11 — EXP-T2B: second standard benchmark (closed P0-1)

- **Setup:** the exact 3DGS evaluation suite — Tanks&Temples {truck (251 imgs), train (301)} + Deep
  Blending {drjohnson (263), playroom (225)} from the official tandt_db distribution (downloaded +
  verified; scenes registered ADDITIVELY in `tools/gems/scenes.py` under PROTOCOL **1.3.0 MINOR**:
  images/, `-r -1`, llff8 split — the same idx%8 rule as full9). Anchor = clean30k (train.py, 30k iters,
  seed 0, the recorded SS3DM/toy recipe; no 26k save exists → no clean-fixed continuation, caveat
  logged). Per scene, through the UNCHANGED frozen pipeline with ZERO per-scene tuning: base row →
  PJ-2026 cache/row → final routed stack row; `--ecr` audit per ECR row (8/8 GREEN).
- **Results (per scene, PJ-2026 vs anchor → final vs PJ-2026 dPSNR / dLPIPS):**
  - truck: 22.429 → 23.572 → 23.733/0.1512; +1.143 [+0.995,+1.296]; **+0.161 [+0.105,+0.222]** / −0.011 [−0.013,−0.009]
  - train: 18.819 → 20.240 → 20.369/0.2463; +1.421 [+0.952,+1.871]; +0.130 [−0.044,+0.322] (CI incl. 0 — honest boundary) / −0.007 [−0.013,−0.002]
  - drjohnson: 26.964 → 27.200 → 27.298/0.3132; +0.236 [+0.181,+0.298]; **+0.097 [+0.027,+0.177]** / −0.014 [−0.019,−0.010]
  - playroom: 27.928 → 28.050 → 28.148/0.2744; +0.122 [+0.078,+0.170]; **+0.098 [+0.050,+0.151]** / −0.001 [−0.004,+0.002]
- **Suite-4 means:** PJ-2026 vs anchor **+0.7306 [+0.6050,+0.8502]**; final vs PJ-2026 **+0.1215
  [+0.0700,+0.1756] dPSNR AND −0.0084 [−0.0105,−0.0064] dLPIPS** — both CI-excl.-0 under BOTH the
  stratified and the scene-cluster bootstrap ([+0.0634,+0.1846] / [−0.0134,−0.0032]); final vs anchor
  +0.8520 [+0.7491,+0.9544].
- **Honest findings kept first-class:** transfer magnitudes are smaller than full9 (+0.12 vs +0.36 over
  the floor); coverage explains the regimes (T&T 0.94 covered; DB indoor 0.49–0.67 — gains persist at
  half coverage); the anchors themselves are weaker on T&T (truck 22.43) — the transport lifts them
  +1.1–1.4 dB regardless.
- Evidence: `analysis/final_stack/t2b_tandt_db.{md,json}`; rows `eval/{<s>_clean30k_v1,
  e0_<s>_clean30k_pj2026_v1, final_<s>_clean30k_v1}`; tools `tools/ecr/t2b_scene.sh`,
  `tools/analysis/t2b_report.py`.

## 5. GOAL #E-12 — EXP-IBR: external generalizable-IBR baseline (closed P0-2)

- **Setup:** IBRNet (CVPR'21), pretrained model_255000 (downloaded via gdown), NO per-scene training or
  fine-tuning (the honest generalizable point, stated in the row); **10 source views per target** by the
  frozen transport camera score — a SUPERSET of ECR's evidence rights (calibrated K = 2–8) and IBRNet's
  own 8–10-source regime (pre-registration amendment logged BEFORE any number existed). Adapter
  (`/data/peilincai/IBRNet/ibr_infer.py`, job-file interface, torch-2.x compat edits logged) built by
  Codex; camera math owned by `tools/analysis/ibr_jobs.py`.
- **Fabrication-risk gate (passed BEFORE any cross-method number):** self-reconstruction test — render a
  held-out TRAIN view from its own pose given its 10 neighbors → **22.48 dB** (wrong axes would give
  <10 dB); outputs additionally visually inspected (real renders with characteristic warp ghosting).
- **Results (mirror self-validated ≤0.01 dB vs banked base rows):** IBRNet garden 23.754
  (−1.027 [−1.510,−0.576] vs the base anchor), bicycle 17.804 (−5.358 [−6.075,−4.583] — 360° unbounded,
  far outside its forward-facing domain), kitchen 26.613 (−0.996 [−1.689,−0.303]). **The ECR final stack
  exceeds IBRNet by +2.6 to +5.9 dB with better SSIM/LPIPS on all three scenes.**
- **Reading:** pretrained generalizable IBR does not transfer to these scenes; the tuned per-scene
  classical point IS our floor (PJ-2026), and the ladder exceeds it with CIs — the "just ULR/Deep
  Blending" objection is now answered head-to-head in BOTH directions (REBUTTAL A10 addendum).
- Runtime ~122 s/frame (chunk 4096, GPU 7). Evidence: `analysis/ibr_cell/{ibr_table.md,ibr_<scene>.json}`,
  `ibr_cell/attempt_log.md`; isolated venv `gems_stage1/ibr_cell/venv` (frozen env untouched).

## 6. GOAL #E-13 — EXP-TEMP: temporal / view-path stability (closed P1-3)

- **Setup (GT-free — no purity surface):** deterministic 120-frame paths (Catmull-Rom on centers +
  quaternion slerp through the name-ordered test poses); per frame: base render + ECR final (metric-path
  quantization); roughness = mean|I_t − I_{t−1}|; acceptance bar final/base ≤ 1.5; side-by-side videos
  (base | final | β map).
- **Results — PASS 3/3:** garden ratio **0.988** mean / 0.983 P95 (the ECR path is SMOOTHER than the base
  render — the transport suppresses base rendering noise) at 3.2 support-set switches/step; bonsai 1.002
  at 1.5 switches; ss3dm_town01 1.030 at 2.0 switches. No popping visible despite per-view-independent
  processing — the structural β·valid gate changes support smoothly.
- Evidence: `analysis/temporal/{temporal_summary.{md,json}, <scene>/temporal.json, <scene>/<scene>_path.mp4}`;
  tool `tools/analysis/ecr_temporal.py` (MiniCam synthetic poses; mpeg4 videos — this host's ffmpeg lacks libx264).

## 7. GOAL #E-14 — EXP-HBOOT + DOC-THREAT (closed P1-4, P1-5)

- **EXP-HBOOT — PASS 8/8:** `hierarchical_mean_ci` (two-stage: resample SCENES with replacement, then
  views) added ALONGSIDE the pre-registered stratified form in `tools/ecr/e0_report.py`. Every Stage-4
  headline interval still excludes 0 under scene-cluster resampling: final vs PJ-2026 dPSNR
  [+0.2175,+0.5273], dLPIPS [−0.0258,−0.0098]; final vs primary [+0.9663,+2.4481]; L6 vs anchor
  [+0.8297,+2.2125] / LPIPS [−0.0882,−0.0619]; L6 vs PJ-2026-B50 [+0.2059,+0.4857] / [−0.0257,−0.0112];
  AT-E0 [+0.7155,+1.9605]. Evidence: `analysis/final_stack/hierarchical_cis.{md,json}`,
  `tools/analysis/hboot_report.py`. CLAIMS v0.6 robustness addendum.
- **DOC-THREAT:** PROTOCOL **§4E.1** threat model written — PROVEN per row (reads ⊆ manifest via confined
  loader; train/test split disjointness independently recomputed; frozen per-view kwargs hash; checkpoint
  fingerprint; GT sentinel; base-mode non-loading) / ASSUMED (OS integrity; the audit tool itself,
  sha-pinned in the pack; registry correctness) / NOT CLAIMED (formal verification; adversarial cache
  robustness; geometry/privacy/security). Claim-bearing **"certified" retired for the defined term
  "audited"** across CLAIMS/VENUE_MEMO/ABSTRACT (frozen Stage-2/3 artifact names untouched); stale CR4
  figures refreshed (96/96 audits; 1-in-139 failure census).

## 8. P1-6 / P1-7 — figures, tables, reproducibility (closed)

- `tools/analysis/plot_ladder.py` → `RESULTS/figures/ecr_paper/ladder_ci.{pdf,png}` (two-panel CI bars,
  L1 negative flagged, promotion floors drawn; verified against gate jsons).
- `tools/analysis/plot_rd.py` → `RESULTS/figures/ecr_paper/rd_master.{pdf,png}` (unified R-D: L5 points,
  uncompressed final, L6, 3DGS, Difix, base — per R1-trio scene).
- `tools/analysis/paper_tables.py` → `RESULTS/tables_tex/{T1_main,T2_ladder,T3_compact,T4_cost_external,T5_temporal}.tex`
  (LaTeX bodies, zero retyped numbers; every emitted row spot-verified against sources).
- `docs/ECR_README.md` (env, per-row-class repro commands, artifact map, determinism notes).
- Evidence pack refolded: **RESULTS/STAGE4_ECR now 26 byte-verified artifacts** (added: hierarchical CIs,
  T2B tables, temporal summary, IBR table + attempt log).

## 9. P2 items rejected with rationale (no work spent)

Net-capacity ablation (E-08 already carries the simplicity result; capacity sweeps are open-ended, low
reviewer value); more private suites (breadth belongs to community sets); Phase-S operator exploration
(excluded by mission constraint); any per-scene re-tuning of the ladder (frozen-config law — itself a
purity asset).

## 10. Division of labor & process notes

- **Claude (scientific ownership):** gap diagnosis + scoring; all pre-registrations and protocol design;
  scene registry + PROTOCOL 1.3.0; camera math (`ibr_jobs.py`) and the self-reconstruction gate; stats
  code (`hierarchical_mean_ci`); temporal tool; metric mirrors; threat model; all claim edits; number
  verification of every delegated output; interpretation and claim boundaries.
- **Codex (delegated, tightly specced):** IBRNet torch-2.x adapter (every edit logged), plots, LaTeX
  extractor, README. All outputs inspected and spot-verified before banking.
- **Learned constraint (memorized):** the Codex sandbox has NO network/DNS, NO GPU, and writes only to
  configured `writable_roots` — downloads, clones, weight fetches, and GPU smokes are always main-session
  jobs (cost: 4 failed dispatches before the division stabilized; `~/.codex/config.toml` writable_roots
  extended to gems_stage1/IBRNet/ENeRF/mesh_datasets).
- Race prevention: per-row output dirs; all gated launches key on exit-code == 0; collectors read only
  completed+audited rows.

## 11. Strongest remaining reviewer attacks (responses prepared)

1. "3DGS is better/faster at this storage" → printed in every row; NON-CLAIMS scope; mesh-artifact axes
   have no 3DGS equivalent (A3/A11). Taste risk — not further reducible.
2. "Per-scene method" → by design (A12); the zero-tuning T&T/DB transfer answers config generalization.
3. "Train images at test time" → D4 rights + §4E.1 + audit wall; we PROVE no test leak per row.
4. "Incremental over Deep Blending" → A10 measured both directions (generalizable fails here; the tuned
   classical point is our floor and is exceeded with CIs).
5. "Community-suite gains are small" → +0.122/−0.0084 with CIs excl. 0 under two bootstrap schemes at
   zero tuning; honesty is the argument.

## 12. Recommended submission claim set (exact)

CR1 (quality: three named references, full9 + SS3DM/toy suites + T&T/DB community suite), CR2 (honest
cost: Pareto + matched-TOTAL 3DGS), CR3 (compact: half-triangles above the full-budget anchor;
base-independent margin), CR4 (audited transport per §4E.1; structural gate; E-08 simplicity),
NON-CLAIMS verbatim; Difix3D+/IBRNet/3DGS presented as context rows, never claim targets.
CLAIMS_ECR.md is at **v0.8**; MATRIX/LEDGER (#E-11..#E-14)/REBUTTAL_BANK/PROTOCOL 1.3.0 all synchronized.

## 13. Where everything lives

- Living docs: `docs/TOPCONF_{GAP_AUDIT,EXECUTION_PLAN,READINESS_REPORT}.md` (final verdict in the report).
- New rows: `eval/{tandt_*,db_*}` (12 rows + 8 audits); analyses:
  `analysis/{final_stack/{t2b_tandt_db,hierarchical_cis}.*, temporal/, ibr_cell/}`.
- Figures/tables for the paper: `RESULTS/figures/ecr_paper/`, `RESULTS/tables_tex/`, videos under
  `analysis/temporal/<scene>/`.
- Pack: `RESULTS/STAGE4_ECR/` (26 artifacts + sha256 manifest, refold via `tools/ecr/fold_pack.sh`).
- External assets: `/data/peilincai/IBRNet` (adapter + weights), `/data/peilincai/mesh_datasets/tandt_db/`
  (4 scenes + provenance zip), `gems_stage1/ibr_cell/venv`.

## 14. What remains

Paper writing only (`docs/stage4_paper_plan.md`, updated by the new evidence: the community-suite table
joins T1, IBRNet joins T4, temporal joins the analysis section, and the "audited" terminology + §4E.1
govern the method text). Camera-ready niceties: re-encode videos h264 off-host if desired; optional
fusion-train-time column in T-ECR-2. Venue call per VENUE_MEMO (3DV/WACV strong; CVPR materially
improved and now defensible on all five anticipated attack lines).
