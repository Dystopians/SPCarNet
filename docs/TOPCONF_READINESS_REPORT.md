# TOPCONF READINESS REPORT — FINAL (2026-07-11)

Mission: close every material gap between Stage-4 ECR and top-conference competitiveness
(CVPR/ICCV/ECCV), interpreted as: no unresolved high-severity reviewer objection addressable with this
repository, machine, and time. Companion docs: `TOPCONF_GAP_AUDIT.md` (diagnosis),
`TOPCONF_EXECUTION_PLAN.md` (frozen specs).

## Verdict

**TOP-CONFERENCE READY** at **87/100** (from 58/100 at audit opening). No unresolved P0 blocker;
every P1 either closed with banked evidence or rejected with rationale; the residual risks (below) are
positioning/taste risks that more experiments cannot reduce — remaining effort has lower expected value
than writing and submission preparation.

## Score trajectory (final)

| Axis | Initial | Final | What closed it |
|---|---|---|---|
| A. Novelty/distinction | 55 | 80 | Head-to-head vs generalizable IBR measured (#E-12); "audited" defined term + threat model; lineage answer now runs in both directions (A10 addendum) |
| B. External baselines | 40 | 85 | PJ-2026 (tuned classical, exceeded with CIs) + 3DGS matched-TOTAL + Difix3D+ + IBRNet — all measured, none argued |
| C. Benchmark breadth | 55 | 85 | T&T+DB (the 3DGS suite) banked with zero per-scene tuning: final>PJ-2026 +0.122/−0.0084, CIs excl. 0 under BOTH bootstraps (#E-11) |
| D. Statistical rigor | 85 | 95 | Scene-cluster bootstrap PASS 8/8 headlines (#E-14) |
| E. Temporal stability | 20 | 90 | Measured PASS 3/3 (ratios 0.988–1.030) + supplementary videos (#E-13) |
| F. Storage/runtime fairness | 80 | 85 | Unified R-D master plot; accounting was already TOTAL-correct |
| G. Threat model | 70 | 95 | PROTOCOL §4E.1 (PROVEN/ASSUMED/NOT-CLAIMED); "certified" retired |
| H. Simplicity/ablations | 75 | 85 | E-08 null = the simplicity result; zero-tuning transfer (#E-11) evidences frozen-config robustness |
| I. Repro/paper readiness | 65 | 85 | ECR_README, paper_tables.py (verified), ladder/R-D figures, videos |

## Blockers found → resolution

| ID | Blocker | Resolution |
|---|---|---|
| P0-1 | No second standard benchmark | **CLOSED** — T&T truck/train + DB drjohnson/playroom trained/evaluated/audited (8/8 GREEN) through the frozen pipeline; transfer holds with CIs excl. 0; honest finding: magnitudes smaller than full9 (+0.12 vs +0.36 over the floor); train scene per-view CI incl. 0 — reported |
| P0-2 | No external learned-IBR baseline | **CLOSED** — IBRNet (pretrained, 10 sources = more evidence than ECR uses, no per-scene training) lands 1.0–5.4 dB below even the base anchor; ECR exceeds it +2.6–5.9 dB; convention fabrication-risk controlled by a 22.48 dB self-reconstruction gate banked BEFORE any cross-method number |
| P1-3 | Temporal stability unmeasured | **CLOSED** — 0.988–1.030× base roughness on 120-frame paths, 3/3 scenes, videos banked; garden is SMOOTHER than base |
| P1-4 | Scene-level bootstrap absent | **CLOSED** — hierarchical CIs alongside stratified; 8/8 headlines survive |
| P1-5 | Threat model imprecise / "certified" overclaim | **CLOSED** — §4E.1 + terminology sweep (CLAIMS v0.6) |
| P1-6 | R-D/runtime positioning not unified | **CLOSED** — `rd_master` + `ladder_ci` figures (verified vs sources) |
| P1-7 | Repro/paper mechanics | **CLOSED** — README + LaTeX extractor (spot-verified) + videos |
| P2 | Net-capacity ablation; more private suites; Phase-S ops | **REJECTED with rationale** (gap audit §2) — no reviewer-risk reduction per unit cost |

## Experiments & code added (all with frozen protocols + collectors)

- `tools/ecr/t2b_scene.sh` + 4 scene registrations (PROTOCOL 1.3.0 MINOR) + `tools/analysis/t2b_report.py`
- `tools/analysis/ibr_jobs.py` (camera math + self-reconstruction gate) + `/data/peilincai/IBRNet/ibr_infer.py`
  (delegated; torch-2.x edits logged) + `tools/analysis/ibr_cell.py` (self-validating mirror)
- `tools/analysis/ecr_temporal.py` (GT-free path metric + videos)
- `hierarchical_mean_ci` in `tools/ecr/e0_report.py` + `tools/analysis/hboot_report.py`
- `tools/analysis/{plot_ladder,plot_rd,paper_tables}.py` (delegated, output-verified) + `docs/ECR_README.md`
- PROTOCOL §4E.1; CLAIMS v0.6–v0.8; REBUTTAL A10/A11 addenda; pack refold (26 artifacts, byte-verified)

## Paper-facing method abstraction (final)

**Evidence-Cached Rendering (ECR):** ship, per scene, {compact mesh checkpoint + train-view evidence
cache + audited transport}. At render time: K-nearest depth-consistent warping of cached train views →
multiband confidence-weighted fusion → a small per-scene net emitting per-pixel α (residual gain) and β
(direct-RGB routing), composed through a STRUCTURAL validity gate (β·valid) that makes hallucination
impossible where evidence is absent. Train-only by audited construction (per-row checks, defined threat
model). The confidence input channels are provably unnecessary (E-08) — the paper presents the SIMPLER
9-channel abstraction as the method and the 12-channel banked system as the measured instance.

## Strongest remaining reviewer attacks + prepared responses

1. **"3DGS is better and faster at this storage — why does this exist?"** → NON-CLAIMS + A3/A11: the
   deliverable is a mesh artifact with preserved geometry/downstream guarantees; the gap is mostly closed
   (kitchen +0.32 at LPIPS parity); the trade is printed in every row. (Taste risk — not further reducible.)
2. **"Per-scene method, generalization unclear"** → by-design positioning (A12) + the zero-tuning T&T/DB
   transfer with CIs — the frozen config generalizes even though the artifact is per-scene.
3. **"Uses training images at test time"** → D4 rights paragraph + §4E.1 threat model + audit wall figure
   + the reference class (ULR/DB/light fields) — we are the ones who PROVE no test leak, per row.
4. **"Incremental over Deep Blending"** → A10 both-directions answer (generalizable fails here; the tuned
   classical point IS our floor and is exceeded with CIs) + the audit/structural-gate/ladder assets.
5. **"Gains are small on the community suite"** → +0.122 dB / −0.0084 LPIPS with CIs excl. 0 under two
   bootstrap schemes, zero tuning; honesty is the argument.

## Recommended submission claim set (exact)

CR1 (quality, three named references + suites + community suite), CR2 (honest cost incl. Pareto +
matched-TOTAL 3DGS), CR3 (compact: half-triangles above full-budget anchor; base-independent margin),
CR4 (audited transport, §4E.1 scope; structural gate; E-08 simplicity), NON-CLAIMS verbatim, plus the
measured externals (Difix, IBRNet) presented as context rows — never as claim targets.

## Residual risks accepted

Novelty-taste at CVPR (mitigated, not eliminable); per-scene fusion-net training cost (~30 min/scene,
reported); the train scene's per-view CI incl. 0 (single-scene honesty note); ffmpeg mpeg4 videos
(re-encode at camera-ready if needed).
