# PROJECT STATUS HANDOFF — complete state for external analysis

Written 2026-07-13 at the pre-submission freeze (`presubmission-freeze-v1`, branch
`neurips-meshsplatopt-repair`). Purpose: a self-contained entry point for analyzing this repository
and planning the path to maximum top-conference competitiveness. Every number cited here is
regenerable from banked artifacts (see §6); nothing is hand-typed at the source.

---

## 1. What this project is (the thesis)

**"Caching evidence beats baking it."** For mesh-first pipelines: instead of pushing training-view
information INTO the representation (three program stages falsified that route: static baking captures
4.76% of the achievable gap, distillation 5–16%, held-out residual cosine ≈0.21), ship a compact
triangle-mesh checkpoint PLUS a train-view evidence cache PLUS an audited render-time transport.
The per-scene artifact class is deliberately that of surface light fields / ULR / Deep Blending —
with three assets that lineage lacks: per-row purity audits under a written threat model, a structural
per-pixel evidence gate with a measured worst case, and (new) edit-consistency of the cache via exact
mesh-face provenance.

One primitive carries the whole story: **per-pixel evidence gating on face provenance** does
(a) quality — the transport ladder; (b) safety — the β·valid compose gate + audits; (c) editability —
provenance invalidation.

## 2. Complete results inventory (headline numbers, all CI-backed)

### Stage 4 — the quality system (CLOSED)
- Final stack (multiband K-source warp → learned per-pixel fusion → residual-vs-RGB routing) vs the
  strong frozen Phase-J floor (PJ-2026), full9/mip-NeRF-360: **+0.361 dB [+0.316,+0.407] AND
  ΔLPIPS −0.0169 [−0.0188,−0.0150]** (8/9 per-scene CI-wins); vs the primary anchor +1.666
  [+1.567,+1.766] (9/9); ladder rungs individually gated incl. one banked NEGATIVE (L1 distillation
  composition: −0.109).
- **L6 compact**: at HALF the triangles, the stack beats the FULL-budget anchor by **+1.488
  [+1.379,+1.593] / −0.0748 LPIPS** (9/9); margin over Phase-J is base-independent (+0.336 on B50).
- Suites: SS3DM towns +0.10..+0.52 over PJ-2026 (3/4 CI-positive; town06 LPIPS-worse honestly);
  toy_parking saturated (CI incl. 0, pre-registered coverage bound).

### Stage 5 — TOPCONF hardening (CLOSED at 87/100)
- **Community benchmark transfer** (T&T truck/train + DB drjohnson/playroom, zero per-scene tuning):
  final vs PJ-2026 **+0.122 [+0.070,+0.176] / −0.0084 [−0.0105,−0.0064]**, CIs excl. 0 under BOTH
  stratified and scene-cluster bootstraps; per-scene: 3/4 PSNR CIs excl. 0 (train +0.130 incl. 0).
- **External baselines all measured**: 3DGS at matched TOTAL storage still ahead +0.32..+1.53 dB at
  40–70× fps (printed, NON-CLAIMS); Difix3D+ enhancer PSNR-negative on all 3 scenes and dominated by
  ECR on both metrics; IBRNet (10 source views, more evidence than ECR uses) lands 1.0–5.4 dB BELOW
  the base anchor (ECR +2.6..+5.9 above it; camera chain verified by a 22.48 dB self-reconstruction
  gate before any number).
- **Temporal stability measured**: 120-frame paths, final/base roughness 0.988–1.030 (3/3 PASS;
  garden smoother than base); videos banked.
- Statistics: scene-cluster (hierarchical) bootstrap PASS 8/8 headline intervals; threat model written
  (PROTOCOL §4E.1); "certified"→"audited" terminology; 96/96 purity audits GREEN at that point
  (more added since; every ECR row ships an audit report).
- Storage honesty: L5 cache Pareto (jpeg95 ≈ free at ~22% savings; halfres −0.9 dB at ~50%);
  TOTAL = checkpoint+cache+net everywhere.

### Route A — edit-aware ECR (CLOSED, red-teamed, oracle-verified)
- **Mechanism**: per-source-pixel validity masks (face-ID pass on the original checkpoint; 1-px
  dilation) multiplied into warp confidence at the single sampling site all fuse modes share;
  default-off and bit-stable (banked view reproduces to the last float).
- **Two non-obvious findings**: (1) REBUILD INVERSION — rebuilding the cache from the edited
  checkpoint is the WORST strategy (+3.09 dB [+2.57,+3.62] ghost; refreshed depths make stale
  photographs depth-consistent); (2) PROTECTION ASYMMETRY — the depth z-test accidentally shields
  deletion staleness but recolors sail through (stale cache repaints old color +1.96 [+1.87,+2.06];
  chained +2.67 [+1.33,+4.09]). No naive strategy survives both edit classes.
- **Oracle-verified novelty** (verified synthetic rebuild without the object; cameras/splits
  byte-identical): exact provenance beats 4-px dilation +0.164, 16-px +0.294, 2D-box +0.383,
  target-side masking +0.263, rebuild +1.133 — **all five survive 99% CIs (Bonferroni)**; region
  genuinely gains +0.49 dB from retained disoccluded-background evidence; coarse masks bleed up to
  −0.74 dB OUTSIDE the region.
- Preservation: unaffected-region TRUE-GT within −0.020..+0.015 dB of the pre-edit system; ECR's
  +0.9–1.7 dB gain survives editing; temporal after edit 0.982.
- Cost/locality (measured wording: "region-proportional"): peripheral object = 57/161 views affected;
  **sparse sidecar 12.8 MB** (18× under dense, ~100× under rebuild), validated same-process bit-equal
  on 72/72 affected views; chained edits supported (parent-mask inheritance).
- Boundaries EVIDENCED: translation figure (shared-vertex tearing + shadowless arrival);
  deformation declared unsupportable without hallucination.
- Cross-process sensitivity study: method-rank ρ=0.983, 7/7 CI conclusions stable.

### Downstream/geometry scope (frozen Stage-2/3 + DS-1)
Consumption IMPOSSIBILITY×4 stands, STRENGTHENED by the DS-1 dense-carve retry (P1 breaks at dense
sampling: false-free 6.2%→19.1%; cause isolated to rendered-depth metric accuracy). The paper cites
this as scope: the same evidence that fails as a world model succeeds as photometric evidence.

## 3. Claims state (CLAIMS_ECR.md v1.1 — the exact submission set)

CR1 quality (3 named references, full9 + 2 suites + community set) · CR2 honest cost (Pareto +
matched-TOTAL 3DGS) · CR3 compact (half-triangles above full-budget anchor; base-independent) ·
CR4 audited transport (defined term, §4E.1 threat model; structural gate; conf-input-redundancy
ablation) · CR5 edit-consistent evidence (deletion+recolor ONLY; oracle-verified; region-proportional
cost; synthetic-only oracle scope stated). NON-CLAIMS verbatim: per-scene method; no arbitrary
editing; no test-GT anywhere; no geometry/planning claims; references always named; the falsification
corpus is the motivation.

## 4. HONEST VULNERABILITY INVENTORY (for the next analysis — attack here)

Ordered by my estimate of reviewer risk:
1. **Novelty taste** (highest residual risk, not evidence-addressable): "polished ULR/Deep-Blending
   descendant". Mitigations in place: A10/A13 measured both directions; the one-gate-three-jobs
   framing. A reviewer who rejects the artifact class premise is not persuadable by more data.
2. **3DGS dominance at matched storage** (+0.32..+1.53 dB at 40–70× fps): printed honestly; the mesh
   artifact + editing + audits are the counter-story. Vulnerable to "why not edit 3DGS then?" —
   answered (no evidence cache to invalidate ⇒ no +1.67 dB to keep), but the argument is subtle.
3. **Absolute quality level**: the base mesh renderer is weak (garden base 24.8 vs 3DGS 27.5);
   all gains are relative to mesh-class anchors. A "numbers too low" desk reaction is possible.
4. **Stale-cache tie on deletion** (C5−C2 = −0.028 [−0.077,+0.007] on oracle): kept honestly;
   recolor/chained break the tie; but a hasty reader may anchor on it.
5. **Oracle is synthetic-only**; real-scene in-region claims are bounded (ghost similarity +
   preservation). Inherent; stated everywhere.
6. **leak_R proxy weakness is self-reported** (ρ=0.50, unstable 0.50→0.27 across processes): a
   strength for honesty, but quotable out of context.
7. **Editing breadth**: 2 classes, 2 scenes + synthetic, 6 cells; no indoor edit cell (rejected with
   rationale); pot-deletion residue = base-representation debris in clutter (banked limitation figure).
8. **Cross-process render nondeterminism** (documented; within-run CIs valid; sensitivity PASS) — a
   determinism purist may still object.
9. **Per-scene fusion-net training** (~30 min/scene) and storage (0.8–3.5 GB TOTAL/scene, Pareto
   mitigations) — deployment-cost objections.
10. **Suite boundary cases**: town06 LPIPS-worse; toy saturated; T&T-train CI incl. 0 — all honest,
    all quotable.

## 5. What is NOT done
- The paper PROSE (LaTeX) — next phase; plan in `docs/stage4_paper_plan.md` (+§6.5 editing addendum);
  all tables/figures/claims machine-generated and frozen.
- Venue deadline verification (3DV primary / WACV backup / CVPR stretch — HUMAN-VERIFY dates).
- Camera-ready niceties: h264 re-encode of supplementary videos; F1 pipeline art (human).
- Optional follow-ups explicitly deferred: translation edit class, 3DGS-editing context row,
  standalone editing paper.

## 6. Repository map (read in this order)
1. This file → `docs/stage4_sum.md` → `docs/stage5_summary.md` → `docs/stage5_A.md` (+addendum).
2. Route A: `docs/EDIT_AWARE_ECR_{FEASIBILITY_AUDIT,PROTOCOL,EXECUTION_PLAN,VALUE_REPORT}.md`,
   then `docs/ROUTE_A_TOPCONF_{GAP_AUDIT,EXECUTION_PLAN,READINESS_REPORT}.md`.
3. Claims/governance: `CLAIMS_ECR.md` (v1.1), `PROTOCOL.md` (v1.3.0, §4E/§4E.1), `LEDGER.md`
   (STAGE 4 → GOAL #E-16, append-only history incl. incidents), `MATRIX.md` (ECR section),
   `docs/PUBLIC_REFERENCE_MAP.md` (internal IDs → paper locations).
4. Evidence pack: `RESULTS/STAGE4_ECR/` (47 byte-verified artifacts + `sha256_manifest.txt` +
   `FREEZE_MANIFEST.md`); canonical Route-A table `RESULTS/STAGE4_ECR/edit_aware/routeA_master_table.md`.
5. Paper assets: `RESULTS/tables_tex/` (LaTeX bodies), `RESULTS/figures/{ecr_paper,ecr_qual,edit_aware}/`,
   `SUBMISSION_HANDOFF/{VENUE_MEMO,ABSTRACT_SKELETON,FIGURE_NOTES,REBUTTAL_BANK}.md` (A1–A13).
6. Reproduction: `docs/ECR_README.md`; generators under `tools/{ecr,analysis,edit}/` (all runnable;
   pipeline sweep green at freeze: tests OK, 78-file compile clean, banked config hash bit-stable).
7. Raw artifacts (off-repo, this machine): `/data/peilincai/gems_stage1/` (models, caches, eval rows
   with per-row audit reports, analyses); datasets incl. the verified oracle
   `gems_stage1/datasets/toy_parking_nocar0`.

## 7. One-paragraph verdict of record

The measurement program is complete and internally attack-hardened: readiness 87/100 (system) and
91/100 (editing) on my harsher scale, with the operational register empty — every P0/P1 either closed
with banked CI evidence or rejected with written rationale, every claim traceable to a byte-verified
artifact, and every known weakness enumerated above rather than hidden. The remaining competitiveness
gap is not experimental: it is the paper's framing quality against vulnerability #1–#3, which is
writing work — plus whatever a fresh adversarial analysis of this handoff finds that I did not.
