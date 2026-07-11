# GEMS Stage 4 — Evidence-Cached Rendering (ECR): Complete Summary

Written 2026-07-11 at Stage-4 closure. Program spec: `docs/GEMS_Stage4_ECR_Prompt.md`.
Every number below is read from a banked artifact (no hand-typed results); pointers are given per section.
Statistics convention throughout: paired per-view bootstrap, 10k resamples, seed 0; full9 means are
stratified mean-of-scene-means; brackets are 95% CIs.

---

## 1. What Stage 4 set out to do

Open a new track on the closed GEMS program (Stage-2/3 evidence pack v4 FROZEN, untouched) and deliver a
system that **strictly exceeds Phase-J** (the archived evidence-lumigraph transport, re-frozen as PJ-2026)
under the modern single-mouth protocol, with every exceedance CI-backed:

- One evaluation mouth: `run_eval.py --renderer {base,ecr}` (PROTOCOL v1.2.0), one frozen config across scenes.
- D4 redefined for Stage 4: **train-view evidence is a legal render-time input** of the shipped artifact;
  test-view GT is absolutely forbidden anywhere in the render path (no per-test-view parameters).
- Delivery bar: final stack > PJ-2026 on full9 mean PSNR **AND** LPIPS, CIs excl. 0; target ≥ +0.15 dB,
  stretch ≥ +0.30 dB — or honest escalation.
- Positioning (frozen): the per-scene shipped artifact is the triple
  **{mesh-splat checkpoint + evidence cache + transport renderer}** — the same representation class as
  surface light fields / unstructured lumigraph / deep blending, compared as such.

## 2. Final verdict (the verbatim closing line)

> **"Stage 4 closed: ECR final stack = PJ-2026 + [+0.3607 dB [+0.3158,+0.4067] / −0.01686 LPIPS
> [−0.01880,−0.01495]] (CIs); deliverable exceeds Phase-J = YES by margin (stretch ≥ +0.30 met at CI-low;
> 8/9–9/9 per-scene CI-wins; exceedance also holds at half the triangle budget: +0.336/−0.0176 over
> PJ-2026 on B50, 9/9); evidence and handoff ready for writing."**

## 3. Infrastructure delivered (GOAL #E-00)

- `tools/ecr/build_cache.py` — evidence cache builder: train-view renders (PNG), real train GT bytes,
  median surf-depths (float32 npy), camera index, manifest with the frozen transport config, train-only
  LOO-calibrated α (and K for multiband), file sizes, and lossless-compressed size (tar+zstd −9).
- `tools/ecr/renderer.py` — `EcrRenderer`: frozen-config transport over one cache. D4 guarantees by
  construction: only pose primitives cross the boundary (never a GT-bearing Camera object);
  `ConfinedFrameLoader` (realpath-confined reads + read log + in-memory target injection +
  raising GT sentinel); per-call kwargs re-hash so audits can prove zero per-test-view parameter injection.
- `run_eval.py` extended (base path byte-identical; PROTOCOL_VERSION 1.2.0): `--renderer ecr`,
  `--ecr-cache`, `--save-renders`; new cost columns cache_mb_raw / cache_mb_compressed /
  transport_ms_per_frame / end_to_end_fps / total_artifact_mb.
- `tools/audit_test_path.py --ecr` — proves per row: all transport reads ⊆ manifest; manifest train views
  disjoint from the independently recomputed test split; checkpoint fingerprint match; frozen per-view
  kwargs hash; no `original_image` token in the render path. Stage-2 base-mode audit preserved and
  dynamically strengthened (base mode must never load tools.ecr).
- Self-test (garden): +1.260 dB [+1.071,+1.444] vs primary anchor — matches the archived Phase-J delta
  (+1.28); both audit modes GREEN.
- **Final audit trail: 96/96 ecr-mode audits GREEN across every reported row.**

## 4. M-E0 floor: PJ-2026 reproduces (GOAL #E-01)

28/28 rows banked (both bases × full9 + 4 SS3DM towns + toy_parking), 28/28 audits GREEN.

- **AT-E0 PASS:** PJ-2026 vs PRIMARY anchor, full9 mean **ΔPSNR +1.3055 [+1.2260,+1.3844]**,
  ΔLPIPS −0.0594; 9/9 CI-positive (min stump +0.273, max bonsai +3.010). Matches the archived +1.33.
  PJ-2026 = incumbent stack v0, the guaranteed floor.
- Suites: SS3DM towns +1.43..+2.88 (4/4 CI-positive); toy_parking +1.091 with ΔLPIPS ≈ 0 — the
  pre-registered ring-coverage weakness, reported honestly.
- Secondary base (B5@B50): +1.1515 [+1.0623,+1.2378] over the *full-budget* primary anchor.
- Calibrated per-scene α (train-LOO, frozen in each manifest): garden/room/counter 0.75;
  bicycle/flowers 0.5; stump/treehill 0.25; kitchen/bonsai/towns/toy 1.0.

## 5. The exceedance ladder (≤2 mechanisms/rung; promotion floor: full9 ΔPSNR ≥ +0.10 OR ΔLPIPS ≥ 0.004, CI excl. 0)

| Rung | Mechanism | Full9 result vs incumbent | Verdict |
|---|---|---|---|
| L1b (#E-02) | PJ transport on a distilled base | **−0.1088 [−0.1291,−0.0901]**, LPIPS worse | **NOT PROMOTED — banked negative:** the transport *subsumes* the distillation channel |
| L2 (#E-03) | joint (K,α) train-LOO over K∈{2,4,8} + 4-band Laplacian confidence-weighted fusion | +0.0963 [+0.0614,+0.1326]; ΔLPIPS −0.00464 [−0.00569,−0.00358] | **PROMOTED → v1** (LPIPS floor, CI excl. 0) |
| L3 (#E-04) | learned per-pixel fusion: FusionNet U-Net-S 844k, 9-ch input, α-map head (bias +1.0), L1+0.2·DSSIM, 3000 steps, seed 0, last iterate | ΔLPIPS −0.00571 [−0.00694,−0.00445]; dPSNR +0.0601 | **PROMOTED → v2**. α ≤ 1 bound recovers low-α scenes but regresses α=1 ceilings (bonsai −0.141) — the measured opening for L4 |
| L4 (#E-05) | routing head: 12-ch input (+K-fused warped GT color), 2-ch head (α, β; β bias −4.0 starts at L3), compose = clamp((1−β·valid)(base+α·signal) + β·valid·color) | **+0.2043 [+0.1836,+0.2253] AND −0.00652 [−0.00723,−0.00583]**, 9/9 CI-positive on both metrics | **PROMOTED ON BOTH FLOORS → v3 = FINAL STACK.** Pre-registered prediction verified: recovers bonsai (+0.550) and kitchen (+0.311) exactly |

**Ladder closed at v3.** Cumulative vs PJ-2026 floor (`l4_vs_floor.json`):
**ΔPSNR +0.3607 [+0.3158,+0.4067] AND ΔLPIPS −0.01686 [−0.01880,−0.01495]** — stretch met at CI-low.
Vs legacy anchor +1.8090 [+1.7061,+1.9112] (9/9); vs PRIMARY anchor +1.6662 [+1.5669,+1.7658] (9/9).

## 6. Suites and the compact variant

- **Suites (final stack, clean30k base):** vs PJ-2026 — town01 **+0.412 [+0.315,+0.512]**, town02 +0.298,
  town03 +0.519 (all CI excl. 0); town06 +0.095 [+0.036,+0.161] (CI-positive but under the +0.10 mean
  floor, ΔLPIPS worse +0.013 — honest boundary case); toy_parking +0.029 (CI incl. 0 — transport already
  saturated at E0 by ring coverage). Net: exceedance extends to 3/5 suite scenes outright.
- **L6 compact rows (final stack on B5@B50, HALF the triangles, full9):**
  vs the FULL-BUDGET primary anchor **+1.4877 [+1.3793,+1.5929] AND ΔLPIPS −0.07478 [−0.07751,−0.07195]** (9/9);
  vs PJ-2026 on the same B50 base **+0.3362 [+0.2935,+0.3799] / −0.01759** (9/9) —
  **the ladder's margin over Phase-J is base-independent** (full-budget: +0.361/−0.0169).

## 7. Honest cost (T-ECR-2) and the storage story

- TOTAL artifact (checkpoint + raw cache): 0.77–3.5 GB/scene; transport 71–1054 ms/frame;
  end-to-end 0.97–8.9 fps (shared-GPU timing caveat).
- **L5 cache Pareto (#E-06; 15 points = {jpeg95, jpeg85, jpeg70, halfres, ksubset50} × {garden, bicycle,
  kitchen}; each variant re-ran the frozen (K,α) recalibration + routed-net training on its OWN bytes;
  15/15 audits GREEN):** jpeg95 is nearly free — garden −0.095 dB / kitchen −0.081 dB at ~22% TOTAL
  savings, bicycle **+0.315 dB** (JPEG re-encode denoises that scene's evidence); halfres = the aggressive
  point (−0.85..−0.95 dB at 44–58% savings); ksubset50 dominated on kitchen (−0.72).
- **Matched-TOTAL-storage 3DGS (#E-07; zero new training needed):** vanilla 3DGS-30k sits UNDER the ECR
  TOTAL budget on all 3 scenes (uses 43%/55%/11% of it) and stays ahead **+1.13 / +1.53 / +0.32 dB**
  (garden/bicycle/kitchen) at ~40–70× the end-to-end FPS — reported plainly (context-only, R1-precedent
  sanctioned exception). The Stage-2 R1 gap (2.1–3.4 dB vs B5@B50) is mostly closed; kitchen is now
  +0.32 dB at near-parity LPIPS (0.1270 vs 0.1321). Exactly the frozen NON-CLAIMS trade.

## 8. External cell: Difix3D+ (#E-09) — measured, not argued

Bounded feasibility (the prompt's 2-day cell) resolved FEASIBLE after two Codex-sandbox false negatives
and one real repo trap: the checkout's `inference_difix.py --model_name` path is **dead code** (loads
sd-turbo + random skip-convs, no Difix weights → garbage output). Correct route:
`src/pipeline_difix.py::DifixPipeline.from_pretrained("nvidia/difix_ref")` (~2.2 s/img single-step, GPU 7).

Result (mirror self-validated ≤0.01 dB against the banked base rows; reference per view = train GT of the
transport's own top support view — the SAME evidence rights ECR has): Difix3D+ on our base renders is
**PSNR-negative on all 3 scenes** (garden −1.543 [−1.738,−1.345] with LPIPS WORSE +0.005;
bicycle −1.128, kitchen −0.886 with LPIPS −0.030), and the **ECR final stack exceeds the Difix-enhanced
base on BOTH metrics on all three scenes** (kitchen 30.486/0.1321 vs 26.723/0.1805).
"Base + generative enhancer" is not a shortcut to this deliverable; REBUTTAL A11 now answers with numbers.

## 9. Ablations, failure cases, qualitative assets

- **E-08 confidence-inputs-off ablation (pre-registered):** zeroing the net's 3 confidence input planes at
  train AND test leaves quality statistically unchanged (bonsai −0.039 [−0.123,+0.020]; garden +0.010
  [−0.033,+0.046]) → **CR4 SHRUNK (CLAIMS v0.3):** the certification attaches to the audits + the
  STRUCTURAL compose gate (β·valid — evidence RGB physically cannot route where the transport lacks
  support), not to the net's confidence input channels. A first-class negative that *strengthens* the
  safety story: the gate is structural, not learned.
- **E9-style failure cases (frozen selection rules; `ecr_failure_cases.md`):** across **139 dumped full9
  test views, exactly 1 is transport-negative** (treehill _DSC8946, −0.06 dB, occlusion-seam close-up);
  the archetype coverage-gap (bicycle _DSC8784, 47% covered — entire foreground unsupported) still gains
  +0.13 dB because the gate withholds the transport there. Graceful degradation, no hallucination.
- **Qualitative grids (frozen E11-style crop rule; `RESULTS/figures/ecr_qual/`):** 5 scenes ×
  {GT, base, PJ-2026, ECR final, β map, confidence} × {best, median, failure} with per-view selection and
  crop windows logged in `manifest.json`. The β maps are the storytelling core: routing concentrates on
  high-frequency well-supported regions and vanishes at occlusion boundaries.

## 10. DS-1 dense-carve retry (#E-10, prompt §6) — FAIL, impossibility STRENGTHENED

The ONE permitted reopening of R3-FINAL (pre-registered: ray stride 16→2, FREE dilation by r_inf,
UNKNOWN = traversable at ×5 cost via a default-off `cell_cost_mult` hook in `planner_loop.astar`;
V1 params frozen, NO recalibration). Verdict: **courtyard 0/100 on both checkpoints — FAIL**, but with a
sharper diagnosis than V1: the failure MODE changed from UNKNOWN starvation to pose invalidation by the
model's own OCCUPIED set (92 start_invalid + 8 goal_invalid; lethal fraction 88.6%), and **P1 breaks at
dense sampling** (raw FREE false-free 6.2% → 19.1%). The binding constraint is the **metric accuracy of
the checkpoint's rendered depth** — not density, semantics, or planner design. Toy: 0→6/100
(0 collisions, far under the bar). `CONSUMPTION_IMPOSSIBILITY.md` amended (strengthened); the §6
reopening is spent. Poignant contrast, now documented: the SAME rendered evidence that cannot form a
collision-grade world model is worth +1.67 dB as render-time photometric evidence.

## 11. Claims state (CLAIMS_ECR.md v0.5 — all instantiated, no open slots)

- **CR1 Quality:** +1.666 [+1.567,+1.766] over PRIMARY (9/9); +0.361/−0.01686 over PJ-2026 (8/9);
  suites 3/5 CI-positive with honest boundary cases.
- **CR2 Honest cost:** matched-TOTAL 3DGS measured; Pareto banked (jpeg95 ~free; halfres −0.9 dB at ~50%).
- **CR3 Mesh retained:** compact stack +1.488/−0.0748 over the full-budget anchor; +0.336/−0.0176 over
  PJ-2026 same-base (9/9); geometry/downstream frozen results cited (depth untouched by construction).
- **CR4 Certified transport (shrunk v0.3):** 96/96 audits GREEN; certification = audits + structural
  compose gate; the explicit confidence input channels are provably redundant (E-08).
- NON-CLAIMS preserved verbatim (per-scene method; no cross-scene generalization; no test-GT anywhere;
  R3 impossibility cited as scope; references named explicitly).

## 12. Incidents log (all resolved, no banked rows affected)

1. **Host reboot 2026-07-09 12:44** silently killed the suite/L6/L5 queue (no .exit files; stale logs
   discovered 07-10). Relaunched from idempotent launcher scripts; all gates re-armed on
   **exit-code == 0** (not file existence — a flaw exposed by incident 2).
2. **GPU-contention OOM** killed the quals chain (foreign 36 GB process on GPU 5); the mere-existence
   gate then fired downstream jobs early. Fixed + hardened; kitchen/garden-PJ dumps re-run.
3. **L4 first launch** crashed on a kwargs leak (`alpha` into the routed feature call) — fixed
   (`if self._fusion_net is None`), relaunched; no results affected.
4. **Difix `--model_name` dead code** (see §8) — root-caused by inspection; correct pipeline route used.
5. **LEDGER placement repairs**: E-06 block once misplaced inside E-05; the E0 pre-registration paragraph
   once stranded at EOF — both moved to their proper goals.

## 13. Where everything lives

- **Repo pack (byte-verified, `tools/ecr/fold_pack.sh`):** `RESULTS/STAGE4_ECR/`
  {tables: final_stack_tables, e0 tables, e07 matched-storage, l5_pareto, failure cases;
  gates: l1–l4 + vs-floor jsons; difix: table + attempt log; sha256_manifest.txt} —
  18 artifacts. Qual grids: `RESULTS/figures/ecr_qual/`.
- **Working artifacts:** `/data/peilincai/gems_stage1/` — caches `ecr_cache/<scene>_<tag>`,
  rows `eval/*` (metrics.json + audits), analyses `analysis/{e0_pj2026,final_stack,difix_cell,quals,ds1_dense_carve}`.
- **Governance docs:** `LEDGER.md` (STAGE 4 section: STACK BOARD, GOALs #E-00..#E-10, incidents, closure
  checklist), `MATRIX.md` (ECR section, all cells DONE), `CLAIMS_ECR.md` v0.5, `PROTOCOL.md` v1.2.0 (§4E),
  `RESULTS/CONSUMPTION_IMPOSSIBILITY.md` (DS-1 addendum),
  `SUBMISSION_HANDOFF/{VENUE_MEMO,ABSTRACT_SKELETON,FIGURE_NOTES,REBUTTAL_BANK}.md` (Stage-4 refresh).
- **Code:** `tools/ecr/` (build_cache, renderer, transport_l2, fusion, train_fusion, e0/e07/l5/final
  report builders, rung_gate, per-rung scene scripts, fold_pack), `tools/analysis/`
  (ecr_dump_quals, ecr_failure_cases, ecr_qual_grids, difix_cell), `tools/gems/ds1_dense_carve.py`,
  default-off hooks in `tools/ecr/{fusion,renderer,train_fusion}.py` (ablate_conf) and
  `tools/gems/planner_loop.py` (cell_cost_mult).

## 14. What remains (paper writing — human/next session)

Prose only; every number is banked. Suggested spine per the refreshed handoff: lead with CR1+CR3
(evidence-cached rendering beats the baked baseline, and does so at half the mesh budget), the ladder as
the method narrative (with L1/E-08 negatives inline), the β/confidence quals as the visual core, honest
cost + Pareto + matched-3DGS + Difix as the trade section, failure cases + impossibility contrast as the
scope section. Venue: 3DV/WACV strong fits; CVPR materially improved (A10 ULR/Deep-Blending-lineage
positioning is the paragraph to get right).
