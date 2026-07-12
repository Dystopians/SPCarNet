# TOPCONF EXECUTION PLAN — frozen specs

Opened 2026-07-11. Every experiment: hypothesis, frozen protocol, acceptance criterion, artifact path,
collector. No open-ended sweeps. Hard laws inherited: no test-GT, single mouth, TOTAL storage accounting,
banked-numbers-only, frozen Stage-2/3 + banked Stage-4 artifacts untouched (new rows only).

## EXP-T2B — second standard benchmark (closes P0-1)

- **Hypothesis:** the frozen ECR ladder (unchanged configs, per-scene train-only calibration) transfers to
  the community T&T/DB suite: final stack > PJ-2026 floor per scene, and PJ-2026 > clean30k anchor by a
  margin comparable to full9 (+1.0-ish dB).
- **Frozen protocol:** scenes = {tandt_truck, tandt_train, db_drjohnson, db_playroom} (the exact 3DGS eval
  suite). Ingestion: COLMAP, `images`, `-r -1` via the repo's standard loader, split = every-8th (llff8
  convention, same as full9). Anchor = clean30k (train.py 30k, seed 0, SS3DM/toy cfg pattern — the
  documented convention for scenes without a 26k save; NO cleanfixed continuation, dual-anchor caveat
  logged). Rows per scene, all through run_eval: (1) base anchor; (2) PJ-2026 cache (build_cache single
  fuse, frozen ADAPTER_CONFIG, train-LOO α); (3) final stack (l2mb multiband cache → routed FusionNet,
  frozen trainer) — identical scripts to the suite chain (`final_stack_scene.sh`). `--ecr` audit per ECR row.
- **Acceptance:** rows + audits GREEN banked for ≥3 of 4 scenes (any scene whose base training diverges is
  reported as-is with its log); per-scene paired CIs + 4-scene stratified mean; NO promotion gate (the
  ladder is closed — these are transfer/validity rows). Honest reporting whatever the numbers say.
- **Artifacts:** `gems_stage1/models/<scene>_clean30k`, `eval/{<scene>_clean30k_v1, e0_<scene>_clean30k_pj2026_v1, final_<scene>_clean30k_v1}` + audits; collector `tools/analysis/t2b_report.py` → `analysis/final_stack/t2b_tandt_db.{md,json}`.
- **Compute:** 4× clean30k training (~4–8 h/scene, GPUs 0/3/4/7 shared) → then the standard cache/eval chain (~2 h/scene).
- Owner: me (scene registry + protocol + report collector); Codex (dataset staging, launcher plumbing).

## EXP-IBR — external learned-IBR baseline cell (closes P0-2)

- **Hypothesis (honest, either-way):** a pretrained generalizable IBR model (ENeRF preferred, IBRNet
  fallback) given the SAME K-nearest train views renders our test views worse than the ECR final stack on
  PSNR and LPIPS at these resolutions (it was not trained per-scene); reported plainly whichever way.
- **Frozen protocol:** Difix/E-09 pattern (sanctioned mirror exception, self-validated vs a banked row —
  metric mirror identical to run_eval conventions). Scenes = R1 trio {garden, bicycle, kitchen} + truck
  once EXP-T2B lands. Source views per test view = the transport's own top-K support views (banked
  support_names — same evidence rights). Published pretrained weights ONLY (no fine-tuning, no per-scene
  training: that is the honest generalizable-IBR point; stated in the row). One config, all scenes.
- **Acceptance:** metrics banked for ≥2 scenes OR an INFEASIBLE filing with full attempt log (weights
  unavailable / hard env wall after a bounded ~1-day effort).
- **Artifacts:** `analysis/ibr_cell/{ibr_table.md,ibr_<scene>.json,attempt_log.md}`; isolated venv under
  `gems_stage1/ibr_cell/` (frozen env untouched).
- Owner: Codex (checkout, env, weights, CPU prep — sandbox does env work; I run GPU smoke + the cell,
  as with Difix); me (mirror/metric correctness review + interpretation).

## EXP-TEMP — temporal / view-path stability (closes P1-3)

- **Hypothesis:** per-view-independent transport does not introduce visible flicker beyond the base
  renderer's own level: temporal roughness of ECR ≤ 1.5× base on smooth paths; support-set switches do
  not produce visible popping (checked qualitatively in the videos).
- **Frozen protocol (GT-free — no purity surface):** per scene in {garden, bonsai, ss3dm_town01}: build a
  smooth pose path by SLERP/Catmull-Rom through the ordered TEST-view poses (120 frames, fixed spacing,
  seed-free/deterministic); render base + ECR final per frame (same quantization); metrics: mean and P95 of
  mean-abs frame difference |I_t − I_{t−1}| for base vs ECR, plus per-frame support-set switch count;
  side-by-side videos (base | final | β map) at 24 fps for supplementary.
- **Acceptance:** metric banked for 3 scenes + videos rendered; if roughness ratio > 1.5×, the finding is
  reported as a limitation with the β-map explanation (no silent fix attempts — any mitigation would be a
  new pre-registered mechanism, out of scope).
- **Artifacts:** `tools/analysis/ecr_temporal.py`; `analysis/temporal/{temporal_summary.{md,json}, <scene>_path.mp4}`.
- Owner: me (tool — touches the renderer boundary); Codex (video encoding polish if needed).

## EXP-HBOOT — hierarchical (scene-level) bootstrap (closes P1-4)

- **Hypothesis:** headline conclusions survive scene-level resampling: the full9 final-vs-PJ-2026 PSNR and
  LPIPS CIs still exclude 0 when scenes are resampled with replacement (two-stage: scenes, then views
  within scene; 10k, seed 0).
- **Frozen protocol:** implement `hierarchical_mean_ci(diffs_by_scene)` alongside the existing
  `stratified_mean_ci` (NOT replacing it — both reported); recompute the four headline aggregates
  (final vs PJ-2026 PSNR/LPIPS; L6 vs anchor PSNR/LPIPS) + AT-E0 from the SAME banked per-view arrays.
- **Acceptance:** if any headline CI now includes 0, the claim text is amended (CI form reported for both
  schemes; no cherry-picking). Expected: wider but still-excluding-0 for the final-stack deltas (9/9
  scene-mean positives make sign flips implausible); AT-E0 trivially safe.
- **Artifacts:** function in `tools/ecr/e0_report.py` (additive), collector
  `tools/analysis/hboot_report.py` → `analysis/final_stack/hierarchical_cis.{md,json}`.
- Owner: me (stats code is claim-bearing).

## DOC-THREAT — threat model + terminology (closes P1-5)

- Write PROTOCOL §4E.1 "Threat model of the ECR purity audit": PROVEN per row (reads ⊆ manifest via
  confined loader + strace; train/test split disjointness recomputed independently; frozen per-view kwargs
  hash; checkpoint fingerprint; GT sentinel; base-mode non-loading) / ASSUMED (OS integrity, honest
  filesystem, the audit tool itself runs unmodified — hash-pinned in the pack) / NOT CLAIMED (formal
  verification of renderer code; robustness to adversarial cache content; anything about geometry).
- Global terminology: "certified" → **"audited"** everywhere claim-bearing (CLAIMS_ECR, MATRIX, handoff),
  with one defined term ("audited train-only transport"). LEDGER historical entries stay verbatim (append
  a note, never rewrite history).
- Owner: me. Acceptance: grep shows no claim-bearing "certified" outside quoted history.

## PLOT-RD / PLOT-LADDER / TOOL-TABLES / DOC-REPRO (closes P1-6, P1-7)

- `tools/analysis/plot_ladder.py`: CI bar chart from `gates/*.json` (+E-08 rows) → `RESULTS/figures/ecr_paper/ladder_ci.pdf/png`.
- `tools/analysis/plot_rd.py`: unified R-D scatter (x = TOTAL MB, y = PSNR; series: L5 points, uncompressed
  final, L6, 3DGS vanilla+pruned from R1/E-07, Difix point at base TOTAL) per R1-trio scene + normalized
  panel → `RESULTS/figures/ecr_paper/rd_master.pdf/png`.
- `tools/analysis/paper_tables.py`: LaTeX bodies for paper T1–T4 from banked json/md (zero retyped numbers).
- `docs/ECR_README.md`: env, one-command repro per row class, artifact map, seeds.
- Owner: Codex (implementation to spec), me (verification of every number against sources before use).
- Acceptance: plots/tables regenerate deterministically; spot-check 10 numbers vs sources each.

## Sequencing & parallelism

1. NOW: dataset staging (Codex) ∥ scenes.py registration (me) ∥ hierarchical bootstrap (me) ∥ threat-model doc (me).
2. Then: launch 4× clean30k training (2 GPU chains, supervised, exit-code-0 gates) ∥ EXP-IBR env prep (Codex) ∥ plots/tables (Codex).
3. When training lands: PJ + final-stack chains per scene (gated) ∥ EXP-TEMP tool + runs (GPU 7-class).
4. Collectors + doc sync (CLAIMS/MATRIX/LEDGER/handoff) at every bank; readiness scores updated in the audit.

Race prevention: each job writes to its own eval/<row> and analysis/<cell>; gated launches key on exit==0;
no two jobs share a cache dir; collectors read only completed rows (existence + audit check).
