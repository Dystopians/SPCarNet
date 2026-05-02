# MeshPrior Remaining Work Prompts

Date: 2026-05-02

## Current State

SPCarNet MeshPrior is about `93%` complete as a research-codebase transformation and about `72%` complete as a NeurIPS-strength empirical paper. The proposal, gate, rollback, patch extraction, recovery-model loading, W&B smoke, clean `origin/main` baseline path, 2000-iteration medium comparison, 7000-iteration single-scene diagnostic, M21.5 topology-controlled current-branch ablation, M22 unified paper-evidence package, M23 claim-risk audit, M23.5 integrated topology-control smoke, M23.6 tuned medium integrated topology control, M24 full-budget integrated topology control, M24.1 late-PRISM Pareto sweep, and M24.2 topology-retention row are implemented. The remaining core is multi-scene evidence and paper-grade visual/failure analysis. The Stage17 MeshPrior resume variant is not viable at 7000 iterations.

Key codebase links:

- Master prompt: `docs/prompts.md`
- Research log: `docs/car_model/SPCarNet_research_log.md`
- Medium baseline report: `docs/car_model/meshprior_parking_medium_baseline_2000iter_report.md`
- Origin/main baseline report: `docs/car_model/meshprior_parking_origin_main_baseline_report.md`
- Current branch 2000 model: `outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model`
- Clean origin/main 2000 model: `outputs/carnet/meshprior/parking_phone_tiny/origin_main_2000iter/model`
- Parking dataset view: `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- Main MeshPrior scripts: `scripts/car_model/meshprior_*.py`
- MeshPrior modules: `ss3dm_prior/meshprior/`
- Paper evidence package: `docs/car_model/meshprior_stage22_paper_evidence_report.md`
- Claim-risk audit: `docs/car_model/meshprior_stage23_claim_risk_audit.md`
- Integrated topology smoke: `docs/car_model/meshprior_stage23_5_integrated_topology_implementation_report.md`
- Tuned medium integrated topology report: `docs/car_model/meshprior_stage23_6_tuned_integrated_topology_report.md`
- Full integrated topology report: `docs/car_model/meshprior_stage24_full_integrated_topology_report.md`
- Late-PRISM Pareto report: `docs/car_model/meshprior_stage24_1_late_prism_pareto_report.md`
- Topology-retention report: `docs/car_model/meshprior_stage24_2_topology_retention_report.md`

Known W&B runs:

- Parking 200-iteration current-branch smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/icjop1fq`
- Cleanup repair 200-iteration smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3swt58x2`
- Clean `origin/main` 2000 external log: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/142memiw`
- Current branch 2000 training-time W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/nk2w04wn`
- Clean `origin/main` 7000 external log: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/yiwb4d2n`
- Current branch 7000 training-time W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/l5buxl3m`
- Stage17 MeshPrior resume 7000 training-time W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w3kczubb`
- M21.5 prune_25 external log: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/evid1gbt`
- M21.5 prune_50 external log: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w1ix6e9a`
- M21.5 prune_66 external log: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xzfqwpgi`
- M23.5 integrated topology debug, protected 800 iter: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/5ekk5gjz`
- M23.5 integrated topology debug, protected 350 iter: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/esyvtvwn`
- M23.5 integrated topology trigger 180 iter: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/an7l2ec0`
- M23.6 tuned medium v2: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j0c8zwkx`
- M24 full v1 early PRISM: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/7i6n8jfj`
- M24 full v2 late 5% safety reject: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ytex9896`
- M24 full v3 late 1% fine prune: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/e92jwttk`
- M24.1 late PRISM 0.5% legacy no-candidate diagnostic: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bqc4w18e`
- M24.1 late PRISM 0.5% retryfix best topology row: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/jnn9yauw`
- M24.1 late PRISM 1% retryfix throttle row: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/0n7kzim5`
- M24.2 topology retention freeze-after-first-commit: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vsv2bs79`

## Operating Rules

Follow the original `docs/prompts.md` rules. These additional rules are mandatory for all prompts below.

1. Work one prompt at a time. Do not move to the next prompt until the current prompt has a design note, code, smoke test, implementation report, output metrics, research-log entry, and explicit `PASS`, `SOFT PASS`, `FAIL`, or `STOP` gate.
2. Before major edits, run repository integrity checks:

```bash
git status --short
python --version
python -m compileall scripts/car_model ss3dm_prior -q
```

3. Before every GPU run, run `nvidia-smi`, choose an idle or light GPU, and record the chosen GPU in the report.
4. Current-branch training runs must use training-time W&B:

```bash
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
```

Use `--enable_wandb`, `--wandb_project spcarnet_meshprior`, a meaningful `--wandb_group`, and a unique `--wandb_name` whenever the current training script supports them.

5. If a historical baseline cannot support training-time W&B, explicitly document the limitation and immediately log an external W&B summary. Do not leave this implicit.
6. Append a dated entry to `docs/car_model/SPCarNet_research_log.md` after each prompt and update this document's status table.
7. Write exact commands and W&B URLs into the prompt report. Do not claim a run succeeded unless checkpoint, logs, metrics JSON, and W&B record exist.
8. Keep inference-time metrics, oracle-only analysis, and GT-dependent evaluation separate.
9. Preserve all old SP-CarNet and MeshPrior baselines. Add new files under `scripts/car_model/meshprior_*`, `ss3dm_prior/meshprior/`, `docs/car_model/meshprior_*`, and `outputs/carnet/meshprior/`.
10. Commit and push after every completed prompt or meaningful failure report.

## Status Table

| item | source | status | note |
|---|---|---|---|
| Real scene proposal, gate, rollback plumbing | original prompts M2-M13 | DONE | Implemented and smoke-tested, but much of it is still dry-run or copied-checkpoint validation. |
| Parking phone tiny scene data path | execution | DONE | Dataset view, COLMAP eval, ROI mining, cluster scoring, patch extraction, recovery eval exist. |
| Clean Mesh Splatting baseline | execution finding | PASS | M19 confirms official `https://github.com/meshsplatting/mesh-splatting.git` main/HEAD equals `1a714f3`; the existing origin/main 2000 run is a valid clean medium baseline. |
| Current branch medium baseline | execution | SOFT PASS | 2000-iteration W&B run exists; better render proxy but much larger topology. |
| Real MeshPrior 2000-iteration variant | original prompts Layer F | PASS | Stage17 resumes a MeshPrior-cleaned checkpoint to 2000 iterations with training-time W&B and improves post-render / sparse geometry proxy metrics, but claim status remains soft due topology inflation. |
| Topology-budget / efficiency-normalized comparison | execution finding | PASS | M18 collector emits JSON/CSV/Markdown and marks Stage17 as `QUALITY_GAIN_NOT_TOPOLOGY_NORMALIZED`; stronger paper claims remain blocked until topology control or budget-matched reporting. |
| Multi-scene validation | original prompts Layer G | STOP | M20 audited parent directories and found no second suitable parking-lot COLMAP/image scene; user data is needed before multi-scene validation can proceed. |
| Long-budget or paper-budget training | original prompts M11/M13/M14 | PASS / negative result | M21 completed 7000-iteration aligned single-scene runs. Clean/current are stable; Stage17 MeshPrior resume collapses at long budget. |
| Topology-controlled current-branch 7000 row | execution finding | PASS | M21.5 post-training checkpoint-copy area pruning shows `prune_50` keeps render metrics above clean with `416888` triangles and depth AbsRel close to clean; use as M22 default topology-controlled row. |
| Unified paper-evidence package | original prompts M22 | SOFT PASS | M22 collector/report/smoke are implemented and reproducible; missing rows stay visible for second scene, integrated topology control, and render-gated full insertion. |
| Claim-risk audit and paper decision | original prompts M23 | PASS | Strongest defensible story is conservative proposal/gate framework plus topology-aware diagnostics, not a full scene-optimization improvement claim. |
| Integrated optimization-time topology control smoke | execution finding / original Layer F | PASS | M23.5 proves PRISM can commit a training-time candidate prune with rollback metadata and eval artifacts; 180-iter relaxed-threshold result is a mechanism proof, not a quality claim. |
| Tuned medium integrated topology-control run | execution finding | PASS | M23.6 v2 commits two training-time PRISM edits at 2000 iterations and improves over current-branch 2000 with far fewer triangles. |
| Full-budget integrated topology-control run | execution finding | PASS | M24-v3 commits two late 1% PRISM edits at 7000 iterations with online W&B and independent metrics. Quality is close to current branch but topology reduction is small. |
| Integrated topology Pareto sweep | execution finding | PASS | M24.1 best row commits five late PRISM edits and reduces final topology to `723438` triangles with near-current independent metrics; still not as topology-efficient as posthoc M21.5 `prune_50`. |
| Topology-retention after integrated edits | execution finding | PASS | M24.2 freezes densification after the first accepted PRISM candidate edit and reaches `254491` triangles with stronger independent metrics than M21.5 prune_50. |
| Metric-path reconciliation | execution finding | TODO | Training internal metrics and `render.py + metrics.py` differ and must stay labeled. |
| Final claim table and failure cases | original prompts Layer G | TODO | Need unified paper-style tables, visual cases, and failure taxonomy. |

## Execution-Discovered Risks

These are not new research directions. They are constraints discovered while implementing the original prompts.

1. `200` iterations is only a smoke budget. It can validate plumbing but must not drive method claims.
2. The current branch's 2000- and 7000-iteration results improve some metrics but use far more triangles and vertices. A paper claim must account for topology and speed.
3. `origin/main` lacks current training-time W&B flags. Historical runs require external summary logging unless the script is backported.
4. The training final-cleanup path previously pruned non-PRISM models destructively. It has been repaired, but every new training run must record final-cleanup state.
5. COLMAP geometry evaluation is a proxy, not ground-truth geometry. It is useful for diagnosis and consistency, not as a sole headline metric.
6. Training internal metrics and post-render `metrics.py` values use different pathways. Reports must label them separately.
7. The current parking scene is small. Generality requires at least one larger parking-lot COLMAP scene or another real scene.
8. The Stage17 MeshPrior resume variant is unstable as a long-budget method: it improves at 2000 iterations but degrades badly by 7000 iterations.
9. M21.5 topology pruning is post-hoc checkpoint-copy pruning, not yet an integrated optimization-time topology controller. It is valid as a diagnostic evidence row, not as the final algorithm.
10. M23.5 shows the integrated PRISM commit path works, but default protection is too conservative for short early smokes. Fully relaxed protection can trigger edits but is not a final paper setting.
11. M24-v1 shows that too many early PRISM rounds can suppress standard Mesh Splatting densification by keeping the controller in freeze/recovery/stat phases. Full runs should use late PRISM or explicitly account for densification windows.
12. M24-v2/v3 show the counterfactual gate is conservative: `5%` late candidate pruning was fully rejected, while `1%` late candidate pruning committed twice and then rejected later rounds.
13. M24.1 shows no-candidate attempts must be logged separately from effective PRISM rounds. The controller now retries without consuming a candidate round and throttles retry attempts.
14. M24.1 also shows that accepted integrated topology edits do not guarantee final topology retention because later densification can restore triangles.
15. M24.2 shows topology-retention is the key missing schedule rule: freezing densification after accepted PRISM commits enables standard pruning to retain a low-topology final checkpoint without hurting render metrics.

---

# Prompt M23.5 — Integrated optimization-time topology-control smoke

## Goal

Move topology control from post-hoc checkpoint-copy pruning into the training loop and verify that a PRISM candidate edit can commit with rollback metadata, W&B, render metrics, COLMAP proxy geometry, and final-cleanup accounting.

## Status

`PASS` on 2026-05-02.

## Result

- successful run: `prism_unprotected_trigger_180iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/an7l2ec0`
- committed PRISM edit: iteration `141`, candidate prune, `64497 -> 63208` triangles
- independent render metrics: PSNR `10.790648`, SSIM `0.284250`, LPIPS `0.645548`
- COLMAP proxy depth AbsRel: `0.327274`
- collector gate: `PASS`

## Caveat

This is a mechanism proof. The successful trigger deliberately relaxed protection thresholds to verify that the commit path works. Do not use the 180-iteration metrics as evidence of paper-quality improvement.

---

# Prompt M23.6 — Tuned medium integrated topology-control run

## Goal

Turn the M23.5 mechanism proof into a meaningful medium-budget method row on `parking_phone_tiny`.

## Status

`PASS` on 2026-05-02.

## Result

- successful run: `tuned_medium_v2_2000iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j0c8zwkx`
- committed PRISM edits: iteration `551` (`64497 -> 63853`) and `922` (`63853 -> 63215`)
- independent render: PSNR `12.046110`, SSIM `0.286099`, LPIPS `0.629034`
- COLMAP proxy: depth AbsRel `0.393866`, normal mean angle `51.945426`
- collector gate: `PASS`

## Required Work

1. Start from M23.5 code and reports.
2. Define a tuned PRISM schedule that is less conservative than the default all-protected short smoke but more defensible than the fully unprotected trigger.
3. Run a short debug if needed, then a medium run with online W&B.
4. Keep final cleanup disabled unless explicitly evaluated as a separate ablation.
5. Evaluate:
   - training internal metrics,
   - independent `render.py + metrics.py`,
   - `evaluate_geometry_colmap.py`,
   - topology counts,
   - PRISM commit/rollback metadata,
   - final-cleanup summary.
6. Compare against clean/current/M21.5 rows without hiding topology count.

## Required Outputs

- `docs/car_model/meshprior_stage23_6_tuned_integrated_topology_design.md`
- `docs/car_model/meshprior_stage23_6_tuned_integrated_topology_report.md`
- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage23_6_tuned_integrated_topology/`
- W&B URL file and command logs under the output root.

## Gate

`PASS` if the medium run completes with online W&B, at least one scheduled topology decision is recorded, all eval artifacts exist, and topology/quality tradeoffs are interpretable.

`SOFT PASS` if the run completes but does not improve over the topology-controlled diagnostic row.

`FAIL` if training crashes, W&B is missing without documented fallback, or rollback/final-cleanup accounting is absent.

---

# Prompt M24 — Full-budget integrated topology-control run

## Goal

Run full-budget 7000-iteration integrated topology control on `parking_phone_tiny`, compare against clean/current/M21.5 baselines, and identify whether the integrated controller is ready for paper evidence.

## Status

`PASS` on 2026-05-02.

## Result

- best current row: `full_v3_late_fine_prune_7000iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/e92jwttk`
- PRISM decisions:
  - iteration `6151`: `612458 -> 606334`, committed
  - iteration `6272`: `606334 -> 600271`, committed
  - iterations `6393` and `6394`: rejected
- independent render: PSNR `17.042757`, SSIM `0.529476`, LPIPS `0.454884`
- COLMAP proxy: depth AbsRel `0.082815`, normal mean angle `43.394721`
- final topology: `823651` triangles, `1058219` vertices
- collector gate: `PASS`

## Decision

M24 is a real integrated optimization-time topology-control milestone, but not yet a decisive paper headline. It proves safe training-time PRISM commits at full budget while preserving near-current render quality. It does not yet match the topology reduction of M21.5 `prune_50`.

## Next Prompt: M24.1

Run a late-PRISM Pareto sweep:

1. Keep W&B online.
2. Use late geometry acquisition (`6000` or later) so normal densification is not suppressed.
3. Sweep candidate ratios `{0.005, 0.01, 0.02}` and candidate rounds `{4, 8}`.
4. Keep counterfactual gate enabled.
5. Report accepted/rejected rounds, topology, render metrics, COLMAP proxy geometry, and final-cleanup state.
6. Stop when a row approaches M21.5 `prune_50` quality/topology from inside training or when the safety gate clearly blocks that Pareto region.

Gate: `PASS` if at least one integrated row improves topology materially over current branch while preserving render metrics near current/M21.5. `SOFT PASS` if only safety-rejection evidence is obtained. `FAIL` if W&B, metrics, rollback, or final-cleanup accounting are missing.

## Status

`PASS` on 2026-05-02.

## Result

- best row: `pareto_ratio0p005_rounds8_retryfix_7000iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/jnn9yauw`
- PRISM decisions: five committed late candidate edits, collector `PASS`
- independent render: PSNR `16.967005`, SSIM `0.530894`, LPIPS `0.465932`
- COLMAP proxy: depth AbsRel `0.082264`, normal mean angle `42.667905`
- final topology: `723438` triangles, `904493` vertices
- report: `docs/car_model/meshprior_stage24_1_late_prism_pareto_report.md`

## Decision

M24.1 is the strongest integrated optimization-time topology-control row so far. It improves substantially over M24-v3 topology while preserving comparable render/geometry metrics. It still does not match the topology budget of M21.5 `prune_50`, so the next work should address topology retention after accepted edits.

---

# Prompt M24.2 — Topology-retention after accepted integrated edits

## Goal

Prevent late densification from erasing accepted PRISM topology-control gains while preserving the M24.1 render and COLMAP proxy quality.

## Required Work

1. Start from the M24.1 `0.005 x 8` retryfix configuration and keep online W&B.
2. Add one minimal topology-retention mechanism:
   - freeze or strongly reduce densification after the first accepted PRISM candidate commit, or
   - select a final checkpoint from the best post-PRISM topology point, or
   - run a gated final-cleanup ablation as a separately labeled row.
3. Keep counterfactual gate, rollback metadata, final-cleanup accounting, and collector output intact.
4. Run a 7000-iteration row if the code change is small and GPU is available.
5. Evaluate independent render metrics, COLMAP proxy geometry, final topology, effective PRISM rounds, retry events, accepted/rejected rounds, and W&B URL.

## Required Outputs

- `docs/car_model/meshprior_stage24_2_topology_retention_design.md`
- `docs/car_model/meshprior_stage24_2_topology_retention_report.md`
- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage24_2_topology_retention/`
- updated `docs/car_model/SPCarNet_research_log.md`

## Gate

`PASS` if a fully integrated 7000-iteration row keeps final topology below M24.1 best (`723438` triangles) while maintaining independent PSNR within about `0.15` of M24.1 best and COLMAP proxy normal no worse than M24-v3.

`SOFT PASS` if topology improves but render/geometry regression makes the row non-headline.

`FAIL` if W&B, checkpoint, rollback, collector, or independent eval artifacts are missing.

## Status

`PASS` on 2026-05-02.

## Result

- run: `freeze_after_first_commit_7000iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vsv2bs79`
- PRISM decisions: two committed late candidate edits, six rollback-protected rejected edits, collector `PASS`
- independent render: PSNR `17.314823`, SSIM `0.559230`, LPIPS `0.442099`
- COLMAP proxy: depth AbsRel `0.078840`, normal mean angle `41.010093`
- final topology: `254491` triangles, `463687` vertices
- report: `docs/car_model/meshprior_stage24_2_topology_retention_report.md`

## Decision

M24.2 is the first integrated row that beats the M21.5 posthoc `prune_50` diagnostic on final topology while improving independent render and COLMAP proxy normal metrics. The next priority is not more single-scene tuning; it is multi-scene validation and paper-grade visual/failure evidence.

---

# Prompt M25 — Multi-scene validation and paper evidence

## Goal

Test whether the M24.2 topology-retention method generalizes beyond `parking_phone_tiny` and prepare the evidence needed for a top-conference paper claim.

## Required Work

1. Add at least one second COLMAP/image scene with enough camera overlap for Mesh Splatting training.
2. Run clean/current/M24.2-aligned rows on the second scene with online W&B.
3. Keep exact metric separation:
   - independent render metrics,
   - COLMAP proxy geometry,
   - topology and speed,
   - PRISM accept/reject/rollback metadata.
4. Build paper tables comparing clean, current, M21.5 diagnostic, M24.1, and M24.2 where applicable.
5. Export visual cases for object/parking-region geometry, failure cases, and topology-efficiency examples.

## Gate

`PASS` if M24.2 keeps a favorable topology/render/geometry tradeoff on at least one second scene.

`SOFT PASS` if the second scene runs but the tradeoff is scene-dependent.

`STOP` if no second scene exists or COLMAP/image data are insufficient.

---

# Prompt M17 — Build the first real MeshPrior 2000-iteration variant

## Goal

Turn the current copied-patch / recovery-model pathway into a real 2000-iteration MeshPrior variant on `parking_phone_tiny`, with training-time W&B and the same evaluation scripts used by the clean and current baselines.

## Required Work

1. Read:
   - `docs/prompts.md`
   - `docs/car_model/meshprior_parking_medium_baseline_2000iter_report.md`
   - `docs/car_model/meshprior_parking_recovery_model_eval_report.md`
   - `docs/car_model/meshprior_parking_patch_proposal_test_report.md`
2. Decide the minimal real variant:
   - either resume/initialize from an accepted MeshPrior-cleaned checkpoint and train to 2000 iterations;
   - or integrate accepted MeshPrior patch actions into the scene-training loop at a fixed iteration boundary.
3. The variant must remain conservative:
   - no oracle GT proposal selection;
   - no proposal without scene evidence gate;
   - rollback snapshot before every geometry edit;
   - final cleanup disabled unless explicitly part of the variant and separately reported.
4. Run a smoke first at `200` iterations.
5. If smoke passes and GPU is light, run `2000` iterations with training-time W&B.

## Required Outputs

- `docs/car_model/meshprior_stage17_real_variant_design.md`
- `docs/car_model/meshprior_stage17_real_variant_implementation_report.md`
- `docs/car_model/meshprior_stage17_real_variant_smoke.md`
- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/`
- W&B URL file: `outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/wandb_url.txt`
- metrics:
  - training internal eval
  - `render.py + metrics.py`
  - `evaluate_geometry_colmap.py`
  - topology counts
  - final-cleanup summary
  - proposal accept/reject/rollback counts

## Gate

`PASS` if the run completes, W&B is online, metrics exist, topology is recorded, and no unguarded geometry edit occurs.

`SOFT PASS` if the method is stable but not better than baselines.

`FAIL` if training crashes, W&B is missing without documented offline fallback, or rollback cannot recover.

`STOP` if no conservative real variant can be defined from available code.

---

# Prompt M18 — Topology-budget and efficiency-normalized comparison

## Goal

Prevent false improvement claims caused by using many more triangles. Compare clean baseline, current branch, and MeshPrior variant under explicit topology and speed accounting.

## Required Work

1. Build a comparison collector that reads:
   - `origin_main_2000iter`
   - `current_branch_2000iter`
   - `stage17_real_variant_2000iter`
2. Add topology-normalized metrics:
   - PSNR / 100k triangles
   - SSIM / 100k triangles
   - LPIPS with triangle count shown beside it
   - FPS with triangle and vertex count
   - depth proxy vs triangle count
3. Add a controlled cleanup or budget-matching option if current topology remains much larger:
   - conservative pruning only;
   - no oracle GT selection;
   - rollback required;
   - before/after render and geometry metrics required.
4. Generate CSV, JSON, and Markdown tables.

## Required Outputs

- `scripts/car_model/meshprior_collect_topology_budget_comparison.py`
- `scripts/car_model/smoke_test_meshprior_topology_budget_comparison.py`
- `docs/car_model/meshprior_stage18_topology_budget_design.md`
- `docs/car_model/meshprior_stage18_topology_budget_implementation_report.md`
- output root: `outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison/`

## Gate

`PASS` if the collector produces reproducible tables and no row hides topology or speed.

`SOFT PASS` if comparison is complete but budget-matched cleanup is diagnostic only.

`FAIL` if the comparison cannot distinguish quality gains from topology inflation.

---

# Prompt M19 — Confirm or replace the clean Mesh Splatting paper baseline

## Goal

Determine whether `origin/main@1a714f3` is a valid clean Mesh Splatting baseline for paper claims. If not, create a stricter baseline path.

## Required Work

1. Inspect remotes, commit history, README, and dependency differences.
2. Compare `origin/main@1a714f3` against the official Mesh Splatting code expected by the paper, if available locally or through the configured remote.
3. Do not overwrite current branch. Use an isolated worktree for any baseline checkout.
4. If a stricter baseline is required, run at least a smoke and a 2000-iteration medium run.
5. Historical scripts without W&B support must be externally logged immediately.

## Required Outputs

- `docs/car_model/meshprior_stage19_clean_baseline_audit.md`
- optional isolated worktree path recorded in the report
- optional output root: `outputs/carnet/meshprior/parking_phone_tiny/official_mesh_splatting_2000iter/`
- W&B URL or documented external summary log

## Gate

`PASS` if baseline validity is resolved and a reproducible baseline path exists.

`SOFT PASS` if `origin/main` remains the best local candidate but official parity is not fully proven.

`STOP` if network or repository access blocks confirmation; write the failure report.

---

# Prompt M20 — Add a second real parking-lot scene

## Goal

Reduce single-scene overfitting risk by adding one larger parking-lot COLMAP/image scene or another real scene supplied by the user.

## Required Work

1. Audit candidate datasets under the parent directory, including larger parking-lot COLMAP/image data if available.
2. Define the expected dataset contract:
   - images directory
   - COLMAP sparse reconstruction
   - camera model compatibility
   - scale/orientation notes
   - optional masks or object annotations
3. Create a repo-local dataset view without copying large raw data.
4. Run dataset audit and baseline smoke.
5. If valid and GPU is light, run a 2000-iteration clean/current baseline with W&B.

## Required Outputs

- `docs/car_model/meshprior_stage20_second_scene_design.md`
- `docs/car_model/meshprior_stage20_second_scene_audit.md`
- `docs/car_model/meshprior_stage20_second_scene_implementation_report.md`
- output root: `outputs/carnet/meshprior/<scene_name>/`
- W&B URLs for any training runs

## Gate

`PASS` if the second scene has valid dataset view, baseline metrics, and documented limitations.

`SOFT PASS` if only audit and smoke are possible.

`STOP` if no usable second scene exists locally; write the data requirement report.

---

# Prompt M21 — Long-budget aligned experiments

Status: `PASS` for aligned execution, `FAIL` for Stage17 MeshPrior resume as a long-budget candidate.

Report: `docs/car_model/meshprior_stage21_long_budget_report.md`

Key result: at 7000 iterations, current branch beats clean `origin/main` on independent render metrics but uses about `2.92x` more triangles; Stage17 MeshPrior resume collapses to PSNR `10.839708`, SSIM `0.285366`, LPIPS `0.662528`, and COLMAP depth AbsRel `0.744099`.

---

# Prompt M21.5 — Topology-controlled current-branch ablation

Status: `PASS`.

Report: `docs/car_model/meshprior_stage21_5_topology_control_implementation_report.md`

Key result: post-training checkpoint-copy area pruning on the current-branch 7000 checkpoint produces a useful topology-control row. `prune_50` reduces triangles from `833775` to `416888`, keeps PSNR / SSIM / LPIPS at `17.051889` / `0.523914` / `0.465400`, and keeps COLMAP depth AbsRel at `0.083265`, slightly better than the clean 7000 baseline `0.084499`.

Use in M22:

- default topology-controlled current row: `prune_50`
- high-compression Pareto endpoint: `prune_66`
- failure case: Stage17 MeshPrior resume at 7000
- caveat: M21.5 is post-hoc topology pruning, not integrated optimization-time topology control

## Goal

Move from medium validation to paper-grade training budget, only after M17-M20 establish that the method and baseline paths are stable.

## Required Work

1. Choose a fixed long budget, such as `7000`, `15000`, or the repository's standard full budget.
2. Use identical data split, image resolution, checkpoint rule, and metric scripts for:
   - clean Mesh Splatting baseline
   - current branch engineering baseline
   - MeshPrior variant
3. Use training-time W&B for current-branch runs.
4. Record all commands, W&B URLs, final cleanup summaries, topology, FPS, render metrics, and geometry proxy metrics.
5. Stop after the first full long-budget run set and write a decision report before launching any sweep.

## Required Outputs

- `docs/car_model/meshprior_stage21_long_budget_design.md`
- `docs/car_model/meshprior_stage21_long_budget_report.md`
- output root: `outputs/carnet/meshprior/<scene_name>/stage21_long_budget/`
- W&B URLs and run metadata

## Gate

`PASS` if all aligned runs finish and comparison tables are generated.

`SOFT PASS` if one aligned pair finishes but the full set is incomplete for documented reasons.

`FAIL` if W&B/logs/checkpoints are missing or the comparison is not aligned.

---

# Prompt M22 — Unified paper table and failure-case package

Status: `SOFT PASS`.

Report: `docs/car_model/meshprior_stage22_paper_evidence_report.md`

Key result: `scripts/car_model/meshprior_collect_paper_evidence.py` regenerates object-prior, synthetic-damage, scene render/geometry/topology, proposal-gate/rollback, failure-case, and missing-row tables under `outputs/carnet/meshprior/paper_evidence/`. The default paper-evidence scene row is `current_branch_prune_50_7000`; `stage17_meshprior_resume_7000` is retained as a failure case. Gate is `SOFT PASS` because important rows remain `MISSING`.

## Goal

Convert scattered engineering reports into a NeurIPS-style evidence package with clear claims, missing rows, failure cases, and metric separation.

## Required Work

1. Extend the M13 matrix runner or add a new collector to include:
   - object prior metrics
   - synthetic damage metrics
   - scene render metrics
   - scene geometry proxy metrics
   - topology and FPS
   - proposal accept/reject/rollback statistics
2. Generate a paper-style Markdown report and machine-readable JSON/CSV.
3. Include failure cases:
   - rejected proposals
   - rollback examples
   - topology inflation examples
   - proxy metric disagreement examples
4. Keep missing rows visible as `MISSING`.

## Required Outputs

- `scripts/car_model/meshprior_collect_paper_evidence.py`
- `scripts/car_model/smoke_test_meshprior_paper_evidence.py`
- `docs/car_model/meshprior_stage22_paper_evidence_report.md`
- output root: `outputs/carnet/meshprior/paper_evidence/`

## Gate

`PASS` if the report can be regenerated from local artifacts and separates all metric classes.

`SOFT PASS` if the report is reproducible but still contains important `MISSING` rows.

`FAIL` if headline tables can silently omit missing or failed experiments.

---

# Prompt M23 — Claim-risk audit and paper decision

Status: `PASS`.

Report: `docs/car_model/meshprior_stage23_claim_risk_audit.md`

Key result: strongest defensible current story is `CLAIM_CONSERVATIVE_FRAMEWORK_NOT_FULL_METHOD`. Supported: object-prior quality, proposal gate/rollback safety, single-scene current-branch/prune_50 diagnostics. Refuted: Stage17 MeshPrior resume as long-budget method. Unsafe to claim: full MeshPrior scene optimization and multi-scene generalization.

## Goal

Make an explicit decision about the strongest defensible claim after M17-M22.

## Required Work

1. Read all stage reports from M17-M22.
2. Classify claims into:
   - supported
   - plausible but under-evidenced
   - refuted
   - unsafe to claim
3. Decide whether the paper story should be:
   - full MeshPrior scene optimization;
   - conservative proposal/gate framework with neutral performance;
   - dataset/tooling contribution;
   - negative result with strong safeguards.
4. Write exact next experiments only if they are necessary to resolve a specific claim.

## Required Outputs

- `docs/car_model/meshprior_stage23_claim_risk_audit.md`
- updated `docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md`
- research-log entry

## Gate

`PASS` if a defensible paper claim and remaining evidence gaps are explicit.

`STOP` if current evidence is insufficient for any strong method claim; recommend the lowest-risk next plan.

---

# Prompt M24 — Optional production hardening after research gate

## Goal

Only after M23, harden the codebase for external use without changing research results.

## Required Work

1. Add CLI help consistency and config examples.
2. Add small regression tests for collectors, gates, rollback, and W&B metadata handling.
3. Add README pointers to the final prompt/report chain.
4. Verify a fresh clone can run smoke tests without local large outputs, marking missing artifacts clearly.

## Required Outputs

- `docs/car_model/meshprior_stage24_production_hardening_report.md`
- updated README or docs index
- smoke-test command log

## Gate

`PASS` if the repo is easier to reproduce and no research outputs are changed.

`FAIL` if hardening changes experimental behavior without rerunning affected metrics.
