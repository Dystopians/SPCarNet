# MeshPrior Remaining Work Prompts

Date: 2026-05-02

## Current State

SPCarNet MeshPrior is about `99%` complete as a research-codebase transformation and about `90%` complete as a NeurIPS-strength empirical paper. The proposal, gate, rollback, patch extraction, recovery-model loading, W&B smoke, clean `origin/main` baseline path, 2000-iteration medium comparison, 7000-iteration single-scene diagnostic, M21.5 topology-controlled current-branch ablation, M22 unified paper-evidence package, M23 claim-risk audit, M23.5 integrated topology-control smoke, M23.6 tuned medium integrated topology control, M24 full-budget integrated topology control, M24.1 late-PRISM Pareto sweep, M24.2 topology-retention row, M25 public multidataset trainability validation, M26 cross-scene medium evidence, M27 topology accounting/schedule tuning, M28 adaptive candidate scheduling, M29 candidate caps, M30 microbatch candidate gating, M31 candidate-quality ranking, M32 measured candidate-impact ranking, M33 calibration-diversity diagnostics, M34 post-commit candidate refresh diagnostics, M35 conservative retained relaxed refresh, M36 metric reconciliation, and M37 visual/failure packaging are implemented. The remaining core is final paper figure/table polishing and an explicit full-budget public-scene training decision. The Stage17 MeshPrior resume variant is not viable at 7000 iterations.

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
- Multidataset validation report: `docs/car_model/meshprior_stage25_multidataset_validation_report.md`
- Cross-scene method evidence report: `docs/car_model/meshprior_stage26_cross_scene_report.md`
- Accounting fix report: `docs/car_model/meshprior_stage27_accounting_fix_report.md`
- Schedule ablation report: `docs/car_model/meshprior_stage27_schedule_ablation_report.md`
- Adaptive schedule smoke report: `docs/car_model/meshprior_stage28_adaptive_schedule_smoke_report.md`
- Adaptive schedule medium report: `docs/car_model/meshprior_stage28_adaptive_schedule_medium_report.md`
- Candidate cap report: `docs/car_model/meshprior_stage29_candidate_cap_report.md`
- Candidate cap medium report: `docs/car_model/meshprior_stage29_candidate_cap_medium_report.md`
- Candidate cap sweep report: `docs/car_model/meshprior_stage29_candidate_cap_sweep_report.md`
- Microbatch candidate gate report: `docs/car_model/meshprior_stage30_microbatch_gate_report.md`
- Candidate-quality ranking report: `docs/car_model/meshprior_stage31_candidate_quality_report.md`
- Measured candidate-rank report: `docs/car_model/meshprior_stage32_measured_candidate_rank_report.md`
- Calibration-diversity report: `docs/car_model/meshprior_stage33_calibration_diversity_report.md`
- Post-commit candidate refresh report: `docs/car_model/meshprior_stage34_post_commit_refresh_report.md`
- Retained relaxed refresh report: `docs/car_model/meshprior_stage35_retained_refresh_report.md`
- Metric reconciliation report: `docs/car_model/meshprior_stage36_metric_reconciliation_report.md`
- Metric reconciliation collector: `scripts/car_model/meshprior_collect_metric_reconciliation.py`
- Visual/failure package report: `docs/car_model/meshprior_stage37_visual_failure_package_report.md`
- Visual/failure package script: `scripts/car_model/meshprior_package_visual_failures.py`

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
- M25 Mip-NeRF 360 bonsai 700-iteration trainability: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/x75zddff`
- M25 Tanks and Temples truck 700-iteration fixed trainability: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/5pre7o19`
- M25 ETH3D courtyard 700-iteration trainability: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/78iu6goq`
- M26 Mip-NeRF 360 bonsai sparse-depth baseline 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xdct9uys`
- M26 Mip-NeRF 360 bonsai M24.2 PRISM 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dmasxcej`
- M26 ETH3D courtyard sparse-depth baseline 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mdan8yc2`
- M26 ETH3D courtyard M24.2 PRISM 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r9zgtuyp`
- M27 accounting smoke ETH3D courtyard 520: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/i6lfgt66`
- M27 Mip-NeRF 360 bonsai ratio0p01 geom1200: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mlftnbt5`
- M27 ETH3D courtyard ratio0p01 geom1200: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qvrnsj2v`
- M27 Mip-NeRF 360 bonsai ratio0p02 geom1400: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/27vl4jnt`
- M27 ETH3D courtyard ratio0p02 geom1400: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ffp07dua`
- M28 adaptive rollback-ratio smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1kmwbu8g`
- M28 Mip-NeRF 360 bonsai adaptive schedule 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/38p6bgw4`
- M28 ETH3D courtyard adaptive schedule 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/piadupsm`
- M29 parking candidate cap smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rgvzhx6k`
- M29 Mip-NeRF 360 bonsai cap512 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ck157wtl`
- M29 ETH3D courtyard cap512 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1ey4qzbd`
- M29 Mip-NeRF 360 bonsai cap256 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mzglj2qw`
- M29 Mip-NeRF 360 bonsai cap1024 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j5v0debo`
- M30 parking microbatch candidate gate smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dioe1cz1`
- M30 Mip-NeRF 360 bonsai microbatch1024x256 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mfvhexjb`
- M30 ETH3D courtyard microbatch1024x256 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ha9qi1ih`
- M31 parking candidate-quality rank smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ucqyou26`
- M31 Mip-NeRF 360 bonsai quality-rank cap512 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/22r3et7s`
- M31 ETH3D courtyard quality-rank cap512 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xt4a2cn0`
- M32 parking measured-rank smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xg4fsvd8`
- M32 Mip-NeRF 360 bonsai measured-rank cap512 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/56l3tz23`
- M32 ETH3D courtyard measured-rank cap512 adaptive 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fb7jfcaj`
- M32 Mip-NeRF 360 bonsai measured+quality diagnostic: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xooe27um`
- M33 parking diverse-calibration smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ms95810g`
- M33 Mip-NeRF 360 bonsai diverse calibration 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kg5htc8u`
- M33 ETH3D courtyard diverse calibration 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w9c0b65f`
- M34 parking post-commit refresh smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rt3cxxhh`
- M34 parking recent0 refresh smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kke60qhc`
- M34 Mip-NeRF 360 bonsai root-cause v1: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/szkqpowq`
- M34 Mip-NeRF 360 bonsai root-cause v2: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/npagb743`
- M34 Mip-NeRF 360 bonsai relaxed-score v3: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/lt1v4652`
- M34 Mip-NeRF 360 bonsai second-edit-only v4: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/zhy368pr`
- M35 Mip-NeRF 360 bonsai retained relaxed retry: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rszvl7gn`
- M35 ETH3D courtyard retained relaxed check: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/u2s15ok0`

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
| Public multidataset trainability validation | original prompts Layer G / execution finding | SOFT PASS | M25 prepares Mip-NeRF 360, ETH3D, and Tanks and Temples-family data. Mip-NeRF 360 and ETH3D are geometry-observable COLMAP scenes; Tanks mirror trains but lacks true sparse tracks, so geometry validation reports `no_sparse_matches`. |
| Cross-scene method evidence | original prompts Layer G / M26 | SOFT PASS | M26 runs aligned sparse-depth baselines and M24.2 PRISM on Mip-NeRF 360 `bonsai` and ETH3D `courtyard` with online W&B, independent render metrics, PRISM decisions, and geometry-observable validation. Direct W&B topology reduction is modest, so full paper claims need schedule tuning. |
| Topology accounting reconciliation | execution finding / M27.0 | PASS | `train.py` now logs post-topology and final-checkpoint W&B counts; 520-iter ETH3D smoke confirms W&B `mesh/triangle_count` and final-cleanup checkpoint counts agree. |
| Cross-scene topology-pressure tuning | execution finding / M27 | SOFT PASS | `ratio0p02_geom1400` strongly reduces ETH3D `courtyard` to `100858` triangles with better independent metrics, but `bonsai` rolls back all six candidate edits and remains near baseline topology. Fixed schedules are not yet cross-scene robust. |
| Adaptive candidate scheduling | execution finding / M28 | SOFT PASS | Added opt-in rollback-driven candidate-ratio decay. Parking smoke verifies `0.04 -> 0.02 -> 0.01`; medium public-scene ablation preserves ETH3D but `bonsai` still rejects even a `0.005` global candidate set. |
| Granular candidate selection | execution finding / M29 | SOFT PASS / diagnostic PASS | Added opt-in `--prism_candidate_max_count_per_round`. Parking smoke passes. Cap sweep shows `bonsai` cap512 is best among `256/512/1024`, giving `633787` triangles and improved SSIM/LPIPS vs M28, but PSNR remains below M28 and slightly below sparse-depth baseline. |
| Microbatch candidate gating | execution finding / M30 | SOFT PASS / diagnostic PASS | Added opt-in `--prism_candidate_microbatch_gate`. Parking smoke passes. `bonsai` accepts `3/4` microbatches from cap1024 and `courtyard` accepts `4/4`, but cap512 remains the better conservative default because independent `bonsai` PSNR/LPIPS are worse. |
| Candidate-quality ranking | execution finding / M31 | SOFT PASS / diagnostic PASS | Added opt-in `--prism_candidate_quality_rank`. Parking smoke passes. `courtyard` improves over M29 cap512, but `bonsai` is mixed, so the ranking is useful diagnostic infrastructure rather than a promoted default. |
| Measured candidate-impact ranking | execution finding / M32 | SOFT PASS / diagnostic PASS | Added opt-in `--prism_candidate_measured_impact_rank`. Parking smoke passes and `courtyard` improves, but `bonsai` does not match Stage29 cap512, so no default promotion. |
| Calibration-diversity diagnostics | execution finding / M33 | SOFT PASS / diagnostic PASS | Added opt-in `--prism_calib_diverse_views` plus manifest/per-view deltas. `bonsai` now exceeds Stage29 cap512 at equal topology; `courtyard` remains better than M29 but worse than M32 measured rank. |
| Post-commit candidate refresh | execution finding / M34 | SOFT PASS / diagnostic PASS | Added opt-in `--prism_post_commit_candidate_refresh` and no-candidate diagnostics. Root cause is `recent_t` protecting all triangles and zeroing prune score after topology sync. Relaxed-score v3 keeps additional `bonsai` topology reduction (`633787 -> 631739`) with PSNR up but SSIM/LPIPS slightly down, so no default promotion. |
| Conservative retained relaxed refresh | execution finding / M35 | PASS | Added retained relaxed commit cap, strict relaxed counterfactual proxy gate, validation-rollback records, final topology audit, and W&B audit scalars. `bonsai` reaches `633275` triangles and improves independent PSNR/SSIM/LPIPS versus Stage33; `courtyard` confirms active relaxed commit retention. |
| Metric-path reconciliation | execution finding / M36 | PASS | Added reproducible JSON/CSV/Markdown collector for M24-M35 evidence rows. Independent `render.py + metrics.py` values are the paper-facing path; training eval fields remain separate diagnostics. |
| Final claim table and failure cases | original prompts Layer G / M37 | PASS | Added three render-vs-GT panels, six concrete failure cases, and paper-safe claim wording. Full-budget public-scene training is deferred until figure/table needs are fixed. |

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
16. M25 shows public dataset support is uneven: Mip-NeRF 360 and ETH3D fit the current COLMAP contract, while the available Tanks and Temples mirror requires real COLMAP tracks before geometry claims are defensible.
17. M26 shows the M24.2 PRISM schedule transfers mechanically to public scenes, but the 2000-iteration cross-scene direct W&B topology reduction is only `0.5%` to `1.5%`. Larger checkpoint-topology deltas must be labeled as schedule/checkpoint accounting effects until metric paths are reconciled.
18. M27.0 identifies the topology mismatch root cause: W&B topology was logged before standard end-of-iteration prune/densify, while checkpoints were saved after it. Future runs now log post-topology and final-checkpoint counts explicitly.
19. M27 schedule tuning shows fixed topology-pressure schedules are not cross-scene robust. `ratio0p02_geom1400` is strong on ETH3D `courtyard`, but Mip-NeRF 360 `bonsai` rolls back all candidates, so the next method step should use adaptive candidate-window selection.
20. M28 implementation smoke shows rollback-driven ratio decay works and is auditable. Short smokes need `--prism_recent_age_iters 0` when the goal is to exercise candidate logic early; otherwise recent-age protection can hide all candidates.
21. M28 medium ablation shows adaptive schedule decay is not enough by itself. On `bonsai`, the ratio decays to `0.005`, but the selected set is still `3171` triangles and remains gate-rejected. The next risk is over-large global candidate sets, not merely timing.
22. M29 candidate-cap smoke shows small candidate edits can pass the same training-loop gate: cap `256` turns the parking smoke's third candidate attempt into an accepted `64497 -> 64241` edit. This is an implementation result; cross-scene validation is still pending.
23. M29 cap512 medium ablation shows candidate caps are the right direction but not yet the final schedule. `bonsai` reaches `633787` triangles with better SSIM/LPIPS than M28 but lower PSNR; `courtyard` stays better than baseline but worse than the M27/M28 best. A cap sweep or microbatch gate is needed before full-budget claims.
24. M29 cap sweep identifies a sharp candidate batch-size threshold on `bonsai`: `512` commits and improves SSIM/LPIPS with large topology reduction; `1024` rolls back and loses the topology benefit. This points to microbatch candidate gating as the next method step.
25. M30 microbatch gating shows that the cap1024 failure is not all-or-nothing: `bonsai` accepts `768/1024` cumulative candidates before rejecting the full set. However, independent metrics do not improve over cap512, so the next bottleneck is candidate quality/ranking, not simply candidate batch size.
26. M31 hand-weighted candidate-quality ranking is mechanically stable and improves ETH3D `courtyard`, but it is not robust enough on Mip-NeRF 360 `bonsai`. The next ranking step should use measured calibration-view candidate impact instead of only local proxy tensors.
27. M32 measured candidate-impact ranking shows local calibration acceptance is still not sufficient for public-scene test metrics: all measured groups can pass on `bonsai` while independent PSNR/SSIM/LPIPS remain worse than Stage29 cap512. The next bottleneck is calibration-set representativeness and view-diverse candidate diversity.
28. M33 calibration diversity improves `bonsai` enough to beat Stage29 cap512 at equal topology, and the parking smoke demonstrates stricter view coverage can reject local degradations that a smaller calibration set misses. It does not solve the next bottleneck: after one cap512 edit, candidate selection often becomes empty, so the controller needs a second-stage post-commit candidate discovery path.
29. M34 identifies the post-commit no-candidate root cause: `_sync_prism_topology_change` makes every surviving triangle recent, and `recent_t` both protects all triangles and zeroes `prune_score_t` through `risk_t`. A relaxed score that removes only recent risk can find more candidates, but retained extra edits need stricter global/held-out metric control before promotion.
30. M35 shows retained relaxed refresh can pass the strict `bonsai` gate, but validation may repeatedly roll back relaxed commits before one survives. Reports must distinguish total relaxed commit attempts from active retained relaxed commits.
31. M35 `courtyard` shows the retained relaxed cap blocks further relaxed attempts once one active commit survives. This is correct for conservative validation but may under-prune easy scenes; future sweeps should report the cap explicitly.

---

# Prompt M35 — Conservative retained-edit control after post-commit refresh

## Goal

Keep M34 default-off, add retained relaxed commit control, prevent silent erasure of relaxed topology edits, and require a stricter proxy gate before relaxed commits.

## Status

`PASS` on 2026-05-02.

## Result

- `bonsai` W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rszvl7gn`
- ETH3D `courtyard` W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/u2s15ok0`
- report: `docs/car_model/meshprior_stage35_retained_refresh_report.md`
- `bonsai`: final `633275` triangles, active relaxed commits `1`, PSNR `12.2673674`, SSIM `0.2776170`, LPIPS `0.6119390`
- Stage33 `bonsai` reference: final `633787` triangles, PSNR `12.1999207`, SSIM `0.2765326`, LPIPS `0.6125830`
- ETH3D `courtyard`: final `101913` triangles, active relaxed commits `1`, PSNR `15.3831606`, SSIM `0.5080911`, LPIPS `0.5846940`

## Gate

`PASS`: the retained relaxed edit lowers final `bonsai` topology and improves all independent metrics versus Stage33.

# Prompt M36 — Paper-facing metric reconciliation and evidence table

## Goal

Convert M24-M35 into a paper-facing evidence package that separates training-time metrics, independent render metrics, topology counts, and validation/audit metadata.

## Status

`PASS` on 2026-05-02.

## Result

- collector: `scripts/car_model/meshprior_collect_metric_reconciliation.py`
- report: `docs/car_model/meshprior_stage36_metric_reconciliation_report.md`
- output: `outputs/carnet/meshprior/stage36_metric_reconciliation/`
- rows: `10`
- visual panels: `2`
- gate: reproducible from local artifacts and metric paths are separated.

## Completed Steps

1. Build a collector that reads selected runs from `outputs/carnet/meshprior/`, including final-cleanup summaries, retained-topology audits, independent `results.json`, train logs, and W&B URLs.
2. Produce one canonical CSV/Markdown table with columns for scene, method row, schedule, W&B run, final checkpoint triangles, active PRISM commits, rolled-back commits, independent PSNR/SSIM/LPIPS, and metric path.
3. Explicitly label Stage35 as the current best `bonsai` retained-edit row and compare against Stage33, M34 relaxed v3, and the relevant courtyard public-scene rows.
4. Include a short failure taxonomy: no-candidate after sync, validation rollback, cap reached, metric-path mismatch, and dataset geometry-observability limits.
5. Add two or more qualitative visual panels from independent renders if available.
6. Run all commands with W&B preserved for any new training. If only post-processing is run, record exact local commands and source W&B links.
7. Update `docs/car_model/SPCarNet_research_log.md`, this prompt file, and commit/push.

## Gate

`PASS`: the evidence table is reproducible from local artifacts and does not mix training-time and independent metric paths.

# Prompt M37 — Visual/failure-case packaging and public-scene full-budget decision

## Goal

Turn M35/M36 from engineering evidence into a paper-ready result section: visual panels, failure taxonomy, claim wording, and a decision on whether to spend GPU time on full-budget public-scene Stage35 runs.

## Status

`PASS` on 2026-05-02.

## Result

- script: `scripts/car_model/meshprior_package_visual_failures.py`
- report: `docs/car_model/meshprior_stage37_visual_failure_package_report.md`
- output: `outputs/carnet/meshprior/stage37_visual_failure_package/`
- visual panels: parking M24.2, `bonsai` M35, `courtyard` M35
- failure cases: post-commit no-candidate, validation rollback, relaxed cap reached, metric-path mismatch, dataset geometry observability, perceptual metric tradeoff
- training decision: do not start full-budget public-scene training yet; polish final paper figures/tables first.

## Completed Steps

1. Use the M36 table to identify the best rows per scene and the rows that expose tradeoffs.
2. Build qualitative panels for at least `bonsai`, `courtyard`, and the parking M24.2 row if renders exist.
3. Add a failure-case table that links each failure type to at least one concrete run or metadata file.
4. Draft paper-safe claim wording that avoids saying Stage35 universally dominates when LPIPS trades off on `courtyard`.
5. Decide whether the next GPU run should be full-budget public-scene Stage35, a cap sweep, or no training until visuals are complete.
6. If training is run, activate W&B, check GPU availability first, and record commands and run URLs.
7. Update research log and commit/push.

## Gate

`PASS`: visual/failure evidence is linked to reproducible artifacts and the next full-budget decision is explicit.

# Prompt M38 — Final paper assets and optional full-budget public-scene run

## Goal

Convert M36/M37 artifacts into final paper assets: camera-ready table text, figure captions, method limitations, and a narrow go/no-go decision for one full-budget Stage35 public-scene run.

## Required Steps

1. Produce a final paper table markdown that includes only independent metrics and selected rows.
2. Produce figure captions for the parking, `bonsai`, and `courtyard` panels.
3. Write a concise method limitations section using the M37 failure taxonomy.
4. Decide whether to run one full-budget Stage35 public-scene experiment. If yes, check GPU with `nvidia-smi`, activate W&B, record command/URL, then run render+metrics and update tables. If no, document why.
5. Update research log and commit/push.

## Gate

`PASS` if the final table/caption/limitations package is usable as a paper draft section, with the full-budget training decision recorded.

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

---

# Prompt M25 — Public multidataset trainability validation

## Status

`SOFT PASS` on 2026-05-02.

## Result

- dataset root: `/data/peilincai/mesh_datasets`
- disk after setup: `/data` has `4.3T` free; M25 data uses about `35G`
- audit JSON: `outputs/carnet/meshprior/stage25_multidataset/dataset_audit.json`
- report: `docs/car_model/meshprior_stage25_multidataset_validation_report.md`
- trainable current-loader scenes: `10`

Representative W&B runs:

- Mip-NeRF 360 `bonsai`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/x75zddff`
- Tanks and Temples `truck` converted from NSVF mirror: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/5pre7o19`
- ETH3D `courtyard`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/78iu6goq`

## Decision

M25 is enough to proceed to cross-scene method validation on Mip-NeRF 360 and ETH3D. ETH3D's all-scene high-resolution training undistorted archive is downloaded, but only `courtyard` has been converted into the current loader layout. Tanks and Temples should not be used for geometry claims until official COLMAP tracks are acquired or generated locally.

## Prompt M26 — Cross-scene method evidence

## Status

`SOFT PASS` on 2026-05-02.

## Result

- report: `docs/car_model/meshprior_stage26_cross_scene_report.md`
- collector: `scripts/car_model/meshprior_collect_stage26_cross_scene.py`
- output root: `outputs/carnet/meshprior/stage26_cross_scene/`
- generated summary: `outputs/carnet/meshprior/stage26_cross_scene/summary/stage26_cross_scene_summary.md`

W&B runs:

- Mip-NeRF 360 `bonsai` sparse-depth baseline 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xdct9uys`
- Mip-NeRF 360 `bonsai` M24.2 PRISM 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dmasxcej`
- ETH3D `courtyard` sparse-depth baseline 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mdan8yc2`
- ETH3D `courtyard` M24.2 PRISM 2000: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r9zgtuyp`

Key results:

- `bonsai`: PRISM training metrics improve by `+0.0960` PSNR, `+0.0027` SSIM, `-0.0036` LPIPS; independent metrics change by `-0.0304` PSNR, `+0.0305` SSIM, `-0.0060` LPIPS; W&B topology reduction is `0.50%`; validation is geometry-observable.
- `courtyard`: PRISM training metrics improve by `+0.0103` PSNR, `+0.0011` SSIM, `-0.0011` LPIPS; independent metrics improve by `+0.1152` PSNR, `+0.0347` SSIM, `-0.0087` LPIPS; W&B topology reduction is `1.49%`; validation is geometry-observable.

## Decision

M26 proves cross-scene trainability and validation for the current method, but it is not enough for the final paper claim. The direct W&B topology reduction is modest at 2000 iterations, and checkpoint-topology deltas are confounded by schedule/checkpoint accounting. Do not launch a full-budget public-scene sweep until M27 tunes topology pressure and reconciles topology metrics.

## Next Prompt: M27

Tune cross-scene PRISM topology pressure and accounting:

## Status

`SOFT PASS` on 2026-05-02.

## Completed

- M27.0 accounting fix report: `docs/car_model/meshprior_stage27_accounting_fix_report.md`
- W&B accounting smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/i6lfgt66`
- verified future W&B `mesh/triangle_count` and `mesh/final_checkpoint_triangle_count` align with `final_cleanup_summary.json`.
- M27 schedule ablation report: `docs/car_model/meshprior_stage27_schedule_ablation_report.md`
- valid W&B schedule runs:
  - `bonsai` ratio `0.01`, geom `1200`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mlftnbt5`
  - `courtyard` ratio `0.01`, geom `1200`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qvrnsj2v`
  - `bonsai` ratio `0.02`, geom `1400`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/27vl4jnt`
  - `courtyard` ratio `0.02`, geom `1400`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ffp07dua`

## Result

Best row: `ratio0p02_geom1400`.

- `bonsai`: final `1357128` triangles, `0` commits, `6` rollbacks, validation `3/3` observable and `2/3` pass, independent PSNR `12.3005`, SSIM `0.2408`, LPIPS `0.6194`.
- `courtyard`: final `100858` triangles, `1` commit, `0` rollbacks, validation `4/4` observable and `3/4` pass, independent PSNR `15.0739`, SSIM `0.4857`, LPIPS `0.5794`.

## Decision

M27 is a `SOFT PASS`: ETH3D achieves meaningful topology reduction with non-worse independent metrics, but `bonsai` rolls back the stronger schedule and remains near baseline topology. Do not scale the fixed schedule directly to a full-budget public sweep.

## Next Prompt: M28

Adaptive PRISM schedule selection:

## Status

`SOFT PASS` on 2026-05-02.

## Completed

- code:
  - `arguments/__init__.py`
  - `train.py`
- report: `docs/car_model/meshprior_stage28_adaptive_schedule_smoke_report.md`
- medium report: `docs/car_model/meshprior_stage28_adaptive_schedule_medium_report.md`
- W&B smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1kmwbu8g`
- W&B `bonsai` medium: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/38p6bgw4`
- W&B `courtyard` medium: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/piadupsm`
- smoke output: `outputs/carnet/meshprior/stage28_adaptive_schedule/parking_adaptive_retry_smoke_v5_140iter/model`
- verified adaptive candidate ratio sequence: `0.04 -> 0.02 -> 0.01`
- medium `bonsai`: final `1357119` triangles, `0` commits, `8` rejected candidate gates, independent PSNR `12.3054`, SSIM `0.2410`, LPIPS `0.6196`.
- medium `courtyard`: final `100858` triangles, `1` commit, `41` no-candidate retries, independent PSNR `15.0919`, SSIM `0.4844`, LPIPS `0.5778`.

## Result

M28 adaptive scheduling preserves the M27 ETH3D result and gives cleaner diagnostics for `bonsai`, but it does not meaningfully reduce `bonsai` topology. The failed `bonsai` case is now precise: after decay to ratio `0.005`, the global candidate selector still selects `3171` triangles and the counterfactual gate rejects the edit.

## Gate

`SOFT PASS`: adaptive scheduling only improves diagnosis and preserves one scene; it does not make the hard `bonsai` scene accept topology edits.

## Next Prompt: M29

Granular PRISM candidate selection:

## Status

`SOFT PASS` on 2026-05-02.

## Goal

Fix the M28 failure mode where a small global candidate ratio still selects too many triangles on dense scenes. The next method change should make candidate edits smaller, auditable, and locally gateable instead of relying only on global ratio decay.

## Required Work

1. DONE: Add an opt-in hard cap `--prism_candidate_max_count_per_round`.
2. DONE: Apply the cap after candidate ranking and before checkpoint mutation, while preserving default behavior when the cap is unset.
3. DONE: Log to metadata and W&B:
   - candidate pool count,
   - ratio target count,
   - cap-limited selected count,
   - rejected/accepted gate result,
   - active adaptive ratio.
4. DONE: Run a parking smoke that forces candidate selection and verifies the cap changes selected count without breaking rollback.
5. DONE: Run `bonsai` / `courtyard` medium ablation against M28:
   - base schedule `ratio0p02_geom1400`,
   - adaptive retry enabled,
   - cap candidates at `512`,
   - online W&B required.
6. DONE: Evaluate with:
   - training internal metrics,
   - independent `render.py + metrics.py`,
   - PRISM validation artifacts,
   - counterfactual gate JSON,
   - final checkpoint topology and cleanup summary.
7. DONE: Promote the next substep to M30 microbatch candidate gating: split selected candidates into small batches, run counterfactual gates per cumulative batch, and commit only accepted batches.
8. DONE: Sweep `bonsai` cap values `256`, `512`, and `1024`; cap512 is the current best Pareto row, cap1024 rolls back.

## Required Outputs

- code changes in `arguments/__init__.py`, `train.py`, and `utils/prism_counterfactual.py`.
- `docs/car_model/meshprior_stage29_candidate_cap_report.md`
- `docs/car_model/meshprior_stage29_candidate_cap_medium_report.md`
- `docs/car_model/meshprior_stage29_candidate_cap_sweep_report.md`
- output root: `outputs/carnet/meshprior/stage29_candidate_selection/`
- W&B URLs and exact command logs under the output root.

## Gate

`PASS` if capped or microbatched candidate selection makes `bonsai` accept a topology edit with non-worse independent SSIM/LPIPS and preserves the ETH3D topology gain.

`SOFT PASS` if it improves diagnosis or one scene but still misses cross-scene topology improvement.

`FAIL` if it causes invalid checkpoints, missing W&B, or hidden topology/accounting mismatch.

---

# Prompt M30 — Microbatch PRISM candidate gating

## Status

`SOFT PASS / diagnostic PASS` on 2026-05-02.

## Goal

Recover useful accepted topology edits from candidate sets that are too large as a single counterfactual mutation. The method should split a cap-limited candidate set into smaller batches, gate each cumulative batch, and commit only accepted batches.

## Result

- Implemented default-off flags:
  - `--prism_candidate_microbatch_gate`
  - `--prism_candidate_microbatch_size`
  - `--prism_candidate_microbatch_max_batches`
- Added per-microbatch counterfactual JSON files, W&B counters, and PRISM round metadata.
- Parking smoke W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dioe1cz1`
- `bonsai` W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mfvhexjb`
- `courtyard` W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ha9qi1ih`
- Report: `docs/car_model/meshprior_stage30_microbatch_gate_report.md`
- Output root: `outputs/carnet/meshprior/stage30_microbatch_gate/`

## Evidence

- Parking smoke: iter `91` selected `644` candidates, accepted `3/3` microbatches, and committed `64497 -> 63853`.
- `bonsai`: iter `1501` selected `1024`, accepted `3/4`, committed `634299 -> 633531`, independent PSNR `12.1423`, SSIM `0.2770`, LPIPS `0.6136`.
- `courtyard`: iter `1501` selected `1024`, accepted `4/4`, committed `102919 -> 101895`, independent PSNR `15.0635`, SSIM `0.4828`, LPIPS `0.5802`.

## Gate

`SOFT PASS / diagnostic PASS`.

The mechanism is stable and useful for diagnosis. It is not promoted to the default schedule because Stage29 cap512 remains the better conservative `bonsai` topology-quality Pareto row.

## Next Prompt: M31

Completed below. The next active prompt is M32.

# Prompt M31 — Candidate-quality calibration and ranking

## Status

`SOFT PASS / diagnostic PASS` on 2026-05-02.

## Goal

Improve cap-limited candidate selection by ranking candidates with a blended quality score instead of raw prune score alone.

## Result

- Implemented default-off flags:
  - `--prism_candidate_quality_rank`
  - `--prism_candidate_quality_prune_weight`
  - `--prism_candidate_quality_render_penalty`
  - `--prism_candidate_quality_geometry_penalty`
  - `--prism_candidate_quality_orientation_penalty`
  - `--prism_candidate_quality_utility_penalty`
  - `--prism_candidate_quality_uncertainty_penalty`
- Added rank-score selection support to `utils/prism_counterfactual.py`.
- Added W&B/TensorBoard/round-metadata fields for candidate-quality score components.
- Parking smoke W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ucqyou26`
- `bonsai` W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/22r3et7s`
- `courtyard` W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xt4a2cn0`
- Report: `docs/car_model/meshprior_stage31_candidate_quality_report.md`
- Output root: `outputs/carnet/meshprior/stage31_candidate_quality/`

## Evidence

- Parking smoke: iter `91` selected `512` candidates and committed `64497 -> 63985`.
- `bonsai`: iter `1501` selected `512`, committed `634299 -> 633787`, independent PSNR `12.1891`, SSIM `0.2756`, LPIPS `0.6136`.
- `courtyard`: iter `1501` selected `512`, committed `102916 -> 102404`, independent PSNR `15.0732`, SSIM `0.4837`, LPIPS `0.5788`.

## Gate

`SOFT PASS / diagnostic PASS`.

The mechanism is stable and useful for diagnosis. It is not promoted to the default schedule because `bonsai` does not robustly improve over Stage29 cap512.

## Next Prompt: M32

Completed below. The next active prompt is M33.

# Prompt M32 — Measured candidate-impact ranking

## Status

`SOFT PASS / diagnostic PASS` on 2026-05-02.

## Goal

Rank candidate groups by measured counterfactual impact on calibration views rather than relying only on raw prune score or local proxy tensors.

## Result

- Implemented default-off flags:
  - `--prism_candidate_measured_impact_rank`
  - `--prism_candidate_measured_pool_multiplier`
  - `--prism_candidate_measured_group_size`
  - `--prism_candidate_measured_max_groups`
- Added per-group counterfactual JSON, W&B/TensorBoard counters, and PRISM round metadata.
- Parking smoke W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xg4fsvd8`
- `bonsai` measured-rank W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/56l3tz23`
- `courtyard` measured-rank W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fb7jfcaj`
- `bonsai` measured+quality diagnostic W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xooe27um`
- Report: `docs/car_model/meshprior_stage32_measured_candidate_rank_report.md`
- Output root: `outputs/carnet/meshprior/stage32_measured_candidate_rank/`

## Evidence

- Parking smoke: iter `91` committed `64497 -> 63985`, with `3/3` measured groups accepted.
- `bonsai` measured-rank: iter `1501` committed `634299 -> 633787`, independent PSNR `12.1742`, SSIM `0.2758`, LPIPS `0.6137`.
- `courtyard` measured-rank: iter `1501` committed `102916 -> 102404`, independent PSNR `15.1390`, SSIM `0.4850`, LPIPS `0.5792`.
- `bonsai` measured+quality diagnostic: independent PSNR `12.1708`, SSIM `0.2760`, LPIPS `0.6133`.

## Gate

`SOFT PASS / diagnostic PASS`.

The mechanism is stable and useful, but it fails the M32 promotion gate because `bonsai` remains below Stage29 cap512 independent metrics.

## Next Prompt: M33

Completed below. The next active prompt is M34.

# Prompt M33 — Calibration-set representativeness and view-diverse candidate selection

## Status

`SOFT PASS / diagnostic PASS` on 2026-05-02.

## Goal

Audit and improve PRISM counterfactual calibration representativeness so measured candidate ranking is not driven only by a small local hard-view set.

## Result

- Implemented default-off flags:
  - `--prism_calib_diverse_views`
  - `--prism_calib_diverse_test_views`
  - `--prism_calib_diverse_train_views`
- Added `prism_debug/calibration_views.json` with image name, source, sparse-match counts, observability reason, and hard-view score where applicable.
- Added per-view counterfactual deltas to `counterfactual_gate_iter_*.json`.
- Parking smoke W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ms95810g`
- `bonsai` diverse-calibration W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kg5htc8u`
- `courtyard` diverse-calibration W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w9c0b65f`
- Report: `docs/car_model/meshprior_stage33_calibration_diversity_report.md`
- Output root: `outputs/carnet/meshprior/stage33_calibration_diversity/`

## Evidence

- Parking smoke: strict diverse calibration rejected all edits; final topology stayed `64497`, and per-view deltas exposed local regressions.
- `bonsai`: iter `1501` committed `634299 -> 633787`, calibration set had `24` views, independent PSNR `12.1999`, SSIM `0.2765`, LPIPS `0.6126`.
- `courtyard`: iter `1501` committed `102919 -> 102407`, calibration set had `17` views, independent PSNR `15.0737`, SSIM `0.4840`, LPIPS `0.5790`.

## Gate

`SOFT PASS / diagnostic PASS`.

Stage33 passes the `bonsai` promotion threshold against Stage29 cap512 with equal topology, and it improves over Stage29 on `courtyard`. It is not the universal default because Stage32 measured rank still gives the stronger `courtyard` PSNR/SSIM row.

## Original Required Work

Calibration-set representativeness and view-diverse candidate selection:

1. Keep Stage29 cap512 as the conservative default.
2. Keep M31/M32 as opt-in diagnostic selectors.
3. Audit calibration views selected for `bonsai` and `courtyard`: image names, train/test split source, sparse-match counts, PSNR/MAE hardness, and whether they cover the final independent test viewpoints.
4. Add an opt-in calibration-diversity mode:
   - include more held-out/test-like views when available;
   - stratify by camera pose or image name order to avoid only local hard train views;
   - keep sparse-depth observability thresholds explicit;
   - log the chosen view list and per-view counterfactual deltas.
5. Rerun parking smoke, then `bonsai` first. Only rerun `courtyard` if `bonsai` reaches or exceeds Stage29 cap512 independent metrics.
6. Gate as `PASS` only if `bonsai` and `courtyard` both match or improve Stage29 cap512 independent metrics with equal or lower topology.

## Next Prompt: M34

Post-commit candidate refresh and second-stage low-risk discovery:

1. Keep Stage29 cap512, Stage32 measured rank, and Stage33 diverse calibration default-off unless explicitly selected.
2. Investigate why candidate selection becomes empty after the first accepted cap512 edit on `bonsai` and `courtyard`.
3. Add an opt-in post-commit candidate refresh mode that can search for a second low-risk candidate set after recovery:
   - log raw eligible candidate count before the cap;
   - log rejection reasons for no-candidate rounds;
   - optionally relax only non-render risk thresholds after a successful recovery window;
   - keep the counterfactual gate and rollback mandatory.
4. Run a parking smoke with online W&B and strict debug JSON.
5. Run `bonsai` first with Stage33 diverse calibration. Only run `courtyard` if `bonsai` keeps or improves Stage33 metrics and finds an additional committed edit or a well-explained no-candidate diagnosis.
6. Gate as `PASS` only if a second useful edit is found without regressing independent `bonsai` metrics; otherwise gate as diagnostic if the no-candidate root cause is measured and documented.

# Prompt M34 — Post-commit candidate refresh and second-stage low-risk discovery

## Status

`SOFT PASS / diagnostic PASS` on 2026-05-02.

## Goal

Investigate why candidate selection becomes empty after the first accepted cap512 edit, then add an opt-in second-stage candidate discovery path that keeps the counterfactual gate mandatory.

## Result

- Implemented default-off flags:
  - `--prism_post_commit_candidate_refresh`
  - `--prism_post_commit_refresh_min_prune_score`
- Added no-candidate blocker diagnostics to PRISM round metadata.
- Added post-commit relaxed candidate scoring that removes only `recent_t` from the risk term while preserving uncertainty, boundary, nonmanifold, ground/ROI, geometry/orientation keep, and render keep risk.
- Parking smoke W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rt3cxxhh`
- Parking recent0 smoke W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kke60qhc`
- `bonsai` root-cause v1 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/szkqpowq`
- `bonsai` root-cause v2 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/npagb743`
- `bonsai` relaxed-score v3 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/lt1v4652`
- `bonsai` second-edit-only diagnostic v4 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/zhy368pr`
- Report: `docs/car_model/meshprior_stage34_post_commit_refresh_report.md`
- Output root: `outputs/carnet/meshprior/stage34_post_commit_refresh/`

## Evidence

- Root cause: after the first commit, later `bonsai` no-candidate rounds recorded `block_recent=633787`, `candidate_pool_count=0`, and normal `prune_score_t` collapsed to zero because `recent_t` enters `risk_t`.
- v3 retained additional edits:
  - iter `1501`: `634299 -> 633787`
  - iter `1592`: relaxed `633787 -> 633275`
  - iter `1683`: relaxed `633275 -> 632763`
  - iter `1774`: relaxed `632763 -> 632251`
  - iter `1956`: relaxed `632251 -> 631739`
- v3 independent metrics: PSNR `12.2019978`, SSIM `0.2757282`, LPIPS `0.6129612`, final topology `631739`.
- Stage33 reference: PSNR `12.1999207`, SSIM `0.2765326`, LPIPS `0.6125830`, final topology `633787`.
- v4 proved that simply limiting candidate rounds is not enough: it logs a second relaxed commit at iter `1592`, but final checkpoint returns to `633787`, so retained topology control still needs a dedicated controller.

## Gate

`SOFT PASS / diagnostic PASS`.

M34 succeeds as a mechanism and root-cause diagnosis, but it is not a hard `PASS`: the retained v3 run improves topology and PSNR while slightly regressing SSIM/LPIPS. Do not promote it as a default schedule.

## Next Prompt: M35

Conservative retained-edit control after post-commit refresh:

1. Keep M34 default-off.
2. Add a separate cap for retained relaxed commits, distinct from total candidate rounds.
3. Add an explicit post-relaxed-commit retained-topology check so later recovery/validation cannot silently erase the second edit without metadata.
4. Add a stricter held-out-view or independent-metric proxy gate for relaxed commits.
5. Rerun `bonsai` with one retained relaxed edit. Only run `courtyard` if all independent `bonsai` metrics match or improve Stage33 while final topology is lower than `633787`.
6. Gate as `PASS` only if the retained second edit lowers final topology without regressing independent PSNR/SSIM/LPIPS; otherwise keep as diagnostic and report the limiting metric.
