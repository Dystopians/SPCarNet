# FinalNeurIPSPrompts.md — MeshSplatOpt Final Plan

Date: 2026-05-04  
Repository: `Dystopians/SPCarNet`  
Target: **a NeurIPS-grade MeshSplatOpt method that improves downstream Mesh Splatting scene optimization: geometry repair, topology efficiency, appearance quality, and auditable rollback.**

---

## 0. Read this first: current evidence and hard pivot

You are working inside `Dystopians/SPCarNet` on `main` or a new final branch.

The current project has advanced beyond the original PRISM pruning line. The strongest current evidence is **not** local snap/fill edit quality. The strongest evidence is:

> **clean long Mesh Splatting → evidence-compatible compaction → strict topology-frozen recovery**.

The current best parking result is:

```text
R53.01:
clean 22k -> remove smallest-area 70% triangles -> strict topology-frozen recovery 22k->26k
triangles: 2,564,473 vs clean 22k 8,548,242
PSNR: 18.706 vs 18.480
SSIM: 0.648 vs 0.635
LPIPS: 0.338 vs 0.347
AbsRel: 0.080 vs 0.082
Depth MAE: 1.854 vs 1.868
Normal angle: 44.261 vs 45.108
```

This is the first row that honestly dominates the strongest clean long baseline while using far fewer primitives.

However, the recent documents also show several negative facts:

```text
- R44 low-topology sparse-decay path loses badly to clean 22k on RGB/depth.
- R45/R46 teacher distillation from the ultra-low-topology R44 checkpoint is rejected.
- R51/R52 direct LPIPS loss is rejected.
- R17-R21 snap selectors are gate-safe but not quality-winning.
- R22/R26 fill patches are gate-safe but fail medium/full recovery versus baseline+sparse.
- R28 proves grid-fill + sparse-depth does not beat matched baseline + sparse-depth.
- R25 proves unfrozen densification after edits explodes topology and still loses render quality.
- The real checkpoint is triangle soup: edge-connected boundary-loop CSEF is invalid for real checkpoint selection unless a spatial/raster substitute is used.
```

Therefore, the final paper story must be:

> **MeshSplatOpt is an evidence-certified compact-repair optimizer. It trains a strong clean mesh, computes a Counterfactual Surface Evidence Field, performs evidence-compatible topology compaction/repair edits under rollback certificates, freezes topology, and recovers appearance/geometry. The edit calculus supports deletion, collapse, snap, split, fill, and appearance repair, but only compaction+recovery is currently headline-validated. Snap/fill remain repair branches that must earn their own evidence.**

Do not write a paper claiming that current snap/fill edits improve quality unless later stages prove it against equal-budget controls.

---

## 1. Final method concept

### Method name

**MeshSplatOpt: Counterfactually Certified Compact-Repair Optimization for Mesh Splatting**

Short name: **MeshSplatOpt**

### One-sentence paper claim

> MeshSplatOpt improves Mesh Splatting scene representations by converting a high-quality over-complete mesh into a compact, evidence-consistent, topology-frozen representation through counterfactually certified compaction and repair, followed by sparse-geometry-guided recovery.

### Core research abstraction

**Counterfactual Surface Evidence Field (CSEF)**:

```text
CSEF(x, n, region) = {
  positive_surface_evidence,      # multi-view visibility, COLMAP support, normal agreement, render contribution
  negative_free_space_evidence,   # camera rays / sparse evidence saying surface should not exist
  explanation_debt,               # residual pixels, missing depth, holes, roughness, discontinuity
  prior_support,                  # ground/plane/object/smoothness prior support
  topology_cost,                  # triangles, vertices, memory, render cost
  uncertainty                     # low coverage, ambiguous geometry, weak prior, bad track support
}
```

MeshSplatOpt uses CSEF to score local edits:

```text
delete/prune         remove weakly supported primitives
collapse/merge       reduce redundant topology while preserving surface evidence
snap/deform          correct dents, roughness, and misalignment
split/subdivide      allocate topology where evidence debt is high
fill/patch           repair certified holes or ground voids
appearance recovery  restore radiance after accepted geometry changes
```

Each edit is reversible and must pass counterfactual certificates before it is committed.

### Current empirical truth

The headline branch should be:

```text
strong clean long baseline
→ evidence-compatible compaction sweep
→ strict topology freeze
→ sparse-COLMAP-depth-guided recovery with validated schedules
→ independent render/metrics/geometry evaluation
→ Pareto table vs clean long, Stage35, PRISM, posthoc decimation, sparse-only, and no-freeze controls
```

The repair branch should be:

```text
defect mining
→ snap/fill/object/ground proposals
→ render-backed counterfactual gate
→ recovery
→ equal-budget control
→ only promote if it beats controls
```

The giant-hole branch should be:

```text
synthetic giant void + injected real-checkpoint void benchmark
→ CSEF ground/plane/depth/boundary certificates
→ fill proposal
→ strict free-space/render/geometry gate
→ topology-frozen recovery
→ compare against no fill, classical fill, plane fill without gate, and sparse-only recovery
```

---

## 2. Non-negotiable operating rules for Codex

Apply to every stage.

```bash
git status --short
python --version
python -m compileall scripts/car_model ss3dm_prior utils -q
```

Before GPU work:

```bash
nvidia-smi
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
```

Rules:

1. Work one prompt at a time.
2. Do not proceed after a failed hard gate.
3. Never mix training-time metrics with independent `render.py + metrics.py` values.
4. Never compare a long method run against a short clean baseline.
5. Always compare against the strongest clean long baseline available for that scene.
6. All edit operations must have rollback snapshots.
7. All runs must write exact commands, W&B URLs, metrics JSON paths, and checkpoint paths.
8. All reports must separate:
   - inference-time metrics,
   - training metrics,
   - oracle diagnostics,
   - prior-only diagnostics.
9. Negative results are first-class evidence.
10. Commit and push after every completed `PASS`, `SOFT PASS`, `FAIL`, or `STOP`.

---

# Prompt F0 — Final current-state audit and claim reset

```text
You are working inside Dystopians/SPCarNet.

Mission:
Audit the current repository state after R0-R56 and reset the final NeurIPS claim to match actual evidence.

Read first:
- README.md
- README.zh.md if present
- docs/car_model/SPCarNet_research_log.md
- docs/car_model/parking_best_clean_long_vs_method_long_report.md
- docs/car_model/parking_clean_to_compact_repair_report.md
- docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md
- docs/car_model/meshsplatopt_stageR15_01_04_multiscene_freeze_medium_report.md
- docs/car_model/meshsplatopt_stageR14_aggregate_decision_report.md
- docs/car_model/meshsplatopt_stageR1_repair_RFC.md
- docs/NeurIPSRepairPrompts.md

Run:
git status --short
python --version
python -m compileall scripts/car_model ss3dm_prior utils -q

Write:
- docs/car_model/final_stageF0_current_state_audit.md

The audit must include:
1. Current git branch and commit.
2. Dirty files.
3. Current strongest validated result.
4. Current strongest negative results.
5. Which goals are partially achieved:
   - topology reduction,
   - render improvement,
   - sparse geometry improvement,
   - real checkpoint edit application,
   - rollback/counterfactual gate,
   - giant-hole repair.
6. Which goals are not achieved:
   - cross-scene clean-to-compact dominance,
   - real giant-hole repair,
   - snap/fill edit-driven full-budget gain,
   - universal trusted sparse sampling,
   - NeurIPS-ready multi-scene evidence.
7. Final claim reset:
   MeshSplatOpt should be framed as evidence-certified compact-repair optimization, not as local snap/fill repair unless future stages prove it.
8. `PROCEED_TO_F1` or `STOP`.

Append a dated entry to docs/car_model/SPCarNet_research_log.md.
Commit and push.

Gate:
PASS only if the report explicitly states that R53/R48/R55 are currently stronger than R44 and that snap/fill are not yet headline methods.
```

---

# Prompt F1 — Final paper story and method spec

```text
Stage F0 must be PASS.

Mission:
Write the final paper-facing method spec for MeshSplatOpt, using current evidence.

Write:
- docs/car_model/final_stageF1_method_spec.md

Required structure:

1. Title options:
   - MeshSplatOpt: Counterfactually Certified Compact-Repair Optimization for Mesh Splatting
   - Evidence-Calibrated Compact Repair for Mesh Splatting
   - From Overcomplete Mesh Splatting to Compact Certified Scene Meshes

2. Main claim:
   MeshSplatOpt improves Mesh Splatting by compacting overcomplete clean meshes into compact evidence-consistent topology, then recovering appearance and sparse geometry under strict topology freeze and counterfactual validation.

3. Core abstraction:
   CSEF with positive evidence, negative free-space, explanation debt, prior support, topology cost, uncertainty.

4. Algorithm:
   A. Train or load a strong clean long Mesh Splatting baseline.
   B. Build CSEF.
   C. Generate compaction and repair edit candidates.
   D. Apply counterfactual certification and rollback.
   E. Run strict topology-frozen recovery.
   F. Report Pareto frontier and repair diagnostics.

5. Current load-bearing empirical branch:
   clean-to-compact:
   clean 22k -> prune 65/70/75/80/90% by evidence-compatible criterion -> freeze topology -> recover -> evaluate.

6. Repair branch:
   snap/fill/object-prior/ground-void edits are optional until equal-budget controls show benefit.

7. What not to claim:
   - Do not claim current snap/fill improves full-budget quality.
   - Do not claim R44 beats clean long on render.
   - Do not claim prior-only giant void fill reconstructs observed geometry.
   - Do not hide scene-specific sparse sampling tuning.

8. Novelty statement:
   Not training-time pruning alone; not sparse depth alone; not posthoc decimation alone.
   The contribution is the certified compact-repair loop:
   CSEF-scored edit proposal + counterfactual gate + topology freeze + sparse-geometry-guided recovery + Pareto-evaluated downstream mesh optimization.

9. Required baselines:
   - clean MeshSplatting long baseline,
   - clean MeshSplatting same-iteration continuation,
   - Stage35 PRISM,
   - delete-only PRISM,
   - clean-to-compact area prune without recovery,
   - clean-to-compact with recovery,
   - sparse-depth-only recovery,
   - posthoc simplification/QEM/area prune,
   - no-freeze recovery,
   - snap/fill branches if promoted.

10. Required figures/tables:
   - Method diagram.
   - CSEF/edit calculus diagram.
   - Pareto frontier: quality vs triangles.
   - Clean long vs R53 qualitative montage.
   - Failure table: rejected directions.
   - Ablation table.

Append research-log entry.
Commit and push.

Gate:
PASS only if the story can be explained in one paragraph and does not overclaim rejected edit primitives.
```

---

# Prompt F2 — Fair baseline registry and metric-integrity collector

```text
Stage F1 must be PASS.

Mission:
Build a single fair-baseline registry and collector so all final results compare against correct baselines.

Create/update:
- scripts/car_model/final_collect_baselines_and_results.py
- docs/car_model/final_stageF2_baseline_registry_design.md
- docs/car_model/final_stageF2_baseline_registry_report.md

The registry must support rows for:
- scene
- method label
- source checkpoint
- training start iteration
- final iteration
- triangle count
- vertex count
- independent PSNR/SSIM/LPIPS
- sparse AbsRel
- sparse Depth MAE
- sparse normal angle
- W&B URL
- exact command path
- metric source path
- whether metric is independent or training-time
- whether topology is frozen
- whether sparse-depth loss is enabled
- sparse sampling mode/fraction/lambda/decay
- whether edit primitives were applied
- edit class
- prior-only flag
- decision

Initial rows must include all locally known:
- clean parking 7k, 22k, 30k
- R44.01/R43.01b/R48.01/R53.01/R55.01/R50.01/R56.01
- courtyard R40.02/R43.02b/R44.02 if present
- bonsai R31/R41/R42 if present
- Stage35 public baselines
- R15 multi-scene medium rows

The collector must write:
- outputs/carnet/meshsplatopt/final_baseline_registry/final_results.json
- outputs/carnet/meshsplatopt/final_baseline_registry/final_results.csv
- outputs/carnet/meshsplatopt/final_baseline_registry/final_results.md

Add integrity checks:
1. Refuse to compare method iteration 26k against clean 7k as headline.
2. Flag missing independent metrics.
3. Flag training-time metrics if accidentally used.
4. Flag topology mismatch where checkpoint triangles and report triangles disagree.
5. Flag any row whose W&B URL is missing for training runs.

Append research-log entry.
Commit and push.

Gate:
PASS only if the collector reproduces the R53 vs clean 22k table and flags R44 as render-losing against clean 22k.
```

---

# Prompt F3 — Cross-scene clean-to-compact feasibility planning

```text
Stage F2 must be PASS.

Mission:
Plan cross-scene clean-to-compact validation before launching expensive runs.

Write:
- docs/car_model/final_stageF3_cross_scene_compact_plan.md

Required scenes:
- parking_phone_tiny
- mipnerf360 bonsai
- ETH3D courtyard
- one additional COLMAP-compatible public scene if available locally
- optional flowers if present and compatible

For each scene, audit:
1. best clean long checkpoint currently available;
2. if missing, exact command to train it;
3. current final triangle count;
4. available Stage35/PRISM baseline;
5. available sparse geometry evaluation;
6. recommended image resolution and split;
7. estimated GPU budget;
8. risk level.

Define compact sweep:
- prune fractions: 50%, 60%, 65%, 70%, 75%, 80%, 90%
- for each fraction:
  - apply evidence-compatible compact edit to clean long checkpoint;
  - render/evaluate immediately after compaction;
  - strict topology-frozen recovery;
  - render/evaluate after recovery.

Start with medium or one-scene dry-run only if missing baseline risk is high.

Hard rule:
Do not launch cross-scene compaction until the plan names exact clean baselines and output paths.

Append research-log entry.
Commit and push.

Gate:
PASS only if the plan identifies exactly which clean long baselines are present/missing and which run should start first.
```

---

# Prompt F4 — Evidence-compatible compaction selector beyond area-only

```text
Stage F3 must be PASS.

Mission:
Upgrade the compaction selector from smallest-area-only to CSEF evidence-compatible compaction.

Current R53 works with area pruning on parking. The final method needs a more research-worthy selector that can be compared against area-only.

Create:
- ss3dm_prior/meshsplatopt/compact_selector.py
- scripts/car_model/meshsplatopt_select_compaction_candidates.py
- scripts/car_model/smoke_test_final_stageF4_compact_selector.py
- docs/car_model/final_stageF4_compact_selector_design.md
- docs/car_model/final_stageF4_compact_selector_report.md

Selector inputs:
- triangle area
- render contribution / visibility if available
- sparse geometry support
- normal/orientation support
- local redundancy / coplanarity
- boundary/edge risk
- recent/topology risk
- CSEF positive evidence
- CSEF negative free-space
- CSEF explanation debt
- topology cost
- uncertainty

Implement selectable modes:
1. `area_smallest`
2. `csef_low_evidence`
3. `csef_low_evidence_boundary_protected`
4. `pareto_area_csef`
5. `random_same_count` control

Output:
- compaction_candidates.json
- compaction_score_table.npz
- compaction_summary.csv
- compaction_report.md

Smoke:
- synthetic mesh with:
  - supported surface,
  - redundant small triangles,
  - boundary hole,
  - floater,
  - large ground patch.
- Verify:
  - supported boundary and hole rims are protected;
  - redundant small triangles are selected;
  - high-debt repair regions are not removed;
  - random control selects same count.

Append research-log entry.
Commit and push.

Gate:
PASS only if selector produces at least one non-area mode and shows different candidate choices from area-only on synthetic data.
```

---

# Prompt F5 — Apply compaction to real Mesh Splatting checkpoints

```text
Stage F4 must be PASS.

Mission:
Apply F4 compaction selectors to real Mesh Splatting checkpoints and make every compact checkpoint renderable.

Create/update:
- scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py
- ss3dm_prior/meshsplatopt/checkpoint_compaction.py
- scripts/car_model/smoke_test_final_stageF5_checkpoint_compaction.py
- docs/car_model/final_stageF5_checkpoint_compaction_report.md

Requirements:
1. Load Mesh Splatting state dict checkpoint.
2. Select triangles by compaction_candidates.json.
3. Remove triangles and synchronize all per-face fields.
4. Preserve schema.
5. Save normal model directory layout.
6. Validate checkpoint loads.
7. Render one smoke model if small enough.
8. Write topology audit:
   - pre triangles,
   - post triangles,
   - pre vertices,
   - post vertices,
   - removed fraction,
   - selector mode,
   - degenerate face count,
   - invalid index count.

Run on parking clean 22k first:
- area_smallest at 70% as R53 reproduction check.
- csef selector at 70% as candidate final method.

Artifacts:
- outputs/carnet/meshsplatopt/final_stageF5_checkpoint_compaction/<run_name>/

Append research-log entry.
Commit and push.

Gate:
PASS only if parking area_smallest 70% reproduces R53 pre-recovery topology count within tolerance and the csef selector produces a renderable checkpoint.
```

---

# Prompt F6 — Strict topology-frozen recovery runner v2

```text
Stage F5 must be PASS.

Mission:
Create the final strict topology-frozen recovery runner and prevent old topology-control ambiguity.

Update/create:
- scripts/car_model/meshsplatopt_run_strict_compact_recovery.py
- docs/car_model/final_stageF6_strict_recovery_design.md
- docs/car_model/final_stageF6_strict_recovery_report.md

The runner must enforce:
- --freeze_topology_updates
- --skip_restricted_delaunay
- no standard prune/densify branch
- no topology mutation after load unless explicitly allowed
- W&B online logging
- exact command saved
- final topology audit saved

Inputs:
- source scene path
- compact model path
- load iteration
- recovery iterations
- sparse-depth options
- topology-freeze options
- output path
- wandb group/name

Recovery presets:
1. `compact_render_only`
   - no sparse depth
   - strict topology freeze
2. `compact_sparse_low_lambda`
   - sparse lambda 0.001-0.002
   - mixed_low_error sampling
   - per-scene fraction
   - optional decay
3. `compact_sparse_decay`
   - low lambda with decay window

For parking, reproduce:
- R53.01: prune70, clean 22k -> 26k, strict topology freeze.
- R48.01: prune80, clean 22k -> 26k.

Write:
- recovery_summary.json
- topology_audit.json
- exact_train_command.txt
- wandb_url.txt
- render/metrics/eval commands

Append research-log entry.
Commit and push.

Gate:
PASS only if R53.01 reproduction or near-reproduction completes and final triangle count is unchanged throughout recovery.
```

---

# Prompt F7 — Parking final compaction Pareto sweep

```text
Stage F6 must be PASS.

Mission:
Produce the final parking Pareto sweep using fair clean-long baselines.

Runs:
For selector modes:
- area_smallest
- csef_low_evidence_boundary_protected
- pareto_area_csef
- random_same_count control

For prune fractions:
- 50%
- 60%
- 65%
- 70%
- 75%
- 80%
- 90%

For each:
1. Apply compaction to clean 22k.
2. Evaluate immediately after compaction.
3. Run strict topology-frozen recovery to 26k.
4. Evaluate independent render metrics.
5. Evaluate sparse COLMAP geometry.
6. Record W&B and exact commands.

Create:
- scripts/car_model/final_run_parking_compact_pareto.py
- scripts/car_model/final_collect_parking_compact_pareto.py
- docs/car_model/final_stageF7_parking_pareto_report.md
- outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/

Required table:
- selector
- prune fraction
- final triangles
- PSNR/SSIM/LPIPS
- AbsRel/DepthMAE/normal
- delta vs clean22k
- delta vs clean30k
- delta vs R53
- W&B

Hard decisions:
- Promote CSEF selector only if it beats area_smallest at the same topology or gives stronger Pareto.
- Keep area_smallest as baseline if CSEF does not beat it.
- Reject 90% if render or LPIPS collapse.
- Early-stop continuation at 26k unless evidence says otherwise.

Append research-log entry.
Commit and push.

Gate:
PASS only if the table includes at least one compact row that beats clean22k on render and geometry while reducing triangles by at least 50%.
```

---

# Prompt F8 — Cross-scene compact-recovery pilot

```text
Stage F7 must be PASS.

Mission:
Test whether compact-recovery transfers beyond parking.

Scenes:
- bonsai
- courtyard
- at least one additional COLMAP-compatible scene if available

For each scene:
1. Identify best clean long baseline.
2. If missing, run clean long baseline first.
3. Apply compact selector at conservative fractions:
   - 50%
   - 60%
   - 70%
   - 80%
4. Run strict topology-frozen recovery.
5. Evaluate independent render metrics and sparse geometry.
6. Compare to clean long baseline, Stage35, and sparse-only recovery.

Create:
- scripts/car_model/final_run_cross_scene_compact_pilot.py
- scripts/car_model/final_collect_cross_scene_compact_pilot.py
- docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md

Hard rules:
- Do not compare to 2000-iteration baseline if method is longer.
- Do not promote if only parking works.
- If clean long baseline is unavailable, mark scene as MISSING_BASELINE and do not claim superiority.

Append research-log entry.
Commit and push.

Gate:
PASS if at least two scenes show:
- >=50% triangle reduction,
- PSNR drop <= 0.2 dB or PSNR gain,
- SSIM drop <= 0.01,
- LPIPS increase <= 0.02 or LPIPS gain,
- no severe sparse geometry regression.

If only parking passes, decision must be SOFT_PASS_SINGLE_SCENE and paper claim must remain workshop/partial.
```

---

# Prompt F9 — Synthetic and injected-real giant-hole repair benchmark

```text
Stage F8 must be PASS or SOFT_PASS with explicit repair-branch continuation approved.

Mission:
Evaluate giant-hole repair honestly without relying on accidentally present real voids.

Create a benchmark with:
1. Synthetic ground plane with giant void.
2. Synthetic parking-like scene with ground/wall/object discontinuities.
3. Injected void into a real Mesh Splatting checkpoint:
   - remove a rectangular or mask-defined ground-like patch,
   - keep a known pre-damage checkpoint as oracle only for evaluation,
   - do not use oracle to choose repair at inference.

Create/update:
- ss3dm_prior/meshsplatopt/injected_damage.py
- scripts/car_model/meshsplatopt_create_injected_void_benchmark.py
- scripts/car_model/meshsplatopt_run_giant_void_repair_benchmark.py
- docs/car_model/final_stageF9_giant_void_repair_benchmark_report.md

Methods to compare:
- no repair
- classical plane fill without render gate
- CSEF ground-void fill without recovery
- CSEF ground-void fill with strict counterfactual gate
- CSEF ground-void fill with topology-frozen recovery
- prior-only fill diagnostic

Metrics:
- hole area repaired
- boundary loop closure
- render metrics
- sparse geometry proxy
- free-space violation
- changed pixel ratio
- hallucination flag
- oracle surface distance for injected benchmark only, labeled as oracle_eval

Hard rules:
- Unknown unobserved void must be rejected in normal mode.
- Prior-only fill must be labeled and excluded from headline.
- If fill loses to no-fill on render/geometry, do not promote giant-hole repair.

Append research-log entry.
Commit and push.

Gate:
PASS only if CSEF fill repairs synthetic and injected-real giant voids without render/geometry collapse and with lower hallucination risk than classical fill.
```

---

# Prompt F10 — Real defect mining and repair attempt on public scenes

```text
Stage F9 must be PASS.

Mission:
Search real public scenes for actual defects and attempt certified repairs only where evidence supports them.

Scenes:
- parking_phone_tiny
- bonsai
- courtyard
- optional flowers / other local COLMAP scene

Run:
1. Build CSEF.
2. Mine defects.
3. Rank defects by explanation debt and evidence strength.
4. Generate snap/fill/object/ground proposals.
5. Run render-backed counterfactual gate.
6. Run strict recovery only for accepted edits.
7. Compare against equal-budget no-edit controls.

Create:
- scripts/car_model/final_run_real_defect_repair_attempts.py
- scripts/car_model/final_collect_real_defect_repair_attempts.py
- docs/car_model/final_stageF10_real_defect_repair_report.md

Outputs:
- defect gallery
- accepted edit gallery
- rejected edit gallery
- before/after videos or panels
- gate JSONs
- W&B links
- independent metrics
- equal-budget control comparison

Hard decision:
- Promote repair branch only if at least one real non-delete edit improves a visible defect and does not lose equal-budget metrics.
- Otherwise keep repair branch as infrastructure/synthetic evidence and make compact-recovery the headline.

Append research-log entry.
Commit and push.

Gate:
PASS only if at least one real repair edit passes equal-budget control.
SOFT_PASS if only rejected/diagnostic repair examples are produced but evidence is well documented.
```

---

# Prompt F11 — Full ablation suite

```text
Stage F8 must be PASS and F9/F10 must have clear decisions.

Mission:
Run the ablations required for a NeurIPS submission.

Create:
- scripts/car_model/final_run_ablation_suite.py
- scripts/car_model/final_collect_ablation_suite.py
- docs/car_model/final_stageF11_ablation_suite_report.md

Ablations:

A. Compact-recovery ablations:
1. clean long baseline
2. compaction only, no recovery
3. recovery only, no compaction
4. compaction + recovery
5. compaction + recovery without strict topology freeze
6. compaction + recovery without sparse depth
7. compaction + recovery without sparse decay
8. compaction + recovery with random sparse sampling
9. compaction + recovery with mixed_low_error sampling
10. area selector vs CSEF selector
11. random same-count compaction control

B. Counterfactual certification ablations:
1. no render gate
2. no sparse geometry gate
3. no free-space gate
4. no changed-pixel gate
5. no rollback
6. prior-only fill allowed as diagnostic

C. Repair operation ablations:
1. delete/collapse only
2. snap only
3. fill only
4. compact + snap
5. compact + fill
6. compact + full portfolio

D. Baseline methods:
1. Stage35 PRISM
2. delete-only PRISM
3. posthoc decimation/QEM if available
4. classical hole fill
5. sparse-depth-only MeshSplatting recovery
6. clean-to-compact area prune baseline

Run ablations first on parking; then replicate the most important rows on at least one public scene.

Append research-log entry.
Commit and push.

Gate:
PASS only if the suite demonstrates which components are load-bearing and which are not. It is acceptable if snap/fill are not load-bearing, but the report must say so.
```

---

# Prompt F12 — Final multi-scene run package

```text
Stage F11 must be PASS.

Mission:
Produce the final multi-scene result package for paper tables.

Required scene groups:
- parking_phone_tiny
- bonsai
- courtyard
- one additional public COLMAP-compatible scene if available

For each scene, include:
1. clean long baseline
2. Stage35/PRISM baseline if available
3. best compact-recovery row
4. strongest posthoc simplification baseline
5. sparse-depth-only recovery baseline
6. repair branch row if F10 promoted one
7. failure row if relevant

Create:
- scripts/car_model/final_run_multiscene_package.py
- scripts/car_model/final_collect_multiscene_package.py
- docs/car_model/final_stageF12_multiscene_package_report.md
- outputs/carnet/meshsplatopt/final_multiscene_package/

Tables:
- Main quantitative table.
- Pareto table.
- Per-scene details.
- Ablation summary.
- Negative result table.

Figures:
- clean vs ours render montage
- error map montage
- topology / triangle count bar chart
- Pareto curve quality vs triangles
- repair examples accepted/rejected
- giant-hole synthetic/injected benchmark visualization if promoted

Append research-log entry.
Commit and push.

Gate:
PASS only if at least two scenes show meaningful compact-recovery benefit over fair clean long baselines or the report explicitly declares the project not NeurIPS-ready.
```

---

# Prompt F13 — Paper assets and figure builder

```text
Stage F12 must be PASS or SOFT_PASS with enough evidence.

Mission:
Create paper-ready assets.

Create/update:
- scripts/car_model/final_make_paper_assets.py
- docs/car_model/final_stageF13_paper_assets_report.md
- outputs/carnet/meshsplatopt/final_paper_assets/

Assets:
1. Method diagram:
   - CSEF
   - edit proposal
   - counterfactual gate
   - topology-frozen recovery
   - rollback
2. Quantitative tables:
   - main table
   - Pareto table
   - ablation table
   - failure/negative table
3. Qualitative:
   - render montage
   - error map montage
   - geometry proxy / sparse depth visualization
   - topology compaction visual
   - accepted/rejected repair examples
4. Reviewer-risk checklist.
5. Reproducibility appendix draft.

Rules:
- All figures must cite source model paths and metrics.
- Do not include training metrics in main table.
- Include failure cases and rejected directions.

Append research-log entry.
Commit and push.

Gate:
PASS only if figures/tables are reproducible by script and trace back to exact output paths.
```

---

# Prompt F14 — Manuscript skeleton and related-work integration

```text
Stage F13 must be PASS.

Mission:
Write a paper skeleton that tells the final story honestly.

Create/update:
- docs/car_model/final_meshsplatopt_neurips_manuscript_skeleton.md
- docs/car_model/final_meshsplatopt_related_work_notes.md
- docs/car_model/final_meshsplatopt_bib_plan.md

Sections:
1. Abstract
2. Introduction
3. Related work
4. Method:
   - Mesh Splatting background
   - CSEF
   - Reversible edit calculus
   - Evidence-compatible compaction
   - Counterfactual certification
   - Strict topology-frozen recovery
   - Optional repair branches
5. Experiments:
   - datasets
   - baselines
   - metrics
   - main results
   - Pareto curves
   - ablations
   - giant-hole benchmark if promoted
   - limitations
6. Conclusion

Required narrative:
- Start from the failure of delete-only PRISM and naive repair edits.
- Explain that overcomplete Mesh Splatting has quality but too much topology.
- MeshSplatOpt compact-repair finds a better quality/topology point.
- Counterfactual certificates keep repair safe.
- Negative results show which operations are not yet load-bearing.

Append research-log entry.
Commit and push.

Gate:
PASS only if the skeleton has a coherent NeurIPS story and does not claim unproven snap/fill wins.
```

---

# Prompt F15 — Final reviewer-risk audit and go/no-go decision

```text
Stage F14 must be PASS.

Mission:
Make the final NeurIPS go/no-go decision.

Write:
- docs/car_model/final_stageF15_neurips_go_no_go.md

Checklist:

1. Evidence strength:
   - How many scenes beat clean long baselines?
   - How many scenes reduce topology significantly?
   - Are independent metrics used?
   - Are baselines fair?

2. Novelty strength:
   - Is this more than posthoc pruning?
   - Is CSEF used in the selector or only in story?
   - Is counterfactual gate load-bearing?
   - Is topology-frozen recovery load-bearing?
   - Are repair operations load-bearing or auxiliary?

3. Reviewer risks:
   - "This is just area pruning + recovery."
   - "Sparse depth supervision is doing the work."
   - "Only one scene works."
   - "Triangle soup means mesh repair is not real mesh repair."
   - "Hole fill is synthetic only."
   - "Object prior is not used in headline."
   - "Baselines are weak."

4. Required mitigations:
   - Include area-only and CSEF selector ablation.
   - Include sparse-depth-only ablation.
   - Include no-freeze ablation.
   - Include clean long baseline.
   - Include cross-scene results.
   - Include honest limitations.

Decision categories:
- `NEURIPS_MAIN_READY`
- `NEURIPS_BORDERLINE_NEEDS_ONE_MORE_SCENE`
- `WORKSHOP_READY_MAIN_NOT_READY`
- `STOP_AND_REFRAME_AS_TECH_REPORT`

Append research-log entry.
Commit and push.

Gate:
PASS only if the decision is brutally honest and tied to exact tables.
```

---

# Prompt F16 — Final implementation cleanup and reproducibility release

```text
Stage F15 must be PASS and decision must not be STOP.

Mission:
Prepare codebase for release-quality reproduction.

Create/update:
- docs/car_model/final_reproducibility_appendix.md
- docs/car_model/final_artifact_checklist.md
- README.md / README.zh.md if needed
- scripts/car_model/final_reproduce_main_table.sh
- scripts/car_model/final_reproduce_pareto_table.sh

Checklist:
1. One-command reproduction for main table where possible.
2. Clear dataset paths and how to override them.
3. All configs saved.
4. W&B optional/offline fallback documented.
5. Random seeds documented.
6. GPU memory expectations documented.
7. Exact commands for clean baseline, compaction, recovery, evaluation.
8. All reported metrics traceable to JSON.
9. No private paths in final public README except example placeholders.
10. Negative result docs preserved but organized.

Append research-log entry.
Commit and push.

Gate:
PASS only if an external user could reproduce the main table with documented data and commands.
```

---

## Final instruction to Codex

The final aim is not to make another small PRISM stage. The aim is to produce a paper-defensible method:

```text
high-quality clean Mesh Splatting
→ CSEF/evidence-compatible compaction and certified repairs
→ strict topology freeze
→ sparse-geometry-guided recovery
→ fair clean-long baseline comparison
→ cross-scene Pareto evidence
```

If CSEF selector does not beat area-only, say so and keep area-only as the baseline.  
If snap/fill does not beat equal-budget controls, say so and keep them as auxiliary repair diagnostics.  
If only parking works, say so and do not claim NeurIPS main readiness.  
If two or more scenes show compact-recovery dominance over fair clean long baselines, build the NeurIPS paper around compact-repair Pareto optimization, not around local edit anecdotes.
