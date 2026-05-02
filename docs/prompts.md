# prompts.md — SPCarNet → Object-Prior Guided Scene Mesh Optimization

## Current Execution Note — 2026-05-01

Status before continuing to Prompt M13:

- DONE: M0-M12 are implemented, smoke-tested, documented, committed, and pushed to `spcarnet/main`.
- DONE: A 200-iteration wandb training smoke ran on GPU 1 and synced successfully.
- BLOCKER FOUND: the training final cleanup path pruned a non-PRISM 200-iteration run from `5706` triangles to `15` triangles.
- DONE BEFORE M13: final cleanup was repaired so ordinary non-PRISM training does not run destructive cleanup by default. A second 200-iteration wandb smoke preserved `5706` triangles and passed COLMAP sparse geometry evaluation.
- DONE: M13 evaluation protocol, experiment matrix, dry-run matrix runner, NeurIPS-style report generator, smoke test, and generated report are implemented. Full dry-run matrix status: `11` total, `7` available, `4` `MISSING`.
- DONE: M13 implementation report, smoke report, research log entry, commit, and push are complete.
- DONE: Pre-M14 stability audit passed after fixing smoke subprocess interpreter drift. Remaining risks are claim/research risks, not immediate code-collapse risks.
- DONE: M14 paper roadmap and claim-risk analysis are complete. Final recommendation: `MORE_SCENE_EVIDENCE_REQUIRED`.
- DONE: M15 retrieval-deformation fallback is implemented and measured. Recommendation: `KEEP_AS_BASELINE`, not a pivot.
- DONE: Pre-M16 scene application bridge is implemented. Accepted dry-run proposals can now be applied to a mesh copy with rollback and recovery command planning.
- DONE: Parking phone tiny scene was audited and a 200-iteration wandb baseline was run. Dataset view is valid and geometry eval is available; this is a short baseline, not a final headline run.

This document is meant to be copied, stage by stage, into Codex / Claude Code or another coding agent working inside `Dystopians/SPCarNet`.

The goal is not merely to improve an object-level car completion benchmark. The goal is to transform the current SP-CarNet work into a top-conference-grade research system for the downstream task the project actually cares about:

> **Use learned object-level shape priors to safely optimize real scene meshes, especially parking-lot scenes: repair holes, remove floaters, protect valid object geometry, improve triangle distribution, and improve geometric accuracy without degrading rendering or speed.**

The guiding slogan for the final method is:

> **Prior proposes; evidence disposes.**

SP-CarNet should propose object-aware mesh repairs. The scene-level evidence and optimizer should accept, reject, or roll back those proposals.

---

## 0. Non-negotiable operating rules for the coding agent

Apply these rules to every prompt below.

### 0.1 Work one stage at a time

Do not jump ahead. A stage is complete only when all of the following are true:

1. The stage design document exists.
2. The relevant code exists and imports cleanly.
3. The smoke test passes.
4. The stage implementation report exists.
5. Metrics and logs are written to the expected output directory.
6. The research log has a dated entry.
7. The stage gate is explicitly marked `PASS`, `SOFT PASS`, `FAIL`, or `STOP`.

If a stage fails a hard gate, write a failure analysis and stop. Do not silently proceed.

### 0.2 Always verify code before editing

Before every major change, run a quick repository integrity check:

```bash
git status --short
python --version
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count', torch.cuda.device_count())
PY
python -m compileall scripts/car_model ss3dm_prior -q
```

If compile/import fails, do not start the research task. Fix the integrity issue first, document it, and then resume.

### 0.3 Use GPUs responsibly but do not avoid training when needed

If a prompt asks for a headline run or meaningful validation, the agent may use an idle GPU.

Before launching training:

```bash
nvidia-smi
```

Pick an idle or lightly used GPU, then set:

```bash
export CUDA_VISIBLE_DEVICES=<gpu_id>
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
```

If wandb is not authenticated or unavailable, set `WANDB_MODE=offline`, but document this in the report. Always save logs under `outputs/carnet/.../logs/` and write the exact command used.

### 0.4 Every large code change must leave readable docs

For every stage, create or update these files:

```text
docs/car_model/meshprior_stageX_<topic>_design.md
docs/car_model/meshprior_stageX_<topic>_implementation_report.md
docs/car_model/meshprior_stageX_<topic>_smoke.md
```

If the stage fails or a key hypothesis is falsified, also create:

```text
docs/car_model/meshprior_stageX_<topic>_failure.md
```

Always append a dated entry to:

```text
docs/car_model/SPCarNet_research_log.md
```

### 0.5 Preserve old baselines

Do not delete or rewrite the v0.7 residual line, v0.8.x point-flow line, Stage 2 auto-decoder, Stage 3 posterior encoder, Stage 4 MAP, or Stage 5 multihypothesis code. They are important baselines and negative results.

New mesh-prior work should live under clearly named new files, for example:

```text
ss3dm_prior/meshprior/
scripts/car_model/meshprior_*.py
configs/ss3dm_prior/meshprior/
docs/car_model/meshprior_*.md
outputs/carnet/meshprior/
```

### 0.6 Separate inference-time metrics from oracle metrics

Never use clean ground truth to choose a proposal at inference time. It may only be used for evaluation.

All reports must separate:

```text
inference_time_metrics
oracle_analysis_metrics
gt_dependent_eval_metrics
```

### 0.7 Default kill condition

If a stage cannot run because required code/data is missing, stop and write a failure report. Do not hallucinate results. Do not claim training succeeded unless logs, checkpoints, and eval JSONs exist.

---

## 1. Current repository state that the agent must verify

Before proposing new work, the agent must read the latest docs and verify code. Current known state from repository documents:

1. Stage 1 object cache / canonicalization audit is marked `DONE`.
   - 2,433 objects total.
   - 1,854 train / 206 val / 373 test.
   - Fields such as `clean_points`, `visible_clean_points`, `hidden_clean_points`, `observed_points`, `query_points_all`, `surface_query_points`, and `free_query_points` are present.
   - Scanner pose is not persisted.

2. Stage 2 shape-field auto-decoder is implemented and smokes pass, but the gate is not cleanly passed.
   - v1 train chamfer around `0.066`, target gate was `≤ 0.05`.
   - v3 bigger decoder did not lift the ceiling.
   - IoU metric has known caveats around sparse point fallback / GLB availability.

3. Stage 3 posterior encoder is the strongest current object-level result.
   - val `recon_chamfer_l1_mean ≈ 0.0664`.
   - `free_space_violation_rate_mean ≈ 0.0335`.
   - zero-corruption chamfer roughly equals normal chamfer, so the v0.7 smoothing-collapse pathology is avoided.
   - Bottleneck is now likely Stage 2 decoder / field representation, not posterior encoder.

4. Stage 4 MAP refinement is a soft pass.
   - It improves free-space violation strongly.
   - It improves chamfer only slightly.
   - It is useful as a safety / proposal validation tool, not as the main headline improvement.

5. Stage 5 multihypothesis sampling shows oracle promise but practical reranking failure.
   - Oracle K=8 improves chamfer by about 0.006.
   - Inference-only reranker chooses worse candidates.
   - Do not use multihypothesis reranking as a headline until the evidence score is fixed.
   - Posterior spread can still be useful as uncertainty for mesh proposals.

6. Important code consistency issue to verify immediately:
   - Several scripts import `ss3dm_prior.models.spcarnet_shape_field` and `ss3dm_prior.models.spcarnet_posterior`.
   - Verify these files exist in the local checkout. If they are missing or untracked, restore/commit them or implement them from the contracts used by the trainer/eval scripts before continuing.

---

## 2. Final target method: SP-CarNet MeshPrior

The final method should be framed as:

> **SP-CarNet MeshPrior: learned object-centric shape posteriors as safe proposal generators for scene mesh optimization.**

It should not be framed as:

> “A better car point-cloud completion model.”

The final system should have these layers:

```text
Layer A — Repository and object-prior integrity
Layer B — Scene/object region mining
Layer C — Object posterior inference in canonical frame
Layer D — Mesh repair proposal generation
Layer E — Scene evidence gates and rollback
Layer F — Alternating scene optimization
Layer G — NeurIPS-grade evaluation and reporting
```

The key proposal types are:

```text
protect: keep valid car-like triangles during pruning / compaction
prune: remove floaters or unsupported triangles inconsistent with the object prior
snap: softly move noisy vertices toward a high-confidence object surface
fill: patch holes only when prior confidence and scene evidence agree
split: allocate more triangles to high-curvature or boundary regions
collapse: reduce redundant triangles on smooth, well-supported surfaces
```

The safe order is:

```text
protect/prune first → snap second → guarded fill third → split/collapse refinement last
```

---

# Prompt M0 — Repository integrity, code audit, and current-state verification

Copy this prompt to the coding agent first.

```text
You are working inside the Dystopians/SPCarNet repository.

Mission of this task:
Verify the actual repository state before any new research changes. The docs indicate SP-CarNet Stage 1–5 work exists, but the local checkout must be validated at the code level. Do not implement the new method yet.

Read these files first, newest/most important first:
- docs/car_model/SPCarNet_research_log.md
- docs/car_model/SPCarNet_radical_RFC.md
- docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md, if present
- docs/car_model/spcarnet_stage4_observation_map_implementation_report.md, if present
- docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md, if present
- docs/car_model/spcarnet_stage2_shape_field_implementation_report.md, if present
- README.md

Then verify these code files exist and import:
- ss3dm_prior/data/spcarnet_object_dataset.py
- ss3dm_prior/models/spcarnet_shape_field.py
- ss3dm_prior/models/spcarnet_posterior.py
- ss3dm_prior/training/spcarnet_autodecoder.py
- ss3dm_prior/training/spcarnet_posterior.py
- ss3dm_prior/losses_spcarnet_observation.py
- scripts/car_model/build_spcarnet_object_index.py
- scripts/car_model/eval_spcarnet_posterior_encoder.py
- scripts/car_model/refine_spcarnet_latent_map.py
- scripts/car_model/eval_spcarnet_multihypothesis.py

Run:

```bash
git status --short
python -m compileall scripts/car_model ss3dm_prior -q
python scripts/car_model/smoke_test_spcarnet_stage1.py || true
python scripts/car_model/smoke_test_spcarnet_stage2.py || true
python scripts/car_model/smoke_test_spcarnet_stage3.py || true
python scripts/car_model/smoke_test_spcarnet_stage4.py || true
python scripts/car_model/smoke_test_spcarnet_stage5.py || true
```

If any smoke test requires unavailable data, record that explicitly instead of treating it as a model failure.

Required output document:
- docs/car_model/meshprior_stage0_repository_audit.md

The audit must contain:
1. Current git branch and dirty files.
2. Python / torch / CUDA availability.
3. Which expected code files are present or missing.
4. Which smoke tests ran, which passed, which failed, and why.
5. Whether the Stage 2/3/4/5 reported metrics are supported by actual output JSONs and checkpoints.
6. A table of blocking issues.
7. A recommendation: `PROCEED`, `FIX_INTEGRITY_FIRST`, or `STOP`.

If `ss3dm_prior/models/spcarnet_shape_field.py` or `ss3dm_prior/models/spcarnet_posterior.py` is missing but imported by existing scripts, this is a blocking integrity issue. Fix it before any new research stage. The fix may either restore the missing file from local/untracked history if available, or implement a minimal compatible version satisfying all imports and existing smoke tests. Document exactly what was done.

Append a dated entry to:
- docs/car_model/SPCarNet_research_log.md

Do not proceed to Prompt M1 until this stage is `PROCEED`.
```

---

# Prompt M1 — Write the mesh-optimization research RFC

```text
You are working inside Dystopians/SPCarNet. Stage M0 must already be marked PROCEED.

Mission:
Write the new research RFC that pivots SP-CarNet from object-only shape completion to object-prior-guided scene mesh optimization.

Do not change model code in this task.

Read:
- docs/car_model/meshprior_stage0_repository_audit.md
- docs/car_model/SPCarNet_research_log.md
- docs/car_model/SPCarNet_radical_RFC.md
- README.md, especially sections on training, custom split, COLMAP geometry evaluation, rendering, PLY creation, and object extraction / segmentation.
- Any existing PRISM / geometry / parking-related scripts if present.

Write:
- docs/car_model/meshprior_stage1_scene_meshprior_RFC.md

The RFC must contain:

1. Current SP-CarNet object-level status
   - Stage 3 posterior encoder is currently strongest.
   - Stage 2 decoder ceiling limits further object-level gains.
   - Stage 4 MAP is useful for free-space safety.
   - Stage 5 multihypothesis oracle shows uncertainty, but inference reranking is not a headline.

2. New central claim
   SP-CarNet should become a learned object prior for scene mesh optimization, not a final point-cloud completion output.

3. Method slogan
   `Prior proposes; evidence disposes.`

4. Proposed final method
   - region mining,
   - object posterior inference,
   - mesh repair proposal generation,
   - scene evidence gates,
   - rollback,
   - alternating optimization.

5. Proposal types
   - protect,
   - prune,
   - snap,
   - fill,
   - split,
   - collapse.

6. Why this serves the downstream task
   Explain how it helps parking-lot mesh distribution, hole repair, floater removal, better geometry, and speed/triangle-budget tradeoffs.

7. Relation to current code
   List which current SP-CarNet modules are reused, which are demoted to baselines, and which new modules are needed.

8. Research risks
   - object prior hallucination,
   - bad object-region mining,
   - canonical-frame mismatch,
   - weak shape-field surface accuracy,
   - proposal acceptance too conservative,
   - mesh-splatting render quality degradation,
   - scene gate cost too high.

9. Kill criteria
   Define hard gates for each future stage.

10. Documentation policy
   Re-state that each stage must have design, implementation, smoke, failure if needed, and research-log entries.

Append a dated entry to docs/car_model/SPCarNet_research_log.md.

Do not proceed to Prompt M2 until this RFC is complete and internally consistent.
```

---

# Prompt M2 — Build a scene/object region mining layer

```text
You are working inside Dystopians/SPCarNet. Stage M1 must be complete.

Mission:
Implement the first bridge from object-level SP-CarNet to scene-level mesh optimization: a scene/object region mining layer that identifies car-like and repair-worthy mesh regions.

Before code:
Write:
- docs/car_model/meshprior_stage2_region_mining_design.md

The design doc must answer:
1. What scene representations are available in this repository?
   Check for trained mesh outputs, PLY creation, segmentation outputs, object extraction scripts, COLMAP sparse points, render residuals, and geometry eval scripts.
2. What is the minimal input format for region mining?
   Define a JSON/NPZ contract that can work even before full PRISM integration.
3. How do we avoid applying car priors to ground / wall / vegetation?
4. What confidence score is required before a region is passed to SP-CarNet?
5. What fallback behavior exists if segmentation masks are unavailable?

Implementation requirements:

1. Create a package:
   - ss3dm_prior/meshprior/
   - ss3dm_prior/meshprior/__init__.py

2. Add a region data contract:
   - ss3dm_prior/meshprior/region_types.py

   Include dataclasses or typed dicts for:
   - SceneMeshRegion
   - RegionEvidence
   - ObjectCanonicalization
   - RegionMiningResult

3. Add a region mining script:
   - scripts/car_model/meshprior_mine_regions.py

   It should support:
   ```bash
   python scripts/car_model/meshprior_mine_regions.py \
     --scene_model <path_to_model_or_scene_output> \
     --scene_source <path_to_colmap_scene> \
     --output_dir outputs/carnet/meshprior/region_mining/<run_name> \
     --mode dry_run
   ```

4. Minimal behavior:
   - If object segmentation artifacts are present, load them.
   - If no segmentation artifacts are present, produce a documented dry-run output and do not crash.
   - If PLY mesh is present, load vertices/faces using trimesh or a robust fallback.
   - Compute basic region diagnostics when possible:
     - triangle count,
     - bounding box,
     - surface area,
     - vertex density,
     - boundary edge count,
     - connected components,
     - approximate hole-boundary score,
     - optional car-likeness heuristic based on bbox aspect ratio / height if enough geometry exists.

5. Output artifacts:
   - regions.json
   - regions_summary.csv
   - region_mining_report.md

6. Add a smoke test:
   - scripts/car_model/smoke_test_meshprior_stage2_region_mining.py

   The smoke test should create a tiny synthetic mesh with two components, run the miner, and verify a nonempty region output.

After code:
Write:
- docs/car_model/meshprior_stage2_region_mining_implementation_report.md
- docs/car_model/meshprior_stage2_region_mining_smoke.md

Stage gate:
- Smoke test passes.
- The script can run in dry-run mode without data and exits cleanly.
- The script can process at least one synthetic or real mesh if available.

Append a dated entry to docs/car_model/SPCarNet_research_log.md.

Do not proceed to M3 until this stage passes.
```

---

# Prompt M3 — Build object posterior inference for scene regions

```text
You are working inside Dystopians/SPCarNet. Stage M2 must be complete.

Mission:
Implement a wrapper that takes a mined scene region, canonicalizes it into SP-CarNet's object frame, runs the Stage 3 posterior encoder, and returns a shape-field posterior plus diagnostics.

Before code:
Write:
- docs/car_model/meshprior_stage3_scene_region_posterior_design.md

The design doc must specify:
1. How a scene-region point cloud is sampled from mesh vertices/faces.
2. How the region is canonicalized.
3. How existing Stage 3 posterior checkpoints are loaded.
4. What happens if the region orientation/front-axis is unknown.
5. How to compute posterior uncertainty without relying on oracle GT.
6. How to expose the shape field for later mesh proposals.

Implementation requirements:

1. Add:
   - ss3dm_prior/meshprior/scene_region_posterior.py

   Implement:
   - sample_region_points(mesh, region, n_points)
   - estimate_canonical_transform(region_points, mode='bbox_pca')
   - canonicalize_region_points(points, transform)
   - run_spcarnet_posterior(checkpoint, points, device)
   - decode_region_field(decoder, z, query_points)
   - estimate_posterior_uncertainty(z_mean, z_logvar, K=8) without using GT

2. Add CLI:
   - scripts/car_model/meshprior_infer_region_posterior.py

   Required CLI:
   ```bash
   python scripts/car_model/meshprior_infer_region_posterior.py \
     --regions_json outputs/carnet/meshprior/region_mining/<run_name>/regions.json \
     --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
     --output_dir outputs/carnet/meshprior/region_posterior/<run_name> \
     --device cuda \
     --limit 16
   ```

3. Output per region:
   - z_mean.npy
   - z_logvar.npy, if variational
   - canonical_transform.json
   - posterior_summary.json
   - sampled_region_points.npy
   - optional occupancy_grid_32.npy

4. Diagnostics:
   - posterior_mu_norm,
   - posterior_logvar_mean,
   - uncertainty_score,
   - field_occupancy_ratio on a coarse grid,
   - extraction_success at 32^3, if Marching-Cubes is available.

5. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py

   The smoke test may use a synthetic car-like ellipsoid / box region if no real checkpoint is present. If no checkpoint is available, it must verify the wrapper fails gracefully with a clear message, not crash obscurely.

After code:
Write:
- docs/car_model/meshprior_stage3_scene_region_posterior_implementation_report.md
- docs/car_model/meshprior_stage3_scene_region_posterior_smoke.md

Stage gate:
- Imports cleanly.
- Smoke test passes.
- If a Stage 3 checkpoint is present, at least one region posterior inference runs and produces diagnostics.
- If no checkpoint is present, the report explicitly says what path is missing and how to train/evaluate it.

Append a dated entry to docs/car_model/SPCarNet_research_log.md.

Do not proceed to M4 until this passes.
```

---

# Prompt M4 — Implement safe protect/prune proposal generation

```text
You are working inside Dystopians/SPCarNet. Stage M3 must be complete.

Mission:
Implement the first safe mesh-prior proposal types: protect and prune. Do not move vertices and do not fill holes yet.

Scientific reason:
Protect/prune is the lowest-risk way to inject object priors into scene mesh optimization. It can improve triangle selection and prevent valid car geometry from being removed, while avoiding hallucinated new surfaces.

Before code:
Write:
- docs/car_model/meshprior_stage4_protect_prune_design.md

The design doc must include:
1. Proposal definitions.
2. Triangle-level scoring formulas.
3. How shape-field support is computed.
4. How free-space violation is computed if free-space queries exist.
5. How uncertainty lowers confidence.
6. How to export scores for PRISM or another mesh optimizer without hard dependency.

Suggested scoring:

```text
surface_support = high if triangle vertices / samples lie near likely object surface
prior_violation = high if triangle samples lie far from likely object support or inside known free-space
uncertainty_penalty = high if posterior uncertainty is high
protect_score = surface_support * observed_support * (1 - uncertainty_penalty)
prune_score = prior_violation + free_space_violation + low_observed_support - protect_score
```

Implementation requirements:

1. Add:
   - ss3dm_prior/meshprior/proposals.py

   Include dataclasses:
   - MeshPriorProposal
   - TriangleScoreTable
   - ProposalBatch

2. Implement protect/prune scorer:
   - ss3dm_prior/meshprior/protect_prune.py

   Functions:
   - sample_triangle_points(vertices, faces, samples_per_face)
   - compute_shape_field_support(decoder, z, samples)
   - compute_triangle_scores(...)
   - build_protect_prune_proposals(...)

3. Add CLI:
   - scripts/car_model/meshprior_make_protect_prune_proposals.py

   Required CLI:
   ```bash
   python scripts/car_model/meshprior_make_protect_prune_proposals.py \
     --regions_json <regions.json> \
     --posterior_dir <region_posterior_dir> \
     --posterior_checkpoint <stage3_checkpoint> \
     --output_dir outputs/carnet/meshprior/proposals_protect_prune/<run_name> \
     --limit 16
   ```

4. Output:
   - triangle_scores.npz
   - proposals.json
   - summary.csv
   - visual_debug.ply if feasible, with score stored as vertex/face attributes or separate CSV.

5. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage4_protect_prune.py

   Synthetic test:
   - create a small cube mesh;
   - define a fake sphere/box occupancy function;
   - verify triangles near support get higher protect score;
   - verify outlier/floater triangles get higher prune score.

After code:
Write:
- docs/car_model/meshprior_stage4_protect_prune_implementation_report.md
- docs/car_model/meshprior_stage4_protect_prune_smoke.md

Stage gate:
- Smoke test passes.
- For at least one synthetic mesh, scores are qualitatively correct.
- No vertex movement or hole filling is performed.
- Output score contract is usable by a downstream optimizer.

Append research-log entry.

Do not proceed to M5 until this passes.
```

---

# Prompt M5 — Add an adapter for PRISM / mesh optimizer score consumption

```text
You are working inside Dystopians/SPCarNet. Stage M4 must be complete.

Mission:
Add an adapter that allows protect/prune proposal scores to be consumed by the existing scene mesh optimizer, PRISM if present, or a standalone placeholder optimizer if PRISM is not present in this repository.

Before code:
Write:
- docs/car_model/meshprior_stage5_optimizer_adapter_design.md

The design doc must answer:
1. Is PRISM code present in this repository? Search for PRISM, prune, geogate, triangle utility, topology mutation, rollback, mesh optimizer.
2. If present, what exact hook should receive protect/prune scores?
3. If not present, what neutral artifact contract can be exported for the mesh- repository or future PRISM code?
4. How do we prevent object-prior scores from overriding scene geometry evidence?

Implementation requirements:

1. Add:
   - ss3dm_prior/meshprior/optimizer_adapter.py

   It should implement:
   - load_triangle_scores(path)
   - normalize_scores_per_region(...)
   - export_prism_score_json(...)
   - export_generic_meshprior_score_npz(...)
   - combine_scores(existing_score, meshprior_score, mode='bounded_add')

2. Add CLI:
   - scripts/car_model/meshprior_export_optimizer_scores.py

   Required CLI:
   ```bash
   python scripts/car_model/meshprior_export_optimizer_scores.py \
     --triangle_scores <triangle_scores.npz> \
     --output_dir outputs/carnet/meshprior/optimizer_scores/<run_name> \
     --format generic_npz
   ```

3. Bounded combination rule:
   MeshPrior may nudge but not dominate optimizer scores by default.

   Example:
   ```text
   keep_score_final = keep_score_base + alpha * clamp(meshprior_protect, 0, 1)
   prune_score_final = prune_score_base + beta * clamp(meshprior_prune, 0, 1)
   alpha, beta default <= 0.25 during early experiments
   ```

4. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py

   It should verify:
   - score normalization finite,
   - no NaNs,
   - bounded_add cannot change a score by more than configured alpha/beta,
   - exported JSON/NPZ can be loaded again.

After code:
Write:
- docs/car_model/meshprior_stage5_optimizer_adapter_implementation_report.md
- docs/car_model/meshprior_stage5_optimizer_adapter_smoke.md

Stage gate:
- Smoke passes.
- Adapter correctly identifies whether PRISM exists in this repository.
- Generic export works even if PRISM is absent.

Append research-log entry.

Do not proceed to M6 until this passes.
```

---

# Prompt M6 — Build a synthetic mesh-damage benchmark for object-prior repair

```text
You are working inside Dystopians/SPCarNet. Stage M5 must be complete.

Mission:
Before touching real parking-lot scenes, build a controlled synthetic object-level mesh-damage benchmark. This is needed to prove that the object prior produces meaningful mesh repair/protect/prune signals under known damage.

Before code:
Write:
- docs/car_model/meshprior_stage6_synthetic_damage_benchmark_design.md

The design doc must define:
1. Damage types.
2. Metrics.
3. Baselines.
4. Expected success thresholds.
5. How to prevent using oracle labels at inference time.

Damage types:
- local hole,
- side-panel removal,
- roof/cabin removal,
- wheel/low-structure removal if identifiable,
- floater triangles,
- noisy vertex displacement,
- density imbalance,
- oversimplified smooth patch.

Baselines:
- damaged input,
- v0.7 residual if available,
- v0.8.2 point-flow if available,
- Stage 3 posterior mesh sample,
- protect/prune proposals only,
- optional retrieval-only baseline if available.

Implementation requirements:

1. Add:
   - ss3dm_prior/meshprior/synthetic_damage.py

   Functions:
   - damage_mesh_local_hole(...)
   - add_floater_triangles(...)
   - perturb_vertices(...)
   - make_density_imbalance(...)
   - compute_hole_boundary_metrics(...)

2. Add benchmark runner:
   - scripts/car_model/meshprior_run_synthetic_damage_benchmark.py

   CLI:
   ```bash
   python scripts/car_model/meshprior_run_synthetic_damage_benchmark.py \
     --object_index outputs/carnet/spcarnet/object_index_v1.json \
     --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
     --output_dir outputs/carnet/meshprior/synthetic_damage/<run_name> \
     --num_objects 64 \
     --damage_types local_hole floater vertex_noise density_imbalance
   ```

3. Metrics:
   - recon_chamfer_l1,
   - hidden_chamfer_l1,
   - visible_preservation_error,
   - free_space_violation_rate,
   - hole_boundary_distance,
   - floater_precision_recall for prune proposals,
   - valid_surface_protect_recall,
   - triangle_count_delta,
   - mesh_extraction_success.

4. Report generator:
   - scripts/car_model/meshprior_make_synthetic_damage_report.py

   Output:
   - metrics.json,
   - metrics.csv,
   - table_by_damage_type.csv,
   - failure_cases.md,
   - docs/car_model/reports/meshprior_synthetic_damage_report.md

5. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py

After code:
Run smoke. If GPU/checkpoint/data are available, run a small benchmark with 8 objects. If an idle GPU is available and Stage 3 checkpoint exists, run 64 objects and log to wandb.

Write:
- docs/car_model/meshprior_stage6_synthetic_damage_benchmark_implementation_report.md
- docs/car_model/meshprior_stage6_synthetic_damage_benchmark_smoke.md

Stage gate:
- Synthetic damage generation works.
- Protect/prune scores identify floaters and preserve valid object-surface triangles on synthetic data.
- Report clearly separates inference-time scores from oracle labels.

Append research-log entry.

Do not proceed to M7 until this passes.
```

---

# Prompt M7 — Implement conservative snap proposals

```text
You are working inside Dystopians/SPCarNet. Stage M6 must be complete.

Mission:
Implement conservative vertex snap proposals. This is the first stage that moves geometry, so it must be risk-gated.

Before code:
Write:
- docs/car_model/meshprior_stage7_conservative_snap_design.md

The design doc must specify:
1. Which vertices are eligible for snap.
2. How snap direction is computed from occupancy/SDF field.
3. Maximum movement per iteration.
4. Boundary and observed-surface preservation rules.
5. How snap is evaluated and rolled back.

Rules:
- Do not snap boundary-loop vertices by default.
- Do not snap high-observed-support vertices unless movement is tiny.
- Do not snap when posterior uncertainty is high.
- Do not snap if free-space violation would increase.
- Default max displacement should be conservative, e.g. `0.005–0.02` canonical units.

Implementation requirements:

1. Add:
   - ss3dm_prior/meshprior/snap.py

   Functions:
   - compute_field_gradient(decoder, z, points)
   - propose_vertex_snap(vertices, decoder, z, confidence, max_disp)
   - apply_snap_proposal(mesh, proposal)
   - evaluate_snap_risk(...)

2. Add CLI:
   - scripts/car_model/meshprior_make_snap_proposals.py

3. Add synthetic benchmark integration:
   - allow Stage M6 runner to evaluate `protect_prune_only` vs `protect_prune_snap`.

4. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage7_snap.py

   Synthetic test:
   - define a sphere/box field,
   - perturb mesh vertices slightly,
   - snap should reduce distance-to-surface,
   - boundary vertices should remain fixed when configured.

After code:
Run smoke and a small synthetic benchmark.

Write:
- docs/car_model/meshprior_stage7_conservative_snap_implementation_report.md
- docs/car_model/meshprior_stage7_conservative_snap_smoke.md

Stage gate:
- Snap improves synthetic surface distance without increasing free-space violation.
- Snap does not move protected/boundary vertices beyond max_disp.
- If snap harms visible preservation by more than 5 percent in the small benchmark, write failure analysis and stop snap line.

Append research-log entry.

Do not proceed to M8 until this passes.
```

---

# Prompt M8 — Implement guarded fill proposals for holes

```text
You are working inside Dystopians/SPCarNet. Stage M7 must be complete.

Mission:
Implement guarded hole-fill proposals using the object shape field, but only under strict evidence gates.

Before code:
Write:
- docs/car_model/meshprior_stage8_guarded_fill_design.md

The design doc must specify:
1. Hole-boundary detection.
2. How the shape-field surface is extracted locally near a hole.
3. How the patch is clipped to the hole support.
4. How to avoid filling visible free-space.
5. How to handle uncertainty and multi-solution ambiguity.
6. Rollback conditions.

Implementation requirements:

1. Add:
   - ss3dm_prior/meshprior/fill.py

   Functions:
   - find_boundary_loops(mesh)
   - score_hole_candidates(mesh, boundary_loops, region_evidence)
   - extract_local_field_patch(decoder, z, local_bbox, resolution)
   - clip_patch_to_hole_boundary(...)
   - build_fill_proposal(...)
   - evaluate_fill_risk(...)

2. Add CLI:
   - scripts/car_model/meshprior_make_fill_proposals.py

3. Add synthetic damage integration:
   - local_hole benchmark should compare damaged input vs guarded fill vs snap+fill.

4. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage8_fill.py

   Synthetic test:
   - create a simple mesh with a known hole;
   - use a simple analytic field;
   - generated patch should close the hole and not add disconnected far-away geometry.

After code:
Run smoke and a small synthetic hole benchmark.

Write:
- docs/car_model/meshprior_stage8_guarded_fill_implementation_report.md
- docs/car_model/meshprior_stage8_guarded_fill_smoke.md

Stage gate:
- Guarded fill closes synthetic holes in at least one controlled case.
- Fill does not create disconnected floaters.
- Fill does not increase free-space violation in benchmark cases.
- If fill fails, do not proceed to scene-level fill; keep protect/prune/snap as the method and write failure analysis.

Append research-log entry.

Do not proceed to M9 until this passes or until the fill line is explicitly killed.
```

---

# Prompt M9 — Build scene evidence gates and rollback

```text
You are working inside Dystopians/SPCarNet. Stage M8 must be complete or fill must be explicitly killed with a failure report.

Mission:
Implement scene evidence gates that decide whether a mesh-prior proposal can be accepted in a real scene.

Before code:
Write:
- docs/car_model/meshprior_stage9_scene_gate_rollback_design.md

The design doc must define:
1. Gate inputs.
2. Gate metrics.
3. Acceptance thresholds.
4. Rollback data structure.
5. How to run gate in dry-run mode without full differentiable training.
6. How to separate object-level evidence from scene-level evidence.

Gate metrics:
- rendering metric delta if render.py / eval outputs are available,
- COLMAP sparse depth AbsRel / MAE delta if evaluate_geometry_colmap.py is available,
- sparse normal proxy delta,
- free-space violation delta,
- triangle count delta,
- connected component / floater count delta,
- hole-boundary score delta,
- controlled FPS proxy if available.

Implementation requirements:

1. Add:
   - ss3dm_prior/meshprior/scene_gate.py

   Implement:
   - ProposalGateResult
   - evaluate_proposal_geometry_delta(...)
   - evaluate_proposal_free_space_delta(...)
   - evaluate_proposal_topology_delta(...)
   - accept_or_reject(...)
   - save_rollback_snapshot(...)
   - restore_rollback_snapshot(...)

2. Add CLI:
   - scripts/car_model/meshprior_evaluate_proposals.py

   Required CLI:
   ```bash
   python scripts/car_model/meshprior_evaluate_proposals.py \
     --scene_source <colmap_scene> \
     --scene_model <trained_scene_model> \
     --proposals <proposals.json> \
     --output_dir outputs/carnet/meshprior/scene_gate/<run_name> \
     --mode dry_run
   ```

3. If full scene rendering is too expensive or unavailable, implement dry-run gates first:
   - topology gate,
   - free-space gate,
   - triangle-count gate,
   - local geometry proxy gate.

4. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py

   Synthetic test:
   - create a mesh and two proposals;
   - one obviously improves topology;
   - one adds a disconnected floater;
   - gate should accept the first and reject the second.

After code:
Write:
- docs/car_model/meshprior_stage9_scene_gate_rollback_implementation_report.md
- docs/car_model/meshprior_stage9_scene_gate_rollback_smoke.md

Stage gate:
- Dry-run gate works.
- Rollback snapshot and restore works.
- Gate report clearly explains accepted/rejected proposals.
- No proposal may be accepted solely because SP-CarNet likes it; scene evidence must be included.

Append research-log entry.

Do not proceed to M10 until this passes.
```

---

# Prompt M10 — Build the alternating mesh-prior optimization runner

```text
You are working inside Dystopians/SPCarNet. Stage M9 must be complete.

Mission:
Implement an orchestration runner that combines region mining, posterior inference, proposal generation, gate evaluation, acceptance/rollback, and optional recovery.

Before code:
Write:
- docs/car_model/meshprior_stage10_alternating_runner_design.md

The design doc must specify:
1. The full pipeline sequence.
2. Which stages are optional.
3. What artifacts are passed from one stage to the next.
4. How recovery is handled if train.py / scene optimizer integration exists.
5. How to resume from intermediate artifacts.
6. How to stop safely if a substage fails.

Implementation requirements:

1. Add runner:
   - scripts/car_model/meshprior_run_pipeline.py

   CLI:
   ```bash
   python scripts/car_model/meshprior_run_pipeline.py \
     --scene_source <colmap_scene> \
     --scene_model <trained_scene_model> \
     --posterior_checkpoint <stage3_checkpoint> \
     --output_dir outputs/carnet/meshprior/pipeline/<run_name> \
     --proposal_types protect prune \
     --mode dry_run
   ```

2. Pipeline stages:
   - region_mining,
   - region_posterior,
   - protect_prune proposals,
   - optional snap,
   - optional fill,
   - scene gate,
   - accepted proposal export,
   - report generation.

3. Add resume flags:
   - --skip_region_mining
   - --regions_json
   - --posterior_dir
   - --proposals_json
   - --eval_only

4. Add safety flags:
   - --dry_run
   - --no_geometry_write
   - --max_regions
   - --max_proposals
   - --require_gate_pass

5. Add smoke test:
   - scripts/car_model/smoke_test_meshprior_stage10_pipeline.py

   It should run the pipeline on synthetic inputs in dry-run mode.

After code:
Write:
- docs/car_model/meshprior_stage10_alternating_runner_implementation_report.md
- docs/car_model/meshprior_stage10_alternating_runner_smoke.md

Stage gate:
- Synthetic dry-run pipeline completes end-to-end.
- Real-scene dry-run completes if a scene model/source path is available.
- The runner does not modify mesh geometry unless `--apply` or equivalent is explicitly enabled.

Append research-log entry.

Do not proceed to M11 until this passes.
```

---

# Prompt M11 — Integrate with actual scene training/evaluation and wandb

```text
You are working inside Dystopians/SPCarNet. Stage M10 must be complete.

Mission:
Run meaningful scene-level experiments. This stage may use idle GPUs and wandb. Do not skip validation.

Before code or training:
Write:
- docs/car_model/meshprior_stage11_scene_experiment_design.md

The design doc must define:
1. Which scene(s) will be used.
2. Which baselines are valid.
3. Which checkpoints/iterations will be compared.
4. Which metrics are primary.
5. How wandb runs are named.
6. What command sequence will be run.

Baseline groups:
- original mesh-splatting / scene baseline,
- baseline + geometry eval only,
- region mining only,
- protect/prune proposals dry-run,
- protect/prune accepted by gate,
- optional snap accepted by gate,
- optional fill accepted by gate.

Primary metrics:
- COLMAP sparse AbsRel,
- sparse DepthMAE,
- sparse normal mean angle,
- PSNR / SSIM / LPIPS / MAE if render eval exists,
- controlled FPS or render time,
- triangle count,
- car-region hole/floater metrics,
- free-space violation,
- accepted/rejected proposal counts.

Before launching training/eval:
```bash
git status --short
python -m compileall scripts/car_model ss3dm_prior -q
nvidia-smi
```

If a GPU is free:
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu>
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
```

Run smoke first. Then run at most one small scene experiment. Do not launch many jobs simultaneously.

Required outputs:
- outputs/carnet/meshprior/scene_experiments/<run_name>/commands.sh
- outputs/carnet/meshprior/scene_experiments/<run_name>/metrics.json
- outputs/carnet/meshprior/scene_experiments/<run_name>/summary.md
- outputs/carnet/meshprior/scene_experiments/<run_name>/wandb_url.txt, if wandb online

After experiment:
Write:
- docs/car_model/meshprior_stage11_scene_experiment_report.md

The report must include:
1. Exact commands.
2. GPU used.
3. Runtime.
4. Wandb links.
5. Metrics table.
6. Qualitative artifacts if generated.
7. Failures and suspected causes.
8. Decision: continue / adjust / stop.

Stage gate:
- A real or dry-run scene experiment completed.
- Metrics are not cherry-picked.
- Any regression is documented.
- If protect/prune improves geometry safety or triangle selection without degrading render/geometry, mark PASS.
- If all scene-level gates reject proposals, mark SOFT FAIL and write why.

Append research-log entry.

Do not proceed to M12 until this stage is fully reported.
```

---

# Prompt M12 — Improve the object prior for mesh optimization, not for Chamfer alone

```text
You are working inside Dystopians/SPCarNet. Stage M11 must be complete.

Mission:
Improve SP-CarNet's object prior specifically for mesh proposal reliability. Do not optimize only for object Chamfer. Optimize for proposal confidence calibration, surface support quality, and free-space safety.

Before code:
Write:
- docs/car_model/meshprior_stage12_prior_calibration_design.md

The design doc must identify which failure occurred in prior stages:
- weak surface localization,
- bad posterior uncertainty,
- bad free-space calibration,
- poor canonicalization,
- poor shape-field resolution,
- bad scene region extraction,
- score thresholds too conservative/aggressive.

Potential upgrades to implement only if motivated by evidence:

1. Surface-distance calibration head
   - Add a lightweight head or post-hoc calibrator that maps occupancy logits / gradients to distance-to-surface confidence.

2. SDF or unsigned-distance ablation
   - If occupancy gradients are too noisy for snap/fill, train an SDF/UDF variant or distillation head.

3. Retrieval-deformation prior
   - If the Stage 3 posterior field is too generic, implement retrieval anchor support for proposal scoring.

4. Symmetry prior for hidden-side confidence
   - Use symmetry only where confidence is high; do not force symmetry everywhere.

5. Uncertainty calibration
   - Use Stage 5 oracle-vs-reranker results to learn when posterior spread is meaningful.

Implementation requirements depend on chosen upgrade, but all upgrades must:
- have a design doc,
- have a smoke test,
- run a small synthetic benchmark,
- update the scene proposal pipeline,
- compare against the uncalibrated Stage 3 prior.

After code/training:
Run a targeted experiment, possibly on an idle GPU with wandb. Do not do a huge sweep unless the small experiment is positive.

Write:
- docs/car_model/meshprior_stage12_prior_calibration_implementation_report.md
- docs/car_model/meshprior_stage12_prior_calibration_smoke.md
- failure doc if the upgrade does not help.

Stage gate:
- Upgrade must improve at least one proposal-relevant metric without harming free-space safety:
  - floater prune precision/recall,
  - valid surface protect recall,
  - snap risk,
  - fill safety,
  - scene gate acceptance rate,
  - free-space violation.
- If it only improves object Chamfer but does not help proposals, do not make it the main method.

Append research-log entry.
```

---

# Prompt M13 — Build the NeurIPS-grade evaluation matrix

```text
You are working inside Dystopians/SPCarNet. At least M10 must be complete; M11/M12 should be complete if possible.

Mission:
Build a rigorous evaluation and reporting system that compares object-level completion, synthetic mesh repair, and scene-level mesh optimization without invalid comparisons.

Before code:
Write:
- docs/car_model/meshprior_stage13_eval_protocol_design.md

The design doc must define:
1. Datasets and splits.
2. Object-level metrics.
3. Synthetic damage metrics.
4. Scene-level metrics.
5. Which metrics are primary vs diagnostic.
6. What counts as inference-time and what is oracle-only.
7. Checkpoint selection rules.
8. Seed protocol.
9. Failure-case reporting.

Implementation requirements:

1. Add experiment registry:
   - configs/ss3dm_prior/meshprior/meshprior_experiment_matrix.yaml

   Include rows for:
   - v0.7 residual baseline, if available,
   - v0.8.2 point-flow baseline, if available,
   - Stage 3 posterior encoder,
   - Stage 4 MAP refinement,
   - Stage 5 oracle K=8 analysis,
   - protect/prune proposals,
   - protect/prune + snap,
   - protect/prune + snap + fill, if fill survived,
   - retrieval-deformation or calibration variants if implemented,
   - scene baseline,
   - scene baseline + meshprior proposals.

2. Add runner:
   - scripts/car_model/meshprior_run_experiment_matrix.py

   Flags:
   - --dry_run
   - --smoke
   - --only <experiment>
   - --group object|synthetic|scene|all
   - --seeds 0,1,2
   - --max_objects
   - --no_train
   - --eval_only

3. Add report generator:
   - scripts/car_model/meshprior_make_neurips_report.py

   It should generate:
   - docs/car_model/reports/meshprior_neurips_main_report.md
   - outputs/carnet/meshprior/reports/object_table.csv
   - outputs/carnet/meshprior/reports/synthetic_damage_table.csv
   - outputs/carnet/meshprior/reports/scene_table.csv
   - outputs/carnet/meshprior/reports/ablation_table.csv
   - outputs/carnet/meshprior/reports/failure_cases.md

Required report tables:

Table 1 — Object prior quality
Columns:
- method,
- output type,
- recon_chamfer_l1,
- hidden_chamfer_l1,
- visible_preservation_error,
- zero_corruption_chamfer,
- free_space_violation,
- mesh_extraction_success,
- inference_time.

Table 2 — Synthetic mesh repair
Columns:
- method,
- damage type,
- hole closure,
- floater prune precision,
- floater prune recall,
- valid surface protect recall,
- visible preservation,
- free-space violation,
- triangle count delta.

Table 3 — Scene mesh optimization
Columns:
- method,
- scene,
- checkpoint/iteration,
- PSNR,
- SSIM,
- LPIPS,
- COLMAP AbsRel,
- sparse DepthMAE,
- normal mean angle,
- triangle count,
- controlled FPS or render time,
- car ROI hole/floater metrics,
- accepted proposals,
- rejected proposals.

Table 4 — Safety ablation
Rows:
- direct insert, no gate,
- prior score only,
- prior + free-space gate,
- prior + geometry gate,
- prior + render gate,
- full gated method.

Scientific safeguards:
- Do not compare oracle best-of-K as headline.
- Do not compare across different damage severities unless labeled diagnostic.
- Do not hide failed mesh extractions.
- Do not use GT clean shape to choose proposals.
- Report rejected proposals as evidence of safety, not as failures.

After code:
Run dry-run report generation. If metrics exist, generate the real report.

Write:
- docs/car_model/meshprior_stage13_eval_protocol_implementation_report.md
- docs/car_model/meshprior_stage13_eval_protocol_smoke.md

Stage gate:
- Dry-run matrix loads.
- Missing experiments are marked `MISSING`, not crashed.
- At least one table can be generated from available metrics.

Append research-log entry.
```

---

# Prompt M14 — Paper-roadmap and claim-risk analysis

```text
You are working inside Dystopians/SPCarNet. Stage M13 must be complete.

Mission:
Write the paper-level roadmap and claim-risk analysis. Do not modify code unless needed to generate missing report artifacts.

Read:
- docs/car_model/meshprior_stage1_scene_meshprior_RFC.md
- docs/car_model/meshprior_stage13_eval_protocol_design.md
- docs/car_model/reports/meshprior_neurips_main_report.md, if present
- docs/car_model/SPCarNet_research_log.md
- all meshprior_stage*_implementation_report.md files

Write:
- docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md

The document must include:

1. Title options
   At least 8 options. Prefer titles around object priors for scene mesh optimization, not just car completion.

2. Central claim
   The claim should be:
   Learned object-centric shape posteriors can safely guide scene mesh optimization when converted into proposals and filtered by scene evidence gates.

3. Negative results that motivate the method
   - v0.7 residual smoothing collapse,
   - v0.8.x point-flow plateau,
   - Stage 2 decoder ceiling,
   - Stage 5 inference reranker failure,
   - Chamfer-only limitations.

4. Method summary
   - region mining,
   - posterior inference,
   - protect/prune/snap/fill proposals,
   - scene gate,
   - rollback,
   - evaluation protocol.

5. Main figures
   - prior proposes / evidence disposes diagram,
   - object prior posterior visualization,
   - proposal types figure,
   - accepted/rejected proposal examples,
   - scene-level before/after.

6. Main tables
   Use the tables from M13.

7. Ablations required before submission
   - no prior,
   - prior without gate,
   - free-space gate removed,
   - geometry gate removed,
   - protect/prune only,
   - +snap,
   - +fill,
   - posterior uncertainty removed,
   - retrieval/symmetry calibration if implemented.

8. Submission risks
   - scene scale too small,
   - weak scene-level improvement,
   - object prior hallucination,
   - object-region miner brittle,
   - lack of real parking scene metrics,
   - PRISM integration incomplete,
   - novelty looks like engineering unless proposal+gate framing is clean.

9. What result is strong enough for NeurIPS
   Define specific thresholds. Example:
   - scene geometry improves over baseline on COLMAP sparse AbsRel or normal proxy,
   - rendering does not meaningfully regress,
   - FPS or triangle budget improves or stays controlled,
   - car ROI holes/floaters decrease,
   - direct prior insertion fails but gated proposal succeeds,
   - safety ablation shows gates matter.

10. What result is not enough
   - object Chamfer improves only,
   - oracle K=8 improves only,
   - qualitative car completions only,
   - synthetic-only repair without scene metrics,
   - proposal scores with no accepted scene-level benefit.

11. Final recommendation
   Decide one of:
   - `SUBMISSION_READY_DIRECTION`,
   - `MORE_SCENE_EVIDENCE_REQUIRED`,
   - `PIVOT_TO_RETRIEVAL_DEFORMATION`,
   - `RETURN_TO_OBJECT_COMPLETION_ONLY`,
   - `STOP_AND_WRITE_FAILURE_REPORT`.

Append research-log entry.
```

---

# Prompt M15 — Optional but high-value: retrieval-deformation fallback

Use this only if Stage M12 or M13 indicates that the learned shape field is too weak for reliable proposals.

```text
You are working inside Dystopians/SPCarNet.

Mission:
Implement a retrieval-deformation fallback for object-prior mesh proposals. This is a serious alternative, not a side toy.

Hypothesis:
For vehicles, retrieving a plausible complete car anchor and deforming it to match observed scene evidence may produce safer mesh proposals than generating from a learned implicit field alone.

Before code:
Write:
- docs/car_model/meshprior_stage15_retrieval_deformation_design.md

Implementation requirements:
1. Build a train-only anchor bank from clean train objects.
2. Prevent validation/test self-retrieval leakage.
3. Implement retrieval-only proposals.
4. Implement optional smooth deformation field.
5. Export the same proposal types as MeshPrior:
   - protect,
   - prune,
   - snap,
   - fill candidate,
   - uncertainty.
6. Run the synthetic damage benchmark and compare against Stage 3 posterior MeshPrior.

Required files:
- ss3dm_prior/meshprior/retrieval_deformation.py
- scripts/car_model/meshprior_build_anchor_bank.py
- scripts/car_model/meshprior_eval_retrieval_deformation.py
- scripts/car_model/smoke_test_meshprior_stage15_retrieval_deformation.py

Stage gate:
- Retrieval-only baseline must be measured before neural deformation.
- If retrieval-only beats learned posterior on proposal metrics, write a pivot recommendation.
- If deformation overfits observed side and increases free-space violation, kill deformation and keep retrieval-only as baseline.

Write docs:
- docs/car_model/meshprior_stage15_retrieval_deformation_implementation_report.md
- docs/car_model/meshprior_stage15_retrieval_deformation_smoke.md
- failure doc if needed.

Append research-log entry.
```

---

# Prompt M16 — Optional but high-value: symmetry and part-aware proposal confidence

Use this only if protect/prune/snap/fill are functional but hidden-side proposal confidence is weak.

```text
You are working inside Dystopians/SPCarNet.

Mission:
Add symmetry and lightweight part-aware confidence to mesh-prior proposals. Do not make it a headline unless it improves proposal metrics.

Before code:
Write:
- docs/car_model/meshprior_stage16_symmetry_part_confidence_design.md

Requirements:
1. Use symmetry only when confidence is high.
2. Do not force symmetry on damaged or asymmetric observations.
3. Add pseudo-part labels only with confidence values.
4. Use part/symmetry to influence proposal confidence, not to blindly add geometry.

Implementation ideas:
- mirror observed region points across estimated symmetry plane;
- compute mirrored surface support;
- add low-height / wheel-like / roof-like region heuristics;
- use part confidence to tune protect/split/collapse scores.

Files:
- ss3dm_prior/meshprior/symmetry_confidence.py
- ss3dm_prior/meshprior/part_confidence.py
- scripts/car_model/meshprior_eval_symmetry_part_confidence.py

Metrics:
- hidden-side proposal precision,
- valid-surface protect recall,
- false fill rate,
- free-space violation,
- scene gate acceptance rate.

Stage gate:
- Must improve proposal metrics, not just object Chamfer.
- If pseudo labels are unstable, keep only qualitative analysis and do not use in main method.

Write implementation, smoke, and failure docs as usual.
Append research-log entry.
```

---

## Recommended execution order

Use this order unless a stage fails:

```text
M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M11 → M12 → M13 → M14
```

Optional fallback / enhancement branches:

```text
M15 if the learned shape field is too weak or retrieval looks stronger.
M16 if hidden-side proposals need symmetry/part confidence.
```

---

## Final desired research story

The final paper should not say:

> “We improve SP-CarNet Chamfer.”

It should say:

> **We show how learned object-centric shape posteriors can safely guide scene mesh optimization. Instead of directly inserting hallucinated object completions, SP-CarNet proposes local mesh operations, while scene-level rendering, sparse geometry, free-space, topology, and budget gates decide whether to accept them. This improves object-region geometry and mesh quality while preserving scene-level rendering and efficiency.**

The final method should be judged by:

```text
scene geometry improvement
hole / floater reduction
valid geometry protection
triangle budget / FPS behavior
render quality preservation
free-space safety
object-level completion as supporting evidence, not the sole claim
```

If the scene-level method cannot show these improvements, the project should not be framed as a top-conference scene-mesh optimization paper yet. It may still be a strong object-prior or negative-result paper, but the claim must be narrowed honestly.
