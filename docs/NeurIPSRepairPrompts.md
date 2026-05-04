# NeurIPSRepairPrompts.md — MeshSplatOpt / PRISM-Repair

Date: 2026-05-03

## Mission

You are working inside `Dystopians/SPCarNet` on a new research branch. The old PRISM line proved that conservative pruning, rollback, and topology-retention can be implemented, but the empirical gains are too small for a top-tier method paper. This prompt file pivots the project from **safe pruning** to **evidence-certified bidirectional mesh surgery** for Mesh Splatting.

Final research target:

> **MeshSplatOpt: Evidence-Certified Bidirectional Mesh Surgery for Mesh Splatting.**

The method should improve geometry and appearance, repair errors and holes, and keep topology efficient. It must not be merely a triangle-count reduction method.

The new headline idea is:

> **Counterfactual Surface Evidence Field (CSEF): every topology or geometry edit is proposed by a field of positive surface evidence, negative free-space evidence, explanation debt, and topology cost; the edit is then certified by counterfactual rendering and geometry validation before it is committed.**

This turns the project from a controller around pruning decisions into a constrained scene-repair optimizer.

---

## 0. Why this pivot is necessary

Read and internalize these repository documents before starting:

```text
docs/car_model/reports/meshprior_prism_deep_retrospective.md
docs/car_model/reports/meshprior_prism_final_handoff.md
docs/car_model/reports/meshprior_prism_reviewer_risk_checklist.md
docs/car_model/meshprior_stage1_scene_meshprior_RFC.md
docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md
docs/car_model/meshprior_stage24_2_topology_retention_report.md
docs/car_model/meshprior_stage35_retained_refresh_report.md
docs/car_model/meshprior_stage36_metric_reconciliation_report.md
docs/prompts.md
docs/car_model/meshprior_remaining_work_prompts.md
```

Core lesson:

- Current PRISM is auditable and robust, but too timid.
- Stage35 proves retained relaxed pruning can work, but the public-scene effect size is too small.
- Stage24.2 shows topology-retention matters, but it is not enough as a general research contribution.
- The original object-prior MeshPrior RFC already listed `protect/prune/snap/fill/split/collapse`; the later PRISM line narrowed too much into delete-centric topology control.
- A NeurIPS-level method needs a stronger scientific formulation and a task where the method unlocks a capability existing Mesh Splatting / 3DGS pruning methods do not address.

The new method must handle:

```text
surface floaters;
local dents / depressions;
rough broken surfaces;
vehicle surface discontinuities;
ground / wall misalignment;
large holes or voids, especially missing parking-lot ground caused by shadows, camera trajectory, under-observation, or bad reconstruction.
```

---

## 1. Research positioning and related-work inspirations

The method must be positioned against these families:

### 1.1 Neural rendering and splatting foundations

- NeRF: differentiable scene optimization from posed images.
- Instant-NGP: fast neural graphics primitive optimization.
- 3D Gaussian Splatting: explicit primitive optimization with densification and pruning.
- Mesh Splatting / Triangle Splatting / 2D Triangle Splatting: mesh or triangle primitive radiance fields.

### 1.2 Surface-aligned and mesh-aware splatting

- SuGaR: Gaussian splats aligned to surfaces and converted to meshes.
- MeshGS / mesh-aligned Gaussian methods: bind splats to mesh structures.
- 2DGS / DN-Splatter: use depth distortion, normal consistency, and geometry priors for better surfaces.

### 1.3 Compact / pruned 3DGS

- LightGaussian, Compact3DGS, EAGLES, Mini-Splatting, EfficientGS, RadSplat, LP-3DGS, MaskGaussian, PUP 3D-GS, GaussianPOP, GaussianSpa, SafeguardGS.
- These methods usually compress, prune, or sparsify Gaussians. They are important baselines, but they do not fully solve evidence-certified mesh repair or giant hole completion.

### 1.4 Classical mesh optimization

- Quadric Error Metrics / edge collapse simplification.
- Remeshing, isotropic remeshing, adaptive remeshing.
- Hole filling and constrained triangulation.
- Poisson surface reconstruction and screened Poisson reconstruction.
- Delaunay / restricted Delaunay surface methods.
- Mesh fairing / Laplacian smoothing / ARAP deformation.

These are strong baselines for geometry operations, but they are not learned differentiable rendering systems and do not usually have render/geometry counterfactual rollback.

### 1.5 Multi-view geometry and priors

- COLMAP / SfM and MVS for sparse / dense scene evidence.
- Plane priors and Manhattan / ground priors for man-made scenes.
- Learned object priors for car / vehicle repair.
- Monocular depth / normal priors may be used as optional support, but must be labeled separately from pure COLMAP evidence.

---

## 2. The core innovation: Counterfactual Surface Evidence Field

Do not implement another heuristic pruning schedule as the main contribution.

Implement a field-level abstraction that every candidate edit must consult:

```text
CSEF(x, n, region) = {
  positive_surface_evidence,
  negative_free_space_evidence,
  explanation_debt,
  prior_support,
  topology_cost,
  uncertainty
}
```

Definitions:

- **Positive surface evidence**: multi-view visibility, sparse COLMAP support, MVS/depth support if available, normal agreement, boundary-loop support, object/plane prior support.
- **Negative free-space evidence**: camera rays / sparse observations indicating that a surface at the candidate location would violate visibility or free space.
- **Explanation debt**: visible pixels, residual regions, boundary holes, missing depth support, or unmatched semantic/object regions that the current mesh fails to explain.
- **Prior support**: learned object prior for vehicles, plane/height-field prior for ground, local smoothness prior for walls/ground, symmetry prior for vehicles.
- **Topology cost**: triangle/vertex/memory/render cost introduced or removed by the edit.
- **Uncertainty**: low evidence, high posterior variance, poor camera coverage, unreliable normal/depth, or under-observed void region.

The edit objective:

```text
maximize  evidence_debt_reduction(edit)
        + render_quality_gain(edit)
        + geometry_consistency_gain(edit)
        - free_space_violation(edit)
        - hallucination_risk(edit)
        - topology_cost(edit)
```

Subject to:

```text
counterfactual_render_gate passes
counterfactual_geometry_gate passes
changed_pixel_gate passes
free_space_gate passes
budget/state-machine gate passes
```

The paper story:

> Existing Mesh Splatting / 3DGS pruning methods ask which primitives can be removed. MeshSplatOpt asks which local surface edit best reduces scene evidence debt while remaining counterfactually certified by held-out rendering and sparse geometry.

---

## 3. Non-negotiable operating rules

Apply to every prompt below.

1. Work one stage at a time.
2. Do not proceed after a failed hard gate.
3. Do not hide negative results.
4. Do not mix training-time metrics with independent `render.py + metrics.py` metrics.
5. Do not use ground truth to choose proposals at inference time.
6. Keep old PRISM stages as baselines; do not overwrite them.
7. All new behavior must be opt-in unless a stage explicitly promotes it after ablation.
8. All new training runs must use W&B when available.
9. Every stage must write design, implementation report, smoke report, and research-log entry.
10. Every edit type must support rollback.
11. Every accepted repair must have an audit trail: proposal JSON, before snapshot, after snapshot, gate report, W&B link if trained, independent metrics if rendered.

Before major edits, run:

```bash
git status --short
python --version
python -m compileall scripts/car_model ss3dm_prior utils -q
```

Before every GPU run:

```bash
nvidia-smi
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
```

Each completed stage must update:

```text
docs/car_model/SPCarNet_research_log.md
```

Commit and push after each meaningful `PASS`, `SOFT PASS`, `FAIL`, or `STOP`.

---

# Prompt R0 — Branch, audit, and pivot lock

```text
You are working inside Dystopians/SPCarNet.

Mission:
Create a new research branch and lock the pivot from delete-centric PRISM to evidence-certified bidirectional mesh surgery.

Recommended branch:

git checkout -b neurips-meshsplatopt-repair

Read:
- docs/car_model/reports/meshprior_prism_deep_retrospective.md
- docs/car_model/reports/meshprior_prism_final_handoff.md
- docs/car_model/reports/meshprior_prism_reviewer_risk_checklist.md
- docs/car_model/meshprior_stage1_scene_meshprior_RFC.md
- docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md
- docs/car_model/meshprior_stage24_2_topology_retention_report.md
- docs/car_model/meshprior_stage35_retained_refresh_report.md
- docs/car_model/meshprior_stage36_metric_reconciliation_report.md
- docs/prompts.md
- docs/car_model/meshprior_remaining_work_prompts.md

Run:

git status --short
python --version
python -m compileall scripts/car_model ss3dm_prior utils -q

Write:
- docs/car_model/meshsplatopt_stageR0_pivot_audit.md

The audit must include:
1. Current branch and commit.
2. Dirty files.
3. Current PRISM strengths.
4. Current PRISM weaknesses.
5. Why Stage35 is a baseline, not the final method.
6. What new operations are required: delete, collapse, snap, split, fill, appearance repair.
7. A go/no-go recommendation: `PROCEED_TO_R1` or `STOP`.

Append research-log entry.
Commit and push if possible.

Gate:
- `PASS` only if repository compiles and the audit explicitly states that the new method is bidirectional repair, not pruning-only.
```

---

# Prompt R1 — Write the NeurIPS repair RFC and novelty manifesto

```text
Stage R0 must be PASS.

Mission:
Write the new research RFC for MeshSplatOpt / PRISM-Repair.

Do not change training code in this stage.

Write:
- docs/car_model/meshsplatopt_stageR1_repair_RFC.md

The RFC must define:

1. New method name:
   `MeshSplatOpt: Evidence-Certified Bidirectional Mesh Surgery for Mesh Splatting`.

2. Core innovation:
   `Counterfactual Surface Evidence Field (CSEF)`.

3. Why this is not just engineering:
   - It reformulates mesh-splatting repair as evidence-debt minimization under counterfactual validation constraints.
   - Edits are not hard-coded heuristic patches; they are actions scored by positive evidence, negative evidence, explanation debt, prior support, topology cost, and uncertainty.
   - The same edit calculus handles deletion, collapse, snapping, splitting, and filling.

4. Problem classes:
   - floaters;
   - dents;
   - rough/broken surfaces;
   - car surface discontinuities;
   - ground/wall misalignment;
   - giant ground voids / parking-lot holes;
   - appearance ghosting after geometry repair.

5. Edit operations:
   - protect;
   - delete/prune;
   - collapse/merge;
   - snap/deform;
   - split/subdivide;
   - fill/patch;
   - appearance reset/recovery.

6. Evidence certificates:
   - render certificate;
   - sparse depth certificate;
   - normal certificate;
   - free-space certificate;
   - boundary-loop certificate;
   - semantic/object certificate;
   - plane/ground certificate;
   - uncertainty certificate.

7. Giant-hole policy:
   - Distinguish observed holes from unobserved unknown voids.
   - A huge hole may be filled only if there is sufficient boundary, plane/height-field, multi-view, semantic, or prior evidence.
   - If the area is truly out-of-trajectory with no evidence, the method may produce a prior-supported candidate for visualization, but headline metrics must label it as prior-supported and uncertain.

8. Paper claim:
   `MeshSplatOpt repairs mesh-splatting scene geometry by proposing bidirectional topology/geometry edits from a surface evidence field and certifying them through counterfactual rendering and geometry validation.`

9. Baselines and ablations:
   - MeshSplatting original;
   - Stage35 PRISM;
   - delete-only PRISM;
   - post-hoc mesh decimation / QEM;
   - hole filling without render gate;
   - plane fill without free-space gate;
   - object prior fill without scene gate;
   - no teacher recovery;
   - no CSEF debt term;
   - no rollback.

10. Kill criteria:
   - If the method remains delete-only after R6, STOP.
   - If giant-hole repair cannot produce a valid candidate on synthetic damage by R8, STOP or demote hole repair.
   - If medium public scenes do not show either repair-quality gains or topology-quality Pareto gains by R13, STOP main-conference framing.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if the RFC clearly separates pruning, repair, and hallucination risk.
```

---

# Prompt R2 — Related work and novelty-threat matrix

```text
Stage R1 must be PASS.

Mission:
Create a paper-facing related-work and novelty-threat matrix before implementing major code.

Write:
- docs/car_model/meshsplatopt_stageR2_related_work_matrix.md
- docs/car_model/meshsplatopt_stageR2_baseline_plan.md

Include at least these related-work groups:

1. Neural rendering / splatting:
   - NeRF
   - Instant-NGP
   - 3DGS
   - Mesh Splatting
   - Triangle Splatting
   - 2D Triangle Splatting

2. Mesh-aware splatting / geometry:
   - SuGaR
   - MeshGS
   - 2DGS
   - DN-Splatter
   - mesh-embedded Gaussian methods

3. 3DGS compression/pruning:
   - LightGaussian
   - Compact3DGS
   - EfficientGS
   - Mini-Splatting
   - EAGLES
   - RadSplat
   - LP-3DGS
   - MaskGaussian
   - PUP 3D-GS
   - GaussianPOP
   - GaussianSpa
   - SafeguardGS

4. Classical mesh processing:
   - QEM edge collapse
   - constrained Delaunay triangulation
   - hole filling
   - Poisson/screened Poisson reconstruction
   - isotropic/adaptive remeshing
   - Laplacian/ARAP deformation

5. Multi-view geometry and priors:
   - COLMAP SfM/MVS
   - plane/Manhattan/ground priors
   - object shape priors
   - monocular depth/normal priors as optional add-ons

For each method/group, record:
- what it does;
- what component of MeshSplatOpt it threatens;
- what it does not cover;
- required baseline or ablation;
- novelty threat: High / Medium / Low.

Key required conclusion:
- Training-time pruning alone is not novel.
- Mesh/triangle pruning alone is not novel.
- Geometry priors alone are not novel.
- Counterfactual validation alone is not enough unless tied to bidirectional repair and evidence-debt formulation.
- The strongest novelty is the unified CSEF + reversible edit calculus + certified huge-hole repair.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if the matrix names concrete baselines for pruning, simplification, hole filling, and geometry-aware repair.
```

---

# Prompt R3 — Implement CSEF data model and diagnostics

```text
Stage R2 must be PASS.

Mission:
Implement the Counterfactual Surface Evidence Field data contract and diagnostic collector.

Before code, write:
- docs/car_model/meshsplatopt_stageR3_csef_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/__init__.py
- ss3dm_prior/meshsplatopt/csef_types.py
- ss3dm_prior/meshsplatopt/csef_builder.py
- scripts/car_model/meshsplatopt_build_csef.py
- scripts/car_model/smoke_test_meshsplatopt_stageR3_csef.py

Data model:

CSEFSample:
- sample_id
- position
- normal
- region_id
- positive_surface_evidence
- negative_free_space_evidence
- explanation_debt
- prior_support
- topology_cost
- uncertainty
- evidence_sources
- notes

CSEFRegion:
- region_id
- defect_type candidates
- bbox
- boundary_loop_ids
- mesh_face_indices
- image_evidence_refs
- sparse_point_refs
- csef summary stats

CSEFBuildResult:
- scene_model
- scene_source
- mesh_path
- regions
- global summary

Minimal implementation:
1. Load mesh or checkpoint PLY when available.
2. Sample faces/vertices/edges.
3. Compute placeholder but meaningful diagnostics:
   - local triangle area;
   - boundary edge score;
   - connected component id;
   - sparse support placeholder if COLMAP parser available;
   - image residual placeholder if render outputs available;
   - topology cost per face;
   - uncertainty based on missing evidence.
4. Do not modify geometry.
5. Write:
   - csef_samples.npz
   - csef_regions.json
   - csef_summary.csv
   - csef_report.md

Smoke test:
- Create synthetic mesh with ground plane, a hole, a floater, and a dent.
- Build CSEF.
- Verify:
  - boundary/hole region has high explanation debt;
  - floater component has high uncertainty or low positive evidence;
  - normal ground region has low debt.

After code, write:
- docs/car_model/meshsplatopt_stageR3_csef_implementation_report.md
- docs/car_model/meshsplatopt_stageR3_csef_smoke.md

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if the synthetic CSEF separates normal surface, floater, and hole/debt region.
```

---

# Prompt R4 — Defect mining: floaters, dents, broken surfaces, and giant voids

```text
Stage R3 must be PASS.

Mission:
Build a defect miner that turns CSEF diagnostics into actionable repair regions.

Before code, write:
- docs/car_model/meshsplatopt_stageR4_defect_mining_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/defect_types.py
- ss3dm_prior/meshsplatopt/defect_mining.py
- scripts/car_model/meshsplatopt_mine_defects.py
- scripts/car_model/smoke_test_meshsplatopt_stageR4_defect_mining.py

Defect types:
- FLOATER_COMPONENT
- LOCAL_DENT
- ROUGH_BROKEN_SURFACE
- VEHICLE_DISCONTINUITY
- GROUND_WALL_MISALIGNMENT
- SMALL_BOUNDARY_HOLE
- GIANT_GROUND_VOID
- UNKNOWN_UNOBSERVED_VOID
- APPEARANCE_GHOSTING_REGION

For each defect, output:
- defect_id
- defect_type
- severity
- confidence
- affected faces/vertices
- boundary loops if present
- candidate edit types allowed
- evidence summary
- uncertainty summary
- reason if no repair is allowed

Giant void logic:
1. Detect large boundary loops and mesh-space voids.
2. Detect image-space unexplained regions if render residuals exist.
3. For ground voids, estimate whether neighboring surfaces support a plane or low-order height field.
4. Separate:
   - observed void: enough views or boundary evidence;
   - prior-supported void: weak views but strong plane/semantic prior;
   - unknown void: too little evidence; do not headline repair.

Smoke test:
- Synthetic parking-ground mesh with a large rectangular missing ground patch.
- Synthetic out-of-trajectory void with no boundary support.
- Verify first becomes `GIANT_GROUND_VOID`, second becomes `UNKNOWN_UNOBSERVED_VOID`.

Artifacts:
- defects.json
- defects_summary.csv
- defect_mining_report.md
- optional debug PLY/OBJ with defect labels

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if huge ground holes are explicitly detected and distinguished from unknown/unobserved voids.
```

---

# Prompt R5 — Unified reversible edit abstraction

```text
Stage R4 must be PASS.

Mission:
Implement a unified reversible edit abstraction for all mesh surgeries.

Before code, write:
- docs/car_model/meshsplatopt_stageR5_reversible_edits_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/edit_types.py
- ss3dm_prior/meshsplatopt/edit_apply.py
- ss3dm_prior/meshsplatopt/edit_snapshot.py
- scripts/car_model/smoke_test_meshsplatopt_stageR5_reversible_edits.py

Edit types:
- PROTECT
- DELETE_TRIANGLES
- EDGE_COLLAPSE
- FACE_MERGE
- SNAP_VERTICES
- SPLIT_TRIANGLES
- FILL_PATCH
- APPEARANCE_RESET

Each edit record must contain:
- edit_id
- edit_type
- defect_id
- affected vertices/faces
- inserted vertices/faces if any
- deleted vertices/faces if any
- attribute changes if any
- topology cost delta
- evidence summary
- risk summary
- rollback snapshot path

Required functions:
- create_snapshot(mesh_or_state)
- apply_edit(mesh_or_state, edit)
- rollback_edit(mesh_or_state, snapshot)
- verify_mesh_integrity(mesh_or_state)
- summarize_topology_delta(before, after)

Minimal implementation can operate on generic numpy mesh arrays first. Integration with Mesh Splatting checkpoints can be added later.

Smoke tests:
1. Delete triangles and rollback exactly.
2. Snap vertices and rollback exactly.
3. Fill a patch and rollback exactly.
4. Edge collapse or face merge preserves valid face indices.
5. Mesh integrity checker catches degenerate faces and invalid indices.

Artifacts:
- edit_smoke_report.json
- before/after/rollback meshes for synthetic tests

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if all edit types are reversible or explicitly marked as not-yet-supported with a failing gate.
```

---

# Prompt R6 — Strong delete/collapse/merge baselines

```text
Stage R5 must be PASS.

Mission:
Build strong topology-reduction baselines so future repair claims cannot hide behind weak comparisons.

Before code, write:
- docs/car_model/meshsplatopt_stageR6_topology_baselines_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/topology_baselines.py
- scripts/car_model/meshsplatopt_run_topology_baselines.py
- scripts/car_model/smoke_test_meshsplatopt_stageR6_topology_baselines.py

Baselines:
1. PRISM score top-k delete.
2. Random same-count delete.
3. Low-visibility delete.
4. Boundary-protected delete.
5. QEM-style edge collapse if available.
6. Planar face merge / coarsening if implementable.
7. Trimesh / pymeshlab simplification if installed; otherwise document missing dependency and keep JSON contract.

Budgets:
- 90%
- 75%
- 50%
- 25%

Outputs:
- topology_baseline_runs.json
- topology_baseline_table.csv
- topology_baseline_report.md

Smoke test:
- Synthetic plane + object mesh.
- Verify each supported baseline produces valid mesh and target triangle count within tolerance.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if at least delete, boundary-protected delete, and one collapse/merge-style baseline run on synthetic mesh.
```

---

# Prompt R7 — Snap/deform proposals for geometry correction

```text
Stage R6 must be PASS.

Mission:
Implement safe snap/deform proposals to repair floaters, dents, rough broken surfaces, vehicle discontinuities, and ground/wall misalignment.

Before code, write:
- docs/car_model/meshsplatopt_stageR7_snap_deform_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/snap_proposals.py
- scripts/car_model/meshsplatopt_make_snap_proposals.py
- scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py

Snap targets:
- local plane fit from neighboring supported triangles;
- sparse COLMAP point-to-plane target if available;
- ground plane / wall plane target if defect type supports it;
- object-prior surface target if region is vehicle-like and posterior confidence is high;
- local smoothing/fairing target for rough broken surfaces.

Proposal rule:
- Never move boundary vertices aggressively.
- Never snap through negative free-space evidence.
- Generate multiple step sizes: 0.1, 0.25, 0.5 of target displacement.
- Cap max displacement by scene scale.
- Store uncertainty and evidence source.

Outputs:
- snap_proposals.json
- snap_summary.csv
- snap_debug_before_after.ply when possible

Smoke test:
1. Synthetic dented plane: snap should reduce surface error.
2. Synthetic floater: snap should not incorrectly attach it unless evidence supports it.
3. Synthetic wall/ground slight misalignment: snap should reduce plane residual.
4. Rollback must restore exact original mesh arrays.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if snap improves synthetic geometry error without increasing free-space violation or corrupting topology.
```

---

# Prompt R8 — Giant ground-void and large-hole fill proposals

```text
Stage R7 must be PASS.

Mission:
Implement large-hole and giant-ground-void fill proposals as first-class operations.

This is a key research stage. It must not be a small local triangle patch only.

Before code, write:
- docs/car_model/meshsplatopt_stageR8_giant_void_fill_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/hole_fill.py
- ss3dm_prior/meshsplatopt/ground_void_fill.py
- scripts/car_model/meshsplatopt_make_fill_proposals.py
- scripts/car_model/smoke_test_meshsplatopt_stageR8_giant_void_fill.py

Fill modes:

1. `boundary_loop_fill`
   - Detect boundary loops.
   - Triangulate loop by constrained 2D projection / fan / Delaunay fallback.
   - Reject loops with poor planarity or ambiguous open-world boundary.

2. `ground_plane_void_fill`
   - Fit robust plane or low-order height field from neighboring ground triangles, sparse points, or ground masks if available.
   - Generate a grid or constrained triangulated patch for the void.
   - Snap patch vertices to height field.
   - Initialize normals and appearance placeholders.

3. `depth_guided_patch_fill`
   - If render/depth evidence exists, backproject candidate support into 3D.
   - Fit local surface patch.
   - Use only where positive evidence is strong.

4. `prior_supported_fill`
   - For vehicle-like regions, use object prior as proposal source.
   - For ground-like regions, use plane/height-field prior.
   - Must be labeled as prior-supported and uncertain if direct multi-view evidence is weak.

Giant void certificate:
Each fill proposal must record:
- boundary_loop_support;
- neighboring_surface_support;
- sparse_depth_support;
- free_space_risk;
- semantic/ground/object support;
- camera_coverage_score;
- prior_only_flag;
- expected topology cost;
- expected area repaired.

Rules:
- Do not fill an `UNKNOWN_UNOBSERVED_VOID` unless explicitly run in `--allow_prior_only_fill` diagnostic mode.
- Prior-only fills cannot be used as headline evidence unless the report labels them separately.
- A fill proposal must be reversible.

Smoke tests:
1. Small hole on plane: fill closes loop.
2. Giant rectangular missing parking ground: fill produces a valid ground patch.
3. Out-of-trajectory unknown void: normal mode rejects; diagnostic prior-only mode proposes but marks `prior_only_flag=true`.
4. Fill rollback restores original mesh.
5. Degenerate boundary loop is rejected.

Artifacts:
- fill_proposals.json
- fill_summary.csv
- fill_certificate_report.md
- before/after debug meshes

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if giant ground void synthetic repair works and unknown void is not silently filled in normal mode.
```

---

# Prompt R9 — Object-prior vehicle repair proposals

```text
Stage R8 must be PASS.

Mission:
Integrate SP-CarNet object posterior as an optional proposal generator for vehicle-region repair.

Do not let the object prior directly overwrite scene geometry.

Before code, write:
- docs/car_model/meshsplatopt_stageR9_object_prior_repair_design.md

Read existing files:
- ss3dm_prior/meshprior/region_types.py
- ss3dm_prior/meshprior/scene_region_posterior.py
- ss3dm_prior/meshprior/protect_prune.py
- ss3dm_prior/meshprior/optimizer_adapter.py
- docs/car_model/meshprior_stage1_scene_meshprior_RFC.md

Implementation files:
- ss3dm_prior/meshsplatopt/object_prior_repair.py
- scripts/car_model/meshsplatopt_make_object_repair_proposals.py
- scripts/car_model/smoke_test_meshsplatopt_stageR9_object_prior_repair.py

Proposal types from object prior:
- vehicle protect mask;
- vehicle floater delete candidates;
- vehicle surface snap candidates;
- vehicle discontinuity fill candidates;
- vehicle boundary split candidates.

Requirements:
- Use posterior uncertainty to downweight all aggressive proposals.
- Never generate object-prior fill for low canonicalization confidence.
- Never commit object-prior proposals without scene counterfactual validation.
- Record `prior_proposes_evidence_disposes=true` in every object-prior proposal.

Smoke test:
- Synthetic car-like box with missing side panel.
- Object-prior module proposes a fill/snap/protect package.
- Uncertain prior case produces only protect/prune or no proposal.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if object-prior proposals are clearly bounded and cannot bypass scene gates.
```

---

# Prompt R10 — Generalized counterfactual validation for all edit types

```text
Stage R9 must be PASS.

Mission:
Generalize PRISM counterfactual validation from delete-only candidate masks to arbitrary reversible edits.

Before code, write:
- docs/car_model/meshsplatopt_stageR10_generalized_counterfactual_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/counterfactual_edit_gate.py
- scripts/car_model/meshsplatopt_validate_edit_counterfactual.py
- scripts/car_model/smoke_test_meshsplatopt_stageR10_counterfactual_edits.py

Reuse or adapt:
- utils/prism_counterfactual.py
- utils/prism_geometry_proxy.py
- existing calibration-view selection
- existing changed-pixel ratio computation

Validation flow:
1. Snapshot state.
2. Apply edit temporarily.
3. Render calibration/held-out views if model/render path exists.
4. Evaluate sparse depth proxy and normal proxy if available.
5. Evaluate free-space and CSEF certificate.
6. Evaluate topology integrity.
7. Optionally run short recovery for appearance/geometry.
8. Accept or reject.
9. Rollback automatically on reject.

Gate metrics:
- delta PSNR;
- delta MAE;
- delta SSIM/LPIPS if available;
- changed pixel ratio;
- sparse AbsRel;
- sparse depth MAE;
- normal mean angle;
- free-space violation;
- topology validity;
- CSEF debt reduction;
- prior-only flag risk.

Edit-specific rules:
- DELETE/COLLAPSE must not increase render/geometry error beyond thresholds.
- SNAP must reduce geometry inconsistency or render residual without visible degradation.
- FILL must reduce hole/debt metrics and must not create free-space violations.
- GIANT_GROUND_VOID fill must pass plane/boundary/camera coverage certificate or be marked prior-only diagnostic.

Smoke tests:
1. Good fill accepted on synthetic hole.
2. Bad floater insertion rejected.
3. Snap through free space rejected.
4. Delete valid supported surface rejected.
5. Rollback exactly restores state.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if at least one non-delete edit is accepted and at least one harmful non-delete edit is rejected in smoke.
```

---

# Prompt R11 — Teacher-guided appearance and geometry recovery

```text
Stage R10 must be PASS.

Mission:
Implement short recovery after accepted or tentative edits so geometry repair does not destroy appearance.

Before code, write:
- docs/car_model/meshsplatopt_stageR11_teacher_recovery_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/teacher_recovery.py
- scripts/car_model/meshsplatopt_run_teacher_recovery.py
- scripts/car_model/smoke_test_meshsplatopt_stageR11_teacher_recovery.py

Core idea:
Before an edit, cache teacher outputs from the current model:
- RGB render;
- depth;
- normal if available;
- alpha/coverage if available;
- visibility mask;
- edit-region mask.

After edit, run a short recovery window:
- optimize new/edited primitive appearance;
- preserve unedited region with teacher RGB/depth/normal distillation;
- allow edited region to match GT images and sparse geometry;
- initialize fill-patch appearance by multi-view color projection or neighbor interpolation;
- prevent global drift.

Required CLI should support:

python scripts/car_model/meshsplatopt_run_teacher_recovery.py \
  --model_path <model> \
  --edit_json <edit_or_proposal_json> \
  --output_dir outputs/carnet/meshsplatopt/recovery/<run_name> \
  --iterations 200

Smoke tests:
- Synthetic or tiny model path if available.
- If no renderable model exists, test cache/recovery contracts and fail gracefully.
- Verify teacher cache files are written.
- Verify recovery report distinguishes edited and unedited regions.

Append research-log entry.
Commit and push.

Gate:
- `PASS` if recovery contract works and either a real tiny recovery runs or missing render path is documented as a clear `SOFT PASS` with no hallucinated metrics.
```

---

# Prompt R12 — Edit portfolio optimizer and budget-aware repair state machine

```text
Stage R11 must be PASS.

Mission:
Implement a state machine that chooses among delete, collapse, snap, split, fill, and appearance recovery under budget and safety constraints.

Before code, write:
- docs/car_model/meshsplatopt_stageR12_edit_portfolio_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/edit_portfolio.py
- ss3dm_prior/meshsplatopt/repair_state_machine.py
- scripts/car_model/meshsplatopt_run_repair_state_machine.py
- scripts/car_model/smoke_test_meshsplatopt_stageR12_portfolio.py

States:
1. GEOMETRY_ACQUISITION
2. DEFECT_MINING
3. LOW_RISK_CLEANUP
4. SNAP_REPAIR
5. GIANT_VOID_REPAIR
6. OBJECT_PRIOR_REPAIR
7. APPEARANCE_RECOVERY
8. TOPOLOGY_RETENTION
9. VALIDATION_ROLLBACK
10. FINAL_AUDIT

Portfolio selection:
- Score each candidate edit by expected CSEF debt reduction per topology/render cost.
- Prefer low-risk cleanup first.
- Allow fill/split only when explanation debt is high and evidence certificate is strong.
- If topology budget is exceeded, require collapse/delete debt repayment after split/fill.
- Freeze or restrict ordinary densification once repair commits begin, unless the state machine records a topology debt plan.

Outputs:
- edit_portfolio.json
- state_machine_trace.json
- accepted_edits.json
- rejected_edits.json
- final_audit.json
- repair_summary.md

Smoke test:
- Synthetic mesh with floater, dent, and giant ground hole.
- State machine should propose cleanup, snap, and fill in a sensible order.
- Bad prior-only fill should be rejected in normal mode.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if the state machine executes at least three edit classes on synthetic data and produces an auditable trace.
```

---

# Prompt R13 — Synthetic repair benchmark suite

```text
Stage R12 must be PASS.

Mission:
Build a controlled synthetic benchmark to prove repair capability before expensive public-scene runs.

Before code, write:
- docs/car_model/meshsplatopt_stageR13_synthetic_repair_benchmark_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/synthetic_damage.py
- scripts/car_model/meshsplatopt_make_synthetic_repair_benchmark.py
- scripts/car_model/meshsplatopt_run_synthetic_repair_benchmark.py
- scripts/car_model/meshsplatopt_collect_synthetic_repair_results.py

Damage types:
- floater triangles;
- local dent;
- noisy rough patch;
- vehicle side discontinuity;
- ground/wall misalignment;
- small hole;
- giant ground void;
- prior-only unobserved void;
- appearance corruption on filled patch.

Methods to compare:
- no repair;
- delete-only PRISM-style cleanup;
- QEM/decimation-only baseline;
- classical hole fill only;
- snap only;
- fill only;
- CSEF without counterfactual gate;
- full MeshSplatOpt repair.

Metrics:
- triangle count;
- surface distance to synthetic clean mesh, evaluation-only;
- hole boundary length reduction;
- giant void area repaired;
- free-space violation;
- normal error;
- topology validity;
- accepted/rejected edits;
- prior-only false fill rate;
- render metrics if synthetic cameras/renders are available.

Important:
- Evaluation clean mesh may be used only for metrics, never for proposal selection.

Append research-log entry.
Commit and push.

Gate:
- `PASS` only if full MeshSplatOpt improves at least 4/7 synthetic damage categories over delete-only baseline and rejects the prior-only unknown void in normal mode.
```

---

# Prompt R14 — Medium public-scene repair pilot

```text
Stage R13 must be PASS.

Mission:
Run the first medium-budget public-scene repair pilot.

Scenes:
Use available COLMAP-compatible scenes already validated in previous reports, for example:
- Mip-NeRF 360 `bonsai`
- ETH3D `courtyard`
- parking_phone_tiny

Before running, write:
- docs/car_model/meshsplatopt_stageR14_medium_scene_pilot_design.md

Required methods:
1. current best clean / sparse-depth baseline;
2. Stage35 PRISM retained relaxed baseline;
3. delete-only PRISM-Budget baseline;
4. topology baseline from R6 where compatible;
5. MeshSplatOpt full repair without giant-hole fill;
6. MeshSplatOpt full repair with giant-hole fill enabled only where certified.

Budget:
- Use medium 2000-iteration or equivalent local budget first.
- Do not launch 7000+ full-budget until medium gate passes.

Commands:
- Record exact train/render/metrics commands.
- Use W&B online for training.
- Use independent `render.py + metrics.py` for paper-facing metrics.
- Use sparse COLMAP geometry proxy separately.

Metrics:
- PSNR/SSIM/LPIPS;
- sparse AbsRel / DepthMAE;
- normal mean angle;
- triangle/vertex count;
- defect counts and repairs;
- giant-hole certificates;
- accepted/rejected edit table;
- runtime and memory if available.

Outputs:
- outputs/carnet/meshsplatopt/stageR14_medium_scene_pilot/...
- docs/car_model/meshsplatopt_stageR14_medium_scene_pilot_report.md

Gate:
- `PASS` if full MeshSplatOpt beats Stage35 or a strong baseline on either:
  a) geometry/repair metrics without render regression, or
  b) topology-quality Pareto frontier,
  on at least two scenes.
- `SOFT PASS` if one scene is strong and another is diagnostic.
- `FAIL` if improvements are only tiny pruning-like deltas.

Append research-log entry.
Commit and push.
```

---

# Prompt R15 — Full-budget scene sweep and Pareto/repair curves

```text
Stage R14 must be PASS or strong SOFT PASS with explicit approval in the report.

Mission:
Run full-budget evaluation only after the medium pilot justifies GPU time.

Before running, write:
- docs/car_model/meshsplatopt_stageR15_full_budget_sweep_design.md

Scenes:
- At least 3 geometry-observable scenes.
- Include parking_phone_tiny only as a domain-specific scene; do not let it be the sole headline.

Budgets:
- 100%
- 75%
- 50%
- 25% if stable

Methods:
- MeshSplatting baseline;
- Stage35 PRISM;
- delete-only PRISM;
- QEM/posthoc simplification;
- classical hole fill / remesh baseline;
- MeshSplatOpt full;
- MeshSplatOpt without giant-hole fill;
- MeshSplatOpt without teacher recovery;
- MeshSplatOpt without rollback.

Outputs:
- full_budget_results.json
- full_budget_table.csv
- pareto_curves.png
- repair_metrics_table.md
- accepted_rejected_edit_gallery.md

Gate:
- `PASS` if MeshSplatOpt shows a clear Pareto or repair-quality advantage on at least 2/3 scenes and all regressions are transparently reported.
- `FAIL` if the result is still only marginal Stage35-like improvement.

Append research-log entry.
Commit and push.
```

---

# Prompt R16 — Ablation suite for scientific contribution

```text
Stage R15 must be PASS.

Mission:
Prove the contribution is scientific, not only engineering.

Before code/runs, write:
- docs/car_model/meshsplatopt_stageR16_ablation_design.md

Required ablations:

1. No CSEF debt term.
2. No negative free-space evidence.
3. No counterfactual render gate.
4. No sparse geometry gate.
5. No changed-pixel gate.
6. No rollback.
7. No teacher recovery.
8. Delete/collapse only.
9. Snap only.
10. Fill only.
11. Giant-hole fill without certificate.
12. Object-prior fill without scene gate.
13. Budget controller disabled.
14. Densification freeze disabled.

For each ablation:
- run at least synthetic benchmark and one public scene where feasible;
- collect metrics and failure cases;
- report accepted harmful edits if any;
- include visual examples.

Gate:
- `PASS` only if at least three core components are empirically necessary or clearly reduce failure rate.

Append research-log entry.
Commit and push.
```

---

# Prompt R17 — Paper package and final NeurIPS go/no-go

```text
Stage R16 must be PASS.

Mission:
Assemble the NeurIPS paper package and make an honest go/no-go decision.

Write:
- docs/car_model/reports/meshsplatopt_neurips_manuscript_skeleton.md
- docs/car_model/reports/meshsplatopt_neurips_method.md
- docs/car_model/reports/meshsplatopt_neurips_experiments.md
- docs/car_model/reports/meshsplatopt_neurips_ablation.md
- docs/car_model/reports/meshsplatopt_neurips_related_work.md
- docs/car_model/reports/meshsplatopt_neurips_reviewer_risk_checklist.md
- docs/car_model/reports/meshsplatopt_neurips_final_go_no_go.md

Required figures:
1. CSEF concept figure.
2. Bidirectional edit calculus figure.
3. Giant-hole repair pipeline figure.
4. Accepted/rejected counterfactual examples.
5. Pareto curves.
6. Before/after repair gallery.
7. Failure cases.

Required tables:
1. Public-scene render/geometry/topology table.
2. Repair metrics table.
3. Synthetic damage benchmark.
4. Baseline comparison.
5. Ablation table.
6. Runtime/memory table.

Go/no-go criteria:

`GO_NEURIPS` only if:
- full method has clear advantage over Stage35 and strong baselines;
- giant-hole repair works on synthetic and at least one real/realistic scene without obvious hallucination;
- ablations prove CSEF/gates/recovery matter;
- independent metrics are separated from training metrics;
- negative results are transparent.

`WORKSHOP_OR_ARXIV` if:
- method is coherent but evidence is limited or scene count is too small.

`STOP_OR_PIVOT` if:
- gains are marginal, delete-centric, or mainly engineering.

Append research-log entry.
Commit and push.
```

---

## Final framing for the eventual paper

Do not frame the paper as:

```text
We prune Mesh Splatting triangles safely.
```

Frame it as:

```text
We introduce an evidence-certified edit calculus for Mesh Splatting. A Counterfactual Surface Evidence Field identifies where the current mesh has surface debt, free-space risk, or topology redundancy. The optimizer proposes reversible bidirectional mesh edits—delete, collapse, snap, split, fill, and appearance recovery—and commits only those edits that pass held-out rendering, sparse-geometry, normal, free-space, and topology certificates. This enables both topology-efficient reconstruction and large-hole scene repair, including parking-lot ground voids, while preventing unsupported prior hallucination.
```

Most important scientific claim:

```text
The method is not a better pruning heuristic; it is a counterfactually certified surface-repair optimizer.
```

