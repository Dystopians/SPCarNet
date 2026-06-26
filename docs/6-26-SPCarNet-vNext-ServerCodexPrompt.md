# SPCarNet vNext Server Codex Prompt

You are working in the SPCarNet / MeshSplatting optimization repository. The goal is to turn the current Phase-J/v106 line into a top-conference-grade method, not to make another small gate or alpha tweak.

## Core Direction

Implement and validate a new representation-level method:

**Evidence-Certified Residual Surface Texturing for MeshSplatting**

The method should distill the strong Phase-J render-time ELA residual repair into a persistent MeshSplatting-compatible surface representation. The target representation is a parent-preserving residual field attached to mesh faces/UV/barycentric coordinates, with adaptive surface capacity and strict train-only certification.

## Current Local Facts To Respect

- Phase-J is the strongest broad RGB endpoint, but it is a render-time guarded ELA portfolio, not a baked representation.
- Phase-J headline: 9/9 scene strict RGB wins vs selected clean MeshSplatting, 244/246 held-out view strict RGB wins, mean +1.3311 PSNR / +0.0347 SSIM / -0.0634 LPIPS, and 7.6479% average triangle reduction.
- v106 POD-MoE base-preserve is the current verified representation-style quality line, but its incremental gain over v104c is tiny. Do not spend this task merely adding another v106 expert unless it is part of the larger residual texturing method.
- v110/v113b showed that a train/odd gate can still falsely accept unsafe candidates; target/test GT must never be used for selection, and unsafe candidates must fall back to the parent.

Important local references:

- README.md
- docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md
- docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md
- docs/car_model/6-25-v106-PODMoE-Technical-Report-Draft.md
- scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
- scripts/car_model/ecsr_apply_surface_residual_lumigraph_adapter.py
- scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py
- utils/evidence_lumigraph_adapter.py

## What To Build

Build a new vNext branch, tentatively named:

`vNext_certified_residual_surface_texture`

The method should have four parts.

1. **Residual Teacher Cache**
   - For train views only, compute parent render, GT-parent RGB residual, depth/normal/face/barycentric evidence, and optionally leave-one-out Phase-J/ELA teacher residual.
   - Do not read held-out test GT except in final evaluation scripts.
   - Save per-scene evidence manifests with exact source paths, hashes or timestamps, split names, and no-test-GT audit fields.

2. **Adaptive Residual Surface Texture**
   - Attach residual capacity to visible mesh faces using face id + barycentric/UV bins.
   - Support variable texture resolution per face or face group, selected from train evidence.
   - Learn or fit a residual decoder with inputs such as face id, barycentric/UV bin, view direction, normal/depth/boundary features, train support count, residual variance, and parent color.
   - Output residual RGB plus a confidence/alpha value.
   - Preserve parent output by default: final RGB must be `parent + confidence * residual`.

3. **Capacity Reallocation**
   - Do not sell this as generic compression. Use geometry-safe compaction only to free low-value surface budget.
   - Allocate residual texture capacity to residual-hot, multiview-consistent, surface-addressable regions.
   - Report triangle count, residual texture size, parameter count, render-time overhead, and model storage.

4. **Train-Only Safety Certificate**
   - Split train into fit/policy-val folds. Candidate generation may use fit views; selection and alpha/confidence thresholds must be decided on policy-val only.
   - Use per-scene and per-region gates: PSNR/SSIM/LPIPS, MSE direction, positive-view fraction, CVaR/tail risk, support count, residual variance, target camera support/OOT, and parent-candidate distance.
   - If the candidate is not certified, write exact no-op parent/fallback outputs and a machine-readable rejection reason.

## Implementation Strategy

Prefer extending the existing surface residual atlas/region texture adapter rather than starting from scratch:

- Reuse `ecsr_apply_surface_residual_region_texture_adapter.py` for face/UV residual atlas fitting and policy-val infrastructure.
- Reuse `ecsr_apply_surface_residual_lumigraph_adapter.py` and `utils/evidence_lumigraph_adapter.py` to build the Phase-J/ELA teacher cache.
- Reuse `meshsplatopt_v109_render_realized_parent_gate.py` ideas for parent-preserving output and OOT fallback, but improve it with per-region uncertainty rather than only render-distance masks.

Add scripts with explicit names, for example:

- `scripts/car_model/ecsr_build_residual_teacher_cache_vnext.py`
- `scripts/car_model/ecsr_apply_certified_residual_surface_texture_vnext.py`
- `scripts/car_model/run_vnext_certified_residual_texture_full9.py`
- `scripts/car_model/assemble_vnext_certified_residual_texture_report.py`

## Required Ablations

Run or prepare scripts for these comparisons:

- clean MeshSplatting parent
- Phase-F compact parent
- Phase-J render-time ELA teacher
- v104c residual field
- v106 POD-MoE base-preserve
- vNext surface texture without adaptive capacity
- vNext surface texture without uncertainty/certificate
- full vNext certified residual surface texture

## Success Criteria

Minimum useful research milestone:

- full9 completed with fixed train-only policy
- no target/test GT used for branch, alpha, texture capacity, fallback, or thresholds
- 9/9 non-regressive/tie vs parent under PSNR/SSIM/LPIPS
- at least 6/9 strict scene RGB wins vs clean MeshSplatting
- mean gain vs clean at least +0.5 PSNR, with LPIPS improvement
- meaningful reduction of the gap to Phase-J, while no longer relying on render-time ELA support warping

Paper-grade milestone:

- representation-level vNext captures a large fraction of Phase-J: ideally >=70% of Phase-J mean PSNR gain over clean
- strict no-regression certificate survives per-view/tail audit
- credible qualitative panels show high-frequency and boundary residual repair
- report includes render speed, storage size, triangle count, texture/residual parameter budget, and fallback rate

## Do Not Do

- Do not tune on held-out test GT.
- Do not present 7.6479% triangle reduction as a compression-paper-level result.
- Do not claim raw score-based pruning or raw geometry residual mapping as novel.
- Do not make another small v106 alpha/gate variant unless it materially advances persistent residual surface texturing.
- Do not silently accept unsafe candidates; fallback must be explicit and auditable.

## Deliverables

1. A technical report under `docs/car_model/` with:
   - method diagram text,
   - exact protocol,
   - train/test leakage audit,
   - full9 table,
   - ablations,
   - limitations.

2. Machine-readable summaries under `outputs/...`:
   - per-scene metrics,
   - per-view metrics,
   - policy decisions,
   - fallback reasons,
   - size/speed accounting.

3. A small qualitative gallery:
   - GT,
   - clean MeshSplatting,
   - Phase-J teacher,
   - v106,
   - vNext,
   - error maps.

4. A final recommendation:
   - promote vNext,
   - keep Phase-J as teacher/upper bound,
   - or reject with diagnosed bottleneck.
