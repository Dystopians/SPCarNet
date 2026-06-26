# SPCarNet v106 POD-MoE Technical Report Draft

Date: 2026-06-25

Audience: mentor discussion and PPT conversion.

Scope: this draft summarizes the current v106 POD-MoE state from local repository files and local result artifacts only. It is intentionally conservative: v106 is promising, but full9 is not complete in the latest verified local assembly.

## 1. Slide-Ready Takeaway

Clean MeshSplatting renders the trained checkpoint directly:

```text
MeshSplatting checkpoint -> held-out test render -> PSNR / SSIM / LPIPS
```

SPCarNet adds a surface-addressed evidence layer:

```text
base MeshSplatting render
+ residual evidence attached to visible mesh triangles
+ guarded field / endpoint logic
-> adapted render
```

The current story has three layers:

| layer | role | current evidence | claim boundary |
|---|---|---|---|
| v101/v102 endpoint/reference | strongest quality ceiling using target-camera delta evidence | full9 endpoint/reference available from prior reports | not a vanilla MeshSplatting checkpoint; not train-only unseen-camera generalization |
| v104c shrink view-affine field | stable representation-field anchor | full9 complete: 9/9 scenes improve over selected local clean baseline | still below endpoint/reference |
| v106 POD-MoE base-preserve | current candidate that adds detail and occlusion-boundary experts on top of v104c-like base | 7/9 scenes currently verified in assembled local partial full9; all 7 improve over v104c by small margins | not final full9 headline until flowers and room are complete |

Safe headline:

> v106 POD-MoE upgrades v104c from one shrink view-affine residual into a base-preserving mixture of certified detail and occlusion-boundary residual experts. It has consistent small gains on the seven verified scenes, but needs the remaining two full9 scenes and qualitative panels before final promotion.

## 2. Problem Being Solved

Clean MeshSplatting has no explicit mechanism for asking:

- Which visible triangle regions have repeatable multi-view residual error?
- Which residuals are texture/detail corrections, and which are occlusion-boundary corrections?
- When should an extra correction be damped or ignored because it increases MSE?
- Can endpoint-quality improvements be baked into a persistent surface field rather than used as per-camera endpoint deltas?

v104c answered part of this by fitting one low-order residual function per triangle. v106 tests the next hypothesis: one residual mode is too compressed, so split the residual into a stable base plus specialized experts.

## 3. Method Modules

### 3.1 Baseline: Clean MeshSplatting

Plain-language role: the unmodified trained model. It gives the base RGB render and the mesh/splat geometry that SPCarNet uses as an address space.

Technical role:

```text
checkpoint render at iteration 26000
held-out test split
metrics: PSNR, SSIM, LPIPS
```

In the current reports, the clean row is the selected local `ours_26000` baseline from `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k`.

### 3.2 v101/v102 Endpoint Reference

Plain-language role: the strongest quality reference. It shows how much improvement is available if we allow a target-camera endpoint delta/evidence path.

Technical role:

- v101 packages train-derived residual/depth/camera/hash evidence behind `render.py`.
- v102 preprojects endpoint deltas for validated target cameras and uses them as a fast reference/teacher.
- This is a ceiling row for the current field work, not a claim that the field has closed the gap.

Important caveat: v101/v102 are target-camera endpoint/reference mechanisms. They use no held-out target GT for policy selection in the reported field line, but they are not train-only unseen-camera generalization.

### 3.3 v104c Shrink View-Affine Field

Plain-language role: current stable field anchor. It stores a compact residual function on mesh triangles, instead of applying a separate endpoint delta image.

Technical role:

```text
input per visible pixel:
  triangle id
  barycentric coordinates u, v
  view direction x, y, z

per-triangle residual:
  [1, u, v, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

v104c stabilizes raw view-affine fitting by shrinking ill-supported or ill-conditioned view-affine coefficients toward a safer barycentric affine fallback. This is why it is the full9 anchor.

### 3.4 v106 POD-MoE Field

Plain-language role: keep the stable v104c-like base, then add two small expert corrections only where there is evidence.

Technical identity:

```text
field_variant: pod_moe
basis_type: affine_barycentric_viewdir_pod_mixture
builder_variant: v106_perceptual_occlusion_detail_moe
base: v104c-like shrink view-affine residual
experts: detail, occlusion_boundary
expert certificate: weighted_normal_equation_lambda_star
pod_base_keep_mode: base_preserving_boundary
```

Stored field tensors include:

```text
triangle_base_coefficients              [T, 6, 3]
triangle_expert_delta_coefficients      [T, 2, 6, 3]
triangle_expert_reliability             [T, 2]
triangle_expert_mse_scale               [T, 2]
triangle_occlusion_base_keep            [T]
triangle_expert_counts / view_counts    [T, 2]
```

The two experts are:

| expert | purpose | cue |
|---|---|---|
| detail | recover high-frequency/perceptual texture residuals that one low-order base may smooth out | luminance/detail score from base render and teacher delta |
| occlusion boundary | handle residuals near triangle/depth boundaries where visibility changes are important | boundary score from rendered triangle/depth discontinuity |

### 3.5 Base-Preserve Rendering

Old POD-MoE variants could let the boundary expert suppress or replace too much of the v104c-like base near occlusion boundaries. That improved some perceptual metrics but made MSE direction worse on most counter views.

v106 base-preserve changes the rendering logic:

```text
rendered residual =
  base residual
  + weighted detail expert delta
  + weighted occlusion-boundary expert delta
```

The key design decision is that the base residual remains present. The experts are additive corrections, not replacements for the stable base.

Runtime expert weighting is approximately:

```text
expert_weight =
  runtime_cue
  * triangle_expert_reliability
  * triangle_expert_mse_scale
  * view_gate
```

Then the render samples the weighted expert deltas and adds them to the base residual before clamping the adapted RGB.

### 3.6 Certificate and Fail-Closed Checks

The v106 runner checks artifact identity so stale or wrong fields do not silently pass:

- field type
- basis type
- builder variant
- field variant
- sha256
- min count / min views
- ridge, residual clip, view std floor
- rank tolerance and condition max
- gate source
- renderer scaling
- residual dtype
- endpoint method

This matters for mentor/PPT discussion because the result is a fixed policy, not an untracked manual rendering artifact.

## 4. Current Results

### 4.1 Stable Full9 Anchor: Clean vs v104c vs Endpoint

Verified from the v104c full9 report.

| method | scenes | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | selected local clean baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | current stable field anchor |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | quality ceiling |

| comparison | dPSNR | dSSIM | dLPIPS | interpretation |
|---|---:|---:|---:|---|
| v104c - clean | +0.677417 | +0.011709 | -0.019073 | full9 field improvement over selected clean |
| v104c - endpoint/reference | -0.652211 | -0.022949 | +0.044243 | remaining endpoint-to-field gap |

### 4.2 v106 Verified Partial Full9 Against v104c

Latest local assembly found:

```text
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_assembled_partial7_20260625.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_assembled_partial7_20260625.json
```

Status: `7 / 9` scenes available. Missing scenes: `flowers`, `room`.

| scene | source | v106 PSNR | v106 SSIM | v106 LPIPS | v104c PSNR | dPSNR | v104c SSIM | dSSIM | v104c LPIPS | dLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | full9 | 23.719175 | 0.675086 | 0.313405 | 23.717649 | +0.001526 | 0.674972 | +0.000115 | 0.313503 | -0.000098 |
| flowers | pending | TBD | TBD | TBD | 20.075844 | TBD | 0.531076 | TBD | 0.374473 | TBD |
| garden | full9 | 25.790945 | 0.799382 | 0.174480 | 25.788094 | +0.002851 | 0.799263 | +0.000119 | 0.174584 | -0.000104 |
| stump | full9 | 25.460457 | 0.714661 | 0.282135 | 25.459311 | +0.001146 | 0.714599 | +0.000061 | 0.282213 | -0.000078 |
| treehill | full9 | 21.245092 | 0.578518 | 0.384177 | 21.243763 | +0.001329 | 0.578418 | +0.000099 | 0.384298 | -0.000121 |
| room | pending | TBD | TBD | TBD | 29.597836 | TBD | 0.891837 | TBD | 0.230664 | TBD |
| counter | counter | 27.499645 | 0.867521 | 0.238847 | 27.498068 | +0.001577 | 0.867420 | +0.000102 | 0.238986 | -0.000139 |
| kitchen | hardtriad | 28.772043 | 0.881652 | 0.187815 | 28.770449 | +0.001595 | 0.881590 | +0.000062 | 0.188021 | -0.000206 |
| bonsai | hardtriad | 30.316090 | 0.907520 | 0.230050 | 30.310877 | +0.005213 | 0.907367 | +0.000154 | 0.230186 | -0.000136 |
| mean over available 7 | selected | 26.114778 | 0.774906 | 0.258701 | 26.112601 | +0.002177 | 0.774804 | +0.000102 | 0.258827 | -0.000126 |

Interpretation:

- The seven available v106 scenes all improve over v104c in PSNR, SSIM, and LPIPS.
- The effect size is small: mean `+0.002177 PSNR`, `+0.000102 SSIM`, `-0.000126 LPIPS` over v104c on the available seven scenes.
- Because `flowers` and `room` are missing, this is a partial full9 update, not a final full9 result.
- Several available rows are sourced from earlier counter/hard-triad artifacts, while four rows are from the full9 run root. This is acceptable for a progress update but should be replaced by one final assembled full9 table when all scenes are complete.

### 4.3 Hard-Triad Result and MSE Direction

Verified hard-triad scenes: `counter`, `kitchen`, `bonsai`.

| scene | v104c PSNR | v106 PSNR | dPSNR | v104c SSIM | v106 SSIM | dSSIM | v104c LPIPS | v106 LPIPS | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | 27.498068 | 27.499645 | +0.001577 | 0.867420 | 0.867521 | +0.000102 | 0.238986 | 0.238847 | -0.000139 |
| kitchen | 28.770449 | 28.772043 | +0.001595 | 0.881590 | 0.881652 | +0.000062 | 0.188021 | 0.187815 | -0.000206 |
| bonsai | 30.310877 | 30.316090 | +0.005213 | 0.907367 | 0.907520 | +0.000154 | 0.230186 | 0.230050 | -0.000136 |
| mean | - | - | +0.002795 | - | - | +0.000106 | - | - | -0.000160 |

MSE-direction diagnostic:

| scene | views | MSE-improved | MSE-worse | mean delta-MSE |
|---|---:|---:|---:|---:|
| counter | 30 | 23 | 7 | -0.00000026 |
| kitchen | 35 | 30 | 5 | -0.00000050 |
| bonsai | 37 | 36 | 1 | -0.00000017 |
| hard-triad total | 102 | 89 | 13 | negative mean |

Why this matters:

```text
delta_mse = MSE(v106, GT) - MSE(v104c, GT)
          = 2 * (v104c - GT) * (v106 - v104c)
            + (v106 - v104c)^2
```

Earlier POD-MoE variants had a serious direction problem on counter: only `4 / 30` views improved MSE and `26 / 30` worsened. Base-preserve changes that to `23 / 30` counter views improved, and `89 / 102` hard-triad views improved.

## 5. Geometry and Topology Boundary

v106 POD-MoE is an appearance residual field module. It does not itself prune, add, or remesh triangles.

Safe statement:

> v106 inherits the geometry/topology of the parent MeshSplatting or compact-parent checkpoint and improves the rendered appearance through a surface-attached residual field.

Unsafe statement:

> v106 itself provides new triangle-count reduction or geometry compression.

If a PPT needs geometry numbers, use a separate parent-checkpoint or PRISM/MeshPrior table and label it separately. Do not merge topology-control claims into v106 unless the exact same v106 plus topology-control run has been validated.

## 6. Qualitative Evidence Plan

Current limitation: this draft did not verify final v106 qualitative panels. Use the numerical tables as the current evidence and treat visuals as pending.

Recommended PPT panel format:

```text
GT | clean MeshSplatting | v104c | v106 POD-MoE base-preserve | endpoint/reference | error maps
```

Use three buckets:

| bucket | examples / source | purpose |
|---|---|---|
| best MSE-improved views | from per-scene delta-MSE JSON/MD | show where v106 actually reduces error |
| worst MSE views | counter `00009.png`, kitchen `00015.png`, bonsai `00028.png` from current notes | honestly show remaining risk |
| detail/boundary crops | select after expert activation/error-map review | explain what detail and boundary experts change |

Panel rules:

- Use the same crop coordinates for clean, v104c, v106, endpoint, and GT.
- Include absolute-error and delta-error maps, not RGB crops only.
- Caption each crop with scene, view id, crop coordinates, and local metric deltas.
- Keep endpoint/reference visually separate as a ceiling row.

## 7. Claim Boundaries for Mentor Discussion

### What We Can Say Now

- v104c is the stable full9 field anchor: 9/9 scenes improve over the selected local clean baseline.
- v106 base-preserve is a real representation change: it adds certified detail and occlusion-boundary experts to a v104c-like base.
- v106 has seven verified scenes in the latest local partial full9 assembly, and all seven improve over v104c on PSNR, SSIM, and LPIPS.
- v106 hard triad passes the fixed-policy gate and has healthier MSE-direction diagnostics than earlier POD-MoE variants.

### What We Should Not Say Yet

- v106 is the final full9 headline.
- v106 closes the endpoint/reference gap.
- v106 is train-only unseen-camera generalization.
- v106 is a vanilla MeshSplatting checkpoint without `render.py` field support.
- v106 provides new topology compression.
- Visual improvement is obvious in every full-frame render.
- The `/dev/shm` partial artifacts are durable paper artifacts.

## 8. Remaining Work Before Promotion

Minimum promotion gate:

1. Complete `flowers` and `room` under the same fixed v106 policy.
2. Reassemble the full9 table from one explicit source root and verify `9 / 9` scenes.
3. Compare v106 vs v104c and vs clean with strict scene win counts.
4. Run or collect delta-MSE diagnostics for all nine scenes.
5. Build qualitative crop/error-map panels for representative best and worst views.
6. Copy or regenerate final summaries under durable `outputs/` paths, not only `/dev/shm`.

Final full9 placeholder table:

| method | scenes OK | mean PSNR | mean SSIM | mean LPIPS | strict wins vs v104c | strict wins vs clean | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline | baseline | selected local clean |
| v104c shrink view-affine | 9 | 25.829099 | 0.760727 | 0.268548 | anchor | 9/9 | stable full9 field |
| v106 POD-MoE base-preserve | TBD: currently 7/9 verified | TBD | TBD | TBD | TBD | TBD | pending flowers and room |
| endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | ceiling | ceiling | target-camera endpoint/reference |

## 9. Suggested PPT Structure

1. Motivation: clean MeshSplatting lacks explicit evidence-aware residual correction.
2. Ladder: clean -> endpoint/reference -> v104c field -> v106 POD-MoE.
3. Module diagram: mesh triangle address, base residual, detail expert, boundary expert, certificate/gate.
4. v104c full9 anchor table.
5. v106 partial full9 table with `flowers` and `room` clearly marked pending.
6. Hard-triad MSE-direction table showing why base-preserve fixed the earlier POD-MoE failure mode.
7. Claim boundary slide: what is proved, what is pending, what not to claim.
8. Next-step slide: finish full9, build qualitative panels, durable artifacts.

## 10. Source Paths Used

Local docs:

```text
docs/car_model/6-25-v104c-Full9-PaperLoop-Technical-Report.md
docs/car_model/6-25-v106-PODMoE-BasePreserve-HardTriad-Log.md
docs/car_model/6-25-v106-PODMoE-Paper-Story-Draft.md
docs/car_model/6-25-v106-PODMoE-Review-And-Story-Draft.md
```

Local result artifacts:

```text
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_assembled_partial7_20260625.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_assembled_partial7_20260625.json
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_compare_20260625.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/
```

Code paths checked for module identity:

```text
scripts/car_model/run_v105_evidence_gated_mixture_scene.py
scripts/car_model/build_v105_evidence_gated_mixture_field.py
scripts/car_model/assemble_v106_basepreserve_full9_report.py
render.py
```
