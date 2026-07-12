# SPCarNet v106 POD-MoE Mentor Technical Report

Date: 2026-06-25

Purpose: a slide-ready technical report for mentor discussion. This version reflects the completed v106 full9 run and should supersede the earlier partial v106 drafts.

## 1. One-Slide Summary

The current representation-level candidate is **v106 POD-MoE base-preserve**.

Plain-language idea:

```text
MeshSplatting gives a base render.
v104c learns one stable residual field on mesh triangles.
v106 keeps that stable base and adds two small, guarded residual experts:
  detail expert + occlusion-boundary expert.
```

Main result:

- v106 is positive against the local v104c representation-field anchor on `9 / 9` selected Mip-NeRF360 scenes.
- Mean full9 gain over v104c is small: `+0.002181 PSNR`, `+0.000103 SSIM`, `-0.000112 LPIPS`.
- Compared with local clean MeshSplatting, v106 still inherits the much larger v104c gain: `+0.679598 PSNR`, `+0.011812 SSIM`, `-0.019185 LPIPS`.
- The result is stable and useful for a method story, but it is not yet a large paper-level breakthrough.

Safe headline:

> v106 turns the single residual-field anchor into a conservative surface-attached mixture of detail and occlusion-boundary experts. It consistently improves the local full9 v104c field, but the small effect size means the next research step must improve expert reliability rather than tune parameters.

## 2. Method Decomposition

### 2.1 Clean MeshSplatting

Clean MeshSplatting is the local baseline checkpoint render. It does not attach an explicit residual correction to mesh triangles.

Role in the report:

- baseline row for local comparison;
- same full9 scenes and same evaluator;
- selected local clean mean: `25.151682 / 0.749018 / 0.287621`.

### 2.2 v104c Shrink View-Affine Field

v104c is the current stable representation-field anchor.

It fits a compact residual function per triangle:

```text
[1, barycentric_u, barycentric_v, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

The shrink logic prevents poorly supported view-affine fits from becoming unstable. This makes v104c reliable across full9:

```text
v104c mean: 25.829099 / 0.760727 / 0.268548
```

### 2.3 v106 POD-MoE Base-Preserve

v106 keeps the v104c-like base, then adds two residual experts:

| module | role | why it exists |
|---|---|---|
| base field | stable shrink view-affine residual | preserves v104c's reliable MSE direction |
| detail expert | high-frequency residual detail | one low-order field can smooth texture residuals |
| occlusion-boundary expert | correction near visibility/depth/triangle boundaries | boundary errors are structured and different from texture |
| reliability + MSE scale | damp unsafe expert deltas | experts should only act where evidence says they help |
| base-preserve renderer | adds experts without replacing the base | previous POD variants damaged PSNR by suppressing the base |

Runtime residual:

```text
adapted residual =
  base residual
  + weighted detail expert delta
  + weighted occlusion-boundary expert delta
```

The field is loaded by `render.py` with:

```text
basis_type: affine_barycentric_viewdir_pod_mixture
field_variant: pod_moe
pod_base_keep_mode: base_preserving_boundary
```

## 3. Quantitative Results

### 3.1 Full9 Mean

| method | scenes | PSNR | SSIM | LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | 0.000000 | 0.000000 | 0.000000 |
| v104c shrink view-affine | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 | +0.011709 | -0.019073 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 | +0.011812 | -0.019185 |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | +1.329628 | +0.034657 | -0.063316 |

### 3.2 v106 vs v104c Per Scene

| scene | v106 PSNR | dPSNR | v106 SSIM | dSSIM | v106 LPIPS | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 23.719175 | +0.001526 | 0.675086 | +0.000115 | 0.313405 | -0.000098 |
| flowers | 20.077723 | +0.001879 | 0.531240 | +0.000163 | 0.374393 | -0.000080 |
| garden | 25.790945 | +0.002851 | 0.799382 | +0.000119 | 0.174480 | -0.000104 |
| stump | 25.460457 | +0.001146 | 0.714661 | +0.000061 | 0.282135 | -0.000078 |
| treehill | 21.245092 | +0.001329 | 0.578518 | +0.000099 | 0.384177 | -0.000121 |
| room | 29.600351 | +0.002516 | 0.891889 | +0.000051 | 0.230616 | -0.000048 |
| counter | 27.499645 | +0.001577 | 0.867521 | +0.000102 | 0.238847 | -0.000139 |
| kitchen | 28.772043 | +0.001595 | 0.881652 | +0.000062 | 0.187815 | -0.000206 |
| bonsai | 30.316090 | +0.005213 | 0.907520 | +0.000154 | 0.230050 | -0.000136 |
| mean | 25.831280 | +0.002181 | 0.760830 | +0.000103 | 0.268435 | -0.000112 |

Evidence files:

- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
- source reports under `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports`

## 4. MSE-Direction Diagnostics

The most useful diagnostic is whether v106's residual delta moves rendered images toward GT relative to v104c.

Known full9 subset diagnostics persisted in the repo:

| scene | test views | improved views | worse views | mean delta MSE |
|---|---:|---:|---:|---:|
| bicycle | 25 | 25 | 0 | -0.00000149 |
| flowers | 22 | 22 | 0 | -0.00000408 |
| garden | 24 | 23 | 1 | -0.00000157 |
| stump | 16 | 16 | 0 | -0.00000080 |
| treehill | 18 | 16 | 2 | -0.00000212 |
| room | 39 | 38 | 1 | -0.00000059 |

Hard-triad diagnostics from the v106 log:

| scene | test views | improved views | worse views | mean delta MSE |
|---|---:|---:|---:|---:|
| counter | 30 | 23 | 7 | -0.00000026 |
| kitchen | 35 | 30 | 5 | -0.00000050 |
| bonsai | 37 | 36 | 1 | -0.00000017 |

Interpretation: v106's corrections are small, but the direction is broadly healthy. This supports v106 as a stable step beyond v104c, while also showing why the visual difference is subtle.

## 5. Qualitative Evidence

Contact sheets compare:

```text
GT | v104c baseline | v106 candidate | |v104c-GT| error | |v106-GT| error
```

Generated assets:

- `docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.png`
- `docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png`
- `docs/car_model/assets/v106_qualitative/treehill_frame00010_bestcrop_contact_sheet.png`
- `docs/car_model/assets/v106_qualitative/room_frame00029_bestcrop_contact_sheet.png`

Recommended PPT usage:

1. Use one full result table first, because the quantitative sign is clearer than the raw RGB difference.
2. Show the crop panel with the two error maps. The RGB difference between v104c and v106 is intentionally small because v106 is base-preserving.
3. State explicitly that v106 is a conservative improvement, not a visually dramatic endpoint.

## 6. Fairness and Claim Boundaries

What we can safely say:

- v106 is a real train/eval/render pipeline change, not just a README edit.
- v106 full9 evaluation is complete for the selected local scene set.
- v106 is positive against the local v104c representation-field anchor on every selected scene and on all three image metrics.
- v106 is also above the local clean MeshSplatting baseline because it inherits the v104c representation-field improvement.

What we should not overclaim:

- v106 is not a large-margin improvement.
- v106 is not yet closing the endpoint/reference gap.
- v106 still depends on target-camera sidecar/distilled evidence in the current diagnostic pipeline, so it should be described carefully rather than as a pure train-only unseen-camera generalization result.
- Geometry/triangle reduction is not improved by v106 itself. Geometry reduction remains associated with earlier Phase-J/compaction lines, not this POD-MoE residual-field step.

## 7. Why v107 Is the Next Step

The main weakness in v106 is not a missing parameter setting. The problem is methodological:

```text
v106 fits expert coefficients and evaluates expert reliability
on the same weighted normal-equation evidence.
```

This can make the reliability certificate optimistic and caps how confidently we can sell the method.

The next mechanism-level improvement is:

```text
v107 cross-fitted POD-MoE expert reliability

1. split target views into even / odd groups;
2. fit detail and boundary experts on one split;
3. score their reliability and MSE scale on the held-out split;
4. swap directions and combine conservatively;
5. keep the same render-time tensor format for fair ablation.
```

Expected benefit:

- more honest expert reliability;
- less self-evaluation bias;
- stronger paper story because the method becomes a reliability-certified expert mixture rather than another small post-hoc adjustment;
- a clean ablation: v104c vs v106 self-scored POD-MoE vs v107 cross-fitted POD-MoE.

## 8. Suggested PPT Structure

1. Problem: MeshSplatting has no triangle-level mechanism for recurring residual repair.
2. v104c anchor: one stable shrink view-affine residual field gives a strong local full9 gain over clean.
3. v106 idea: keep the v104c base, add detail and boundary experts with conservative weights.
4. Quantitative result: v106 improves all 9 scenes over v104c, but by small margins.
5. Qualitative result: show contact sheet and error maps; emphasize subtle but consistent correction.
6. Honest limitation: effect size and target-sidecar diagnostic boundary.
7. Next step: v107 cross-fitted expert reliability to move from small stable gains to stronger research evidence.

