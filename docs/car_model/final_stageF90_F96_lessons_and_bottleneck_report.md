# F90-F96 Repair Lessons and Current Bottleneck Report

Date: 2026-05-05

This note records the repair lessons that were not yet captured in the earlier F85-F89 progress report. It focuses on the courtyard repair branch because this is where the accepted F82 fixed policy still exposes the clearest gap between visual/RGB gains and strict parent-Pareto geometry validation.

## Current Status

The work is not fully solved. The accepted baseline remains F82 fixed adaptive policy v5. F90-F96 added several recovery mechanisms and produced meaningful improvements, but no candidate has passed the strict parent-Pareto gate against F82 on courtyard.

The strongest current candidate is F95:

| Run | Main change | Gate status | RGB | Per-view | Geometry |
|---|---|---:|---|---|---|
| F90 | vertex checkpoint anchor 0.02, 26000->28000 | rejected | PSNR +0.0847, SSIM +0.0086, LPIPS -0.0018 | 1 negative view, min -0.00018 | AbsRel, depth, normal worse |
| F91 | strong vertex checkpoint anchor 2.0 | rejected | PSNR/SSIM still better, LPIPS slightly worse | 2 negative views | AbsRel slightly better, depth/normal worse |
| F92 | vertex LR = 0, appearance-only recovery | rejected | PSNR/SSIM better, LPIPS worse | 2 negative views | depth better, normal worse |
| F93 | vertex LR = 0 and weight LR = 0 | rejected | same as F92 | 2 negative views | same as F92 |
| F94 | shorter 1000-iter recovery with vertex anchor | rejected | PSNR +0.0686, SSIM +0.0064, LPIPS -0.0021 | 1 tiny negative view | depth/normal worse |
| F95 | render-space depth+normal checkpoint anchor 0.01 | rejected | PSNR +0.0780, SSIM +0.0067, LPIPS -0.0013 | 0 negative views | normal better, depth/AbsRel worse |
| F96 | stronger render-depth anchor 0.1, normal anchor 0.01 | rejected | PSNR +0.0794, SSIM +0.0066, LPIPS -0.0012 | 0 negative views | depth worse, normal no longer better |

F95 is the first candidate in this round that simultaneously improves RGB mean metrics, improves every fixed per-view PSNR sample, and improves normal angle over F82. It still fails because sparse depth/AbsRel becomes worse.

## Code and Interface Changes Added

Three commits were pushed to `spcarnet/main`:

- `03aa7d4 Add checkpoint geometry anchor recovery loss`
  - Adds a vertex-space checkpoint anchor.
  - Purpose: keep optimized vertices close to the loaded F82 checkpoint during teacher/render recovery.
- `8b51963 Expose recovery optimizer learning rates`
  - Adds wrapper-level control for `lr_triangles_points_init`, `feature_lr`, and `weight_lr`.
  - Purpose: allow geometry-frozen or appearance-only recovery without editing training code.
- `3b76e52 Add checkpoint render geometry anchor`
  - Adds render-space checkpoint depth/normal distillation from the loaded checkpoint.
  - Purpose: directly constrain the rendered geometry surfaces used by recovery, closer to the actual evaluation failure mode than raw vertex L2.

## What Worked

Render-space normal anchoring is useful.

F95 improved the normal gate while keeping RGB and per-view improvements:

- PSNR: 12.1986 -> 12.2766
- SSIM: 0.30865 -> 0.31532
- LPIPS: 0.56669 -> 0.56540
- per-view min delta: +0.01276, negative views: 0
- normal mean angle: 40.2157 -> 40.1670

This is the clearest evidence so far that a geometry-aware recovery mechanism can improve visual quality without necessarily destroying normal consistency.

Shorter recovery is better than long blind recovery for RGB/per-view stability.

F94/F95 at 27000 were more useful than the 28000 F90/F91 style runs. Longer recovery tends to keep improving aggregate RGB but increases the chance of geometry drift.

Checkpoint anchoring must be applied in render space, not only parameter space.

Vertex L2 anchoring alone did not align well with the failure metric. Weak vertex anchor did not stop depth/normal drift; strong vertex anchor reduced useful appearance gains and damaged LPIPS/per-view stability. Render-space anchoring is more promising because it acts on the projected quantities used by the validation logic.

Per-view robustness can be fixed.

F87-F89 already showed 0 negative per-view on courtyard but lost strict geometry. F95 now gets 0 negative per-view while also improving normal. This means the per-view brittleness is no longer the main unresolved blocker.

## What Failed

Stronger vertex anchoring is not a solution.

F91 improved AbsRel slightly, but harmed LPIPS, normal, and per-view robustness. This shows that directly increasing the vertex anchor weight trades away visual quality and does not reliably fix sparse depth.

Freezing vertices is not enough.

F92 froze vertex LR and improved depth/AbsRel, but normal and LPIPS degraded. F93 additionally froze vertex weight and produced nearly identical behavior. The failure is therefore not explained only by vertex displacement or opacity/weight drift.

Naively increasing render-depth anchor does not fix depth.

F96 raised render-depth anchor from 0.01 to 0.1. It preserved RGB and per-view gains, but depth/AbsRel became worse than F95 and normal regressed. This means the current render-depth anchor is not correctly aligned with the sparse COLMAP depth gate, or the sampled gate points are dominated by locations where dense render-depth preservation is the wrong target.

Teacher-render recovery alone is insufficient.

Teacher render constraints reliably improve RGB/per-view metrics, but they can move geometry away from sparse COLMAP consistency. Teacher RGB supervision must be paired with a depth mechanism targeted to the actual failing sparse correspondences, not just a global rendered depth anchor.

## Current Bottleneck

The remaining bottleneck is sparse depth/AbsRel on courtyard. F95 demonstrates that RGB quality, per-view robustness, LPIPS, and normal can all be improved at the same time. The strict blocker is:

- AbsRel: F82 0.3018837 vs F95 0.3034414
- Depth MAE: F82 3.3398725 vs F95 3.3787072

The failure is small in absolute terms but decisive under the current parent-Pareto gate. It is also not solved by simply increasing render-depth regularization.

The likely reason is mismatch between three depth notions:

1. Render-space dense `surf_depth` used by checkpoint render anchoring.
2. Sparse COLMAP correspondence depth used by the geometry gate.
3. RGB teacher-induced changes that improve image reconstruction but alter geometry around sparse points.

The current depth anchor supervises dense rendered depth over training views, but the gate evaluates sparse projected points and can be sensitive to a small subset of views or correspondence clusters. A global depth anchor can preserve average surfaces while still worsening the gate-critical sparse points.

## Lessons for Next Work

Do not continue global lambda sweeps as the main strategy.

F90-F96 show that global weights are not enough. We can move RGB, normal, and depth in different directions, but the depth failure is localized and gate-specific. Continuing to scan global `depth_anchor_lambda` values risks overfitting and wasting GPU time.

Next diagnostic should be per-view and per-correspondence.

The next required tool should compare F82 vs F95 at the sparse depth correspondence level:

- Which test/train views contribute most to the depth MAE/AbsRel regression?
- Which projected COLMAP points have the largest candidate-parent regression?
- Are regressions concentrated near boundaries, low-alpha/low-weight surfaces, far-depth regions, or occlusion/disocclusion areas?
- Are the failing points present in training views used by the render-depth anchor?

The depth fix should target failing sparse geometry points.

A better policy is likely:

- keep F95 as the visual/normal repair base;
- add a checkpoint-vs-current sparse depth rollback or sparse-depth targeted loss on the exact correspondence distribution used by the gate;
- apply it only where the candidate becomes worse than the checkpoint by a margin;
- avoid penalizing depth everywhere, because global dense depth anchoring did not align with the gate.

The recovery policy should become a measured decision rule, not a manual parameter choice.

A robust fixed policy could be:

1. Run compact F82-style policy.
2. Run short teacher + render-normal anchored recovery.
3. Evaluate a small training-view sparse-depth sentinel against the checkpoint.
4. If sentinel depth worsens, activate targeted sparse rollback only on regressed correspondence clusters.
5. Accept only if RGB/per-view/normal gains remain nonnegative and sparse depth sentinel is nondegrading.

This is closer to an intelligent repair policy than a scene-specific parameter search.

## Evidence and Run IDs

W&B runs from this round:

- F90 courtyard anchor0.02 26000->28000: `tgpxhoqt`
- F91 courtyard anchor2.0 26000->28000: `u14y4skf`
- F92 courtyard vertexlr0 26000->28000: `6m6pqdjq`
- F93 courtyard vertexlr0 weightlr0 26000->28000: `dmr47qw0`
- F94 courtyard anchor0.02 26000->27000: `76gb1yxg`
- F95 courtyard rendergeom0.01 26000->27000: `y9yl906t`
- F96 courtyard renderdepth0.1 normal0.01 26000->27000: `frb6c9rx`

Important output files:

- `outputs/carnet/meshsplatopt/final_stageF95_render_geometry_anchor_repair/courtyard_vs_f82_pareto_gate.json`
- `outputs/carnet/meshsplatopt/final_stageF96_render_geometry_anchor_repair/courtyard_vs_f82_pareto_gate.json`
- `outputs/carnet/meshsplatopt/final_stageF95_render_geometry_anchor_repair/courtyard/adaptive_global_policy_v5_teacher0p001_sparse0p001_rendergeom0p01_27000_seed0/recovery_model/geometry_eval_colmap/iter_27000_max500.json`

## Recommended Next Step

Build a sparse-depth regression analyzer comparing F82 vs F95. It should emit a table of per-view and per-correspondence deltas, plus a candidate mask for targeted rollback. Do this before launching more full runs.

Only after that diagnostic should the next medium run be started. The next run should be F95 plus targeted sparse-depth rollback, not another uniform global depth-anchor sweep.

