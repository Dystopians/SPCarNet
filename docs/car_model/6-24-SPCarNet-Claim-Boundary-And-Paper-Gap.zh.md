# SPCarNet Claim Boundary And Paper Gap

Date: 2026-06-24

This note is the current claim boundary for mentor/PPT use and for future paper
planning. It prevents the strongest verified result from being overstated.

## Safe Main Claim

SPCarNet is best described as:

```text
Evidence-certified post-training repair and compaction for MeshSplatting checkpoints.
```

The method starts from a trained MeshSplatting checkpoint, builds train/policy-val
surface evidence, then decides which geometry can be compacted and which
surface-bound residuals can be repaired. Held-out test views are used for final
evaluation only.

## Current Verified Result

The promoted endpoint is **Phase-J**.

Under the local Mip-NeRF360 full9 selected-clean MeshSplatting protocol:

- scene-level PSNR/SSIM/LPIPS strict wins: `9 / 9`;
- held-out view strict wins: `244 / 246`;
- mean delta: `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS;
- mean triangle reduction: `7.6479%`;
- geometry-safe scenes under the current closure audit: `9 / 9`;
- strict sparse-geometry wins: `6 / 9`.

Recommended evidence:

```text
docs/car_model/6-24-SPCarNet-Current-Complete-Method-Experiment-Report-With-Render-Comparisons.zh.md
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
```

## Claims That Are Not Yet Safe

Do not claim:

- SPCarNet fully solves MeshSplatting;
- SPCarNet is a fully baked replacement renderer;
- the paper-level representation loop is complete;
- the MeshSplatting paper-table comparison is fully same-protocol verified;
- geometry is strictly better on every scene and every geometry metric;
- v82b/v83 is promoted as a paper endpoint despite failed or mixed validation.

The precise current status is:

```text
Phase-J local full9 RGB loop: COMPLETE
paper-level representation loop: NOT COMPLETE
paper-table same-protocol loop: NOT COMPLETE
```

## Why Visual Gains Can Look Subtle

Phase-J gains are real but often local. They mainly reduce residual-level errors
on high-frequency or surface-specific details. Full-frame comparisons are useful
for fairness, but the best visual explanation is:

```text
GT crop / clean MeshSplatting crop / SPCarNet crop / error reduction map
```

For paper-quality qualitative evidence, the current selected crop showcase must
be complemented by:

- a fixed-rule or random held-out qualitative grid;
- the two non-strict held-out views from the `244 / 246` audit;
- side-by-side full-frame fairness panels;
- rate-distortion tables with triangles, vertices, storage, and FPS.

## Representation-Level Gap

The current strongest visual/RGB endpoint is still a render-time self-auditing
adapter. It is surface-bound and train-evidence gated, so it is not ordinary
2D post-processing, but it is also not yet a fully baked representation-level
method.

Recent representation-level probes show the gap:

- v64 fixed auto bin-alpha is the best broad fixed reference, but full9 gains
  are very small;
- v81 view-conditioned basis regresses versus the anchor;
- v82 patch-mixture teacher basis runs but falls back to legacy teacher and
  regresses;
- v82b gives a strict `counter` micro-win, but raw hard-triad validation fails
  promotion on `kitchen/bonsai`;
- v83 reaches `26.756147385 / 0.862125337 / 0.251688808` on `counter`, improving
  PSNR/LPIPS but regressing SSIM versus the v56/v64/v79 anchor.

v82b/v83 are therefore validation diagnostics, not paper endpoints.

## Required Next Evidence

Minimum next steps before a stronger paper claim:

1. Treat v83 as completed mixed evidence, not as a promoted endpoint:

```text
v83:    26.756147385 / 0.862125337 / 0.251688808
anchor: 26.756130219 / 0.862126231 / 0.251691371
delta:  +0.000017166 / -0.000000894 / -0.000002563
```

2. If another fixed policy wins all three counter metrics, run hard-triad with
   the identical policy:

```text
counter, kitchen, bonsai
```

3. If hard-triad is non-regressive, run full9:

```text
bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai
```

4. Save all commands, W&B run IDs, configs, results, per-view metrics, audit
   files, and qualitative renders.

5. Add the ablation table:

```text
clean MeshSplatting
compact-only
ELA without gate
ELA with gate
fallback
Phase-J full
new representation-level method
```

## Current Verdict

The current work is strong enough for a mentor update and a serious research
direction, but not yet a fully closed top-conference submission package.

The honest story is:

```text
We have a strong train-evidence post-training repair/compaction result over
local clean MeshSplatting. The next milestone is to internalize this reliability
into a broadly validated representation-level endpoint.
```
