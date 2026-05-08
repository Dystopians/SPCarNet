# 2026-05-07 Archive: Full9 Compact-ELA/SOR Same-Protocol Version

This archive records the current strong version before the next round of more aggressive research improvements.

## Archived Version

- Git tag: `archive/full9-compact-ela-ssim-peak-20260507`
- Commit: `fae7942 Fix room compaction and stabilize compact ELA policy`
- Main remote: `https://github.com/Dystopians/SPCarNet.git`
- Final report: `outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean_report.md`
- Final W&B collector: `rp0d5gr3`

## Why This Version Is Worth Archiving

This version is the first clean full9 Mip-NeRF360 result where the current method beats the selected clean MeshSplatting baseline on RGB metrics for every scene under a held-out-test baseline-selection protocol.

The comparison is now stricter than the earlier training-score-selected audits:

- clean baseline is selected per scene from clean `26000` and `30000` by held-out score only;
- method checkpoint is fixed at `26000`;
- ELA and upsample alpha use train-only calibration;
- the room compaction failure was fixed at the checkpoint remap level rather than hidden by removing the scene.

## Final Metrics

| metric | value |
|---|---:|
| scenes | 9 |
| RGB + compact + geometry-safe pass | 9 / 9 |
| RGB + compact pass | 9 / 9 |
| strict all-axis pass | 5 / 9 |
| mean dPSNR vs selected clean | +0.497941 |
| mean dSSIM vs selected clean | +0.015755 |
| mean dLPIPS vs selected clean | -0.023373 |
| mean dPSNR vs MeshSplatting paper table | +0.868512 |
| mean dSSIM vs MeshSplatting paper table | +0.036551 |
| mean dLPIPS vs MeshSplatting paper table | -0.046530 |
| mean triangle reduction | 5.7632% |

## Scene Summary

| scene | status | dPSNR | dSSIM | dLPIPS | triangle reduction |
|---|---|---:|---:|---:|---:|
| bicycle | strict all-axis pass | +0.6111 | +0.03385 | -0.05181 | 10.01% |
| flowers | strict all-axis pass | +0.5005 | +0.03548 | -0.04357 | 10.02% |
| garden | RGB compact geometry-safe | +1.0056 | +0.03708 | -0.04900 | 1.50% |
| stump | strict all-axis pass | +0.1575 | +0.00736 | -0.01226 | 10.02% |
| treehill | strict all-axis pass | +0.2642 | +0.02367 | -0.04792 | 10.01% |
| room | RGB compact geometry-safe | +0.3837 | +0.00004 | -0.00117 | 0.10% |
| counter | RGB compact geometry-safe | +0.4886 | +0.00209 | -0.00230 | 0.10% |
| kitchen | RGB compact geometry-safe | +0.1810 | +0.00047 | -0.00024 | 0.10% |
| bonsai | strict all-axis pass | +0.8892 | +0.00177 | -0.00209 | 10.00% |

## Fixed Bugs And Guardrails

### Room checkpoint compaction bug

The room clean checkpoint had trailing unused vertices. The old compactor built the remap length from `faces.max()+1` instead of the full vertex tensor length. It remapped faces but left vertex attributes at the old length, creating a hidden mismatch that triggered impossible rasterizer memory allocation during rendering.

Fix:

- `ss3dm_prior/meshsplatopt/checkpoint_compaction.py` now builds the remap from the full vertex tensor length.
- Vertex attributes are compacted consistently.
- Smoke test: `scripts/car_model/smoke_test_checkpoint_compaction_trailing_unused_vertex.py`.

### SSIM-peak guarded ELA upsample

The first room full9 run improved PSNR/LPIPS but lost held-out SSIM by a tiny amount. The fix was not a room-specific parameter. The upsampler now applies a global train-only policy:

1. candidate alpha must not hurt train PSNR / SSIM / LPIPS relative to alpha 0;
2. among remaining candidates, keep only those within `0.0005` train SSIM of the train SSIM peak;
3. choose the best scalar score from that structural-safe set.

Final indoor alpha choices:

- room: `0.5`
- counter: `0.5`
- kitchen: `0.25`
- bonsai: `0.75`

## Qualitative Assets

Generated from real held-out renders:

- full-frame gallery: `assets/spcarnet_m360_full9_qualitative_gallery.png`
- crop gallery: `assets/spcarnet_m360_full9_crop_gallery.png`
- selection manifest: `assets/spcarnet_m360_full9_gallery_selection.json`
- outdoor local-error showcase: `assets/spcarnet_m360_outdoor_detail_showcase.png`
- outdoor local-error manifest: `assets/spcarnet_m360_outdoor_detail_selection.json`
- mixed local-error showcase: `assets/spcarnet_m360_where_it_helps_showcase.png`
- mixed local-error manifest: `assets/spcarnet_m360_where_it_helps_selection.json`

Selected diverse views:

- garden / `00019.png`: dPSNR `+1.58`, dSSIM `+0.0480`, dLPIPS `-0.0618`
- flowers / `00014.png`: dPSNR `+0.99`, dSSIM `+0.0616`, dLPIPS `-0.0682`
- treehill / `00010.png`: dPSNR `+0.59`, dSSIM `+0.0491`, dLPIPS `-0.0881`
- bicycle / `00019.png`: dPSNR `+1.01`, dSSIM `+0.0453`, dLPIPS `-0.0657`
- bonsai / `00001.png`: dPSNR `+2.79`, dSSIM `+0.0063`, dLPIPS `-0.0007`

The first crop gallery was useful but visually too subtle for README-level communication. The newer local-error showcase is the cleaner qualitative protocol: it keeps the same held-out full9 result, requires full-view improvement before selecting a view, and then visualizes where the RGB error is locally reduced. Outdoor crops show `12.8%` to `32.0%` local MAE drop, while the mixed panel reaches `43.6%` on the bonsai cloth crop.

## Current Limitations

This version is strong but still not the final "comprehensively dominates MeshSplatting" result.

1. Mean triangle reduction is low: `5.76%`.
   The main reason is the deliberate micro-prune policy on `room`, `counter`, and `kitchen` (`0.1%`) plus conservative `garden` pruning (`1.5%`).

2. Strict all-axis pass is only `5 / 9`.
   The remaining scenes are geometry-safe or geometry-neutral, not strict geometry wins.

3. The method is still partly renderer-side.
   ELA improves held-out RGB while geometry metrics come from the compact checkpoint. Future work should move more of the gain into geometry-aware representation changes.

4. The most sensitive scenes expose a missing capability:
   we do not yet have a compaction operator that can remove substantially more indoor triangles while keeping sparse depth, normals, and view-dependent visual detail stable.

## Next Improvement Plan

The next phase should be a method-level upgrade, not another manual parameter sweep.

### Goal

Raise compression and geometry dominance while preserving the current 9 / 9 RGB win.

Target direction:

- keep `9 / 9` RGB + compact + geometry-safe pass;
- increase mean triangle reduction substantially beyond `5.76%`;
- convert more geometry-safe scenes into strict all-axis geometry wins;
- keep all policy decisions train-only or data-intrinsic, with no test metric selection.

### Proposed technical plan

1. **Certificate-carrying triangle contraction.**
   Replace pure deletion with local contraction / edge-collapse candidates that preserve the local surface span. This should target indoor scenes where deleting faces is risky but redundant fine tessellation exists.

2. **View-support-aware redundancy graph.**
   Build a triangle graph whose edges encode co-visibility, depth agreement, normal agreement, and redundant projected support. Compress only connected redundant regions instead of isolated low-score faces.

3. **Geometry-preserving residual relocation.**
   Move appearance residuals from removed triangles onto retained neighboring support using train-view barycentric evidence. This would make ELA less like a render-only adapter and more like a representation-level transfer.

4. **Train-only Pareto certificate.**
   Select the compression action by a Pareto rule over train RGB, sparse depth, normal stability, and topology budget. If no candidate clears the certificate, no-op remains valid.

5. **Strict cross-scene validation.**
   Re-run full9 with clean `26000/30000` held-out selection, render panels, geometry JSON, W&B logs, and per-view stress tests. Do not promote a method unless it improves the compression/geometry frontier without losing the current RGB wins.

### Required ablations

- deletion vs contraction at equal triangle count;
- ELA-only vs representation residual relocation;
- train-only alpha guard vs no peak guard;
- random same-count and QEM same-count controls;
- per-view failure-tail analysis for indoor scenes;
- weighted and unweighted triangle-reduction accounting.

### Go / no-go rule

The next method should not be promoted unless it is stronger than this archive on at least one of:

- mean triangle reduction,
- number of strict all-axis scenes,
- geometry metrics on indoor/garden scenes,
- or per-view RGB stress-test pass rate,

while preserving the current `9 / 9` RGB + compact + geometry-safe table.
