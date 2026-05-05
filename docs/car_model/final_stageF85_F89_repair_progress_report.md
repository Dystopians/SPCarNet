# F85-F89 Repair Progress Report

Date: 2026-05-05

## Accepted Reference

F82 remains the accepted fixed-policy reference. It is still the only version in
this round that has both:

- all-metric wins over the clean MeshSplat baseline on the selected multi-scene
  benchmark, and
- two-seed robustness evidence from F81/F82.

## New Experiments

All runs used W&B online logging and topology-frozen recovery unless noted.

| stage | scene | change | W&B | status vs F82 | key finding |
|---|---|---|---|---|---|
| F85 | bonsai | conservative CSEF 25% -> 20% | `7ssd5rz1` | rejected | Render improves slightly, but depth/normal and 4/37 views regress. |
| F85 | courtyard | conservative CSEF 72% -> 70% | `s0h7jnpi` | rejected | Geometry improves, but render collapses vs F82 and all 5 views regress. |
| F86 | bonsai | F82 continuation 26000 -> 30000 | `4m7vqeb8` | rejected | PSNR/depth slightly improve, but SSIM/LPIPS and 14/37 views regress. |
| F86 | courtyard | F82 continuation 26000 -> 30000 | `v0ibcrmr` | rejected | PSNR/SSIM improve, but LPIPS/depth regress and 1/5 views regress. |
| F87 | courtyard | teacher 0.005 + sparse 0.003, 26000 -> 28000 | `g4f03dxr` | rejected | 5/5 views and render metrics beat F82, but AbsRel/Depth regress. |
| F88 | courtyard | teacher 0.001 + sparse 0.001, 26000 -> 28000 | `ejsq087t` | rejected | Best balance so far: 5/5 views and render metrics beat F82; depth gap is smaller but still nonzero. |
| F89 | courtyard | teacher 0.001 + sparse 0.005, 26000 -> 28000 | `eydx84yn` | rejected | Stronger sparse improves normal but worsens depth/AbsRel tradeoff. |

## Parent-Pareto Deltas

Positive means candidate improves over F82. For LPIPS, AbsRel, Depth, and Normal,
the table reports F82 minus candidate, so positive is also better.

| stage | scene | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | min per-view dPSNR | negative views |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F85 | bonsai | +0.0084 | +0.0005 | +0.0010 | -0.0031 | -0.0200 | -0.2956 | -0.0067 | 4 |
| F85 | courtyard | -0.4640 | -0.0051 | +0.0071 | +0.0137 | +0.2722 | +1.5457 | -1.2302 | 5 |
| F86 | bonsai | +0.0085 | -0.0142 | -0.0115 | +0.0001 | +0.0052 | -0.0715 | -0.0825 | 14 |
| F86 | courtyard | +0.1011 | +0.0077 | -0.0040 | -0.0045 | -0.0605 | +0.2120 | -0.0466 | 1 |
| F87 | courtyard | +0.1052 | +0.0094 | +0.0010 | -0.0053 | -0.0717 | +0.3384 | +0.0174 | 0 |
| F88 | courtyard | +0.1008 | +0.0092 | +0.0011 | -0.0029 | -0.0597 | +0.2955 | +0.0118 | 0 |
| F89 | courtyard | +0.1007 | +0.0093 | +0.0008 | -0.0053 | -0.0742 | +0.4510 | +0.0164 | 0 |

## Diagnosis

1. Conservative pruning is not a complete fix. F85 shows that lowering the
   deletion budget can trade render quality for geometry, but does not dominate
   F82.
2. Long continuation is not stable by itself. F86 improves some averages but
   creates visible/per-view regressions.
3. Teacher-render recovery is now useful on courtyard. F87-F89 all achieve
   positive per-view PSNR deltas over F82 on every courtyard test view, and F88
   improves PSNR, SSIM, and LPIPS at the same topology.
4. The remaining blocker is geometry depth drift. F88 reduces the AbsRel gap to
   -0.0029 and the Depth MAE gap to -0.0597, but the strict parent-Pareto gate
   correctly rejects it.

## Current Decision

Do not replace F82 yet. Use F82 as the accepted method in tables. Treat F88 as
the best current repair candidate for qualitative/render-improved courtyard
evidence only, with a clear caveat that it does not pass geometry Pareto.

## Next Technical Target

The next repair should target depth drift directly instead of increasing sparse
loss weight globally. The evidence suggests the sparse COLMAP term is not
pinning the same geometry support used by the evaluation, so simply increasing
lambda moves normal and render but does not improve Depth MAE. A better next
step is a geometry-stability regularizer from the F82 checkpoint:

- add a checkpoint-depth/vertex displacement consistency term during recovery,
- apply it only after teacher warmup starts to move render quality,
- gate against F82 with the existing parent-Pareto script before accepting.

