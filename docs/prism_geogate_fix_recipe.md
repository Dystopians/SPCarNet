# PRISM GeoGate Fix Recipe

## Why we no longer gate on pseudo depth/normal

The final acceptance geometry benchmark uses the COLMAP sparse proxy (`sparse depth + sparse PCA normal`), not training-time pseudo supervision (`invdepthmap` / `normal_map`).
If PRISM gates use pseudo geometry while evaluation uses sparse COLMAP geometry, we can optimize to the wrong target and accept prune rounds that look safe in pseudo space but regress official geometry.
For this round, all geometry decisions are aligned to the official sparse proxy path.

## Official geometry definition in current PRISM

PRISM counterfactual gate, PRISM validation gate, and offline benchmark all share one geometry implementation path:

- sparse COLMAP depth metrics: `AbsRel`, `Delta<1.25`, `Depth MAE`
- sparse COLMAP normal proxy metrics: `MeanAngle`, `AbsCos`

This shared proxy is the only geometry gate/eval source used for the three experiments below.

## Three high-information experiments

### 1) PRISM-GeoGateFix (`prism_geogate_fix`)

Purpose: isolate the effect of geogate-unified supervision with sparse COLMAP depth loss, while keep-protect logic is effectively disabled.

Expected insight: whether geometry alignment alone already recovers most official geometry regressions.

### 2) PRISM-LatePrune (`prism_late_prune`)

Purpose: test whether delaying candidate prune start (later geometry acquisition + longer stats collection) improves stability without sparse depth supervision.

Expected insight: whether schedule shift alone is enough, or geogate/sparse-depth alignment is still required.

### 3) PRISM-GeoGateFixKeep (`prism_geogate_fix_keep`)

Purpose: geogate fix + sparse depth supervision + keep/protected dilation enabled.

Expected insight: whether explicit geometry keep signals and neighborhood protection improve `MeanAngle` and `Depth MAE` over plain geogate fix.

## Recommended execution order

1. `PRISM-GeoGateFix` (establish aligned baseline)
2. `PRISM-LatePrune` (measure schedule-only effect)
3. `PRISM-GeoGateFixKeep` (measure additive keep/protect benefit)

Run all with fixed checkpoint saves at:
`15000, 16000, 18000, 20000, 21000, 24000, 30000`.

Then benchmark all three runs with unified multi-checkpoint render + image metrics + official geometry eval.

