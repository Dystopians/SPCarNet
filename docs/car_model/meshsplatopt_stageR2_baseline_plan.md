# MeshSplatOpt Stage R2 Baseline Plan

Date: 2026-05-02

## Gate

`PASS`.

The baseline plan covers pruning, mesh simplification, hole filling, geometry-aware repair, and prior hallucination risk.

## Baseline Principles

All paper-facing comparisons must separate:

- training-time W&B metrics;
- independent `render.py + metrics.py` metrics;
- sparse COLMAP proxy metrics;
- synthetic clean-mesh evaluation metrics;
- oracle or GT-dependent analysis.

Clean meshes or synthetic ground truth may be used only for evaluation, never for proposal selection.

## Required Baselines

| category | baseline | purpose | expected artifact |
|---|---|---|---|
| base method | clean Mesh Splatting | establish render/topology starting point | train/render/metrics logs |
| internal pruning | Stage35 retained PRISM | strongest current retained delete-centric baseline | existing and refreshed rows |
| internal pruning | delete-only PRISM budget sweep | isolate deletion from repair | budget tables at 90/75/50/25% |
| random pruning | random same-count delete | sanity check candidate quality | synthetic and public mesh outputs |
| visibility pruning | low-visibility delete | test if simple visibility explains gains | topology baseline report |
| protected pruning | boundary-protected delete | test whether preservation alone helps | topology baseline report |
| simplification | QEM edge collapse | strong classical topology baseline | same budgets as PRISM |
| mesh processing | planar face merge/coarsen | collapse/merge baseline | valid mesh + count table |
| hole repair | classical boundary-loop fill | compare fill geometry without CSEF gates | hole-fill baseline report |
| hole repair | plane/height-field fill without free-space gate | test free-space certificate value | ablation and failure cases |
| reconstruction | Poisson/screened Poisson where compatible | dense geometry repair baseline | optional if inputs exist |
| geometry-aware splatting | SuGaR/MeshGS/2DGS/DN-Splatter style baseline | compare surface-aware geometry methods | cite or run compatible method |
| prior risk | object-prior fill without scene gate | show hallucination risk | rejected/failed ablation |
| recovery | no teacher recovery | isolate appearance recovery value | ablation table |
| safety | no rollback | show harmful edit persistence risk | ablation table |

## Method Variants

Required MeshSplatOpt variants:

| variant | isolates |
|---|---|
| full MeshSplatOpt | CSEF + portfolio + gates + rollback + recovery |
| no giant-hole fill | ordinary repair without high-risk fill |
| no CSEF debt term | whether evidence debt drives useful repairs |
| no negative free-space evidence | hallucination/free-space failures |
| no counterfactual render gate | render-safety failures |
| no sparse geometry gate | geometry-safety failures |
| no changed-pixel gate | visible change control |
| delete/collapse only | whether pruning explains all gains |
| snap only | geometry correction without topology growth |
| fill only | hole repair without cleanup/snap |
| prior-only diagnostic fill | labeled uncertainty and hallucination analysis |

## Budget Plan

Topology budgets:

- 100%;
- 90%;
- 75%;
- 50%;
- 25% if stable.

Training budgets:

- smoke: synthetic and tiny, no paper claims;
- medium: 2000 iterations or equivalent, W&B online;
- full: 7000+ only after medium gate passes.

## Scene Plan

Use scenes that already support the repository's COLMAP-compatible pipeline:

- Mip-NeRF 360 `bonsai`;
- ETH3D `courtyard`;
- `parking_phone_tiny` as domain-specific parking evidence.

Do not use a scene for sparse geometry claims unless COLMAP sparse tracks or compatible proxy data exist.

## Metrics

Render:

- PSNR;
- SSIM;
- LPIPS;
- MAE if available.

Geometry:

- sparse AbsRel;
- sparse DepthMAE;
- normal mean angle;
- free-space violation;
- surface distance to synthetic clean mesh for synthetic evaluation only.

Repair:

- floater count;
- dent residual;
- roughness/normal variance;
- boundary length reduction;
- repaired giant-void area;
- prior-only false-fill rate;
- accepted and rejected edit counts.

Topology and cost:

- triangle count;
- vertex count;
- topology cost delta;
- runtime;
- memory;
- W&B run link for training/recovery.

## Minimum Evidence For Main Claim

MeshSplatOpt needs at least one of:

1. repair-quality improvement without meaningful render regression on at least two medium public scenes;
2. clear topology-quality Pareto advantage over Stage35, delete-only PRISM, and QEM on at least two scenes;
3. synthetic benchmark win on at least four damage categories plus one real/realistic certified giant-hole case.

If medium results are only tiny pruning-like deltas, the main-conference framing should stop or be demoted.
