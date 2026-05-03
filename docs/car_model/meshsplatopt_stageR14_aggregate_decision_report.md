# MeshSplatOpt Stage R14 Aggregate Decision Report

Date: 2026-05-02

## Decision

`SOFT PASS`.

R14 now has one W&B-logged medium-budget recovery result with clear single-scene gains, plus two additional public-scene render-backed gate diagnostics. This is enough to continue method development, but not enough to claim a full NeurIPS-level repair result.

## Evidence Summary

| stage | scene | run type | status | key result |
|---|---|---|---|---|
| R14.10 | `parking_phone_tiny` | W&B medium recovery, iter 200 -> 2000 | `SOFT PASS_SINGLE_SCENE` | Render and sparse-depth geometry improve over current-branch 2000iter baseline |
| R14.11 | `bonsai` | posthoc edit + render-backed gate, iter 2000 | `PASS_DIAGNOSTIC` | Conservative area-outlier edit is accepted with negligible render/geometry deltas |
| R14.12 | `courtyard` | posthoc edit + render-backed gate, iter 2000 | `PASS_DIAGNOSTIC` | Conservative area-outlier edit is accepted with negligible render/geometry deltas |
| R14.13 | `bonsai` | W&B post-edit recovery diagnostic, iter 2000 -> 2200 | `PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET` | Recovery is stable and improves metrics, but uses 200 extra steps |

## Parking Medium Result

W&B:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/81kwhzr3
```

| metric | current branch 2000 | MeshSplatOpt R14.10 | delta |
|---|---:|---:|---:|
| triangles | `782982` | `783509` | `+527` |
| vertices | `820107` | `822064` | `+1957` |
| PSNR | `11.599437713623047` | `13.276764869689941` | `+1.6773271560668945` |
| SSIM | `0.2702677547931671` | `0.30384060740470886` | `+0.03357285261154175` |
| LPIPS | `0.6347319483757019` | `0.6081721186637878` | `-0.026559829711914062` |
| AbsRel | `0.42787965657189714` | `0.3640420630578014` | `-0.06383759351409572` |
| Depth MAE | `4.414160625200222` | `3.806375643108584` | `-0.6077849820916381` |
| normal mean deg | `52.565184963415106` | `52.672900862227785` | `+0.10771589881267915` |

## Public-Scene Gate Diagnostics

| scene | triangles delta | PSNR delta | SSIM delta | LPIPS delta | AbsRel delta | Depth MAE delta |
|---|---:|---:|---:|---:|---:|---:|
| `bonsai` | `-1` | `-0.0003681182861328125` | `-0.000012442469596862793` | `-0.0000036954879760742188` | `0.0` | `0.0` |
| `courtyard` | `-1` | `-0.0005950927734375` | `0.000011831521987915039` | `0.00007200241088867188` | `0.0` | `0.0` |

## Bonsai Recovery Diagnostic

W&B:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/z498br53
```

| metric | baseline 2000 | post-edit recovery 2200 | delta |
|---|---:|---:|---:|
| PSNR | `12.201611518859863` | `13.276382446289062` | `+1.0747709274291992` |
| SSIM | `0.20731531083583832` | `0.24055197834968567` | `+0.03323666751384735` |
| LPIPS | `0.6242585182189941` | `0.6113873720169067` | `-0.012871146202087402` |
| AbsRel | `0.49587362441894434` | `0.4733479577347401` | `-0.022525666684204215` |
| Depth MAE | `4.907808996255763` | `4.762276469029142` | `-0.14553252722662113` |
| normal mean deg | `50.118300749023625` | `49.21947049923495` | `-0.8988302497886749` |

This diagnostic is useful because it verifies recovery stability on a second public scene. It is not an equal-budget win because it evaluates at iteration 2200 against a 2000iter baseline.

## What This Proves

- Real checkpoint edit materialization works for delete and fill.
- Render-backed acceptance works on three scenes.
- Teacher/recovery training can resume edited checkpoints with W&B online.
- A medium-budget parking run can improve independent render metrics and sparse-depth geometry.
- Edge-connected CSEF is invalid for real Mesh Splatting checkpoint proposal selection because the saved representation is triangle soup.

## What This Does Not Prove Yet

- It does not prove bidirectional repair beyond conservative delete/fill path validation.
- It does not show a second W&B medium-scene recovery improvement.
- It does not beat Stage35 across public scenes.
- It does not justify full-budget 7000+ sweeps yet.
- It does not validate giant-hole repair on real public scenes.

## Next Gate

Proceed to R14.13/R14.14 only if:

1. a second W&B recovery diagnostic finishes without render collapse;
2. Stage35 comparison tables are added for `parking_phone_tiny`, `bonsai`, and `courtyard`;
3. a non-delete spatial-adjacency or render-residual proposal is implemented and tested through the same gate.

Full-budget R15 should remain blocked until at least one non-delete real edit or a second W&B medium-scene improvement exists.
