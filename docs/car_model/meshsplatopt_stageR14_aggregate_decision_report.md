# MeshSplatOpt Stage R14 Aggregate Decision Report

Date: 2026-05-02

## Decision

`SOFT PASS`.

R14 now has one W&B-logged medium-budget parking recovery result with clear single-scene gains, two public-scene delete gate diagnostics, two public-scene recovery diagnostics, a three-scene non-delete `SNAP_VERTICES` gate pass, one short equal-step public-scene control, and one medium public-scene control. This is enough to continue method development, but not enough to claim a full NeurIPS-level repair result.

## Evidence Summary

| stage | scene | run type | status | key result |
|---|---|---|---|---|
| R14.10 | `parking_phone_tiny` | W&B medium recovery, iter 200 -> 2000 | `SOFT PASS_SINGLE_SCENE` | Render and sparse-depth geometry improve over current-branch 2000iter baseline |
| R14.11 | `bonsai` | posthoc edit + render-backed gate, iter 2000 | `PASS_DIAGNOSTIC` | Conservative area-outlier edit is accepted with negligible render/geometry deltas |
| R14.12 | `courtyard` | posthoc edit + render-backed gate, iter 2000 | `PASS_DIAGNOSTIC` | Conservative area-outlier edit is accepted with negligible render/geometry deltas |
| R14.13 | `bonsai` | W&B post-edit recovery diagnostic, iter 2000 -> 2200 | `PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET` | Recovery is stable and improves metrics, but uses 200 extra steps |
| R14.14-R14.16 | `parking_phone_tiny`, `bonsai`, `courtyard` | non-delete `SNAP_VERTICES` + render-backed gate, iter 2000 | `PASS_DIAGNOSTIC_CROSS_SCENE` | Real checkpoint area-outlier snap edits pass all render/geometry gates |
| R14.17 | `bonsai` | W&B non-delete snap recovery diagnostic, iter 2000 -> 2200 | `PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET` | Non-delete snap recovery is stable and improves over 2000iter baseline, but is not equal-budget |
| R14.18 | `bonsai` | W&B unedited baseline continuation, iter 2000 -> 2200 | `CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN` | Equal-step control is slightly stronger than snap recovery on most metrics |
| R14.19-R14.20 | `bonsai` | W&B medium continuation, iter 2000 -> 4000 | `MEDIUM_CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN` | Baseline continuation beats snap on render/depth metrics; snap only improves normal proxy |

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

## Non-Delete Snap Gate Diagnostics

| scene | selected face | area before | area after | PSNR delta | SSIM delta | LPIPS delta | AbsRel delta | Depth MAE delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `parking_phone_tiny` | `727102` | `247.02622985839844` | `15.439142227172852` | `0.00000286102294921875` | `-0.0000012516975402832031` | `-0.000002086162567138672` | `0.0` | `0.0` |
| `bonsai` | `2462659` | `164.05824279785156` | `10.253642082214355` | `-0.00019073486328125` | `-0.000013679265975952148` | `-0.00005561113357543945` | `0.0` | `0.0` |
| `courtyard` | `404443` | `873.2474365234375` | `54.57794952392578` | `-0.005673408508300781` | `0.000041097402572631836` | `0.0000642538070678711` | `0.0` | `0.0` |

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

## Bonsai Non-Delete Snap Recovery Diagnostic

W&B:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/8qdzfu6h
```

| metric | baseline 2000 | snap recovery 2200 | delta |
|---|---:|---:|---:|
| PSNR | `12.201611518859863` | `13.273988723754883` | `+1.0723772048950195` |
| SSIM | `0.20731531083583832` | `0.24039088189601898` | `+0.033075571060180664` |
| LPIPS | `0.6242585182189941` | `0.6116319894790649` | `-0.0126265287399292` |
| AbsRel | `0.49587362441894434` | `0.47445281696526337` | `-0.02142080745368097` |
| Depth MAE | `4.907808996255763` | `4.772623802825101` | `-0.1351851934306621` |
| normal mean deg | `50.118300749023625` | `49.315686202793366` | `-0.8026145462302589` |

This diagnostic is useful because it verifies W&B recovery stability for a real non-delete edit. It is not an equal-budget win because it evaluates at iteration 2200 against a 2000iter baseline.

## Bonsai Equal-Step Control

W&B:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kic0euiq
```

| metric | baseline continuation 2200 | snap recovery 2200 | snap minus control |
|---|---:|---:|---:|
| PSNR | `13.274771690368652` | `13.273988723754883` | `-0.0007829666137695312` |
| SSIM | `0.2403060346841812` | `0.24039088189601898` | `+0.00008484721183776855` |
| LPIPS | `0.6113919019699097` | `0.6116319894790649` | `+0.00024008750915527344` |
| AbsRel | `0.47338970412280024` | `0.47445281696526337` | `+0.0010631128424631347` |
| Depth MAE | `4.765895956720541` | `4.772623802825101` | `+0.006727846104559882` |
| normal mean deg | `49.19677426124215` | `49.315686202793366` | `+0.11891194155121613` |

The equal-step control is negative for a snap performance-gain claim. It strengthens the paper discipline by forcing the R14 snap path to be framed as real-edit safety/stability infrastructure unless a better selector or equal-budget run changes this result.

## Bonsai Medium Continuation Control

W&B:

```text
snap:     https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fjzy6lun
baseline: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gxeskhta
```

| metric | snap 4000 | baseline continuation 4000 | snap minus control |
|---|---:|---:|---:|
| triangles | `5090526` | `5090601` | `-75` |
| vertices | `4270548` | `4270293` | `+255` |
| PSNR | `15.81759262084961` | `15.834700584411621` | `-0.01710796356201172` |
| SSIM | `0.33459141850471497` | `0.33469849824905396` | `-0.00010707974433898926` |
| LPIPS | `0.5731096863746643` | `0.5714929699897766` | `+0.0016167163848876953` |
| AbsRel | `0.40904864176963485` | `0.40514114339865287` | `+0.00390749837098198` |
| Depth MAE | `4.261201179402033` | `4.241773913061498` | `+0.019427266340535047` |
| normal mean deg | `47.83674765098326` | `48.11943889631045` | `-0.2826912453271897` |

This medium control blocks a full-budget R15 run from the current snap selector. Both rows grow to about `5.09M` triangles by iteration 4000, and the unedited continuation is better on render and sparse-depth metrics.

## What This Proves

- Real checkpoint edit materialization works for delete, fill, and non-delete vertex snap.
- Render-backed acceptance works on three scenes.
- Teacher/recovery training can resume delete and non-delete edited checkpoints with W&B online.
- A medium-budget parking run can improve independent render metrics and sparse-depth geometry.
- Edge-connected CSEF is invalid for real Mesh Splatting checkpoint proposal selection because the saved representation is triangle soup.

## What This Does Not Prove Yet

- It does not prove equal-budget training gains from non-delete edits.
- It does not show a second W&B medium-scene recovery improvement.
- The current non-delete snap selector does not beat equal-step unedited continuation on `bonsai`.
- Medium continuation from the current non-delete snap selector also does not beat unedited continuation on `bonsai`.
- It does not beat Stage35 across public scenes.
- It actively argues against full-budget 7000+ sweeps for this selector without topology retention or a stronger proposal signal.
- It does not validate giant-hole repair on real public scenes.

## Next Gate

Proceed to R14.17 public-scene W&B recovery only if:

1. public-scene recovery uses W&B online logging;
2. Stage35 comparison tables are added for `parking_phone_tiny`, `bonsai`, and `courtyard`;
3. the non-delete snap selector remains behind render-backed acceptance.

Full-budget R15 should remain blocked until a second W&B medium-scene improvement exists.
