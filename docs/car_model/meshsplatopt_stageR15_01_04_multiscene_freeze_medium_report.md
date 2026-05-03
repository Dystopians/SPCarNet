# MeshSplatOpt Stage R15.01-R15.04 Multi-Scene Freeze Medium Report

Date: 2026-05-03

## Decision

`MULTI_SCENE_SCHEDULE_PASS_SNAP_SELECTOR_WEAK`.

R15 extends the R14 freeze-densify/skip-Delaunay recovery schedule from `bonsai` to `courtyard` and `parking_phone_tiny`. All runs use online W&B logging and the same medium continuation policy:

```text
--densify_until_iter 2000 --skip_restricted_delaunay
```

The schedule is now supported on three scenes. It improves render metrics and depth proxy metrics on `bonsai`, `courtyard`, and `parking_phone_tiny` while preventing the severe topology growth observed in unfrozen R14.19-R14.20. The current `SNAP_VERTICES` selector remains weak: under the same freeze schedule it is mixed on `bonsai` and slightly negative on `courtyard` and `parking_phone_tiny`.

## W&B

```text
bonsai baseline freeze:    https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qdwbbpob
bonsai snap freeze:        https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/srdr58z6
courtyard baseline freeze: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/cvf6t7do
courtyard snap freeze:     https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/d3h2ruj3
parking baseline freeze:   https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/evj36lvp
parking snap freeze:       https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3r7inkj0
```

## Results

| scene | row | iter | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bonsai` | R14.21b baseline freeze | `4000` | `2487474` | `2478890` | `17.429750442504883` | `0.43235236406326294` | `0.5064895749092102` | `0.27106212926722306` | `2.897163412813164` | `43.347689336379396` |
| `bonsai` | R14.22 snap freeze | `4000` | `2487474` | `2478890` | `17.437725067138672` | `0.4337323307991028` | `0.5067973732948303` | `0.2728521602266819` | `2.8930862576166856` | `43.570728874963045` |
| `courtyard` | baseline 2000 | `2000` | `410254` | `444301` | `14.946162223815918` | `0.4387754499912262` | `0.5924432873725891` | `0.3547996069696563` | `3.647069967658135` | `35.32471188743233` |
| `courtyard` | R15.01 baseline freeze | `4000` | `410254` | `444301` | `17.819637298583984` | `0.5783027410507202` | `0.46039170026779175` | `0.24305365085457115` | `2.6916776369705433` | `37.96788445741664` |
| `courtyard` | R15.02 snap freeze | `4000` | `410254` | `444301` | `17.81588363647461` | `0.5780841112136841` | `0.4617271423339844` | `0.24501803376316553` | `2.707712333593413` | `38.46088560987557` |
| `parking_phone_tiny` | baseline 2000 | `2000` | `782982` | `820107` | `11.599437713623047` | `0.2702677547931671` | `0.6347319483757019` | `0.42787965657189714` | `4.414160625200222` | `52.565184963415106` |
| `parking_phone_tiny` | R15.03 baseline freeze | `4000` | `782982` | `820107` | `14.251087188720703` | `0.38379988074302673` | `0.5697492957115173` | `0.32479430564316225` | `3.6368910610211134` | `51.043450901931564` |
| `parking_phone_tiny` | R15.04 snap freeze | `4000` | `782982` | `820107` | `14.190221786499023` | `0.38185572624206543` | `0.5716038942337036` | `0.3291848750374292` | `3.623574344332695` | `51.30038566250242` |

## Schedule Gains

Baseline-freeze versus the scene's 2000-iteration baseline:

| scene | triangles delta | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `courtyard` | `0` | `+2.8734750747680664` | `+0.13952729105949402` | `-0.13205158710479736` | `-0.11174595611508515` | `-0.9553923306875917` | `+2.643172569984308` |
| `parking_phone_tiny` | `0` | `+2.6516494750976562` | `+0.11353212594985962` | `-0.06498265266418457` | `-0.1030853509287349` | `-0.7772695641791085` | `-1.521734061483542` |

The `courtyard` normal proxy regresses while render and depth improve strongly. `parking_phone_tiny` improves render, depth, and normal proxy. `bonsai` was documented in R14.21-R14.22 and showed the strongest all-metric schedule improvement versus the unfrozen medium baseline.

## Snap Selector Delta

Snap-freeze minus baseline-freeze:

| scene | triangles delta | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bonsai` | `0` | `+0.007974624633789062` | `+0.0013799667358398438` | `+0.0003077983856201172` | `+0.0017900309594588437` | `-0.004077155196478444` | `+0.22303953858364878` |
| `courtyard` | `0` | `-0.003753662109375` | `-0.0002186298370361328` | `+0.001335442066192627` | `+0.001964382908594381` | `+0.016034696622869493` | `+0.4930011524589304` |
| `parking_phone_tiny` | `0` | `-0.06086540222167969` | `-0.0019441545009613037` | `+0.0018545985221862793` | `+0.004390569394266963` | `-0.013316716688418495` | `+0.2569347605708572` |

This blocks any top-level claim that the current area-outlier `SNAP_VERTICES` selector improves quality. It is safe and trainable, but not strong enough.

## Interpretation

The paper-quality claim should pivot away from the current snap selector and toward topology-retained recovery as a training schedule. The strongest statement supported today is:

> Freezing densification at the edited/loaded checkpoint and skipping the delayed restricted-Delaunay refresh gives stable medium-budget recovery across three scenes, preserving checkpoint topology while substantially improving render metrics and depth proxy metrics.

The next method step should improve proposal selection. The current selector is only a safety mechanism for real checkpoint edits, not a performance mechanism.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR15_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR15_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/geometry_eval_colmap/iter_4000_max500.json`
- `outputs/carnet/meshsplatopt/stageR15_02_courtyard_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR15_02_courtyard_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/geometry_eval_colmap/iter_4000_max500.json`
- `outputs/carnet/meshsplatopt/stageR15_03_parking_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR15_03_parking_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/geometry_eval_colmap/iter_4000_max500.json`
- `outputs/carnet/meshsplatopt/stageR15_04_parking_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR15_04_parking_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/geometry_eval_colmap/iter_4000_max500.json`

## Next Gate

Run one full-budget freeze schedule on the strongest public scene and build a stronger proposal selector before claiming edit-driven improvement. The current multi-scene evidence is enough to justify a full schedule validation, but not enough to sell the area-outlier snap selector as the method.
