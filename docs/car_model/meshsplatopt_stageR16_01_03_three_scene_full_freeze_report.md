# MeshSplatOpt Stage R16.01-R16.03 Three-Scene Full Freeze Report

Date: 2026-05-03

## Decision

`THREE_SCENE_FULL_SCHEDULE_PASS`.

R16 now validates the freeze-densify/skip-Delaunay recovery schedule at full 7000-iteration budget on three scenes: `courtyard`, `bonsai`, and `parking_phone_tiny`. All rows used online W&B logging and preserved checkpoint topology exactly.

## W&B

```text
courtyard full freeze: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/z2i5ndyu
bonsai full freeze:    https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/nsj76h7d
parking full freeze:   https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dq8urgr7
```

## Results

| scene | row | iter | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `courtyard` | baseline | `2000` | `410254` | `444301` | `14.946162223815918` | `0.4387754499912262` | `0.5924432873725891` | `0.3547996069696563` | `3.647069967658135` | `35.32471188743233` |
| `courtyard` | freeze medium | `4000` | `410254` | `444301` | `17.819637298583984` | `0.5783027410507202` | `0.46039170026779175` | `0.24305365085457115` | `2.6916776369705433` | `37.96788445741664` |
| `courtyard` | freeze full | `7000` | `410254` | `444301` | `18.321130752563477` | `0.5942807793617249` | `0.4400215744972229` | `0.17145306790424905` | `2.0675095533889585` | `37.57569562265334` |
| `bonsai` | baseline | `2000` | `2487474` | `2478890` | `12.201611518859863` | `0.20731531083583832` | `0.6242585182189941` | `0.49587362441894434` | `4.907808996255763` | `50.118300749023625` |
| `bonsai` | freeze medium | `4000` | `2487474` | `2478890` | `17.429750442504883` | `0.43235236406326294` | `0.5064895749092102` | `0.27106212926722306` | `2.897163412813164` | `43.347689336379396` |
| `bonsai` | freeze full | `7000` | `2487474` | `2478890` | `18.303302764892578` | `0.45555639266967773` | `0.49066033959388733` | `0.22088828680246533` | `2.3921979382504768` | `41.23361092248522` |
| `parking_phone_tiny` | baseline | `2000` | `782982` | `820107` | `11.599437713623047` | `0.2702677547931671` | `0.6347319483757019` | `0.42787965657189714` | `4.414160625200222` | `52.565184963415106` |
| `parking_phone_tiny` | freeze medium | `4000` | `782982` | `820107` | `14.251087188720703` | `0.38379988074302673` | `0.5697492957115173` | `0.32479430564316225` | `3.6368910610211134` | `51.043450901931564` |
| `parking_phone_tiny` | freeze full | `7000` | `782982` | `820107` | `15.570565223693848` | `0.4482119083404541` | `0.5280523300170898` | `0.2578147524632813` | `3.085023261488907` | `49.78974907411433` |

## Full-Budget Gains

Full freeze minus 2000 baseline:

| scene | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `courtyard` | `0` | `0` | `+3.3749685287475586` | `+0.15550532937049866` | `-0.1524217128753662` | `-0.18334653906540725` | `-1.5795604132691766` | `+2.250983735221011` |
| `bonsai` | `0` | `0` | `+6.101691246032715` | `+0.24824108183383942` | `-0.1335981786251068` | `-0.274985337616479` | `-2.5156110580052863` | `-8.884689826538405` |
| `parking_phone_tiny` | `0` | `0` | `+3.971127510070801` | `+0.177944153547287` | `-0.1066796183586121` | `-0.17006490410861584` | `-1.329137363711315` | `-2.7754358893007775` |

Full freeze minus medium freeze:

| scene | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `courtyard` | `0` | `0` | `+0.5014934539794922` | `+0.01597803831100464` | `-0.020370125770568848` | `-0.0716005829503221` | `-0.6241680835815848` | `-0.3921888347633004` |
| `bonsai` | `0` | `0` | `+0.8735523223876953` | `+0.023204028606414795` | `-0.015829235315322876` | `-0.05017384246475773` | `-0.5049654745626872` | `-2.1140784138941747` |
| `parking_phone_tiny` | `0` | `0` | `+1.3194780349731445` | `+0.06441202759742737` | `-0.04169696569442749` | `-0.06697955317988097` | `-0.5518677995322063` | `-1.253701827817232` |

## Interpretation

The freeze-densify/skip-Delaunay schedule is now the strongest validated contribution in this branch:

- three-scene full-budget support;
- exact topology retention on every row;
- consistent improvement over the 2000 checkpoint in PSNR, SSIM, LPIPS, depth AbsRel, and depth MAE;
- consistent additional gain from medium 4000 to full 7000 budget;
- normal proxy improves with more full-budget continuation versus medium on all three scenes, but remains worse than the 2000 baseline on `courtyard`.

The paper claim should therefore focus on topology-retained recovery/continuation under a frozen mesh connectivity budget. The current `SNAP_VERTICES` selector remains useful as a safe edit materialization path, but it is not strong enough to headline the method.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR16_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR16_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/geometry_eval_colmap/iter_7000_max500.json`
- `outputs/carnet/meshsplatopt/stageR16_02_bonsai_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR16_02_bonsai_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/geometry_eval_colmap/iter_7000_max500.json`
- `outputs/carnet/meshsplatopt/stageR16_03_parking_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR16_03_parking_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/geometry_eval_colmap/iter_7000_max500.json`

## Next Gate

The next meaningful gate is method novelty, not schedule proof:

1. add a stronger edit/proposal selector that can beat the topology-retained freeze baseline;
2. add a normal-aware or surface-consistency recovery term to address the remaining sparse-normal proxy regressions;
3. produce a final paper-grade comparison table against Stage35/current-branch baselines and posthoc pruning.
