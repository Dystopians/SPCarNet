# MeshSplatOpt Stage R16.01-R16.02 Two-Scene Full Freeze Report

Date: 2026-05-03

## Decision

`TWO_SCENE_FULL_SCHEDULE_PASS`.

R16 validates the freeze-densify/skip-Delaunay schedule at full 7000-iteration budget on two public scenes: `courtyard` and `bonsai`. Both rows use online W&B logging and preserve checkpoint topology exactly.

## W&B

```text
courtyard full freeze: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/z2i5ndyu
bonsai full freeze:    https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/nsj76h7d
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

## Full-Budget Gains

Full freeze minus 2000 baseline:

| scene | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `courtyard` | `0` | `0` | `+3.3749685287475586` | `+0.15550532937049866` | `-0.1524217128753662` | `-0.18334653906540725` | `-1.5795604132691766` | `+2.250983735221011` |
| `bonsai` | `0` | `0` | `+6.101691246032715` | `+0.24824108183383942` | `-0.1335981786251068` | `-0.274985337616479` | `-2.5156110580052863` | `-8.884689826538405` |

Full freeze minus medium freeze:

| scene | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `courtyard` | `0` | `0` | `+0.5014934539794922` | `+0.01597803831100464` | `-0.020370125770568848` | `-0.0716005829503221` | `-0.6241680835815848` | `-0.3921888347633004` |
| `bonsai` | `0` | `0` | `+0.8735523223876953` | `+0.023204028606414795` | `-0.015829235315322876` | `-0.05017384246475773` | `-0.5049654745626872` | `-2.1140784138941747` |

## Interpretation

The full-budget schedule claim is now much stronger than before:

- It has two-scene public full-budget validation.
- It preserves topology exactly on both scenes.
- It improves render and depth proxy metrics versus 2000 baseline and medium freeze.
- On `bonsai`, it also strongly improves the sparse normal proxy.
- On `courtyard`, the sparse normal proxy remains worse than baseline but improves over medium.

The method should be framed as topology-retained recovery/continuation. The current `SNAP_VERTICES` selector is still weak and should not be the headline contribution without a stronger proposal signal.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR16_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR16_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/geometry_eval_colmap/iter_7000_max500.json`
- `outputs/carnet/meshsplatopt/stageR16_02_bonsai_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR16_02_bonsai_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/geometry_eval_colmap/iter_7000_max500.json`

## Next Gate

The next meaningful improvement is no longer more proof that freezing works. It is:

1. a stronger edit/proposal selector that beats the freeze baseline;
2. a normal-aware recovery term or geometry regularizer for `courtyard`;
3. a paper-grade comparison table against Stage35, current branch, and posthoc pruning.
