# Stage23.6 Tuned Integrated Topology-Control Report

Date: 2026-05-02

## Status

Gate: `PASS`

Stage23.6 turns the Stage23.5 trigger smoke into a medium-budget integrated topology-control diagnostic on `parking_phone_tiny`. The useful row is `tuned_medium_v2_2000iter`; the first run is kept as a failure diagnostic.

## Runs

| run | W&B | result |
|---|---|---|
| `tuned_medium_2000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3209wi9z` | Diagnostic failure. `orientation_keep=1.0` protected every triangle when the threshold was `0.85`; no useful candidate pool. |
| `tuned_medium_v2_2000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j0c8zwkx` | Useful medium row. Two PRISM candidate edits committed with counterfactual acceptance and no rollback. |

## V2 Configuration

- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage23_6_tuned_integrated_topology/tuned_medium_v2_2000iter/`
- iterations: `2000`
- GPU: `1`
- PRISM candidate rounds: `2`
- candidate prune ratio per round: `0.01`
- `--prism_keep_orientation_threshold 1.1`
- `--prism_keep_geometry_threshold 0.85`
- `--prism_keep_render_threshold 0.85`
- counterfactual gate enabled
- final cleanup disabled
- W&B online

## PRISM Decisions

| iteration | pre triangles | post triangles | committed | rollback |
|---:|---:|---:|---|---:|
| 551 | 64497 | 63853 | true | 0 |
| 922 | 63853 | 63215 | true | 0 |

Collector output:

- gate: `PASS`
- rounds: `2`
- committed rounds: `2`

## Metrics

Training internal test@2000:

- PSNR `15.975685`
- SSIM `0.474268`
- LPIPS `0.544082`
- L1 `0.117759`
- FPS `253.114508`

Independent render / geometry at 2000:

| row | PSNR | SSIM | LPIPS | depth AbsRel | normal deg | triangles | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|
| origin/main 2000 | 11.047660 | 0.219931 | 0.641706 | 5.611905 | 52.198939 | 39079 | 58458 |
| current branch 2000 | 11.599438 | 0.270268 | 0.634732 | 0.427880 | 52.565185 | 782982 | 820107 |
| Stage17 2000 | 13.278273 | 0.303979 | 0.607610 | 0.366691 | 52.169584 | 777251 | 816498 |
| Stage23.6 v2 2000 | 12.046110 | 0.286099 | 0.629034 | 0.393866 | 51.945426 | 415334 | 507649 |

## Interpretation

Stage23.6 v2 proves that a non-trivial, counterfactual-gated PRISM schedule can commit topology edits during training without rollback or final-cleanup ambiguity. It improves over the current-branch 2000 row in independent render and geometry while using far fewer triangles. It does not beat the Stage17 2000 quality row, but Stage17 later fails at the 7000-iteration budget.

The key tuning lesson is that orientation keep protection must not be interpreted with a threshold below the observed saturated `1.0` signal on this scene. Using `--prism_keep_orientation_threshold 1.1` restores an editable candidate pool while keeping geometry/render keep checks active.

## Decision

Stage23.6 is a `PASS` as a medium integrated-topology mechanism row. It justifies a full-budget Stage24 run, but does not by itself provide a final paper-quality claim.

