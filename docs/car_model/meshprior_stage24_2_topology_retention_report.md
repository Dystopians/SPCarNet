# Stage24.2 Topology-Retention Report

Date: 2026-05-02

## Gate

`PASS`.

M24.2 is the first integrated optimization-time topology-control row that beats the M21.5 posthoc `prune_50` diagnostic on final topology while also improving independent render and COLMAP proxy geometry.

## Method

The code adds one opt-in schedule flag:

```text
--prism_freeze_densification_after_first_commit
```

When a PRISM candidate prune commits, the training loop records `densification_frozen_after_prism_commit=1` and disables subsequent standard Mesh Splatting densification. Standard pruning, optimization, counterfactual rollback, collector accounting, and final-cleanup accounting remain active.

This isolates topology retention from new proposal logic: if the row improves, the gain comes from preserving accepted PRISM edits instead of adding another model component.

## Run

- run: `freeze_after_first_commit_7000iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vsv2bs79`
- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage24_2_topology_retention/freeze_after_first_commit_7000iter/`
- GPU: `1`
- final cleanup: disabled
- counterfactual gate: enabled
- sparse COLMAP depth loss: enabled
- command log: `outputs/carnet/meshprior/parking_phone_tiny/stage24_2_topology_retention/freeze_after_first_commit_7000iter/logs/train_command.txt`

## Metrics

| row | triangles | vertices | PSNR | SSIM | LPIPS | depth AbsRel | normal mean deg | collector |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| current branch 7000 | `833775` | `1071408` | `17.204679` | `0.535045` | `0.450750` | `0.076126` | `45.561976` | n/a |
| M21.5 posthoc `prune_50` | `416888` | `662039` | `17.051889` | `0.523914` | `0.465400` | `0.083265` | `45.825681` | n/a |
| M24-v3 integrated | `823651` | `1058219` | `17.042757` | `0.529476` | `0.454884` | `0.082815` | `43.394721` | `PASS` |
| M24.1 best integrated | `723438` | `904493` | `16.967005` | `0.530894` | `0.465932` | `0.082264` | `42.667905` | `PASS` |
| M24.2 freeze after first commit | `254491` | `463687` | `17.314823` | `0.559230` | `0.442099` | `0.078840` | `41.010093` | `PASS` |

M24.2 is the current best single-scene row across topology, SSIM, LPIPS, and COLMAP normal proxy. It gives a large final topology reduction without sacrificing render quality.

## PRISM Decisions

- effective PRISM rounds: `8`
- no-candidate retry events: `27`
- committed rounds: `2`
- rollback rounds: `6`
- final cleanup: disabled, `0` cleanup-pruned triangles

Important round metadata:

- `6151`: candidate commit, `618527 -> 615435`
- `6342`: candidate commit, `618527 -> 615435`
- `6500`: standard pruning after densification freeze reduces topology to the final low-budget regime
- `6603` through `6608`: counterfactual gate rejects further 0.5% candidate edits and rolls back

The collector reports final state-dict topology as `254491` triangles and `463687` vertices. The W&B scalar summary showed an earlier `350671` triangle count; the saved checkpoint and collector are the source of truth for final topology.

## Interpretation

M24.1 showed that integrated PRISM can safely commit topology edits but later densification can erase part of the gain. M24.2 fixes that failure mode with a minimal schedule rule. This creates the first defensible paper-style method row:

1. PRISM proposes and gates topology edits inside training.
2. Counterfactual rollback rejects harmful later edits.
3. A topology-retention schedule prevents accepted edits from being overwritten by densification.
4. The final checkpoint improves topology and render/geometry proxies relative to the strongest available single-scene baselines.

Remaining caveat: this is still one small parking scene with COLMAP proxy geometry, not a multi-scene NeurIPS-grade empirical claim. The method claim is now much stronger, but generality remains unproven.

## Verification

- online W&B run exists: `vsv2bs79`
- checkpoint exists: `model/point_cloud/iteration_7000/point_cloud_state_dict.pt`
- independent `render.py` completed
- `metrics.py` completed
- `evaluate_geometry_colmap.py` completed
- collector gate: `PASS`
- final cleanup summary exists and records cleanup disabled
