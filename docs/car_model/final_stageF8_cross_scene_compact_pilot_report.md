# Final Stage F8 Cross-Scene Compact Pilot Report

Date: 2026-05-04

## Decision

`IN_PROGRESS`.

F8 is now structured and the first required missing clean-long run is launched. The stage is not allowed to claim cross-scene superiority yet because bonsai, courtyard, room, and counter do not all have completed fair clean-long baselines. Parking remains the only fully validated scene until at least one non-parking compact recovery beats its own clean-long baseline.

## Implemented

```text
scripts/car_model/final_run_cross_scene_compact_pilot.py
scripts/car_model/final_collect_cross_scene_compact_pilot.py
outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/
```

The manifest covers `bonsai`, `courtyard`, `room`, and `counter` with conservative 50/60/70/80 percent pruning for `csef_low_evidence_boundary_protected` and `area_smallest`. The collector marks scenes without clean-long baselines as `MISSING_BASELINE` and refuses to count them as wins.

## Active Clean-Long Run

```text
scene: bonsai
source baseline: outputs/carnet/meshsplatopt/stageR58_02_bonsai_clean_continue_7000to9000/recovery_model
target clean-long: outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000
dataset: /data/peilincai/mesh_datasets/mipnerf360/bonsai
images: images_4
load iteration: 9000
final iteration: 22000
W&B run: r8ozggn1
W&B URL: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r8ozggn1
```

## Current Scene Status

| scene | clean-long status | next action |
| --- | --- | --- |
| parking | PASS from F7 | keep as validated anchor, not sufficient for F8 PASS alone |
| bonsai | running 9k->22k | render, metrics, geometry, then run CSEF/area compact pilot |
| courtyard | missing clean-long | run only after bonsai establishes a non-parking path |
| room | missing clean-long | candidate third scene if dataset path is available |
| counter | missing clean-long | candidate third scene if dataset path is available |

## Gate

PASS requires at least two scenes with fair clean-long comparisons and compact rows satisfying:
- at least 50 percent triangle reduction;
- PSNR drop no worse than 0.2 dB;
- SSIM drop no worse than 0.01;
- LPIPS increase no worse than 0.02;
- no severe sparse geometry regression.

Current gate state: `NOT_PASSED_YET`. The correct next step is to finish bonsai clean-long, then launch a bonsai CSEF70 or CSEF60 strict topology-frozen compact recovery before broader sweeps.
