# Final Stage F25 - Parking Posthoc QEM Baseline Report

Decision: `FINAL_F25_PARKING_QEM70_REJECT_UNMATCHED_COMPRESSION`.

## Goal

Test whether the Open3D QEM posthoc simplification baseline can match the parking headline topology target. The fair target is the same `2,564,473` triangles used by R53/F7, starting from the clean-long 22k checkpoint.

## Command

```bash
CUDA_VISIBLE_DEVICES=7 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py \
  --source_model outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model \
  --iteration 22000 \
  --output_model outputs/carnet/meshsplatopt/final_stageF25_parking_posthoc_qem_baseline/prune70/compact_model \
  --target_faces 2564473
```

## Result

| field | value |
| --- | ---: |
| source triangles | 8,548,242 |
| requested target triangles | 2,564,473 |
| produced triangles | 8,125,970 |
| removed triangles | 422,272 |
| removed fraction | 4.939870% |
| source vertices | 2,286,499 |
| produced vertices | 1,897,393 |
| invalid indices | 0 |
| degenerate faces | 0 |

The produced checkpoint is valid, but Open3D QEM did not reach the matched compression target on this 8.55M-triangle parking mesh. It removed only `4.94%` of triangles instead of the required `70.0%`.

## Decision

No W&B recovery was launched for this row because the topology budget is not comparable to R53/F7. Using it as a positive baseline would create a fairness error: it would retain `3.17x` more triangles than the accepted compact method. The correct interpretation is a rejected posthoc simplification control for parking, while the completed bonsai/courtyard/room/counter QEM rows remain valid matched-target controls on smaller public-scene meshes.

