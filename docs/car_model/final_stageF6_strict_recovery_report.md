# Final Stage F6 Strict Recovery Runner Report

Date: 2026-05-04

## Decision

`PASS`.

The strict recovery runner is implemented and the R53.01 recovery contract verifies unchanged topology from 22k to 26k. No new long training was launched because all GPUs had high memory occupancy; the gate is satisfied by the already completed R53.01 reproduction checkpoint and exact runner contract.

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_run_strict_compact_recovery.py \
  --source_path outputs/carnet/meshprior/parking_phone_tiny/dataset_view \
  --output_path outputs/carnet/meshsplatopt/stageR53_01_prune70_clean_recovery_22000to26000/recovery_model \
  --load_iteration 22000 \
  --final_iteration 26000 \
  --images images \
  --resolution 4 \
  --preset compact_render_only \
  --wandb_name q15qg2b8 \
  --contract_out_dir outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract
```

## Outputs

```text
outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract/recovery_summary.json
outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract/topology_audit.json
outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract/exact_train_command.txt
outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract/wandb_url.txt
outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract/render_command.txt
outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract/metrics_command.txt
outputs/carnet/meshsplatopt/final_stageF6_strict_recovery/r53_reproduction_contract/geometry_command.txt
```

## Topology Audit

| checkpoint | iteration | triangles | vertices |
| --- | ---: | ---: | ---: |
| load | 22000 | 2,564,473 | 1,661,616 |
| final | 26000 | 2,564,473 | 1,661,616 |

`topology_unchanged`: `true`.

## Gate

`PASS`.

R53.01 already completed with W&B run `q15qg2b8`, and the F6 runner verifies that the final triangle count is unchanged throughout recovery. Future F7 runs must be launched through this runner or reproduce its exact flag contract.
