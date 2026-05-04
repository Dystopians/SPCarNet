# Final Stage F27 - Bonsai QEM No-Freeze Control Report

Decision: `FINAL_F27_BONSAI_QEM_NO_FREEZE_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`.

## Goal

Replicate the strict-topology-freeze failure mode on a third scene using the strong Open3D QEM50 compact operator. This control deliberately omits `--freeze_topology_updates` while otherwise matching the QEM50 recovery schedule.

## Run

| field | value |
| --- | --- |
| source compact checkpoint | `outputs/carnet/meshsplatopt/final_stageF22_bonsai_posthoc_qem_baseline/prune50/compact_model` |
| recovery checkpoint | `outputs/carnet/meshsplatopt/final_stageF27_bonsai_qem_no_freeze_control/prune50/recovery_model` |
| schedule | `22000->26000` |
| W&B | `0wskvq3h` |
| deliberate control | omitted `--freeze_topology_updates` |

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 88,460 | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 |
| QEM50 frozen 26k | 44,230 | 11.082405 | 0.243249 | 0.570177 | 0.182966 | 1.793852 | 42.889339 |
| QEM50 no-freeze 26k | 17,962 | 10.560091 | 0.176992 | 0.609218 | 0.229736 | 1.718488 | 46.233158 |

## Decision

No-freeze collapses the compact topology from `44,230` to `17,962` triangles and loses badly to frozen QEM50 on PSNR, SSIM, LPIPS, AbsRel, and normal. Together with the counter and room no-freeze controls, this establishes strict topology freezing as a replicated load-bearing mechanism across three scenes and two compact operators.

