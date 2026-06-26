# v106 Base-Preserve Full9 Assembly

- v104c anchors: `outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.json`
- source priority: `counter, hardtriad, full9`
- available scenes: `9` / `9`
- warnings: `0`

| scene | source | method | PSNR | SSIM | LPIPS | v104c PSNR | dPSNR | v104c SSIM | dSSIM | v104c LPIPS | dLPIPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | full9 | v106_podmoe_basepreserve | 23.719175 | 0.675086 | 0.313405 | 23.717649 | +0.001526 | 0.674972 | +0.000115 | 0.313503 | -0.000098 |
| flowers | full9 | v106_podmoe_basepreserve | 20.077723 | 0.531240 | 0.374393 | 20.075844 | +0.001879 | 0.531076 | +0.000163 | 0.374473 | -0.000080 |
| garden | full9 | v106_podmoe_basepreserve | 25.790945 | 0.799382 | 0.174480 | 25.788094 | +0.002851 | 0.799263 | +0.000119 | 0.174584 | -0.000104 |
| stump | full9 | v106_podmoe_basepreserve | 25.460457 | 0.714661 | 0.282135 | 25.459311 | +0.001146 | 0.714599 | +0.000061 | 0.282213 | -0.000078 |
| treehill | full9 | v106_podmoe_basepreserve | 21.245092 | 0.578518 | 0.384177 | 21.243763 | +0.001329 | 0.578418 | +0.000099 | 0.384298 | -0.000121 |
| room | full9 | v106_podmoe_basepreserve | 29.600351 | 0.891889 | 0.230616 | 29.597836 | +0.002516 | 0.891837 | +0.000051 | 0.230664 | -0.000048 |
| counter | counter | v106_podmoe_basepreserve | 27.499645 | 0.867521 | 0.238847 | 27.498068 | +0.001577 | 0.867420 | +0.000102 | 0.238986 | -0.000139 |
| kitchen | hardtriad | v106_podmoe_basepreserve | 28.772043 | 0.881652 | 0.187815 | 28.770449 | +0.001595 | 0.881590 | +0.000062 | 0.188021 | -0.000206 |
| bonsai | hardtriad | v106_podmoe_basepreserve | 30.316090 | 0.907520 | 0.230050 | 30.310877 | +0.005213 | 0.907367 | +0.000154 | 0.230186 | -0.000136 |
| mean | selected |  | 25.831280 | 0.760830 | 0.268435 | 25.829099 | +0.002181 | 0.760727 | +0.000103 | 0.268548 | -0.000112 |

## Selected Sources

| scene | source path |
|---|---|
| bicycle | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/bicycle/bicycle_v105_evidence_gated_mixture_report.json` |
| flowers | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/flowers/flowers_v105_evidence_gated_mixture_report.json` |
| garden | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/garden/garden_v105_evidence_gated_mixture_report.json` |
| stump | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/stump/stump_v105_evidence_gated_mixture_report.json` |
| treehill | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/treehill/treehill_v105_evidence_gated_mixture_report.json` |
| room | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/room/room_v105_evidence_gated_mixture_report.json` |
| counter | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/counter/counter_v105_evidence_gated_mixture_report.json` |
| kitchen | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/kitchen/kitchen_v105_evidence_gated_mixture_report.json` |
| bonsai | `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/bonsai/bonsai_v105_evidence_gated_mixture_report.json` |
