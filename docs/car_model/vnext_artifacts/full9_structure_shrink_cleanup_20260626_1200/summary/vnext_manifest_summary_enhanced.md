# vNext Certified Residual Surface Texture Summary

- run root: `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200`
- compact artifact root: `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_compact_20260626_1200`
- scenes found: `9`
- completed metric scenes: `9`
- accepted scenes: `6`
- fallback/rejected scenes: `3`
- mean changed fraction: `0.002756271`
- mean PSNR: `25.067699`
- mean SSIM: `0.741260`
- mean LPIPS: `0.306689`

| scene | status | protocol | accepted | policy | alpha | changed frac | policy gain | SSIM gain | L1 gain | PSNR | SSIM | LPIPS | report |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bicycle | COMPLETE | True | True | accepted_atlas | 0.015625 | 0.000173916 | 0.000000000 | 0.000000571 | 0.000000277 | 23.293516 | 0.659651 | 0.332269 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/bicycle/reports/bicycle_vnext_certified_residual_texture_report.md` |
| bonsai | COMPLETE | True | True | accepted_atlas | 0.250000 | 0.001489739 | 0.007119612 | 0.000016049 | 0.000003021 | 28.865479 | 0.896003 | 0.259323 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/bonsai/reports/bonsai_vnext_certified_residual_texture_report.md` |
| counter | COMPLETE | True | True | accepted_atlas | 0.125000 | 0.012343567 | 0.007941791 | 0.000050878 | 0.000006301 | 26.751171 | 0.862042 | 0.251955 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/counter/reports/counter_vnext_certified_residual_texture_report.md` |
| flowers | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.224441335 | -0.000082279 | -0.000000106 | 19.519194 | 0.490780 | 0.424170 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/flowers/reports/flowers_vnext_certified_residual_texture_report.md` |
| garden | COMPLETE | True | True | accepted_atlas | 0.125000 | 0.002050379 | 0.003581617 | 0.000002772 | 0.000000317 | 24.741142 | 0.754052 | 0.248015 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/garden/reports/garden_vnext_certified_residual_texture_report.md` |
| kitchen | COMPLETE | True | True | accepted_atlas | 0.125000 | 0.003549714 | 0.006050986 | 0.000020564 | 0.000002900 | 27.817173 | 0.876445 | 0.199172 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/kitchen/reports/kitchen_vnext_certified_residual_texture_report.md` |
| room | COMPLETE | True | True | accepted_atlas | 0.062500 | 0.005199120 | 0.001577048 | 0.000012413 | 0.000001152 | 28.739571 | 0.884797 | 0.249909 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/room/reports/room_vnext_certified_residual_texture_report.md` |
| stump | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.172453639 | 0.000000762 | 0.000000298 | 25.043329 | 0.689480 | 0.349850 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/stump/reports/stump_vnext_certified_residual_texture_report.md` |
| treehill | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.053640114 | -0.000009413 | 0.000008294 | 20.838715 | 0.558089 | 0.445541 | `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/treehill/reports/treehill_vnext_certified_residual_texture_report.md` |
