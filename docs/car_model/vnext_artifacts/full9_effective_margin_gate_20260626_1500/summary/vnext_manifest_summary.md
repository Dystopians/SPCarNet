# vNext Certified Residual Surface Texture Summary

- run root: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500`
- compact artifact root: `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500`
- scenes found: `9`
- completed metric scenes: `9`
- accepted scenes: `1`
- fallback/rejected scenes: `8`
- mean changed fraction: `0.001371507`
- mean PSNR: `25.067410`
- mean SSIM: `0.741259`
- mean LPIPS: `0.306695`

| scene | status | protocol | accepted | policy | alpha | changed frac | policy gain | SSIM gain | L1 gain | PSNR | SSIM | LPIPS | report |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bicycle | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.023694383 | 0.000035246 | 0.000030915 | 23.293507 | 0.659651 | 0.332269 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/bicycle/reports/bicycle_vnext_certified_residual_texture_report.md` |
| bonsai | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.002950285 | 0.000065108 | 0.000036387 | 28.864376 | 0.896010 | 0.259334 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/bonsai/reports/bonsai_vnext_certified_residual_texture_report.md` |
| counter | COMPLETE | True | True | accepted_atlas | 0.125000 | 0.012343567 | 0.007941791 | 0.000050878 | 0.000006301 | 26.751171 | 0.862042 | 0.251955 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/counter/reports/counter_vnext_certified_residual_texture_report.md` |
| flowers | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.224441335 | -0.000082279 | -0.000000106 | 19.519194 | 0.490780 | 0.424170 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/flowers/reports/flowers_vnext_certified_residual_texture_report.md` |
| garden | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.088839874 | -0.000009457 | 0.000002707 | 24.741003 | 0.754049 | 0.248023 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/garden/reports/garden_vnext_certified_residual_texture_report.md` |
| kitchen | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.006631135 | -0.000039970 | 0.000009977 | 27.816387 | 0.876443 | 0.199201 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/kitchen/reports/kitchen_vnext_certified_residual_texture_report.md` |
| room | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | 0.072159743 | 0.000035971 | 0.000027366 | 28.739004 | 0.884790 | 0.249916 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/room/reports/room_vnext_certified_residual_texture_report.md` |
| stump | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.997520509 | 0.000005305 | 0.000002033 | 25.043329 | 0.689480 | 0.349850 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/stump/reports/stump_vnext_certified_residual_texture_report.md` |
| treehill | COMPLETE | True | False | fallback_noop | 0.000000 | 0.000000000 | -0.028940849 | 0.000001470 | 0.000017347 | 20.838715 | 0.558089 | 0.445541 | `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/treehill/reports/treehill_vnext_certified_residual_texture_report.md` |
