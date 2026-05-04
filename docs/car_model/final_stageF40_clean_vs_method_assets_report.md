# Final Stage F40 - Clean-Long Baseline vs Method-Best Assets

Decision: `FINAL_F40_FAIR_QUALITATIVE_AND_CLAIM_AUDIT_PASS`.

## Assets

- montage: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/clean_long_22k_vs_method_best_26k_montage.png`
- manifest: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/clean_long_22k_vs_method_best_manifest.json`
- parking_phone_tiny panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/panels/parking_phone_tiny_clean_long_vs_method_best.png`
- bonsai panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/panels/bonsai_clean_long_vs_method_best.png`
- courtyard panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/panels/courtyard_clean_long_vs_method_best.png`
- room panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/panels/room_clean_long_vs_method_best.png`
- counter panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/panels/counter_clean_long_vs_method_best.png`

## Fairness Contract

This package compares each final method row only against the scene-matched clean-long 22k baseline from F12. It intentionally excludes old 7k parking baselines and excludes control rows from the main qualitative panel, so the visual comparison matches the final quantitative claim.

## Quantitative Audit

| scene | clean-long triangles | method triangles | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| parking_phone_tiny | 8,548,242 | 2,564,473 | 70.0% | 0.232330 | 0.012730 | -0.008741 | -0.002929 | -0.013985 | -1.072292 | 00005.png |
| bonsai | 88,460 | 44,230 | 50.0% | 0.137266 | 0.020400 | -0.016500 | -0.012551 | -0.036627 | -2.932622 | 00005.png |
| courtyard | 1,677,484 | 838,742 | 50.0% | 0.452301 | 0.041625 | -0.024231 | -0.032415 | -0.220612 | 0.008508 | 00002.png |
| room | 84,506 | 42,253 | 50.0% | 0.802811 | 0.080218 | -0.062114 | -0.025153 | -0.135009 | -0.541874 | 00005.png |
| counter | 83,834 | 50,300 | 40.0% | 0.272587 | 0.034768 | -0.031847 | -0.008982 | -0.030858 | -0.701820 | 00005.png |

## Safe Claims

- Render quality: `5/5` scenes improve PSNR and SSIM while reducing LPIPS versus clean-long 22k.
- Sparse depth proxies: `5/5` scenes improve AbsRel and Depth MAE versus clean-long 22k.
- Normal proxy: `4/5` scenes improve normal angle; courtyard is essentially tied but slightly worse by `0.008508` degrees, so the paper should not claim all-scene normal dominance.
- Topology: reductions range from `40.0%` to `70.0%` while keeping the render/depth wins above.

## Unsafe Claims To Avoid

- Do not compare the final method to parking clean 7k; F40 and F12 use clean-long 22k.
- Do not claim universal dominance over every geometry proxy; F37 fast-QEM improves sparse geometry proxies on parking but collapses render quality.
- Do not claim longer recovery always helps; F34 shows 30k continuation hurts render quality on parking.

## Gate

PASS. `all_render_wins=True`, `all_depth_wins=True`, `normal_wins=4/5`. The final qualitative package is now aligned with the strongest clean-long baselines and with the F12 quantitative table.
