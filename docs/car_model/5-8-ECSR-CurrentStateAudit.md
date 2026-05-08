# 5-8 ECSR Current-State Audit


This audit is generated before new ECSR implementation, following `docs/car_model/5-7-FinalDecision.md`. It locks the current protocol, result surface, bottleneck diagnosis, and leakage risks.


## Current Protocol Table


| scene | type | selected clean iter | score 26000 | score 30000 | score gap | method | policy | clean path | method path |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | outdoor | 26000 | 29.857 | 28.894 | +0.963 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/bicycle | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/bicycle/sor_adaptive_geo/compact_model |
| flowers | outdoor | 26000 | 22.027 | 21.060 | +0.968 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/flowers | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/flowers/sor_adaptive_geo/compact_model |
| garden | outdoor | 26000 | 36.604 | 35.623 | +0.981 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/garden | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/garden/sor_adaptive_geo/compact_model |
| stump | outdoor | 26000 | 33.428 | 32.347 | +1.081 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/stump | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/stump/sor_adaptive_geo/compact_model |
| treehill | outdoor | 26000 | 24.104 | 23.124 | +0.980 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/treehill | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/treehill/sor_adaptive_geo/compact_model |
| room | indoor | 26000 | 41.446 | 40.575 | +0.871 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/room | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/room/sor_adaptive_geo/compact_model |
| counter | indoor | 26000 | 38.953 | 37.772 | +1.181 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/counter | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/counter/sor_adaptive_geo/compact_model |
| kitchen | indoor | 26000 | 41.364 | 39.940 | +1.424 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/kitchen | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/kitchen/sor_adaptive_geo/compact_model |
| bonsai | indoor | 26000 | 41.633 | 40.156 | +1.477 | ours_26000_sor_adaptive_geo_compact_ela | sor_adaptive_geo | outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/bonsai | outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/bonsai/sor_adaptive_geo/compact_model |


## Current Result Table


| scene | clean PSNR/SSIM/LPIPS | SPCarNet PSNR/SSIM/LPIPS | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | tri red. | vertex red. | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 23.3016 / 0.6599 / 0.3321 | 23.9127 / 0.6937 / 0.2803 | +0.6111 | +0.0338 | -0.0518 | -0.000241 | -0.0204 | -0.0119 | 10.01% | 4.57% | STRICT ALL AXIS PASS |
| flowers | 19.6823 / 0.5118 / 0.3946 | 20.1828 / 0.5473 / 0.3510 | +0.5005 | +0.0355 | -0.0436 | -0.003356 | -0.1250 | -0.0439 | 10.02% | 4.64% | STRICT ALL AXIS PASS |
| garden | 25.0292 / 0.7800 / 0.2013 | 26.0348 / 0.8171 / 0.1523 | +1.0056 | +0.0371 | -0.0490 | -0.000007 | -0.0002 | -0.0010 | 1.50% | 2.69% | RGB COMPACT PASS GEOMETRY SAFE |
| stump | 25.2050 / 0.7052 / 0.2940 | 25.3625 / 0.7125 / 0.2817 | +0.1575 | +0.0074 | -0.0123 | -0.005878 | -0.3507 | -0.0260 | 10.02% | 4.57% | STRICT ALL AXIS PASS |
| treehill | 20.9342 / 0.5645 / 0.4060 | 21.1984 / 0.5882 / 0.3581 | +0.2642 | +0.0237 | -0.0479 | -0.001246 | -0.0747 | -0.0122 | 10.01% | 4.86% | STRICT ALL AXIS PASS |
| room | 28.7473 / 0.8848 / 0.2499 | 29.1310 / 0.8849 / 0.2487 | +0.3837 | +0.0000 | -0.0012 | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.03% | RGB COMPACT PASS GEOMETRY SAFE |
| counter | 26.7518 / 0.8621 / 0.2520 | 27.2404 / 0.8641 / 0.2497 | +0.4886 | +0.0021 | -0.0023 | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.10% | RGB COMPACT PASS GEOMETRY SAFE |
| kitchen | 27.8186 / 0.8765 / 0.1992 | 27.9996 / 0.8769 / 0.1989 | +0.1810 | +0.0005 | -0.0002 | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.29% | RGB COMPACT PASS GEOMETRY SAFE |
| bonsai | 28.8952 / 0.8964 / 0.2595 | 29.7844 / 0.8982 / 0.2574 | +0.8892 | +0.0018 | -0.0021 | -0.000368 | -0.0045 | -0.0254 | 10.00% | 3.16% | STRICT ALL AXIS PASS |


## Bottleneck Diagnosis Table


| scene | diagnosis | tri red. | selected faces | dPSNR | dLPIPS | local MAE drop | reason |
|---|---|---|---|---|---|---|---|
| bicycle | appearance-sensitive, occlusion-sensitive, texture-detail-sensitive, compression-friendly | 10.01% | 943594 | +0.611 | -0.0518 | 17.5% | train sparse geometry has enough occluder/depth residual evidence for SOR budget; local crop MAE drop 17.5%; 1302 sparse-occluder faces touched |
| flowers | appearance-sensitive, occlusion-sensitive, texture-detail-sensitive, compression-friendly | 10.02% | 966583 | +0.501 | -0.0436 | 24.2% | train sparse geometry has enough occluder/depth residual evidence for SOR budget; local crop MAE drop 24.2%; 1623 sparse-occluder faces touched |
| garden | appearance-sensitive, geometry-sensitive, occlusion-sensitive, texture-detail-sensitive, compression-hostile | 1.50% | 173579 | +1.006 | -0.0490 | 27.6% | train sparse geometry is already reliable; preserve geometry with a conservative low-evidence-only budget; low prune budget selected by geometry evidence; local crop MAE drop 27.6%; 58 sparse-occluder faces touched |
| stump | occlusion-sensitive, texture-detail-sensitive, compression-friendly | 10.02% | 930011 | +0.157 | -0.0123 | n/a | train sparse geometry has enough occluder/depth residual evidence for SOR budget; 2302 sparse-occluder faces touched |
| treehill | appearance-sensitive, occlusion-sensitive, texture-detail-sensitive, compression-friendly | 10.01% | 953798 | +0.264 | -0.0479 | 32.0% | train sparse geometry has enough occluder/depth residual evidence for SOR budget; local crop MAE drop 32.0%; 1034 sparse-occluder faces touched |
| room | geometry-sensitive, occlusion-sensitive, compression-hostile | 0.10% | 11173 | +0.384 | -0.0012 | n/a | train-only sparse evidence indicates high-confidence geometry but a rasterization/overdraw risk; use a micro topology budget; micro-prune chosen to preserve indoor geometry |
| counter | appearance-sensitive, geometry-sensitive, occlusion-sensitive, compression-hostile | 0.10% | 9851 | +0.489 | -0.0023 | 21.1% | train sparse geometry is already ultra stable; use a micro topology budget to preserve normals while keeping the ELA recovery path; micro-prune chosen to preserve indoor geometry; local crop MAE drop 21.1% |
| kitchen | geometry-sensitive, compression-hostile | 0.10% | 9716 | +0.181 | -0.0002 | n/a | train sparse geometry is already ultra stable; use a micro topology budget to preserve normals while keeping the ELA recovery path; micro-prune chosen to preserve indoor geometry |
| bonsai | appearance-sensitive, occlusion-sensitive, texture-detail-sensitive, compression-friendly | 10.00% | 1083638 | +0.889 | -0.0021 | 43.6% | train sparse geometry has enough occluder/depth residual evidence for SOR budget; local crop MAE drop 43.6%; 220 sparse-occluder faces touched |


## Leakage Risk Table


| step | evidence source | test involved? | current status | required replacement / guard |
|---|---|---|---|---|
| clean baseline selection | held-out test score over clean 26000/30000 | yes, for baseline selection only | acceptable evaluation protocol; not a method hyperparameter | keep fixed and report candidate envelope |
| SOR compaction candidate selection | train sparse geometry / low-evidence selector | no | valid train-evidence policy | keep; future ECSR must use policy-val certificates |
| ELA alpha / policy calibration | train rendered RGB/depth/camera evidence | no | valid for current archived method | future main method should attach residual to surface and report ELA as teacher/upper bound |
| README local crop showcase | held-out render/GT error reduction | yes, for presentation crop selection | presentation-only; invalid for method selection or paper local-metric protocol | replace with train-evidence top support masks projected to test in Phase A/D |
| final full9 report | held-out test metrics | yes, final evaluation | valid final reporting only | do not use for candidate rollback, threshold tuning, or alpha selection |


## Summary


- Git branch: `neurips-meshsplatopt-repair`
- Git commit: `27fef39`
- Tags at commit: `none`
- W&B collector: `rp0d5gr3`
- Report CSV: `outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean.csv`
- Scenes: `9`
- Strict all-axis pass: `5/9`
- RGB + compact + geometry-safe pass: `9/9`
- Mean delta vs selected clean: `+0.497941` PSNR, `+0.015755` SSIM, `-0.023373` LPIPS
- Mean delta vs MeshSplatting paper table: `+0.868512` PSNR, `+0.036551` SSIM, `-0.046530` LPIPS
- Mean geometry delta: `-0.001233` AbsRel, `-0.063943` DepthMAE, `-0.013388` Normal
- Mean triangle reduction: `5.7632%`
- Mean vertex reduction: `3.4332%`


## One-Paragraph Conclusion


The archived Compact-ELA/SOR version is a valid same-protocol RGB win over the selected clean MeshSplatting baseline, but it is not sufficient as the final top-conference contribution. The main evidence is that visual gains are still mostly localized residual corrections, mean triangle reduction remains only `5.7632%`, indoor scenes are protected by `0.1%` micro-pruning, and the current strongest RGB component is an image-space ELA adapter. Under FinalDecision, the next phase must prove train-evidence surface addressability of residuals and then move the recovery into representation-attached surface state with policy-val certificates, while keeping held-out test views final-report-only.
