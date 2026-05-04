# Final Stage F46 - Unified CSEF Sparse-Depth Fairness Repair

Date: 2026-05-04

Decision: `F46_VALIDATION_BUDGET_CSEF_REPAIR_PASS_WITH_FIXED50_LIMITATION`.

## Goal

Respond to the F45 fairness audit by running a single CSEF selector family with explicit sparse-depth strict topology-frozen recovery. The first batch tests fixed CSEF50; the second batch tests conservative validation-selected CSEF budgets on the scenes where fixed CSEF50 was weak.

All rows use online W&B, `22000->26000`, `--freeze_topology_updates`, `--skip_restricted_delaunay`, `--enable_sparse_colmap_depth_loss`, low-error sparse COLMAP samples, independent render metrics, independent COLMAP geometry, and exact topology audit.

## Results vs Clean-Long

| scene | row | W&B | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bonsai | fixed CSEF50 + sparse-depth | `xpv6dd08` | 44,230 | 50.0% | 10.956095 | 0.224766 | 0.586397 | 0.185832 | 1.751883 | 43.179775 | +0.011747 | +0.001918 | +0.000239 | -0.008417 | -0.064527 | -2.178581 | `MIXED` |
| room | fixed CSEF50 + sparse-depth | `7fq1dnqk` | 42,253 | 50.0% | 14.387774 | 0.415018 | 0.567809 | 0.222497 | 1.583369 | 54.572975 | +0.129395 | +0.014154 | -0.011110 | +0.016215 | +0.103139 | -0.869678 | `MIXED` |
| room | validation-budget CSEF20 + sparse-depth | `v7ld1o0x` | 67,605 | 20.0% | 14.968359 | 0.466475 | 0.534423 | 0.203593 | 1.472770 | 53.973366 | +0.709980 | +0.065611 | -0.044496 | -0.002689 | -0.007460 | -1.469287 | `PASS_ALL_METRIC_CLEAN_WIN` |
| counter | fixed CSEF50 + sparse-depth | `vuvaul2s` | 41,917 | 50.0% | 14.074306 | 0.499067 | 0.468812 | 0.093763 | 0.433353 | 43.725219 | -0.061876 | -0.013735 | +0.016763 | +0.016767 | +0.063380 | -0.561816 | `FAIL` |
| counter | validation-budget CSEF40 + sparse-depth | `ihoyzp1a` | 50,300 | 40.0% | 14.212742 | 0.518842 | 0.450441 | 0.084020 | 0.399804 | 43.403673 | +0.076560 | +0.006040 | -0.001608 | +0.007024 | +0.029831 | -0.883362 | `MIXED` |
| counter | validation-budget CSEF30 + sparse-depth | `panxl9lh` | 58,684 | 30.0% | 14.286163 | 0.529584 | 0.440765 | 0.077596 | 0.377906 | 43.090673 | +0.149981 | +0.016782 | -0.011284 | +0.000600 | +0.007933 | -1.196362 | `MIXED` |
| counter | validation-budget CSEF20 + sparse-depth | `pijpv7ny` | 67,067 | 20.0% | 14.345424 | 0.536152 | 0.435706 | 0.074277 | 0.364962 | 43.105598 | +0.209242 | +0.023350 | -0.016343 | -0.002719 | -0.005011 | -1.181437 | `PASS_ALL_METRIC_CLEAN_WIN` |

## Reference Against Earlier CSEF Rows

| scene | row | comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonsai | fixed CSEF50 + sparse-depth | vs previous CSEF prune50 | -0.001402 | +0.000008 | -0.000018 | +0.000652 | +0.014068 | -0.314200 |
| room | fixed CSEF50 + sparse-depth | vs previous CSEF prune50 | +0.000611 | +0.000064 | -0.000472 | -0.002530 | -0.019661 | -0.069818 |
| counter | fixed CSEF50 + sparse-depth | vs previous CSEF prune50 | -0.003253 | +0.000093 | +0.000421 | -0.000968 | -0.005579 | -0.098171 |
| counter | validation-budget CSEF40 + sparse-depth | vs previous CSEF prune40 | +0.000709 | +0.000441 | -0.000040 | -0.001522 | -0.006569 | -0.073299 |

## Interpretation

F46 does not rescue the claim that one fixed CSEF50 hyperparameter is universally enough. Fixed CSEF50 remains weak on counter and mixed on room depth. That limitation should stay visible in the paper.

F46 does, however, materially repairs the fairness story: using the same CSEF selector family, sparse-depth strict recovery, and a conservative validation-selected budget, both previously weak public scenes now have all-metric clean-long wins. Room CSEF20 improves PSNR, SSIM, LPIPS, AbsRel, Depth MAE, and normal while keeping 20% topology reduction. Counter CSEF20 does the same while keeping 20% topology reduction. Counter CSEF30 is also near-all-metric, missing only small depth margins, and CSEF40 improves every tracked metric over the earlier CSEF40 row.

The safe claim is now: MeshSplatOpt supports a validation-selected CSEF-family compact-recovery protocol with conservative fallback budgets, and this protocol can produce all-metric clean-long wins on the formerly weak room/counter scenes without switching to QEM. The stronger F12 table may still use QEM rows as a posthoc simplification/operator baseline, but the method no longer depends on QEM to pass those scenes.
