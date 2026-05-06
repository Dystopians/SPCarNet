# Stage ELA9/10 Strict Multi-Axis Audit

Decision: `STRICT_MULTIAXIS_PARTIAL_ROOM_SOLVED_NOT_FULLY_CROSS_SCENE`.

This audit uses the stricter definition requested after the ELA7 RGB-only result: a method must improve PSNR, SSIM, LPIPS, sparse-depth AbsRel, sparse Depth MAE, sparse normal angle, and reduce triangle count against the strongest clean baseline for that scene. Under this definition, the selected-scene problem is only partially solved: room now has strict full-pass rows, but the same fixed policy still needs full replication on bonsai, courtyard, and counter.

Strict full-pass rows on selected clean9000 scenes: `2/18`.
Selected scenes with at least one strict full-pass row: `room`.

## Selected Scenes vs Strong Clean9000

| scene | method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction | RGB | geom | topo | full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| bonsai | ELA7 Pareto evidence portfolio | 1.338095 | 0.058450 | -0.024570 | 0.000000 | 0.000000 | 0.000000 | 0.00% | `True` | `False` | `False` | `False` |
| bonsai | legacy compact-recovery | -7.459511 | -0.220248 | 0.086393 | -0.019841 | -0.412007 | 1.769674 | 98.22% | `False` | `False` | `True` | `False` |
| courtyard | ELA7 Pareto evidence portfolio | 0.203512 | 0.011715 | -0.018647 | 0.000000 | 0.000000 | 0.000000 | 0.00% | `True` | `False` | `False` | `False` |
| courtyard | legacy compact-recovery | -5.942103 | -0.263586 | 0.121747 | 0.191396 | 2.026047 | 2.883891 | -104.44% | `False` | `False` | `False` | `False` |
| room | ELA7 Pareto evidence portfolio | 2.751766 | 0.043678 | -0.052810 | 0.000000 | 0.000000 | 0.000000 | 0.00% | `True` | `False` | `False` | `False` |
| room | legacy compact-recovery | -11.155910 | -0.408290 | 0.381717 | 0.148104 | 0.960950 | 5.168381 | 78.45% | `False` | `False` | `True` | `False` |
| counter | ELA7 Pareto evidence portfolio | 2.413528 | 0.060425 | -0.059244 | 0.000000 | 0.000000 | 0.000000 | 0.00% | `True` | `False` | `False` | `False` |
| counter | legacy compact-recovery | -10.393161 | -0.296881 | 0.260965 | 0.045113 | 0.163487 | 7.289541 | 68.85% | `False` | `False` | `True` | `False` |
| room | clean9000 area50 teacher+rollback recovery pilot | 0.315981 | 0.006854 | -0.007655 | 0.000040 | 0.001023 | 0.176077 | 50.00% | `True` | `False` | `True` | `False` |
| room | clean9000 area50 sparse+teacher+rollback recovery pilot | 0.315977 | 0.006854 | -0.007656 | 0.000040 | 0.001023 | 0.176077 | 50.00% | `True` | `False` | `True` | `False` |
| room | clean9000 QEM50 sparse-depth teacher+rollback recovery | 0.153080 | 0.003044 | -0.002497 | 0.000052 | 0.001910 | -0.213866 | 50.00% | `True` | `False` | `True` | `False` |
| room | clean9000 QEM50 compact checkpoint | 0.063013 | -0.002101 | 0.006003 | 0.000033 | -0.001857 | -0.088153 | 50.00% | `False` | `False` | `True` | `False` |
| room | clean9000 QEM50 compact + ELA safe | 2.820644 | 0.043782 | -0.053010 | 0.000033 | -0.001857 | -0.088153 | 50.00% | `True` | `False` | `True` | `False` |
| room | clean9000 QEM30 sparse-depth teacher+rollback recovery | -0.062401 | -0.001039 | 0.000479 | 0.000221 | 0.004322 | -0.132440 | 30.00% | `False` | `False` | `True` | `False` |
| room | clean9000 QEM30 compact + ELA safe | 2.773758 | 0.044134 | -0.052992 | 0.000074 | -0.000198 | -0.044555 | 30.00% | `True` | `False` | `True` | `False` |
| room | clean9000 QEM20 sparse-depth teacher+rollback recovery | -0.127014 | -0.002682 | 0.001640 | 0.000230 | 0.004847 | -0.141198 | 20.00% | `False` | `False` | `True` | `False` |
| room | clean9000 QEM50 sparse parent-rollback recovery | 0.692919 | 0.013745 | -0.015990 | -0.002331 | -0.019509 | -1.824378 | 50.00% | `True` | `True` | `True` | `True` |
| room | clean9000 QEM50 sparse parent-rollback + ELA safe | 3.304691 | 0.050085 | -0.062170 | -0.002331 | -0.019509 | -1.824378 | 50.00% | `True` | `True` | `True` | `True` |

## Cross-Dataset Existing Evidence

Parking is included as an additional dataset/scene. It demonstrates that compact-recovery can win all tracked axes against its fair clean-long baseline, but it is not the same as proving the ELA7 clean9000 selected-scene method is fully solved.

| scene | method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction | full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| parking_phone_tiny | CSEF70+sparse-depth compact recovery | 0.232340 | 0.013107 | -0.008653 | -0.003106 | -0.014383 | -1.072729 | 70.00% | `True` |

## Interpretation

- ELA7 is a strong RGB method, but it inherits clean9000 geometry and topology. It cannot satisfy geometry/triangle-count superiority.
- Legacy compact-recovery rows often reduce triangles, but when compared to the stronger clean9000 baselines on bonsai/courtyard/room/counter, they lose RGB by large margins and often lose sparse geometry too.
- ELA10 changes the branch from parameter scanning to a fixed recovery policy: strong clean9000 checkpoint -> QEM compact topology -> topology-frozen recovery with train-only sparse parent rollback, checkpoint geometry anchoring, parent render rollback -> ELA-style appearance evidence.
- On room, the QEM50 sparse parent-rollback checkpoint is the first strict full-pass against clean9000 across RGB, sparse geometry, and triangle count. The parent-rollback + ELA row is the strongest current room result.
- This is not yet a global selected-scene claim. The fixed ELA10 policy must now be replicated on bonsai, courtyard, and counter, and cross-dataset rows should remain separated from selected-scene claims.

## Artifacts

- JSON: `outputs/carnet/meshsplatopt/stageELA9_strict_multiaxis_audit/strict_multiaxis_audit.json`
- selected-scene CSV: `outputs/carnet/meshsplatopt/stageELA9_strict_multiaxis_audit/selected_scene_strict_rows.csv`
- cross-dataset CSV: `outputs/carnet/meshsplatopt/stageELA9_strict_multiaxis_audit/cross_dataset_rows.csv`
