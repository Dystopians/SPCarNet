# Parking clean-to-compact repair report

Date: 2026-05-03

## Why this report exists

The corrected clean-long comparison showed that R44 was not a render-quality win. R44.01 is useful as a low-topology / normal-proxy Pareto point, but it loses badly to the strongest clean long baseline on RGB and depth. The repair question was therefore not "can teacher loss polish R44?", but "can we keep most of the clean long quality while removing most of the clean long topology?"

## Negative controls

| run | start | intervention | W&B | iter | PSNR ↑ | SSIM ↑ | LPIPS ↓ | decision |
|---|---|---|---|---:|---:|---:|---:|---|
| R45.01 | R44.01 22k | full-image clean-render teacher, lambda 0.5, DSSIM 0.2 | `1vmbmftd` | 26k | 16.975 | 0.539 | 0.454 | rejected |
| R45.02 | R44.01 22k | full-image clean-render teacher, lambda 1.0, DSSIM 0.4 | `1lsrbnys` | 26k | 16.926 | 0.532 | 0.462 | rejected |
| R46.01 | R44.01 22k | counterfactual teacher mask, teacher-better pixels only | `awwaei5j` | 26k | 16.968 | 0.535 | 0.456 | rejected |

Decision: `LOW_TOPOLOGY_TEACHER_DISTILLATION_REJECTED`. Starting from the 0.78M-triangle R44 checkpoint is too constrained; teacher render supervision pulls appearance but cannot recover the missing representational capacity or geometry.

## Clean-to-compact path

| run | source | intervention | iter | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal ° ↓ | triangles |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean current branch | clean long | none | 22k | 18.480 | 0.635 | 0.347 | 0.082 | 1.868 | 45.108 | 8,548,242 |
| clean current branch | clean long | none | 30k | 18.409 | 0.632 | 0.351 | 0.082 | 1.866 | 44.839 | 8,548,242 |
| R44.01 | low-topology sparse decay | recovery | 22k | 17.170 | 0.549 | 0.442 | 0.187 | 2.919 | 42.218 | 782,982 |
| R47 prune80 | clean 22k | remove smallest-area 80% triangles | 22k | 17.976 | 0.600 | 0.387 | 0.081 | 1.849 | 45.000 | 1,709,648 |
| R47 prune90 | clean 22k | remove smallest-area 90% triangles | 22k | 16.093 | 0.503 | 0.462 | n/a | n/a | n/a | 854,824 |
| R48.01 | R47 prune80 | recovery 22k -> 26k | 26k | 18.620 | 0.642 | 0.349 | 0.080 | 1.847 | 44.743 | 1,709,648 |
| R49.01 | R48 26k | continuation 26k -> 30k, legacy topology controls | 30k | 18.361 | 0.629 | 0.361 | 0.082 | 1.836 | 45.356 | 934,205 |
| R50.01 | R48 26k | true fixed-topology continuation 26k -> 30k | 30k | 18.455 | 0.629 | 0.361 | 0.081 | 1.845 | 45.319 | 1,709,648 |

Decision: `CLEAN_TO_COMPACT_RECOVERY_PASS_EARLY_STOP_AT_26K`. R48.01 is the current parking mainline. It keeps only 20.0% of the clean long triangles, beats clean 22k on independent PSNR (+0.140 dB), SSIM (+0.007), AbsRel (-0.0019), and Depth MAE (-0.021), while LPIPS is nearly tied but slightly worse (+0.0025). It is also much stronger than R44.01 on render and depth while using about 2.18x the R44 topology.

R49.01 exposed a control bug in the old "fixed topology" recipe: `--skip_restricted_delaunay` skipped only the Delaunay refresh, while the standard 500-step pruning branch still ran before `densify_until_iter + 1000`. The new `--freeze_topology_updates` flag disables the standard prune/densify branch and the Delaunay refresh. R50.01 verifies the fix by preserving exactly 1,709,648 triangles through 30k, but it still loses render quality versus R48.01. Therefore the accepted checkpoint remains R48.01 at 26k, not the longer R50 continuation.

## Interpretation

The repaired claim is no longer "local hole/snap edits beat clean training." It is:

1. learn a strong clean long mesh,
2. compress geometry aggressively with evidence-compatible area pruning,
3. freeze topology,
4. recover appearance with long-horizon training,
5. verify against the best clean long baseline with independent render, metrics, and sparse COLMAP geometry.

This gives a defensible topology-quality Pareto: R48.01 is not the smallest topology point, but it is the first point that restores clean-level render quality with a major topology reduction.

## Reproducibility notes

Use `--freeze_topology_updates --skip_restricted_delaunay` for strict topology-frozen continuation. `--skip_restricted_delaunay` alone is insufficient for fixed-topology evidence.

W&B runs:

- R48.01: `1n6jv232`
- R49.01: `xdaixz33`
- R50.01: `zwafhpte`
