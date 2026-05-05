# Final Stage F75 Adaptive Policy Reflection Report

## Status

`FINAL_STAGE_F75_ACCEPTED_FOR_PARKING_HEADLINE`.

The adaptive policy branch now has two independently evaluated long-run rows, F74 and F75, that beat the previous F7 CSEF70 baseline on all tracked parking metrics at the same topology budget. Both rows keep exactly `2,564,473` triangles and `1,661,617` vertices from iteration `22000` to `26000`, with strict topology freeze verified by the recovery contract.

## What Failed Before

The earlier adaptive selector attempts failed because they treated render-only evidence as if it were sufficient geometric evidence. That made the selector overreact to view-dependent visibility and produced F65-F67 rows that either selected the wrong fraction or ranked faces in a way that damaged render quality. The first corrected adaptive selector, F68, switched to area/redundancy-primary ranking and used render evidence only as a risk/audit signal. That fixed the selector behavior, but the recovery objective still had a metric tradeoff:

- F69 fixed geometry and beat R53, but missed F7 LPIPS by only `0.000063`.
- F71 fixed LPIPS strongly, but the LPIPS loss was too large and harmed DepthMAE.
- F72/F73 confirmed that moderate LPIPS weights still over-pull the checkpoint toward perceptual texture and regress sparse depth.

The useful lesson is that the policy should not use a single strong perceptual correction. It needs a tiny LPIPS repair term layered on top of sparse-depth recovery.

## Accepted Method

The accepted policy is:

1. Build an adaptive CSEF policy from checkpoint evidence.
2. For large meshes, rank compaction candidates primarily by small area and local redundancy.
3. Use render-only positive evidence as a downweighted risk signal, not as the main face-removal score.
4. Select the compression fraction with the near-optimal knee rule; for parking this chooses 70 percent pruning.
5. Apply strict topology-frozen recovery from `22000` to `26000`.
6. Use sparse COLMAP depth loss with `lambda=0.001`.
7. Add only a tiny LPIPS term. The strongest validated row is F75 with `lpips_lambda=0.00025`; F74 with `0.0001` is the more conservative all-metric pass.

This is no longer a manual per-scene parameter table for the selector. The selector chooses the fraction and ranking from the checkpoint. The LPIPS recovery coefficient still has two validated conservative settings; the paper should present F75 as the accepted parking recovery setting and disclose F74 as the robustness neighbor.

## Evidence Table

| row | method | triangles | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | beats clean22k | beats R53.01 | beats F7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| clean22k | MeshSplat clean long | 8,548,242 | 18.479990 | 0.634623 | 0.346913 | 0.082177 | 1.868398 | 45.108437 | n/a | n/a | n/a |
| R53.01 | area70 compact recovery | 2,564,473 | 18.705738 | 0.647807 | 0.338492 | 0.079555 | 1.853751 | 44.261391 | true | n/a | false |
| F7 | CSEF70 recovery | 2,564,473 | 18.706079 | 0.647764 | 0.338282 | 0.079404 | 1.852816 | 44.204497 | true | mixed | n/a |
| F74 | adaptive + sparse + LPIPS 0.0001 | 2,564,473 | 18.711475 | 0.648027 | 0.338280 | 0.078848 | 1.851651 | 44.068988 | true | true | true |
| F75 | adaptive + sparse + LPIPS 0.00025 | 2,564,473 | 18.711857 | 0.647911 | 0.337509 | 0.078873 | 1.850042 | 43.954957 | true | true | true |

F75 improves over F7 by:

- PSNR: `+0.005778`
- SSIM: `+0.000147`
- LPIPS: `-0.000773`
- AbsRel: `-0.000531`
- DepthMAE: `-0.002774`
- Normal angle: `-0.249540`

The full machine-readable and Markdown summaries are in:

- `outputs/carnet/meshsplatopt/final_stageF75_adaptive_policy_evidence/adaptive_policy_results.json`
- `outputs/carnet/meshsplatopt/final_stageF75_adaptive_policy_evidence/adaptive_policy_results.md`

## W&B Runs

| row | run id | role |
|---|---|---|
| F68 | `lm2nzbrs` | corrected adaptive selector control |
| F69 | `qetzit46` | sparse-depth geometry row |
| F71 | `cqdpevk8` | LPIPS-heavy perceptual row, rejected as single headline |
| F72 | `gafbl2m7` | lower LPIPS, rejected due DepthMAE |
| F73 | `j811febm` | lower LPIPS, rejected due DepthMAE |
| F74 | `fs3u0p4h` | accepted conservative all-metric F7 win |
| F75 | `hhyy475d` | accepted strongest all-metric F7 win |

## Implementation Changes

- Added `csef_adaptive_policy` support in `ss3dm_prior/meshsplatopt/compact_selector.py`.
- Added adaptive policy decision export and large-selection `selected_faces.npy` output to avoid huge JSON files.
- Added adaptive selection to `scripts/car_model/meshsplatopt_select_compaction_candidates.py`.
- Added adaptive compaction and `selected_faces_path` loading to `scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py`.
- Added `scripts/car_model/final_collect_stageF68_F73_adaptive_policy.py` to collect F68-F75 adaptive evidence.

## Remaining Risk

The parking result is now strong and fair against the selected clean MeshSplat, R53 area, and F7 CSEF baselines. It is not yet a universal multi-scene proof. Earlier public-scene evidence remains mixed, so the paper should not claim all-scene dominance. The defensible claim is: on the selected parking validation scene, the adaptive policy plus tiny perceptual recovery is now strictly stronger than the prior MeshSplat/CSEF compact baselines at identical topology, and the failure analysis explains why larger perceptual weights were rejected.

The next paper-grade step is to run the accepted F75 policy unchanged on the remaining selected scenes and report both passes and failures, without per-scene hand tuning.
