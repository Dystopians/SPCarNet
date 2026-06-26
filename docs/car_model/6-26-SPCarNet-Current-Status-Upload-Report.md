# SPCarNet Current Status and Clone Report

Date: 2026-06-26

This note is the latest clone-facing status addendum for mentor/PPT preparation. It updates the 2026-06-25 report package with the newest live experiment state and the current claim boundary.

Fresh clone:

```bash
git clone https://github.com/Dystopians/SPCarNet.git
cd SPCarNet
```

Start from:

```text
SPCARNET_REPORT_INDEX.md
docs/car_model/6-26-SPCarNet-Clone-PPT-Technical-Summary.zh.md
docs/car_model/6-26-SPCarNet-Mentor-PPT-Status-And-vNext-Strict-Report.zh.md
docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md
docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md
docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md
docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md
```

## Executive Status

The current verified quality line is still `v106 POD-MoE base-preserve`. It is a MeshSplatting-compatible surface residual method: keep the trained MeshSplatting parent, attach a triangle-addressed residual field, and use reliability gates so experts only act where train evidence supports them.

The main verified result is positive versus the local clean MeshSplatting baseline on the assembled selected full9 table:

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 |

The honest paper status is not final-complete. `v110/v110b` exposed strict train/even -> train/odd -> test failures; `v113b/v113c` repair safety by falling back to v106 where support is weak, but they do not create a stronger candidate than v106. `v114` is the active candidate-side attempt and has not produced a completed evaluation yet.

## Progress Achieved

- Built a cloneable report package with root index, manifest, bilingual README pointers, quantitative tables, qualitative contact sheets, and method logs.
- Implemented the current representation line: v104c shrink view-affine residual field plus v106 base-preserve POD-MoE detail and occlusion-boundary experts.
- Added train/test delta-bank split support and field-builder view subsets for strict split experiments.
- Added render-realized parent gates with calibration view subsets, lower-tail checks, out-of-trajectory support certificates, and frame-level fallback.
- Added v110 strict runner, v111 end-to-end strict runner, v113b/v113c replay runner, smoke tests, and report collectors.
- Completed the v106 selected full9 table and stored both Markdown/JSON/CSV summaries and qualitative contact-sheet assets in the repo.
- Completed v110/v110b flowers/garden diagnostics showing that a gate can beat clean MeshSplatting while still being worse than v106 parent.
- Completed v113b/v113c safety repairs on flowers/garden; useful as risk control, not as a new quality headline.
- Implemented vNext strict no-target-GT apply protocol, completed frozen-policy face-softshrink diagnostics on `counter,bonsai,room`, added the fixed structure-aware shrink policy, and extended the strict structure-aware table to ready scenes `counter,bonsai,room,garden`.
- Rebuilt the missing `stump` vNext input chain under `/dev/shm`, fixed a carrier-prune runner interface break, and completed a strict no-target-GT stump run. The run was safely rejected to fallback/no-op, so it improves full9 readiness but not the accepted-quality table.
- Rebuilt the missing `treehill` vNext input chain under `/dev/shm` and completed a strict no-target-GT treehill run. The run was safely rejected to fallback/no-op by lower-tail/SSIM/L1 certificates, moving input coverage to ready6 while preserving the claim boundary.
- Rebuilt the missing `flowers` vNext input chain under `/dev/shm`, completed strict no-target-GT evaluation with W&B offline, and added a same-evidence parent export. The run is also safely rejected to fallback/no-op, but the same-evidence comparison proves that fallback is exact under the rebuilt `images_2` target-evidence resolution.
- Rebuilt the missing `kitchen` vNext input chain under `/dev/shm`, completed strict no-target-GT evaluation with W&B offline, and added a same-evidence parent export. This is the first accepted nonzero result among the rebuilt missing scenes: `alpha=0.125`, `changed_fraction=0.003549714`, with same-evidence held-out delta `+0.000786 PSNR / +0.00000256 SSIM / -0.00002818 LPIPS`.

## Latest Live Experiment State

This snapshot has been extended after commit `93c8376` with local kitchen vNext rebuild evidence; the next pushed commit records the exact kitchen artifact package.

| job | local root | latest state |
|---|---|---|
| v110 counter strict candidate | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/counter` | **FAILED** during `build_train_even_candidate_field`; return code `-9`; no candidate field was written |
| v110 bonsai strict candidate | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/bonsai` | still running at the latest process check; field artifact not yet produced |
| v111 flowers end-to-end strict | `/dev/shm/peilincai_spcarnet_v111_end_to_end_strict_parent_gate_20260625/flowers` | still running at the latest process check; field artifact not yet produced |
| v114 garden OOF-refit POD-MoE | `/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625/garden` | still running at the latest process check; field artifact not yet produced |

The v110 counter failure report is local-only because it lives under `/dev/shm`:

```text
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/counter/reports/counter_v110_strict_split_parent_gate_report.json
```

Key failure facts:

- failed step: `build_train_even_candidate_field`
- return code: `-9`
- elapsed time: `15695.6s`
- expected missing field: `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/counter/fields/ours_26000_v110_strict_train_even_candidate_counter_field.pt`
- likely cause: CPU/shared-memory/storage pressure during final POD-MoE tensor assembly and save, not a metric-level rejection

Resource pressure at the same check was severe:

```text
RAM available: about 29 GiB
swap: 9 GiB used / 0 free
/dev/shm: 216 GiB used / 252 GiB total
/data: 100% used, about 420 MiB free
```

## Latest vNext Artifact State

| run | repo artifact | state |
|---|---|---|
| garden face-softshrink | `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/` | COMPLETE; protocol audit passed; `accepted=true`; `alpha=0.0625`; `changed_fraction=0.002080`; tiny three-metric gain versus no-op/fallback parent |
| counter strict face-softshrink | `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/` | COMPLETE; protocol audit passed; `target_gt_visible_to_apply=false`; `accepted=true`; `alpha=0.25`; `changed_fraction=0.01177355`; test `26.752003 / 0.862004 / 0.251912`; delta vs Phase-F compact parent `+0.00213 / -0.000047 / -0.000085` |
| bonsai strict face-softshrink | `docs/car_model/vnext_artifacts/bonsai_strict_face_softshrink_20260626_052500/` | COMPLETE; protocol audit passed; `target_gt_visible_to_apply=false`; `accepted=true`; `alpha=0.25`; `changed_fraction=0.00151333`; test `28.865564 / 0.896002 / 0.259322`; delta vs Phase-F compact parent `+0.001225 / -0.000010 / -0.000018` |
| room strict face-softshrink | `docs/car_model/vnext_artifacts/room_strict_face_softshrink_20260626_052500/` | COMPLETE; protocol audit passed; `target_gt_visible_to_apply=false`; `accepted=false`; fallback/no-op; `changed_fraction=0`; test `28.739004 / 0.884790 / 0.249916`; tiny delta vs Phase-F compact parent `-0.000097 / -0.000003 / -0.000007` is parent-level eval noise |
| strict frozen-policy aggregate | `docs/car_model/vnext_artifacts/strict_frozen_policy_multiscene_20260626_052500/strict_frozen_policy_multiscene_summary.md` | 3/3 complete; 3/3 protocol pass; 3/3 target GT hidden from apply; 2/3 nonzero accepted; mean delta `+0.001086 / -0.000020 / -0.000037` |
| structure-aware shrink counter | `docs/car_model/vnext_artifacts/counter_structure_shrink_tau002_20260626_0558/` | COMPLETE; protocol audit passed; `target_gt_visible_to_apply=false`; `accepted=true`; `alpha=0.125`; `changed_fraction=0.01234357`; delta vs Phase-F compact parent `+0.00129890 / -0.00000906 / -0.00004268` |
| structure-aware shrink bonsai | `docs/car_model/vnext_artifacts/bonsai_structure_shrink_tau002_20260626_0718/` | COMPLETE; protocol audit passed; `target_gt_visible_to_apply=false`; `accepted=true`; `alpha=0.25`; `changed_fraction=0.00148974`; delta vs Phase-F compact parent `+0.00113869 / -0.00000954 / -0.00001693` |
| structure-aware shrink room | `docs/car_model/vnext_artifacts/room_structure_shrink_tau002_20260626_0718/` | COMPLETE; protocol audit passed; `target_gt_visible_to_apply=false`; `accepted=true`; `alpha=0.0625`; `changed_fraction=0.00519912`; delta vs Phase-F compact parent `+0.00046921 / +0.00000334 / -0.00001399` |
| structure-aware shrink aggregate | `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_multiscene_20260626_0718/strict_structure_aware_shrink_multiscene_summary.md` | 3/3 complete; 3/3 protocol pass; 3/3 target GT hidden from apply; 3/3 nonzero accepted; mean delta `+0.00096893 / -0.00000509 / -0.00002453`; converts room from old fallback/no-op to accepted nonzero |
| structure-aware shrink garden | `docs/car_model/vnext_artifacts/garden_structure_shrink_tau002_20260626_071413/` | COMPLETE; protocol audit passed; `target_gt_visible_to_apply=false`; `accepted=true`; `alpha=0.125`; `changed_fraction=0.00205038`; delta vs Phase-F compact parent `+0.00013924 / +0.00000316 / -0.00000791`; also improves old garden face-softshrink by `+0.00006294 / +0.00000119 / -0.00000468` |
| structure-aware shrink ready4 aggregate | `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md` | 4/4 complete; 4/4 protocol pass; 4/4 target GT hidden from apply; 4/4 nonzero accepted; mean delta `+0.00076151 / -0.00000302 / -0.00002038` |
| manifest runner and full9 preflight | `docs/car_model/6-26-vNext-ManifestRunner-and-Full9Gap-Log.md` | per-scene manifest runner implemented; ready4 preflight is `4/4`; full9 gap preflight is `4/9` ready and `5/9` missing fit/target evidence plus carrier |
| ready4 scene config | `docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_scene_config_20260626.json` | machine-readable manifest runner config for `bonsai,counter,garden,room` |
| full9 gap scene config | `docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_scene_config_20260626.json` | machine-readable target full9 config; five missing scenes point at the planned normalized input tree |
| ready4/full9 preflight JSON | `docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.json`, `docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.json` | machine-readable readiness audit for clone-side review |
| stump input rebuild and strict rejection | `docs/car_model/6-26-vNext-StumpInputRebuild-Ready5-and-Rejection-Log.md` | `stump` fit/target evidence and carrier rebuilt locally; strict run completed with W&B offline, protocol pass, `accepted=false`, `fallback_noop`, `changed_fraction=0` |
| after-stump preflight | `docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.md` | local full9 preflight moved to `5/9` ready; remaining missing scenes are `bicycle,flowers,kitchen,treehill` |
| treehill input rebuild and strict rejection | `docs/car_model/6-26-vNext-TreehillInputRebuild-Ready6-and-Rejection-Log.md` | `treehill` fit/target evidence and carrier rebuilt locally; strict run completed with W&B offline, protocol pass, `accepted=false`, `fallback_noop`, `changed_fraction=0`; rejection reason includes `cvar20_view_relative_gain=-0.053640`, `min_view_relative_gain=-0.077837`, `ssim_gain=-0.000009413` |
| after-treehill preflight | `docs/car_model/vnext_artifacts/full9_gap_after_treehill_preflight_20260626/vnext_manifest_runner_summary.md` | local full9 preflight moved to `6/9` ready; remaining missing scenes are `bicycle,flowers,kitchen` |
| flowers input rebuild and strict same-evidence fallback | `docs/car_model/6-26-vNext-FlowersInputRebuild-Ready7-and-SameEvidenceFallback-Log.md` | `flowers` fit/target evidence and carrier rebuilt locally; strict run completed with W&B offline, protocol pass, `accepted=false`, `fallback_noop`, `changed_fraction=0`; same-evidence parent and fallback metrics are identical at `19.519194 / 0.490780 / 0.424170` |
| after-flowers preflight | `docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.md` | local full9 preflight moved to `7/9` ready; remaining missing scenes are `bicycle,kitchen` |
| kitchen input rebuild and strict accepted milestone | `docs/car_model/6-26-vNext-KitchenInputRebuild-Ready8-and-AcceptedMilestone-Log.md` | `kitchen` fit/target evidence and carrier rebuilt locally; strict run completed with W&B offline, protocol pass, `accepted=true`, `alpha=0.125`, `changed_fraction=0.003549714`; same-evidence delta is `+0.000786 PSNR / +0.00000256 SSIM / -0.00002818 LPIPS` |
| after-kitchen preflight | `docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/preflight/vnext_manifest_runner_summary.md` | local full9 preflight moved to `8/9` ready; remaining missing scene is `bicycle` |

The old strict frozen-policy face-softshrink result is a no-target-GT apply protocol milestone across three scenes, but it has 2/3 nonzero accepted and room fallback/no-op. The newer structure-aware shrink result is the preferred vNext milestone for PPT: it keeps strict no-target-GT apply, makes all four ready scenes accepted/nonzero, converts room into a positive three-metric row versus its Phase-F compact parent, and adds garden as a fourth strict scene. The effect size is still tiny and counter/bonsai still have extremely small SSIM regressions, so it should not be described as full9 closure or proof of superiority over v106/clean MeshSplatting.

## Current Claim Boundary

What is safe to say:

- SPCarNet currently has a real method line that beats the local clean MeshSplatting baseline on the assembled selected full9 evaluator.
- The method difference is substantive: MeshSplatting renders the parent checkpoint; SPCarNet attaches train-evidence residual experts to visible mesh triangles and gates them by reliability.
- The report package is cloneable and includes the quantitative/qualitative artifacts needed for a mentor presentation.

What is not safe to say:

- Do not claim the strict paper-final branch is closed.
- Do not claim v113b/v113c are quality breakthroughs; they are safety repairs that preserve or partially recover v106.
- Do not claim v114 improves quality until field build, render, eval, and collector summaries complete.
- Do not conflate v106 residual-field gains with earlier Phase-J triangle-reduction claims; v106 itself is not the triangle-pruning result.
- Do not claim vNext is superior to v106 or clean MeshSplatting; current vNext counter/bonsai/room/garden ready4 pilots are proof-of-life and protocol evidence with tiny or mixed deltas, stump/treehill/flowers are safety-rejected fallback/no-op negative results, and kitchen is an accepted nonzero micro-gain rather than a large visual breakthrough.

## Best PPT Story

The cleanest story is:

```text
MeshSplatting gives a strong parent render.
SPCarNet treats the mesh surface as an address space for train-evidence residuals.
v106 adds a base-preserving mixture of detail and boundary residual experts.
The result beats the local clean MeshSplatting baseline on selected full9.
Strict split experiments then reveal where naive candidate gates fail.
vNext then starts converting the Phase-J render-time teacher into strict no-target-GT residual surface texture.
The current next step is rebuilding the final bicycle evidence/carrier input, then turning the ready8 proof-of-life into fixed-policy full9, visible, three-metric gains.
```

For slides, use the v106 full9 table and the committed contact sheets under:

```text
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/
docs/car_model/assets/v106_qualitative/
```

## Next Engineering Step

The immediate blocker is not another hyperparameter scan. The field builder needs a lower-memory POD-MoE finalization path before rerunning v110 counter and other large strict scenes:

- avoid dense duplicated expert tensors during `expert_delta` construction;
- free intermediate normal-equation and split-stat tensors before payload assembly;
- run garbage collection before `torch.save`;
- relaunch counter after `/data` and `/dev/shm` pressure is relieved or after the builder is patched.

For the v106 strict branch, the required closure sequence is:

1. rerun v110 counter field/render/eval;
2. finish or rerun v110 bonsai, v111 flowers, and v114 garden;
3. recollect `spcarnet_v110_v111_v114_package`;
4. compare clean baseline, v106 parent, improved candidate, and fallback/gate ablations;
5. update README/report tables only after completed metrics exist.

For vNext, the required closure sequence is:

1. rebuild or recover missing fit/target evidence and policy-val pruned carrier for `bicycle`;
2. run full9 with the exact frozen structure-aware shrink policy through the manifest runner;
3. add a fixed comparison table with clean MeshSplatting, Phase-F parent, v104c, v106, Phase-J teacher, old face-softshrink, structure-aware shrink, no-certificate ablation, and exact fallback;
4. generate changed-region qualitative panels where residual changes are visually interpretable;
5. add budget accounting: triangle count, residual texture storage, parameter count, render overhead, fallback rate;
6. only promote vNext if it beats the chosen parent and clean baseline under the same frozen protocol.

## Final Status

`NOT COMPLETE`.

The report package is uploaded and useful for PPT preparation, but the research loop is not fully closed because strict branch long jobs are unfinished or failed, and vNext has delivered strict ready4 structure-aware proof-of-life metrics plus four rebuilt input chains (`stump`, `treehill`, `flowers`, `kitchen`) rather than full9, v106/clean-baseline superiority. The kitchen accepted row is real but tiny; `bicycle` remains the only missing-input scene.
