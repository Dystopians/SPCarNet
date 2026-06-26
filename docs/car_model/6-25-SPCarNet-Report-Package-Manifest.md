# SPCarNet Report Package Manifest

Date: 2026-06-25

This is the clone-facing manifest for the current SPCarNet technical-report package. Use it as the first file when preparing a mentor/PPT briefing from a fresh clone of:

```bash
git clone https://github.com/Dystopians/SPCarNet.git
cd SPCarNet
```

## Current Status

The verified quality line is `v106 POD-MoE base-preserve`. It improves over the local clean MeshSplatting baseline on the assembled selected full9 table:

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 |

The strict-fairness branch is not closed. `v110/v110b` exposed train/even -> train/odd -> test generalization failures; `v113b` adds lower-tail and out-of-trajectory support certificates; `v113c` narrows OOT fallback to frame level and partially improves garden, but it still does not beat v106. `v114_oof_refit_pod_moe` is the active candidate-side attempt and is still running locally.

## Read Order

| order | purpose | file |
|---:|---|---|
| 1 | root entry point | `SPCARNET_REPORT_INDEX.md` |
| 2 | clone/PPT executive technical summary | `docs/car_model/6-26-SPCarNet-Clone-PPT-Technical-Summary.zh.md` |
| 3 | latest 2026-06-26 live status addendum | `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md` |
| 4 | vNext structure-aware shrink milestone | `docs/car_model/6-26-vNext-StructureAwareShrink-Strict-Multiscene-Log.md` |
| 5 | vNext artifact index | `docs/car_model/vnext_artifacts/README.md` |
| 6 | vNext first implementation milestone | `docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md` |
| 7 | vNext feasibility and execution plan | `docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md` |
| 8 | cloneable report index | `docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md` |
| 9 | current mentor/PPT technical report | `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md` |
| 10 | long Chinese mentor report for slide preparation | `docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md` |
| 11 | v106 quality-line table | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md` |
| 12 | vNext structure-aware shrink table | `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_multiscene_20260626_0718/strict_structure_aware_shrink_multiscene_summary.md` |
| 13 | v113b strict-gate safety summary | `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md` |
| 14 | v113c/v114 continuation summary | `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md` |
| 15 | detailed v113c/v114 log | `docs/car_model/6-25-v113c-FrameFallback-and-v114-OOFRefit-Log.md` |
| 16 | strict branch mechanical collector output | `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md` |
| 17 | car-model docs catalog | `docs/car_model/README.md` |

## Quantitative Artifacts

| artifact | path |
|---|---|
| clone/PPT executive technical summary | `docs/car_model/6-26-SPCarNet-Clone-PPT-Technical-Summary.zh.md` |
| latest status addendum | `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md` |
| v106 assembled Markdown | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md` |
| v106 assembled JSON | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.json` |
| v106 assembled CSV | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.csv` |
| v106 comparison Markdown | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md` |
| v106 comparison JSON | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.json` |
| v113b summary Markdown | `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md` |
| v113b summary JSON | `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.json` |
| v113c/v114 summary Markdown | `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md` |
| v113c/v114 summary JSON | `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.json` |
| v110/v111/v114 strict branch package Markdown | `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md` |
| v110/v111/v114 strict branch package JSON | `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.json` |
| vNext structure-aware shrink aggregate Markdown | `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_multiscene_20260626_0718/strict_structure_aware_shrink_multiscene_summary.md` |

## Qualitative Artifacts

The current lightweight qualitative assets are committed under:

```text
docs/car_model/assets/v106_qualitative/
```

| scene | contact sheet |
|---|---|
| flowers | `docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.png` |
| garden | `docs/car_model/assets/v106_qualitative/garden_frame00000_crop_contact_sheet.png` |
| garden best crop | `docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png` |
| room | `docs/car_model/assets/v106_qualitative/room_frame00029_bestcrop_contact_sheet.png` |
| treehill | `docs/car_model/assets/v106_qualitative/treehill_frame00010_bestcrop_contact_sheet.png` |

Each PNG has a sibling JSON manifest with source render paths and crop metadata.

## Implemented Method Interfaces

| component | file | status |
|---|---|---|
| vNext scene runner | `scripts/car_model/run_vnext_certified_residual_texture_scene.py` | implemented; dry-run verified; real metric pilot pending resources |
| vNext full9 wrapper | `scripts/car_model/run_vnext_certified_residual_texture_full9.py` | implemented; two-scene dry-run verified; real full9 pending resources |
| vNext protocol helpers | `scripts/car_model/ecsr_vnext_protocol.py` | implemented; records command manifest and no-test-GT audit |
| vNext no-test-GT certificate smoke | `scripts/car_model/smoke_test_vnext_no_test_gt_certificate_schema.py` | implemented and passing |
| vNext report assembler | `scripts/car_model/assemble_vnext_certified_residual_texture_report.py` | implemented and dry-run verified |
| train/test delta-bank split | `scripts/car_model/run_v102_preprojected_delta_scene.py` | implemented |
| field-builder view subsets and v114 OOF reliability | `scripts/car_model/build_v105_evidence_gated_mixture_field.py` | implemented and smoke-tested |
| render-realized parent gate with lower-tail/OOT/frame fallback | `scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py` | implemented and smoke-tested |
| strict v110 runner | `scripts/car_model/run_v110_strict_split_parent_gate_scene.py` | implemented and smoke-tested |
| end-to-end strict v111 runner | `scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py` | implemented and smoke-tested |
| v113b/v113c replay runner | `scripts/car_model/run_v113b_oot_tail_gate_replay_scene.py` | implemented and smoke-tested |
| v110 report collector | `scripts/car_model/collect_v110_strict_split_report.py` | implemented |
| strict branch package collector | `scripts/car_model/collect_v110_v111_v114_package.py` | implemented; current output is pending until long jobs finish |

## Latest Running Jobs

These large local jobs are intentionally not committed because their artifacts live under `/dev/shm`:

| job | local root | status at report time |
|---|---|---|
| v110 counter strict candidate | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/counter` | failed during field build with return code `-9`; field artifact not produced |
| v110 bonsai strict candidate | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/bonsai` | latest log check around 105/128 train/even views; field artifact not yet produced |
| v111 flowers end-to-end strict | `/dev/shm/peilincai_spcarnet_v111_end_to_end_strict_parent_gate_20260625/flowers` | latest log check around 121/151 train/all views; field artifact not yet produced |
| v114 garden OOF-refit POD-MoE | `/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625/garden` | latest log check around 26/161 train/all views; field artifact not yet produced |

## PPT-Ready Takeaway

Compared with the start of the major rebuild, the project now has a reproducible method ladder, a local clean MeshSplatting comparison table, committed quantitative/qualitative artifacts, strict split interfaces, a documented safety failure, and a repaired OOT gate. The strongest honest claim is:

```text
SPCarNet adds a MeshSplatting-compatible, surface-addressed residual layer with reliability gates.
It beats the local clean MeshSplatting baseline in the current assembled full9 table,
but the strict paper-final branch is still not closed because the newest safe gates preserve v106
rather than producing a clearly stronger candidate.
```

## Chinese Brief

当前最可信的主线是 `v106 POD-MoE base-preserve`：它把 MeshSplatting 的 mesh/surface 当成地址空间，在三角形可见区域上挂接受控残差场，并通过可靠性 gate 避免明显破坏。它在本地 selected full9 表上相对 clean MeshSplatting 的 PSNR/SSIM/LPIPS 均值都更好。

必须诚实说明：严格公平分支还没有彻底闭环。`v110/v110b` 证明单纯 train/odd gate 会在 garden 发生 held-out 回退；`v113b/v113c` 修复了安全性，但不是比 v106 更强的质量突破。`v114` 已经把改进转向 candidate 生成侧，目前长程实验仍在跑。
