# SPCarNet Cloneable Report Index

Date: 2026-06-25

This index is the entry point for a fresh clone of `https://github.com/Dystopians/SPCarNet.git`. It lists the files needed for a mentor/PPT technical analysis without depending on large transient `/dev/shm` render trees.

## Read First

| purpose | file |
|---|---|
| report package manifest | `docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md` |
| latest 2026-06-26 live status addendum | `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md` |
| vNext first implementation milestone | `docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md` |
| vNext feasibility and execution plan | `docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md` |
| vNext garden soft-shrink milestone | `docs/car_model/6-26-SPCarNet-vNext-SoftShrink-Garden-Milestone-Log.md` |
| vNext PPT technical report and artifact index | `docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md` |
| vNext artifact directory index | `docs/car_model/vnext_artifacts/README.md` |
| current complete technical report | `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md` |
| long Chinese mentor/PPT report | `docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md` |
| v110 strict-split technical report draft | `docs/car_model/6-25-SPCarNet-v110-StrictSplit-Technical-Report-Draft.md` |
| v110 execution log | `docs/car_model/6-25-v110-StrictSplitParentGate-Log.md` |
| v113b OOT tail-safe gate log | `docs/car_model/6-25-v113b-OOT-Tail-Safe-Gate-Log.md` |
| v113c/v114 continuation log | `docs/car_model/6-25-v113c-FrameFallback-and-v114-OOFRefit-Log.md` |
| v106 quality-line technical report | `docs/car_model/6-25-v106-PODMoE-Technical-Report-Draft.md` |
| v106 implementation/progress log | `docs/car_model/6-25-v106-PODMoE-BasePreserve-HardTriad-Log.md` |
| v109 parent-gate log | `docs/car_model/6-25-v109-RenderRealizedParentGate-Log.md` |

## Quantitative Result Package

| result | file |
|---|---|
| v106 full9 assembled table | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md` |
| v106 vs v104c and clean comparison | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md` |
| v106 assembled JSON | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.json` |
| v106 comparison JSON | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.json` |
| v110 default strict-split summary | `docs/car_model/results/v110_strict_split_20260625/summary/v110_strict_split_parent_gate_summary.md` |
| v110b flowers/garden strict diagnostic | `docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md` |
| v113b OOT tail-safe strict diagnostic | `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md` |
| v113c frame-fallback and v114 running summary | `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md` |
| v110/v111/v114 strict branch package | `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md` |
| vNext initial garden fallback result | `docs/car_model/vnext_artifacts/garden_20260626_004134/garden_ours_26000_vnext_certified_residual_surface_texture_test_results.json` |
| vNext hard-bin soft-shrink diagnostic | `docs/car_model/vnext_artifacts/garden_hardbin_softshrink_20260626_035631/garden_ours_26000_vnext_certified_residual_surface_texture_test_results.json` |
| vNext face-softshrink accepted result | `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_ours_26000_vnext_certified_residual_surface_texture_test_results.json` |
| vNext face-softshrink structured summary | `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_summary.json` |

## Qualitative Result Package

| scene | contact sheet |
|---|---|
| flowers | `docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.png` |
| garden | `docs/car_model/assets/v106_qualitative/garden_frame00000_crop_contact_sheet.png` |
| garden best crop | `docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png` |
| vNext garden face-softshrink | `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png` |
| room | `docs/car_model/assets/v106_qualitative/room_frame00029_bestcrop_contact_sheet.png` |
| treehill | `docs/car_model/assets/v106_qualitative/treehill_frame00010_bestcrop_contact_sheet.png` |

Each contact sheet has a sibling `.json` manifest with source render paths and crop metadata.

## Implemented Scripts

| script | role |
|---|---|
| `scripts/car_model/build_v105_evidence_gated_mixture_field.py` | builds v105/v106/v108-style surface residual fields; now supports `--view_subset all/even/odd` for strict split experiments |
| `scripts/car_model/run_v102_preprojected_delta_scene.py` | builds preprojected delta banks; now supports `--target_split train/test` |
| `scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py` | render-realized parent gate; now supports calibration view subsets |
| `scripts/car_model/run_v110_strict_split_parent_gate_scene.py` | strict candidate/gate runner with v106 parent and train/even -> train/odd -> test protocol |
| `scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py` | end-to-end strict parent/candidate/gate runner; parent is rebuilt from train/all |
| `scripts/car_model/run_v113b_oot_tail_gate_replay_scene.py` | gate/eval-only replay runner for prebuilt strict candidates using the v113b lower-tail and OOT support certificates |
| `scripts/car_model/smoke_test_v114_oof_refit.py` | synthetic verification for the v114 OOF positive-cap reliability rule |
| `scripts/car_model/collect_v110_strict_split_report.py` | collects clean/v106/v110 strict-split metrics into Markdown/JSON summaries |
| `scripts/car_model/collect_v110_v111_v114_package.py` | collects the current strict-branch package status and metrics for v110 counter/bonsai, v111 flowers, and v114 garden |
| `scripts/car_model/run_vnext_certified_residual_texture_scene.py` | vNext certified residual surface texture scene runner; now supports face-softshrink by enabling soft bin uncertainty shrink and optionally disabling hard bin allowlisting |
| `scripts/car_model/run_vnext_certified_residual_texture_full9.py` | vNext multi-scene wrapper with `{scene}` templates |
| `scripts/car_model/assemble_vnext_certified_residual_texture_report.py` | assembles vNext scene manifests and metrics |
| `scripts/car_model/smoke_test_vnext_no_test_gt_certificate_schema.py` | verifies vNext protocol audit schema and no-test-GT certificate fields |

## Smoke and Static Checks

The current report package records these checks as passing:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py \
  scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py \
  scripts/car_model/run_v113b_oot_tail_gate_replay_scene.py \
  scripts/car_model/build_v105_evidence_gated_mixture_field.py \
  scripts/car_model/smoke_test_v110_strict_runner_args.py \
  scripts/car_model/smoke_test_v111_runner_args.py \
  scripts/car_model/smoke_test_v109_oot_gate.py \
  scripts/car_model/smoke_test_v113b_replay_runner_args.py \
  scripts/car_model/smoke_test_v114_oof_refit.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v110_strict_runner_args.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v111_runner_args.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v109_oot_gate.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v113b_replay_runner_args.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v114_oof_refit.py
```

## Current Local Runtime Roots

The repo stores lightweight report artifacts. Large render/model trees remain local:

| root | meaning |
|---|---|
| `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625` | detached model package, parent/candidate/gated renders, scene `results.json` |
| `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_field` | v106 field tensors |
| `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625` | v110/v110b strict-split working directory |
| `/dev/shm/peilincai_spcarnet_v111_end_to_end_strict_parent_gate_20260625` | v111 end-to-end strict working directory |
| `/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625` | v114 OOF-refit POD-MoE working directory |
| `/dev/shm/peilincai_spcarnet_vnext_softshrink_garden_20260626_035631` | vNext hard-bin soft-shrink garden diagnostic, copied in lightweight form under `docs/car_model/vnext_artifacts/` |
| `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558` | vNext face-softshrink accepted garden run, copied in lightweight form under `docs/car_model/vnext_artifacts/` |

These roots are not committed because they contain large render/model artifacts.

## One-Line Status

`v106` is the current quality line and beats the local clean MeshSplatting baseline on the assembled selected full9 table. `v113b` is the current strict-gate safety repair. `v113c` improves garden v110b but remains below v106, and `v114` is the active candidate-side attempt now running. The paper-final method is not closed. Latest live update: v110 counter failed during field build with return code `-9`, likely due to memory/shared-storage pressure, so it must be rerun after a lower-memory field-builder fix or resource cleanup.
