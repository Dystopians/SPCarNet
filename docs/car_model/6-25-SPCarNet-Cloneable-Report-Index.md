# SPCarNet Cloneable Report Index

Date: 2026-06-25

This index is the entry point for a fresh clone of `https://github.com/Dystopians/SPCarNet.git`. It lists the files needed for a mentor/PPT technical analysis without depending on large transient `/dev/shm` render trees.

## Read First

| purpose | file |
|---|---|
| current complete technical report | `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md` |
| v110 strict-split technical report draft | `docs/car_model/6-25-SPCarNet-v110-StrictSplit-Technical-Report-Draft.md` |
| v110 execution log | `docs/car_model/6-25-v110-StrictSplitParentGate-Log.md` |
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

## Qualitative Result Package

| scene | contact sheet |
|---|---|
| flowers | `docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.png` |
| garden | `docs/car_model/assets/v106_qualitative/garden_frame00000_crop_contact_sheet.png` |
| garden best crop | `docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png` |
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
| `scripts/car_model/collect_v110_strict_split_report.py` | collects clean/v106/v110 strict-split metrics into Markdown/JSON summaries |

## Smoke and Static Checks

The current report package records these checks as passing:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py \
  scripts/car_model/smoke_test_v110_strict_runner_args.py \
  scripts/car_model/smoke_test_v111_runner_args.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v110_strict_runner_args.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v111_runner_args.py
```

## Current Local Runtime Roots

The repo stores lightweight report artifacts. Large render/model trees remain local:

| root | meaning |
|---|---|
| `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625` | detached model package, parent/candidate/gated renders, scene `results.json` |
| `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_field` | v106 field tensors |
| `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625` | v110/v110b strict-split working directory |
| `/dev/shm/peilincai_spcarnet_v111_end_to_end_strict_parent_gate_20260625` | v111 end-to-end strict working directory |

These roots are not committed because they contain large render/model artifacts.

## One-Line Status

`v106` is the current quality line and beats the local clean MeshSplatting baseline on the assembled selected full9 table. `v110/v110b/v111` are the fairness/safety validation line; they have exposed a real out-of-trajectory gate-generalization weakness and are not yet promoted as the final paper method.
