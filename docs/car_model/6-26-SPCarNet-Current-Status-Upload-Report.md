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

## Latest Live Experiment State

This snapshot was checked after the report package commit `87f8387`.

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

## Best PPT Story

The cleanest story is:

```text
MeshSplatting gives a strong parent render.
SPCarNet treats the mesh surface as an address space for train-evidence residuals.
v106 adds a base-preserving mixture of detail and boundary residual experts.
The result beats the local clean MeshSplatting baseline on selected full9.
Strict split experiments then reveal where naive candidate gates fail.
The current next step is candidate-side improvement plus lower-memory long-run execution.
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

After that, the required closure sequence is:

1. rerun v110 counter field/render/eval;
2. finish or rerun v110 bonsai, v111 flowers, and v114 garden;
3. recollect `spcarnet_v110_v111_v114_package`;
4. compare clean baseline, v106 parent, improved candidate, and fallback/gate ablations;
5. update README/report tables only after completed metrics exist.

## Final Status

`NOT COMPLETE`.

The report package is uploaded and useful for PPT preparation, but the research loop is not fully closed because strict branch long jobs are unfinished or failed, and the newest candidate-side method has not yet delivered completed metrics.
