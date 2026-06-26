# SPCarNet Current Technical Report for Mentor/PPT

Date: 2026-06-25

## Executive Summary

SPCarNet is currently best described as a MeshSplatting-compatible surface residual system. It keeps MeshSplatting's trained geometry and renderer as the parent, then attaches a guarded residual field to visible mesh triangles. The current verified quality line is `v106 POD-MoE base-preserve`; the current strict-fairness line is `v110/v110b/v111`.

The main progress since the large-method rebuild started is concrete:

- a local clean MeshSplatting baseline table is available under the official-style reproduction root;
- a full9 assembled v106 result package is now stored in the repo;
- v106 improves over the selected local clean MeshSplatting baseline on mean PSNR, SSIM, and LPIPS;
- strict train/even -> train/odd -> test interfaces were added to the field builder, delta-bank builder, parent gate, and orchestration scripts;
- v110/v110b exposed an important weakness: a gate that looks good on train/odd can still harm held-out test relative to the v106 parent;
- v113b repairs that weakness on flowers/garden by adding lower-tail metric checks and a target-GT-free out-of-trajectory support certificate;
- v111 now exists as the end-to-end strict runner where even the parent field is rebuilt from train/all rather than inherited from a target-sidecar artifact.

The honest current conclusion is: SPCarNet has a real, measurable improvement line over the local clean baseline, and the strict-gate branch now has a stronger safety repair. The paper-final claim is still not closed because v113b restores unsafe candidates to v106 rather than improving beyond v106.

## Method in Plain Language

MeshSplatting renders a trained mesh/point representation directly. SPCarNet adds a correction layer:

```text
MeshSplatting parent render
  + triangle-addressed residual evidence
  + reliability/gate logic
  -> corrected render
```

The triangle mesh acts like an address book. If a pixel comes from a known triangle, SPCarNet can store and replay a small color correction for that triangle and viewing condition. This is different from a free image-space filter because the correction is attached to scene surface structure.

The current v106 correction is:

```text
v106 residual =
  stable base residual
  + detail expert residual
  + occlusion/boundary expert residual
```

The parent-preserving gate is:

```text
output = parent + mask * (candidate - parent)
```

If the candidate is not proven safe, the mask can be zero and the output becomes exactly the parent.

## Current Quantitative Status

### v106 vs Local Clean MeshSplatting

Source files:

```text
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md
```

| method | scenes | PSNR | SSIM | LPIPS | mean delta vs clean |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline |
| v104c shrink view-affine | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 |

v106 vs v104c:

| dPSNR | dSSIM | dLPIPS |
|---:|---:|---:|
| +0.002181 | +0.000103 | -0.000112 |

Interpretation: v106 is the best verified quality-line table in the current package. The improvement over clean is large because it includes the residual-field endpoint line; the incremental gain of v106 over v104c is small but consistently favorable on the assembled table.

### Representative Four-Scene Baseline Snapshot

These values use the local clean baseline from:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/<scene>/results.json
```

| scene | clean MeshSplatting | v106 parent | v106 minus clean |
|---|---:|---:|---:|
| flowers | 19.682257 / 0.511822 / 0.394563 | 20.077723 / 0.531240 / 0.374393 | +0.395466 / +0.019418 / -0.020170 |
| garden | 25.029211 / 0.780035 / 0.201314 | 25.790945 / 0.799382 / 0.174480 | +0.761734 / +0.019347 / -0.026834 |
| counter | 26.751774 / 0.862055 / 0.252003 | 27.499645 / 0.867521 / 0.238847 | +0.747871 / +0.005466 / -0.013156 |
| bonsai | 28.895233 / 0.896400 / 0.259493 | 30.316090 / 0.907520 / 0.230050 | +1.420856 / +0.011120 / -0.029443 |

## Strict-Split Diagnostics

The v110 protocol separates fitting, selection, and final evaluation:

```text
fit candidate field on train/even
calibrate parent gate on train/odd
evaluate on test
```

### Flowers

Default v110 gate:

| method | PSNR | SSIM | LPIPS | relation |
|---|---:|---:|---:|---|
| clean MeshSplatting | 19.682257 | 0.511822 | 0.394563 | baseline |
| v106 parent | 20.077723 | 0.531240 | 0.374393 | current parent |
| default v110 gate | 19.966076 | 0.522843 | 0.380387 | beats clean but regresses vs v106 |

v110b gain-margin gate on flowers falls back to parent:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v110b flowers | 20.077723 | 0.531240 | 0.374393 |

### Garden

The corrected v110b gate accepted a nonzero mask on train/odd, but held-out test still regressed relative to v106:

| method | PSNR | SSIM | LPIPS | relation |
|---|---:|---:|---:|---|
| clean MeshSplatting | 25.029211 | 0.780035 | 0.201314 | baseline |
| v106 parent | 25.790945 | 0.799382 | 0.174480 | current parent |
| v110b garden | 25.430321 | 0.783703 | 0.186970 | beats clean but regresses vs v106 |

The v110b two-scene diagnostic is stored here:

```text
docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md
```

v113b adds that first stronger risk model. It repairs the garden regression by falling back when target masks land outside the empirical train/odd-to-train/even camera support:

```text
docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md
```

Interpretation: strict train/odd calibration is not enough by itself. v113b makes the gate non-regressive on the two completed representative scenes, but it still does not replace v106 with a better nonzero candidate.

## Qualitative Assets

The current cloneable qualitative package is:

```text
docs/car_model/assets/v106_qualitative/
```

Available contact sheets:

| scene | file |
|---|---|
| flowers | `flowers_frame00001_bestcrop_contact_sheet.png` |
| garden | `garden_frame00000_crop_contact_sheet.png` |
| garden best crop | `garden_frame00004_bestcrop_contact_sheet.png` |
| room | `room_frame00029_bestcrop_contact_sheet.png` |
| treehill | `treehill_frame00010_bestcrop_contact_sheet.png` |

These are better PPT assets than showing only aggregate numbers, but they should be presented carefully: the visual improvement is localized and subtle, not a dramatic full-frame reconstruction change.

## What Was Implemented

| component | file | status |
|---|---|---|
| train/test delta-bank split | `scripts/car_model/run_v102_preprojected_delta_scene.py` | implemented |
| field-builder view subsets | `scripts/car_model/build_v105_evidence_gated_mixture_field.py` | implemented |
| render-realized parent gate calibration subset | `scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py` | implemented |
| strict v110 runner | `scripts/car_model/run_v110_strict_split_parent_gate_scene.py` | implemented and smoke-tested |
| end-to-end strict v111 runner | `scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py` | implemented and smoke-tested |
| v110 collector | `scripts/car_model/collect_v110_strict_split_report.py` | implemented |

## Commands and Repro Pointers

Smoke/static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py \
  scripts/car_model/smoke_test_v110_strict_runner_args.py \
  scripts/car_model/smoke_test_v111_runner_args.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v110_strict_runner_args.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v111_runner_args.py
```

Representative v110 run command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  --scene flowers \
  --gpu <gpu> \
  --merge_model_results \
  --wandb \
  --output_root /dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625
```

Representative v111 command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py \
  --scene flowers \
  --gpu <gpu> \
  --merge_model_results \
  --wandb \
  --output_root /dev/shm/peilincai_spcarnet_v111_end_to_end_strict_parent_gate_20260625
```

## Current Running or Unfinished Items

| item | status |
|---|---|
| v110 counter | candidate field build still running in the local `/dev/shm` workspace at the last audit |
| v110 bonsai | candidate field build relaunched after stale preflight issue; old report is not a final result |
| v111 flowers | parent train/all field build running; no completed end-to-end result yet |
| garden v110b | completed as a negative strict-gate diagnostic |

## Weaknesses to State Honestly

1. v106 is the current quality line, but part of its assembled full9 provenance is mixed from full9/counter/hard-triad report roots. It is acceptable for internal progress reporting if provenance is shown; a final paper table should be regenerated from one frozen pipeline.
2. v110/v110b do not yet improve over v106 under strict split. They are useful because they revealed a real gate-generalization failure.
3. Visual improvements are localized. The method should be presented with crop/error panels, not only full-frame comparisons.
4. The current method does not yet demonstrate triangle-count reduction superiority as a primary claim. It mainly improves rendering quality through residual appearance fields while inheriting MeshSplatting topology.
5. The MeshSplatting paper-number comparison must be treated as a protocol bridge, not a definitive paper-table victory, unless scene set, split, checkpoint, and metric implementation are all matched.

## Suggested PPT Story

1. Start from the problem: MeshSplatting is strong, but leaves localized surface/view-dependent residual errors.
2. Introduce SPCarNet: use the mesh surface as an address space for guarded residual correction.
3. Show v106 numbers: full9 selected table beats local clean MeshSplatting in PSNR/SSIM/LPIPS.
4. Show qualitative crops: improvements are local and surface-attached.
5. Be candid about strict validation: v110/v110b discovered out-of-trajectory gate risk.
6. End with next research step: replace single split-gate trust with cross-view risk certificates or trajectory-aware uncertainty.

## Final Current Assessment

Compared with the start of the major rebuild, the project has moved from ad hoc candidate tuning to a documented, reproducible method ladder with clear result packages, strict split interfaces, and known failure modes. For a mentor PPT, the progress is substantial and defensible.

For a top-conference final method, the work is not complete. The current quality headline is real but not yet strong enough to close the paper alone, and the strict-split branch needs a stronger mechanism that improves over v106 instead of merely preserving or regressing from it.
