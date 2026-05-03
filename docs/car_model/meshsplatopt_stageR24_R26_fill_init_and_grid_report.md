# MeshSplatOpt R24-R26 Fill Initialization And Grid Fill Report

Date: 2026-05-03

## Summary

Stages R24-R26 tested three fixes for the Stage R22 boundary-fill weakness:

1. initialize new `FILL_PATCH` face runtime fields from nearest existing faces instead of zeros;
2. diagnose whether post-edit densification can recover the fill;
3. replace the one-center fan fill with a denser plane-grid Delaunay fill.

All gate/integrity checks passed, and all medium/short training runs used W&B online on GPU 4. The results are useful but not yet headline quality: the edits are counterfactually safe and auditable, but they do not beat the strong 4000-iteration parking baseline.

Decision: `FILL_INIT_GRID_ENGINEERING_PASS_MEDIUM_REPAIR_FAIL`.

## Implementation

### R24: nearest-face field initialization

Changed `ss3dm_prior/meshsplatopt/checkpoint_adapter.py`:

- added nearest old-face lookup for appended `FILL_PATCH` faces;
- initializes appended `importance_score`, `image_size`, and `pixel_count` from the nearest old face;
- default scale is `0.5`, controlled by edit attributes:
  - `face_field_init`
  - `face_field_init_scale`
- retains nearest-vertex initialization for radiance fields.

Smoke:

- `outputs/carnet/meshsplatopt/stageR24_00_checkpoint_adapter_nearest_face_stats_smoke/checkpoint_adapter_smoke_report.json`
- status: `PASS`

### R25: densification-on diagnostic

R25 intentionally removed the freeze/skip recovery restrictions after the edit to test whether free densification can repair the patch.

Result: `FAIL`.

The run exploded topology:

- final triangles: `5,889,468`
- final vertices: `4,964,968`
- PSNR: `12.031141`
- SSIM: `0.310603`
- LPIPS: `0.641519`

This confirms that unbounded post-repair densification is not a valid recovery strategy.

### R26: plane-grid Delaunay fill

Added `scripts/car_model/meshsplatopt_expand_boundary_fill_to_grid.py`.

The script expands an existing fan `FILL_PATCH` edit into a denser, checkpoint-compatible plane-grid fill:

- recovers the ordered boundary loop from fan faces;
- fits a local plane to boundary vertices;
- samples interior grid points inside the projected polygon;
- triangulates by 2D Delaunay and keeps triangles whose centroids lie inside the polygon;
- emits a normal `FILL_PATCH` JSON with global checkpoint indices.

Parking R26 grid fill:

- source edit: R22 boundary fan fill
- loop vertices: `6`
- added vertices: `51`
- added faces: `106`
- projected area: `37.097569`
- grid spacing: `0.879128`

## Gate Results

### R24 nearest-face init gate

Artifact:

- `outputs/carnet/meshsplatopt/stageR24_01_parking_boundary_fill_nearest_face_init_gate/render_backed_checkpoint_gate_report.json`

Result:

- status: `PASS`
- topology: `+1` vertex, `+6` triangles
- PSNR delta: `+0.000097`
- SSIM delta: `-0.00000039`
- LPIPS delta: `+0.00000364`
- AbsRel delta: `0.0`
- DepthMAE delta: `0.0`
- normal delta: `+0.00000110`

### R26 grid-fill gate

Artifact:

- `outputs/carnet/meshsplatopt/stageR26_02_parking_boundary_grid_fill_gate/render_backed_checkpoint_gate_report.json`

Result:

- status: `PASS`
- topology: `+51` vertices, `+106` triangles
- PSNR delta: `+0.0000925`
- SSIM delta: `-0.00000003`
- LPIPS delta: `+0.00000262`
- AbsRel delta: `0.0`
- DepthMAE delta: `0.0`
- normal delta: `+0.00000103`

## Recovery Results

### Short 2200

| Method | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|---:|---:|---:|
| R17 baseline | 12.331465 | 0.298222 | 0.622323 | 0.409263 | 4.300273 | 52.595639 |
| R22 fan fill | 12.354150 | 0.298658 | 0.621934 | 0.410232 | 4.302468 | 52.328850 |
| R24 fan fill + nearest-face stats | 12.347798 | 0.297994 | 0.621984 | 0.409399 | 4.302556 | 52.568240 |
| R26 grid fill + nearest-face stats | 12.347396 | 0.298338 | 0.622142 | 0.408940 | 4.303084 | 52.550777 |

W&B:

- R24 short: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1iam7x3c`
- R26 short: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bg5cflp8`

### Medium 4000

| Method | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|---:|---:|---:|
| R15 baseline | 14.251087 | 0.383800 | 0.569749 | 0.324794 | 3.636891 | 51.043451 |
| R20 residual snap | 14.207231 | 0.383298 | 0.570288 | 0.323844 | 3.589209 | 51.225949 |
| R22 fan fill | 14.224104 | 0.381926 | 0.570877 | 0.329337 | 3.645573 | 51.527010 |
| R26 grid fill | 14.212496 | 0.383164 | 0.570729 | 0.329141 | 3.667578 | 51.594204 |

W&B:

- R25 diagnostic: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/hkzqqedj`
- R26 medium: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/phki0fj4`

## Interpretation

The core positive result is engineering maturity:

- checkpoint `FILL_PATCH` can now initialize topology, vertex radiance, and face statistics coherently;
- denser fill geometry can be generated and passed through the existing render/geometry gate;
- W&B-backed short and medium runs finish without checkpoint corruption;
- unbounded recovery has been falsified as a strategy.

The core negative result is scientific:

- neither nearest-face field initialization nor denser plane-grid fill resolves the public-scene medium-budget gap;
- the best short result remains R22 fan fill, but its medium result fails;
- the current real-scene fill proposal is not yet a top-conference headline result.

## Next Required Fix

The next high-probability direction is not more local triangulation. The repair stack needs true recovery supervision:

1. cache pre-edit teacher renders from the unedited baseline model, not the already-edited candidate;
2. apply unedited-region RGB/depth distillation during external edit recovery;
3. optimize edited regions against image evidence and sparse geometry separately;
4. add an edit-region metric so global PSNR does not hide local repair.

Until that is implemented, topology-adding repairs are safe but underpowered.
