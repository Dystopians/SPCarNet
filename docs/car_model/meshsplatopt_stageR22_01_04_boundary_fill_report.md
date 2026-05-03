# MeshSplatOpt R22.01-R22.04 Boundary Fill Report

Date: 2026-05-03

## Objective

Move beyond snap-only repair by validating a real checkpoint `FILL_PATCH` path on a parking boundary loop, including held-out gate, short W&B recovery, and medium W&B recovery.

## Implementation

Added:

`scripts/car_model/meshsplatopt_select_checkpoint_boundary_fill_edit.py`

The selector:

- loads checkpoint mesh vertices/faces;
- finds boundary loops;
- filters by loop length and XY area;
- creates a checkpoint-compatible `FILL_PATCH` edit using a centroid vertex and fan triangles;
- writes selection report, edit JSON, and fill certificate.

The existing checkpoint adapter already supports `FILL_PATCH` by appending vertices/faces and initializing new vertex radiance from nearest existing vertices.

## Candidate

Selected parking boundary loop:

- loop count: `48858`
- candidates after filtering: `4545`
- selected loop vertices: `6`
- selected XY area: `24.723803`
- inserted vertices: `1`
- inserted faces: `6`
- topology: `782982 -> 782988` triangles, `820107 -> 820108` vertices

Edit:

`outputs/carnet/meshsplatopt/stageR22_02_parking_boundary_fill_medium_selection/selected_boundary_fill_edit.json`

Certificate:

- boundary loop support: `true`
- neighboring surface support: `true`
- prior only: `false`
- free-space risk: `0.1`
- expected topology cost: `6`

## Held-Out Gate

Gate output:

`outputs/carnet/meshsplatopt/stageR22_02_parking_boundary_fill_medium_selection/gate/render_backed_checkpoint_gate_report.json`

Result: `PASS`

| metric | delta |
| --- | ---: |
| triangles | `+6` |
| vertices | `+1` |
| PSNR | `+0.000097` |
| SSIM | `-0.00000039` |
| LPIPS | `+0.00000364` |
| AbsRel | `0.0` |
| Depth MAE | `0.0` |
| Normal mean deg | `+0.00000110` |

## Short Recovery 2000->2200

Output:

`outputs/carnet/meshsplatopt/stageR22_03_parking_boundary_fill_medium_freeze_skip_delaunay_2000to2200`

W&B:

`https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/jzxzz4g2`

| run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal mean deg ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline continuation | `12.331465` | `0.298222` | `0.622323` | `0.409263` | `4.300273` | `52.595639` |
| single residual snap | `12.342549` | `0.298893` | `0.622299` | `0.408892` | `4.302941` | `52.354489` |
| patch residual snap | `12.329646` | `0.298382` | `0.622157` | `0.409988` | `4.303037` | `52.586082` |
| boundary fill | `12.354150` | `0.298658` | `0.621934` | `0.410232` | `4.302468` | `52.328850` |

Short-run decision: `FILL_SHORT_RECOVERY_APPEARANCE_NORMAL_PASS_DEPTH_FAIL`.

## Medium Recovery 2000->4000

Output:

`outputs/carnet/meshsplatopt/stageR22_04_parking_boundary_fill_medium_freeze_skip_delaunay_2000to4000`

W&B:

`https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1tqd66ah`

| run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal mean deg ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 2000->4000 | `14.251087` | `0.383800` | `0.569749` | `0.324794` | `3.636891` | `51.043451` |
| residual snap 2000->4000 | `14.207231` | `0.383298` | `0.570288` | `0.323844` | `3.589209` | `51.225949` |
| boundary fill 2000->4000 | `14.224104` | `0.381926` | `0.570877` | `0.329337` | `3.645573` | `51.527010` |

Medium-run decision: `FILL_MEDIUM_RECOVERY_FAIL`.

## Decision

`BOUNDARY_FILL_GATE_PASS_SHORT_PROMISING_MEDIUM_FAIL`

R22 fixes a major architectural gap: real checkpoint `FILL_PATCH` now has a selector, gate, W&B short recovery, and W&B medium recovery evidence. The primitive is trainable and counterfactually safe for a small real boundary fill. However, the naive centroid fan fill is not yet a robust quality improvement under medium training. The next fill selector must be residual/depth-aware and place/fair inserted geometry using local surface evidence rather than loop centroid alone.
