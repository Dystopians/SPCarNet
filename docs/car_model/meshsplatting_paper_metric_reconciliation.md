# MeshSplatting Paper Metric Reconciliation

Date: 2026-05-07

## Why This Exists

The previous internal comparison treated a train-selected clean checkpoint as the paper-facing MeshSplatting baseline. That was wrong for reviewer-facing claims: train metrics can prefer longer continuations that overfit train views while doing worse on held-out test views. The corrected ELA12 audit now selects the coherent clean checkpoint by held-out test score and keeps train scores only as diagnostics.

This document separates three different baselines that must not be mixed:

1. **Internal clean checkpoint baseline**: our local clean MeshSplatting checkpoints on the current method-artifact scene set.
2. **Corrected ELA12 selected-clean baseline**: the held-out-test-selected local clean checkpoint used in the current README table.
3. **Paper MeshSplatting baseline**: the official MeshSplatting paper protocol on the full Mip-NeRF360 benchmark.

## What 24.78 Means In The MeshSplatting Paper

The MeshSplatting paper reports `24.78 / 0.310 / 0.728` for PSNR / LPIPS / SSIM on Mip-NeRF360. This is not one scene. It is the arithmetic mean over the nine Mip-NeRF360 scenes:

| group | scenes | paper PSNR |
|---|---|---:|
| outdoor | Bicycle, Flowers, Garden, Stump, Treehill | 23.04, 19.34, 24.70, 24.78, 20.53 |
| indoor | Room, Counter, Kitchen, Bonsai | 28.52, 26.51, 27.42, 28.19 |
| mean | all 9 scenes | 24.781 |

The corresponding mean LPIPS and SSIM from the paper's per-scene tables are `0.3108` and `0.7282`, matching the headline table after rounding.

Important protocol notes from the paper/code:

- Official Mip-NeRF360 full evaluation uses outdoor `images_4`, indoor `images_2`, `--eval`, and final 30k render/metrics.
- The official repository says exported RGB-only PLY files lose about 2 dB PSNR on average; the paper's highest visual quality is the viewer/SH/supersampling path, not a plain RGB-only PLY comparison.
- Their Table 11 reports `w/o SH` as `-2.07 PSNR`, which is consistent with the repo warning that RGB-only exports are not the paper-quality rendering path.

Sources:

- arXiv abstract and paper page: https://arxiv.org/abs/2512.06818
- official repository evaluation instructions and PLY warning: https://github.com/meshsplatting/mesh-splatting

## Current Local Status

The corrected ELA12 audit is **not yet a same-protocol reproduction of the MeshSplatting paper table**. It covers only the current method-artifact set:

- Mip-NeRF360: `bonsai`, `room`, `counter`
- ETH3D: `courtyard`
- phone COLMAP: `parking_phone_tiny`

That set is useful for internal method development, but it cannot be reported as the Mip-NeRF360 benchmark. In particular, the local clean checkpoints currently used by ELA12 are not the official 9-scene, 30k, full_eval protocol. Their numbers are much lower than the paper's per-scene table for several Mip-NeRF360 scenes, so any direct "beats MeshSplatting 24.78" claim would be invalid.

## Same-Protocol Reproduction Plan

The same-protocol benchmark must be built in this order:

1. Reproduce the official clean MeshSplatting baseline with the official command structure:

```bash
python full_eval.py \
  --mipnerf360 /data/peilincai/mesh_datasets/mipnerf360 \
  --output_path outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k
```

2. Because the local dataset initially had only 7 / 9 Mip-NeRF360 scenes, first report whatever scenes are complete and separately mark incomplete scenes. `flowers` and `treehill` have now been imported from the official `refraw360` release, so the queue can move to the full nine-scene benchmark.
3. For each reproduced clean scene, run our method from the exact same input images, split, resolution, and final render path.
4. Compare only under the same render path: if the paper uses SH/viewer/supersampling, our method must use the same renderer or explicitly report an RGB-only/exported-mesh ablation for both sides.
5. Use the official 9-scene mean only after all nine scene artifacts exist.

## Fixed-Budget Method Protocol

The method comparison must not gain credit from extra recovery iterations. The reviewer-facing fixed-budget protocol is therefore:

1. Train clean MeshSplatting with official Mip-NeRF360 settings and save both `26000` and `30000`.
2. Apply the fixed CSEF adaptive compaction policy only to the clean `26000` checkpoint.
3. Recover the compact model from `26000 -> 30000` with the same source images, split, and image scale. The default method policy also uses a train-only ATR parent-render rollback cache from clean `26000`, so recovery is penalized only where it becomes worse than the parent on train-view residual tails.
4. Compare method `30000` against clean `30000`.

Implemented interfaces:

- clean queue: `scripts/car_model/run_paper_m360_official_clean30k_available7.sh`
- fixed-budget method queue: `scripts/car_model/run_paper_m360_fixedbudget_method_available7.sh`
- fixed-budget method collector: `scripts/car_model/collect_paper_m360_fixedbudget_method_metrics.py`

The queue defaults now follow the paper's nine-scene order: `bicycle`, `flowers`, `garden`, `stump`, `treehill`, `room`, `counter`, `kitchen`, `bonsai`. They still skip scenes whose source or checkpoints are missing.

Any `30000 -> 34000` method recovery is only a diagnostic longer-budget experiment unless a clean `34000` continuation is also evaluated.

## Garden Calibration Result

The first official-protocol reproduction run is complete:

| scene | local PSNR | paper PSNR | local SSIM | paper SSIM | local LPIPS | paper LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| garden | 24.697647 | 24.70 | 0.761070 | 0.762 | 0.216561 | 0.217 |

The deltas are `-0.002353` PSNR, `-0.000930` SSIM, and `-0.000439` LPIPS.  This is close enough to treat the local official clean protocol as calibrated for Garden.  It is still only one scene; the paper headline is the nine-scene mean, so no method claim should use this as a full benchmark result.

The same checkpoint's sparse COLMAP geometry record is:

- AbsRel: `0.007413`
- Depth MAE: `0.112305`
- normal mean angle: `30.568113` degrees
- valid sparse samples: `11998`

W&B runs:

- training: `el3kj209`
- metric collection: `2vlfanty`

## Active Run

The full official clean queue has now been launched:

- Scene order: `bicycle`, `flowers`, `garden`, `stump`, `treehill`, `room`, `counter`, `kitchen`, `bonsai`
- Clean checkpoints saved: `26000` and `30000`
- Output root: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k`
- W&B group: `paper_m360_official_clean30k`
- Current first scene: `bicycle`, W&B run `c08zvifs`

Note: the completed Garden calibration run was launched before the clean queue was updated to save the `26000` split checkpoint. It is valid for clean paper-protocol reproduction, but fixed-budget method validation on Garden requires the queue rerun to provide the `26000` clean checkpoint.

## Claim Discipline

Allowed today:

- "On the current internal artifact set, the corrected ELA12 audit is 5 / 5 aggregate strict full-pass against held-out-test-selected clean checkpoints."
- "Parking OUT2 improves aggregate PSNR/SSIM/LPIPS over clean22000 and clean30000, but it still has one PSNR tail view versus clean22000."

Not allowed today:

- "We beat the MeshSplatting paper Mip-NeRF360 result."
- "We are above PSNR 24.78 on Mip-NeRF360."
- "We have a full 9-scene Mip-NeRF360 same-protocol comparison."
- "Our method beats the calibrated Garden baseline." Garden currently calibrates only the clean baseline; the fixed-budget method row has not been run under this protocol yet.
